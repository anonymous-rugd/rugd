"""Validation-only temperature scaling and split-conformal prediction sets."""
from __future__ import annotations

import numpy as np
import torch


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> TemperatureScaler:
        x = torch.tensor(logits, dtype=torch.float64)
        target = torch.tensor(y, dtype=torch.long)
        log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], max_iter=100, line_search_fn="strong_wolfe")
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(x / log_t.exp(), target)
            loss.backward()
            return loss
        optimizer.step(closure)
        self.temperature = float(log_t.exp().detach().clamp(.05, 20))
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        z = logits / self.temperature
        z -= z.max(1, keepdims=True)
        exp = np.exp(z)
        return exp / exp.sum(1, keepdims=True)


class APSConformal:
    """Randomization-free Adaptive Prediction Sets with finite-sample quantile."""
    def __init__(self, alpha: float = .1) -> None:
        self.alpha, self.qhat = alpha, None

    @staticmethod
    def scores(p: np.ndarray, y: np.ndarray) -> np.ndarray:
        order = np.argsort(-p, axis=1)
        ranked = np.take_along_axis(p, order, axis=1)
        cumulative = ranked.cumsum(1)
        rank_y = (order == y[:, None]).argmax(1)
        return cumulative[np.arange(len(y)), rank_y]

    def fit(self, p: np.ndarray, y: np.ndarray) -> APSConformal:
        scores = self.scores(p, y)
        level = min(1., np.ceil((len(scores) + 1) * (1 - self.alpha)) / len(scores))
        self.qhat = float(np.quantile(scores, level, method="higher"))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        if self.qhat is None:
            raise RuntimeError("fit must be called first")
        order = np.argsort(-p, axis=1)
        cumulative = np.take_along_axis(p, order, axis=1).cumsum(1)
        ranked_set = cumulative - np.take_along_axis(p, order, axis=1) < self.qhat
        result = np.zeros_like(p, dtype=bool)
        np.put_along_axis(result, order, ranked_set, axis=1)
        return result

    def evaluate(self, p: np.ndarray, y: np.ndarray) -> dict[str, float]:
        sets = self.predict(p)
        return {"conformal_coverage": float(sets[np.arange(len(y)), y].mean()),
                "conformal_mean_set_size": float(sets.sum(1).mean()),
                "conformal_severe_coverage": float(sets[y == 2, 2].mean()) if (y == 2).any() else float("nan")}
