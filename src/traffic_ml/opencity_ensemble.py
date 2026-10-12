"""Fixed-seed predictive ensembles and epistemic mutual-information diagnostics."""
from pathlib import Path

import numpy as np
import pandas as pd

from .opencity_metrics import evaluate_open


def aggregate_seed_ensembles(root, bins=15):
    """Average all registered seed probabilities; never select a favorable subset."""
    root = Path(root)
    rows = []
    for model_dir in root.glob("*/*"):
        if not model_dir.is_dir() or model_dir.parent.name in {"figures", "tuning"}:
            continue
        files = sorted(model_dir.glob("seed_*/predictions.npz"))
        if len(files) < 3:
            continue
        bundles = [np.load(path) for path in files]
        targets, times = bundles[0]["targets"], bundles[0]["time_ids"]
        if any(not np.array_equal(item["targets"], targets) or
               not np.array_equal(item["time_ids"], times) for item in bundles[1:]):
            continue
        stack = np.stack([item["probabilities"] for item in bundles])
        mean_probability = stack.mean(0)
        predictive_entropy = -(mean_probability * np.log(mean_probability.clip(1e-9))).sum(1)
        expected_entropy = -(stack * np.log(stack.clip(1e-9))).sum(2).mean(0)
        mutual_information = predictive_entropy - expected_entropy
        metrics = evaluate_open(targets, mean_probability, predictive_entropy, times, bins)
        metrics.update({"target_city": model_dir.parent.name, "model": model_dir.name,
                        "ensemble_size": len(files),
                        "mean_epistemic_mi": float(mutual_information.mean()),
                        "p95_epistemic_mi": float(np.quantile(mutual_information, .95))})
        rows.append(metrics)
        np.savez_compressed(model_dir / "ensemble_predictions.npz",
                            probabilities=mean_probability, targets=targets, time_ids=times,
                            predictive_entropy=predictive_entropy,
                            expected_entropy=expected_entropy,
                            mutual_information=mutual_information)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "ensemble_results.csv", index=False)
    return frame
