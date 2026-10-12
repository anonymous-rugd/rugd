import numpy as np
import torch

from traffic_ml.calibration import APSConformal, TemperatureScaler
from traffic_ml.losses import effective_number_weights
from traffic_ml.metrics import evaluate_probabilities
from traffic_ml.statistics import paired_score_inference


def test_metrics_calibration_and_conformal():
    rng = np.random.default_rng(1); logits = rng.normal(size=(120, 3)); y = rng.integers(0, 3, 120)
    p = TemperatureScaler().fit(logits[:60], y[:60]).transform(logits)
    metrics = evaluate_probabilities(y, p)
    cp = APSConformal(.1).fit(p[:60], y[:60]); metrics.update(cp.evaluate(p[60:], y[60:]))
    assert 0 <= metrics["ece"] <= 1 and metrics["conformal_mean_set_size"] >= 1
    assert np.isclose(effective_number_weights(torch.tensor([10, 20, 30])).mean(), 1)



def test_paired_score_inference_keeps_city_as_a_group():
    reference = np.array([.8, .82, .81, .7, .72, .71])
    baseline = np.array([.7, .71, .70, .6, .61, .62])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    result = paired_score_inference(reference, baseline, groups, repeats=200, seed=3)
    assert result["wins"] == 6 and result["n_pairs"] == 6
    assert result["group_wins"] == 2 and result["n_groups"] == 2
    assert result["rank_biserial"] == 1.0
    assert result["ci95_low"] > 0
