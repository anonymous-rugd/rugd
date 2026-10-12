"""Protocol-level tests for OpenCity splits, source statistics, and model ablations."""
import numpy as np
import torch

from traffic_ml.opencity_data import OpenCityStore, fit_source_statistics, temporal_indices
from traffic_ml.opencity_experiment import _corrupt_flat, training_objective
from traffic_ml.opencity_models import build_open_model

CFG = {
    "target": {"severe_quantile": .1, "warning_quantile": .3},
    "model": {"hidden_dim": 8, "dropout": 0., "diffusion_steps": 1},
    "training": {"focal_gamma": 2., "evidence_kl_weight": .001, "domain_weight": .1,
                 "auxiliary_weight": .2},
}


def _store(value):
    speed = np.tile(np.linspace(value, value + 10, 100, dtype=np.float32)[:, None], (1, 4))
    adjacency = np.eye(4, dtype=np.float32)
    return OpenCityStore("x", speed, adjacency, np.zeros((4, 4), np.float32), np.arange(100))


def test_temporal_split_is_ordered_and_embargoed():
    split = temporal_indices(200, 12, 1, (.6, .1, .1, .2), 12)
    assert split.train.max() < split.calibration.min() - 11
    assert split.calibration.max() < split.validation.min() - 11
    assert split.validation.max() < split.test.min() - 11


def test_source_statistics_do_not_consult_target():
    source = _store(10.)
    target = _store(999.)
    split = temporal_indices(100, 6, 1, (.6, .1, .1, .2), 2)
    before = fit_source_statistics([source], [split], CFG)
    target.speed[:] = 1.
    after = fit_source_statistics([source], [split], CFG)
    assert before == after


def test_all_deep_variants_accept_unseen_node_count():
    batch = {"history": torch.randn(2, 5, 6), "history_mask": torch.ones(2, 5, 6).bool(),
             "static": torch.randn(2, 5, 4), "calendar": torch.randn(2, 4)}
    adjacency = torch.eye(5)
    variants = ("gru", "graph_gru", "rugd", "rugd_no_graph", "rugd_no_topology",
                "rugd_no_domain", "rugd_no_evidence", "rugd_no_focal",
                "rugd_no_class_balance")
    for name in variants:
        output = build_open_model(name, CFG, 2)(batch, adjacency)
        assert output["logits"].shape == (2, 5, 3)
        assert torch.isfinite(output["logits"]).all()


def test_flat_corruptions_preserve_static_features():
    x = np.arange(4 * 20, dtype=float).reshape(4, 20)
    for kind, level in (("missing", .5), ("noise", .2), ("shift", 1)):
        changed = _corrupt_flat(x, kind, level, history_steps=6)
        np.testing.assert_array_equal(changed[:, 12:], x[:, 12:])



def test_rugd_local_evidence_is_positive():
    model = build_open_model("rugd", CFG, 2)
    batch = {"history": torch.randn(2, 5, 6),
             "history_mask": torch.ones(2, 5, 6, dtype=torch.bool),
             "static": torch.randn(2, 5, 4), "calendar": torch.randn(2, 4)}
    output = model(batch, torch.eye(5))
    assert torch.all(output["pre_alpha"] >= 1.0)
    assert torch.isfinite(output["pre_logits"]).all()


def test_missing_history_produces_finite_evidence():
    model = build_open_model("rugd", CFG, 2)
    batch = {"history": torch.zeros(2, 5, 6),
             "history_mask": torch.zeros(2, 5, 6, dtype=torch.bool),
             "static": torch.randn(2, 5, 4), "calendar": torch.randn(2, 4)}
    output = model(batch, torch.eye(5))
    assert torch.isfinite(output["pre_alpha"]).all()
    assert torch.isfinite(output["logits"]).all()


def test_new_objective_is_finite_and_backpropagates():
    model = build_open_model("rugd", CFG, 2)
    batch = {"history": torch.randn(2, 5, 6),
             "history_mask": torch.ones(2, 5, 6, dtype=torch.bool),
             "target": torch.randint(0, 3, (2, 5)), "city": torch.tensor([0, 1]),
             "static": torch.randn(2, 5, 4), "calendar": torch.randn(2, 4)}
    loss = training_objective(model, batch, torch.eye(5), .5, torch.ones(3), CFG)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert model.pre_evidence.weight.grad is not None

def test_modern_models_and_revised_rugd_accept_unseen_graphs():
    cfg = {
        **CFG,
        "data": {"history_steps": 12},
        "model": {**CFG["model"], "attention_heads": 2, "attention_layers": 1,
                  "patch_length": 3},
        "training": {**CFG["training"], "recovery_mask_rate": .25,
                     "recovery_weight": .1},
    }
    batch = {
        "history": torch.randn(2, 5, 12),
        "history_mask": torch.ones(2, 5, 12, dtype=torch.bool),
        "target": torch.randint(0, 3, (2, 5)),
        "city": torch.tensor([0, 1]),
        "static": torch.randn(2, 5, 4),
        "calendar": torch.randn(2, 4),
        "history_tod": torch.randint(0, 48, (2, 12)),
        "history_dow": torch.randint(0, 7, (2, 12)),
    }
    adjacency = torch.eye(5)
    for name in ("rugd", "uignn_inductive", "staeformer_inductive", "opencity_scratch"):
        model = build_open_model(name, cfg, 2)
        output = model(batch, adjacency)
        assert output["logits"].shape == (2, 5, 3)
        loss = training_objective(model, batch, adjacency, .5, torch.ones(3), cfg)["loss"]
        loss.backward()
        assert torch.isfinite(loss)
