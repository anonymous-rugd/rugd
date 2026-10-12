"""End-to-end leave-one-city-out OpenCity experiments with a common protocol."""
from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calibration import APSConformal, TemperatureScaler
from .losses import effective_number_weights, focal_evidential_loss
from .opencity_classical import Persistence, SourcePrior, build_baseline, flatten_loader
from .opencity_data import (
    CityWindowDataset,
    OpenCityStore,
    fit_source_statistics,
    temporal_indices,
    write_data_manifest,
)
from .opencity_metrics import evaluate_open
from .opencity_models import SUPPORTED_DEEP_MODELS, build_open_model
from .progress import progress
from .reproducibility import seed_everything, write_provenance

CLASSICAL_MODELS = {"prior", "persistence", "logistic", "hist_gbdt", "lightgbm"}
DEFAULT_MODELS = ["prior", "persistence", "logistic", "hist_gbdt", "lightgbm",
                  "gru", "graph_gru", "rugd", "uignn_inductive",
                  "staeformer_inductive", "opencity_scratch"]


def prepare(cfg, target):
    """Create source-only train/calibration/validation and target-only future test sets."""
    names = cfg["data"]["cities"]
    stores = {name: OpenCityStore.open(cfg["data"]["root"], name,
              cfg["data"]["start_timestamp"], cfg["data"]["interval_minutes"])
              for name in names}
    split_cfg = cfg["split"]
    fractions = tuple(split_cfg[x] for x in ("train_fraction", "calibration_fraction",
                                              "validation_fraction", "test_fraction"))
    splits = {name: temporal_indices(len(store.speed), cfg["data"]["history_steps"],
              cfg["data"]["horizon_steps"], fractions, split_cfg["embargo_steps"])
              for name, store in stores.items()}
    sources = [name for name in names if name != target]
    stats = fit_source_statistics([stores[name] for name in sources],
                                  [splits[name] for name in sources], cfg)
    datasets = {}
    for city_index, name in enumerate(sources):
        for part in ("train", "calibration", "validation"):
            datasets[name, part] = CityWindowDataset(
                stores[name], getattr(splits[name], part), cfg["data"]["history_steps"],
                cfg["data"]["horizon_steps"], stats, city_index)
    datasets[target, "test"] = CityWindowDataset(
        stores[target], splits[target].test, cfg["data"]["history_steps"],
        cfg["data"]["horizon_steps"], stats, 0)
    return stores, sources, stats, datasets


def loader(dataset, cfg, shuffle=False):
    """Batch complete graph snapshots; nodes remain the spatial dimension."""
    return DataLoader(dataset, batch_size=cfg["training"]["time_batch_size"],
                      shuffle=shuffle, num_workers=0,
                      pin_memory=torch.cuda.is_available())


def _to(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _deep_loss(model, out, batch, weights, cfg):
    """Primary objective plus supervised local-evidence learning for RUGD."""
    valid = batch["target"] >= 0
    flat = {"logits": out["logits"][valid], "alpha": out["alpha"][valid],
            "domain_logits": out["domain_logits"]}
    domain = batch["city"] if model.use_domain and out["domain_logits"].shape[1] > 1 else None
    gamma = 0.0 if model.variant.endswith("_no_focal") else cfg["training"]["focal_gamma"]
    kl_weight = cfg["training"]["evidence_kl_weight"] if model.use_evidence else 0.0
    domain_weight = cfg["training"]["domain_weight"] if model.use_domain else 0.0
    losses = focal_evidential_loss(flat, batch["target"][valid], weights, gamma, kl_weight,
                                   domain, domain_weight)
    if model.pre_evidence is not None:
        preliminary = {"logits": out["pre_logits"][valid], "alpha": out["pre_alpha"][valid],
                       "domain_logits": out["domain_logits"]}
        auxiliary = focal_evidential_loss(preliminary, batch["target"][valid], weights, gamma,
                                          cfg["training"]["evidence_kl_weight"], None, 0.0)
        losses["local_evidence"] = auxiliary["loss"]
        losses["loss"] = losses["loss"] + cfg["training"]["auxiliary_weight"] * auxiliary["loss"]
    return losses


def training_objective(model, batch, adjacency, domain_scale, weights, cfg):
    """Compute the identical registered training objective for tuning and final fits."""
    result = model(batch, adjacency, domain_scale)
    losses = _deep_loss(model, result, batch, weights, cfg)
    if getattr(model, "use_recovery", False):
        recovery_view = dict(batch)
        rate = float(cfg["training"].get("recovery_mask_rate", 0.20))
        drop_nodes = (torch.rand(batch["history"].shape[:2],
                                 device=batch["history"].device) < rate)[..., None]
        held_out = drop_nodes & batch["history_mask"]
        recovery_view["history_mask"] = batch["history_mask"] & ~drop_nodes
        recovery_view["history"] = batch["history"].masked_fill(held_out, 0)
        recovered = model(recovery_view, adjacency, domain_scale)["reconstruction"]
        if held_out.any():
            recovery = torch.nn.functional.mse_loss(
                recovered[held_out], batch["history"][held_out])
            losses["recovery"] = recovery
            losses["loss"] = (losses["loss"] +
                              cfg["training"].get("recovery_weight", 0.10) * recovery)
    return losses

@torch.inference_mode()
def predict_deep(model, data_loader, adjacency, device, corruption=None):
    """Predict nodes and preserve timestamp IDs for block-aware inference."""
    model.eval()
    probabilities, labels, uncertainty, time_ids, node_ids = [], [], [], [], []
    generator = torch.Generator(device=device).manual_seed(9182)
    for raw in data_loader:
        batch = _to(raw, device)
        if corruption:
            history = batch["history"].clone()
            kind, level = corruption
            if kind == "missing":
                dropped = torch.rand(history.shape, generator=generator, device=device) < level
                history[dropped] = 0
                batch["history_mask"] = batch["history_mask"] & ~dropped
            elif kind == "noise":
                history += torch.randn(history.shape, generator=generator, device=device) * level
            elif kind == "shift":
                history = torch.roll(history, int(level), dims=-1)
                batch["history_mask"] = torch.roll(batch["history_mask"], int(level), dims=-1)
            batch["history"] = history
        output = model(batch, adjacency)
        valid = batch["target"] >= 0
        prob = output["logits"].softmax(-1)
        unc = (3 / output["alpha"].sum(-1) if model.use_evidence else
               -(prob * prob.clamp_min(1e-9).log()).sum(-1))
        probabilities.append(prob[valid].cpu().numpy())
        labels.append(batch["target"][valid].cpu().numpy())
        uncertainty.append(unc[valid].cpu().numpy())
        times = batch["time_id"][:, None].expand_as(valid)
        nodes = torch.arange(valid.shape[1], device=device)[None].expand_as(valid)
        time_ids.append(times[valid].cpu().numpy())
        node_ids.append(nodes[valid].cpu().numpy())
    return tuple(np.concatenate(items) for items in
                 (probabilities, labels, uncertainty, time_ids, node_ids))


def fit_calibration(probabilities, labels, alpha):
    """Fit scalar temperature and APS threshold on source calibration only."""
    logits = np.log(probabilities.clip(1e-9))
    temperature = TemperatureScaler().fit(logits, labels)
    calibrated = temperature.transform(logits)
    conformal = APSConformal(alpha).fit(calibrated, labels)
    return temperature, conformal


def zip_predictions(items):
    return tuple(np.concatenate(values) for values in zip(*items))


def source_validation_objective(model, sources, loaders, adjacencies, device, cfg):
    """City-balanced clean/rare/missing validation score using source cities only."""
    clean_bundles = []
    clean_city_metrics, missing_city_metrics = [], []
    missing_rate = float(cfg["training"].get("validation_missing_rate", 0.30))
    for city in sources:
        clean = predict_deep(model, loaders[city, "validation"], adjacencies[city], device)
        clean_bundles.append(clean)
        clean_city_metrics.append(evaluate_open(clean[1], clean[0], clean[2], clean[3],
                                                cfg["evaluation"]["ece_bins"]))
        missing = predict_deep(model, loaders[city, "validation"], adjacencies[city], device,
                               ("missing", missing_rate))
        missing_city_metrics.append(evaluate_open(missing[1], missing[0], missing[2], missing[3],
                                                  cfg["evaluation"]["ece_bins"]))
    combined = zip_predictions(clean_bundles)
    clean_metric = evaluate_open(combined[1], combined[0], combined[2], combined[3],
                                 cfg["evaluation"]["ece_bins"])
    clean_macro = float(np.mean([m["macro_f1"] for m in clean_city_metrics]))
    clean_severe = float(np.mean([m["severe_auprc"] for m in clean_city_metrics]))
    missing_macro = float(np.mean([m["macro_f1"] for m in missing_city_metrics]))
    missing_severe = float(np.mean([m["severe_auprc"] for m in missing_city_metrics]))
    selection = cfg["training"].get("selection_weights", {})
    weights = np.asarray([selection.get("clean_macro", .40),
                          selection.get("clean_severe", .30),
                          selection.get("missing_macro", .15),
                          selection.get("missing_severe", .15)], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("selection_weights must have positive total weight")
    weights = weights / weights.sum()
    components = np.asarray([clean_macro, clean_severe, missing_macro, missing_severe])
    score = float(weights @ components)
    diagnostics = {"selection_score": score,
                   "source_clean_macro_f1": clean_macro,
                   "source_clean_severe_auprc": clean_severe,
                   "source_missing_macro_f1": missing_macro,
                   "source_missing_severe_auprc": missing_severe}
    return score, clean_metric, diagnostics

def _corrupt_flat(x, kind, level, history_steps, seed=9182):
    """Apply the same registered corruptions to flattened classical features."""
    out = x.copy()
    rng = np.random.default_rng(seed)
    histories = out[:, :history_steps]
    masks = out[:, history_steps:2 * history_steps]
    if kind == "missing":
        dropped = rng.random(histories.shape) < level
        histories[dropped] = 0
        masks[dropped] = 0
    elif kind == "noise":
        histories += rng.normal(0, level, histories.shape)
    elif kind == "shift":
        histories[:] = np.roll(histories, int(level), axis=1)
        masks[:] = np.roll(masks, int(level), axis=1)
    return out


def train_deep(cfg, target, name, seed):
    """Train on source cities, select on source validation, test target exactly once."""
    seed_everything(seed)
    stores, sources, stats, sets = prepare(cfg, target)
    output_dir = Path(cfg["project"]["output_dir"]) / target / name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(output_dir / "provenance.json", cfg)
    (output_dir / "target_definition.json").write_text(json.dumps(stats, indent=2))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_open_model(name, cfg, len(sources)).to(device)
    adjacencies = {city: stores[city].normalized_adjacency().to(device) for city in stores}
    loaders = {(city, part): loader(dataset, cfg, part == "train")
               for (city, part), dataset in sets.items()}
    labels = np.concatenate([batch["target"].numpy().reshape(-1)
                             for city in sources for batch in loaders[city, "train"]])
    if name.endswith("_no_class_balance"):
        weights = torch.ones(3, device=device)
    else:
        counts = torch.tensor(np.bincount(labels[labels >= 0], minlength=3))
        weights = effective_number_weights(counts, cfg["training"]["effective_beta"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["training"]["amp"] and device.type == "cuda")
    best, best_state, bad_epochs, history = -1.0, None, 0, []
    started = time.perf_counter()
    for epoch in progress(range(cfg["training"]["epochs"]),
                          f"{target}/{name}/seed_{seed}", "epoch", leave=False):
        model.train()
        total, batches = 0.0, 0
        domain_scale = (2 / (1 + math.exp(-10 * epoch /
                        max(1, cfg["training"]["epochs"] - 1))) - 1) if model.use_domain else 0.0
        for city in sources:
            for raw in loaders[city, "train"]:
                batch = _to(raw, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    losses = training_objective(model, batch, adjacencies[city], domain_scale,
                                                weights, cfg)
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                total += losses["loss"].detach().item()
                batches += 1
        selection_score, metric, diagnostics = source_validation_objective(
            model, sources, loaders, adjacencies, device, cfg)
        history.append({"epoch": epoch, "train_loss": total / max(1, batches),
                        **metric, **diagnostics})
        if selection_score > best:
            best, best_state, bad_epochs = selection_score, deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["training"]["patience"]:
                break
    train_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)
    calibration = zip_predictions([predict_deep(model, loaders[city, "calibration"],
                                  adjacencies[city], device) for city in sources])
    temperature, conformal = fit_calibration(calibration[0], calibration[1],
                                              cfg["evaluation"]["conformal_alpha"])
    test_loader = loaders[target, "test"]
    started = time.perf_counter()
    probability, label, unc, time_id, node_id = predict_deep(
        model, test_loader, adjacencies[target], device)
    inference_seconds = time.perf_counter() - started
    probability = temperature.transform(np.log(probability.clip(1e-9)))
    metrics = evaluate_open(label, probability, unc, time_id, cfg["evaluation"]["ece_bins"])
    metrics.update(conformal.evaluate(probability, label))
    metrics.update({"model": name, "seed": seed, "target_city": target,
                    "temperature": temperature.temperature, "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "predictions_per_second": len(label) / max(inference_seconds, 1e-9),
                    "parameters": sum(parameter.numel() for parameter in model.parameters())})
    torch.save({"state_dict": best_state, "model_name": name, "config": cfg, "stats": stats},
               output_dir / "model.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    np.savez_compressed(output_dir / "predictions.npz", probabilities=probability,
                        targets=label, uncertainty=unc, time_ids=time_id, node_ids=node_id)
    robustness = []
    for kind, levels in (("missing", cfg["robustness"]["missing_rates"]),
                         ("noise", cfg["robustness"]["noise_std"]),
                         ("shift", cfg["robustness"]["temporal_shifts"])):
        for level in levels:
            rp, ry, ru, rt, _ = predict_deep(model, test_loader, adjacencies[target], device,
                                             (kind, level))
            rp = temperature.transform(np.log(rp.clip(1e-9)))
            robustness.append({"model": name, "seed": seed, "target_city": target,
                               "corruption": kind, "severity": level,
                               **evaluate_open(ry, rp, ru, rt, cfg["evaluation"]["ece_bins"])})
    pd.DataFrame(robustness).to_csv(output_dir / "robustness.csv", index=False)
    return output_dir


def run_classical(cfg, target, name, seed):
    """Fit and evaluate a classical baseline with identical windows and corruptions."""
    seed_everything(seed)
    _, sources, stats, sets = prepare(cfg, target)
    root = Path(cfg["project"]["output_dir"]) / target / name / f"seed_{seed}"
    root.mkdir(parents=True, exist_ok=True)
    write_provenance(root / "provenance.json", cfg)
    train_parts = [flatten_loader(loader(sets[city, "train"], cfg),
                   cfg["data"]["classical_train_cap"] // len(sources), seed)
                   for city in sources]
    calibration_parts = [flatten_loader(loader(sets[city, "calibration"], cfg))
                         for city in sources]
    xtrain, ytrain, _ = tuple(np.concatenate(values) for values in zip(*train_parts))
    xcal, ycal, _ = tuple(np.concatenate(values) for values in zip(*calibration_parts))
    xtest, ytest, time_ids = flatten_loader(loader(sets[target, "test"], cfg))
    started = time.perf_counter()
    if name == "prior":
        model = SourcePrior().fit(ytrain)
    elif name == "persistence":
        model = Persistence(cfg["data"]["history_steps"], stats)
    else:
        model = build_baseline(name, seed).fit(xtrain, ytrain)
    train_seconds = time.perf_counter() - started
    temperature, conformal = fit_calibration(model.predict_proba(xcal), ycal,
                                              cfg["evaluation"]["conformal_alpha"])
    started = time.perf_counter()
    probability = model.predict_proba(xtest)
    inference_seconds = time.perf_counter() - started
    probability = temperature.transform(np.log(probability.clip(1e-9)))
    uncertainty = -(probability * np.log(probability.clip(1e-9))).sum(1)
    metrics = evaluate_open(ytest, probability, uncertainty, time_ids,
                            cfg["evaluation"]["ece_bins"])
    metrics.update(conformal.evaluate(probability, ytest))
    metrics.update({"model": name, "seed": seed, "target_city": target,
                    "temperature": temperature.temperature, "train_seconds": train_seconds,
                    "inference_seconds": inference_seconds,
                    "predictions_per_second": len(ytest) / max(inference_seconds, 1e-9),
                    "parameters": np.nan})
    pd.DataFrame([metrics]).to_csv(root / "metrics.csv", index=False)
    joblib.dump(model, root / "model.joblib")
    np.savez_compressed(root / "predictions.npz", probabilities=probability, targets=ytest,
                        uncertainty=uncertainty, time_ids=time_ids)
    robustness = []
    for kind, levels in (("missing", cfg["robustness"]["missing_rates"]),
                         ("noise", cfg["robustness"]["noise_std"]),
                         ("shift", cfg["robustness"]["temporal_shifts"])):
        for level in levels:
            corrupted = _corrupt_flat(xtest, kind, level, cfg["data"]["history_steps"])
            rp = model.predict_proba(corrupted)
            rp = temperature.transform(np.log(rp.clip(1e-9)))
            ru = -(rp * np.log(rp.clip(1e-9))).sum(1)
            robustness.append({"model": name, "seed": seed, "target_city": target,
                               "corruption": kind, "severity": level,
                               **evaluate_open(ytest, rp, ru, time_ids,
                                               cfg["evaluation"]["ece_bins"])})
    pd.DataFrame(robustness).to_csv(root / "robustness.csv", index=False)
    return root


def run_suite(cfg, models=None, targets=None):
    """Run every requested target/model/registered seed and aggregate only afterward."""
    models = models or DEFAULT_MODELS
    targets = targets or cfg["data"]["cities"]
    unknown = set(models) - CLASSICAL_MODELS - SUPPORTED_DEEP_MODELS
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    manifest = Path(cfg["project"]["output_dir"]) / "data_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_data_manifest(cfg["data"]["root"], manifest, cfg["data"]["cities"])
    for target in targets:
        for name in models:
            for seed in cfg["project"]["seeds"]:
                runner = train_deep if name in SUPPORTED_DEEP_MODELS else run_classical
                runner(cfg, target, name, seed)
    aggregate_suite(cfg)
    return Path(cfg["project"]["output_dir"])


def aggregate_suite(cfg):
    """Produce per-seed, city-level, and overall tables without selecting seeds."""
    root = Path(cfg["project"]["output_dir"])
    metric_files = list(root.glob("*/*/seed_*/metrics.csv"))
    if not metric_files:
        raise FileNotFoundError("No suite metrics")
    frame = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    frame.to_csv(root / "results_all.csv", index=False)
    metrics = ["macro_f1", "severe_auprc", "severe_recall", "nll", "brier", "ece",
               "conformal_coverage", "conformal_mean_set_size", "train_seconds"]
    city_summary = frame.groupby(["target_city", "model"])[metrics].agg(["mean", "std"])
    city_summary.columns = [f"{metric}_{stat}" for metric, stat in city_summary.columns]
    city_summary.reset_index().to_csv(root / "results_summary.csv", index=False)
    overall = frame.groupby("model")[metrics].agg(["mean", "std"])
    overall.columns = [f"{metric}_{stat}" for metric, stat in overall.columns]
    overall.reset_index().to_csv(root / "results_overall.csv", index=False)
    robust_files = list(root.glob("*/*/seed_*/robustness.csv"))
    if robust_files:
        robust = pd.concat([pd.read_csv(path) for path in robust_files], ignore_index=True)
        robust.to_csv(root / "robustness_all.csv", index=False)
        robust_summary = robust.groupby(
            ["target_city", "model", "corruption", "severity"])[
            ["macro_f1", "severe_auprc", "nll", "ece"]].agg(["mean", "std"])
        robust_summary.columns = [f"{metric}_{stat}" for metric, stat in robust_summary.columns]
        robust_summary.reset_index().to_csv(root / "robustness_summary.csv", index=False)
