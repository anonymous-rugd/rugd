"""Leakage-safe Optuna search using source-city chronological validation only."""
from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch

from .losses import effective_number_weights
from .opencity_experiment import (
    _to,
    loader,
    prepare,
    source_validation_objective,
    training_objective,
)
from .opencity_models import build_open_model
from .reproducibility import seed_everything


def tune_source_validation(cfg, target, trials=20, model_name="rugd"):
    """Tune without evaluating, calibrating on, or selecting against the target test set."""
    output = Path(cfg["project"]["output_dir"]) / "tuning" / target / model_name
    output.mkdir(parents=True, exist_ok=True)
    base_seed = cfg["project"]["seeds"][0]

    def objective(trial):
        trial_cfg = deepcopy(cfg)
        trial_cfg["model"]["hidden_dim"] = trial.suggest_categorical("hidden_dim", [48, 64, 96, 128])
        trial_cfg["model"]["dropout"] = trial.suggest_float("dropout", .05, .4)
        if model_name in {"graph_gru", "rugd", "uignn_inductive"}:
            trial_cfg["model"]["diffusion_steps"] = trial.suggest_int("diffusion_steps", 1, 3)
        trial_cfg["training"]["lr"] = trial.suggest_float("lr", 2e-4, 3e-3, log=True)
        trial_cfg["training"]["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        trial_cfg["training"]["focal_gamma"] = trial.suggest_float("focal_gamma", 1., 3.)
        if model_name == "rugd":
            trial_cfg["training"]["domain_weight"] = trial.suggest_float(
                "domain_weight", .01, .25, log=True)
            trial_cfg["training"]["evidence_kl_weight"] = trial.suggest_float(
                "evidence_kl_weight", 1e-5, 1e-2, log=True)
            trial_cfg["training"]["auxiliary_weight"] = trial.suggest_categorical(
                "auxiliary_weight", [.05, .10, .20])
        if model_name == "uignn_inductive":
            trial_cfg["training"]["evidence_kl_weight"] = trial.suggest_float(
                "evidence_kl_weight", 1e-5, 1e-2, log=True)
            trial_cfg["training"]["recovery_weight"] = trial.suggest_categorical(
                "recovery_weight", [.05, .10, .25])
            trial_cfg["training"]["recovery_mask_rate"] = trial.suggest_categorical(
                "recovery_mask_rate", [.10, .20, .30])
        if model_name in {"staeformer_inductive", "opencity_scratch"}:
            trial_cfg["training"]["time_batch_size"] = int(
                cfg["training"].get("transformer_time_batch_size", 24))
        trial_cfg["training"]["epochs"] = min(cfg["training"]["epochs"], 15)
        seed_everything(base_seed)
        stores, sources, _, sets = prepare(trial_cfg, target)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_open_model(model_name, trial_cfg, len(sources)).to(device)
        adj = {city: stores[city].normalized_adjacency().to(device) for city in sources}
        loaders = {(city, part): loader(sets[city, part], trial_cfg, part == "train")
                   for city in sources for part in ("train", "validation")}
        labels = np.concatenate([batch["target"].numpy().reshape(-1)
                                 for city in sources for batch in loaders[city, "train"]])
        counts = torch.tensor(np.bincount(labels[labels >= 0], minlength=3))
        weights = effective_number_weights(counts, trial_cfg["training"]["effective_beta"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=trial_cfg["training"]["lr"],
                                      weight_decay=trial_cfg["training"]["weight_decay"])
        best = -1.0
        for epoch in range(trial_cfg["training"]["epochs"]):
            model.train()
            scale = 2 / (1 + math.exp(-10 * epoch / max(1, trial_cfg["training"]["epochs"] - 1))) - 1
            for city in sources:
                for raw in loaders[city, "train"]:
                    batch = _to(raw, device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = training_objective(
                        model, batch, adj[city], scale, weights, trial_cfg)["loss"]
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trial_cfg["training"]["grad_clip"])
                    optimizer.step()
            score, _, _ = source_validation_objective(
                model, sources, loaders, adj, device, trial_cfg)
            best = max(best, score)
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=base_seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(objective, n_trials=trials, show_progress_bar=True)
    pd.DataFrame(study.trials_dataframe()).to_csv(output / "trials.csv", index=False)
    (output / "best_params.json").write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    return study.best_params
