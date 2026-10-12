"""Professional, resumable equal-budget tuning launcher for all LOCO deep models."""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path

import optuna
import yaml

from traffic_ml.config import load_config
from traffic_ml.opencity_models import SUPPORTED_DEEP_MODELS
from traffic_ml.opencity_tuning import tune_source_validation
from traffic_ml.progress import TqdmLoggingHandler, progress

TARGETS = ("TrafficCD", "TrafficHZ", "TrafficSH")
MODELS = ("gru", "graph_gru", "rugd", "uignn_inductive")
MODERN_EXTENSION_MODELS = ("rugd", "uignn_inductive",
                           "staeformer_inductive", "opencity_scratch")


def logger_for(path: Path) -> logging.Logger:
    logger = logging.getLogger("opencity-tuning")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    console = TqdmLoggingHandler(); console.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8"); file_handler.setFormatter(formatter)
    logger.addHandler(console); logger.addHandler(file_handler)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Tune all deep models without target labels.")
    parser.add_argument("--study-id", required=True, help="letters, digits, dot, dash, underscore")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--targets", help="comma-separated subset; default is all")
    parser.add_argument("--models", help="comma-separated subset; default is all")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.study_id):
        raise ValueError("Invalid study ID")
    study_root = Path("outputs/studies") / args.study_id
    config_dir = study_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    if any(config_dir.glob("final_*.yaml")) and not args.resume:
        raise FileExistsError(f"Study {args.study_id} already has tuned configs; use --resume or a new study ID")
    log = logger_for(study_root / "tuning.log")
    cfg = load_config(args.config)
    cfg["project"]["output_dir"] = str(study_root)
    (config_dir / "base_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False),
                                                     encoding="utf-8")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    targets = args.targets.split(",") if args.targets else TARGETS
    models = args.models.split(",") if args.models else MODELS
    if set(targets) - set(TARGETS) or set(models) - SUPPORTED_DEEP_MODELS:
        raise ValueError("Unknown target or model in requested subset")
    total = len(targets) * len(models)
    log.info("STUDY %s | equal-budget tuning | %d searches x %d trials",
             args.study_id, total, args.trials)
    started = time.perf_counter()
    searches = [(t, m) for t in targets for m in models]
    for index, (target, model) in enumerate(progress(searches, "Tuning searches", "search"), 1):
        final_config = config_dir / f"final_{target}_{model}.yaml"
        if args.resume and final_config.exists():
            log.info("[%d/%d] SKIP | target=%s model=%s | frozen config exists",
                     index, total, target, model)
            continue
        log.info("[%d/%d] START | target=%s model=%s", index, total, target, model)
        search_started = time.perf_counter()
        best = tune_source_validation(cfg, target, args.trials, model)
        frozen = load_config(args.config)
        for key, value in best.items():
            section = "model" if key in {"hidden_dim", "dropout", "diffusion_steps"} else "training"
            frozen[section][key] = value
        if model in {"staeformer_inductive", "opencity_scratch"}:
            frozen["training"]["time_batch_size"] = int(
                frozen["training"].get("transformer_time_batch_size", 24))
        frozen["data"]["target_city"] = target
        frozen["project"]["output_dir"] = str(study_root / "main")
        final_config.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
        log.info("[%d/%d] DONE  | target=%s model=%s | %.1fs",
                 index, total, target, model, time.perf_counter() - search_started)
        log.info("[%d/%d] PARAMS | %s", index, total, json.dumps(best, sort_keys=True))
    log.info("COMPLETE | tuning elapsed %.1f minutes | configs=%s",
             (time.perf_counter() - started) / 60, config_dir)


if __name__ == "__main__":
    main()
