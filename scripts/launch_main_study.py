"""Run and report the complete, identifiable, resumable OpenCity benchmark."""
from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
import yaml

from traffic_ml.config import load_config
from traffic_ml.opencity_ensemble import aggregate_seed_ensembles
from traffic_ml.opencity_experiment import (
    CLASSICAL_MODELS,
    DEFAULT_MODELS,
    aggregate_suite,
    run_classical,
    train_deep,
)
from traffic_ml.opencity_models import SUPPORTED_DEEP_MODELS
from traffic_ml.opencity_reporting import create_report
from traffic_ml.opencity_statistics import compare_suite
from traffic_ml.progress import TqdmLoggingHandler, progress

TARGETS = ("TrafficCD", "TrafficHZ", "TrafficSH")


def logger_for(path: Path) -> logging.Logger:
    logger = logging.getLogger("opencity-main")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    console = TqdmLoggingHandler(); console.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8"); file_handler.setFormatter(formatter)
    logger.addHandler(console); logger.addHandler(file_handler)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Run all cities, models, seeds, and reports.")
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--targets", help="comma-separated subset; default is all")
    parser.add_argument("--models", help="comma-separated subset; default is all")
    parser.add_argument("--untuned", action="store_true",
                        help="Use the common base config instead of requiring frozen tuning configs")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.study_id):
        raise ValueError("Invalid study ID")
    study_root = Path("outputs/studies") / args.study_id
    main_root = study_root / "main"
    config_dir = study_root / "configs"
    main_root.mkdir(parents=True, exist_ok=True); config_dir.mkdir(parents=True, exist_ok=True)
    log = logger_for(study_root / "main.log")
    base = load_config(args.config)
    base["project"]["output_dir"] = str(main_root)
    (config_dir / "main_base_resolved.yaml").write_text(yaml.safe_dump(base, sort_keys=False),
                                                         encoding="utf-8")
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (study_root / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the main study; run the README verification first")
    targets = args.targets.split(",") if args.targets else TARGETS
    models = args.models.split(",") if args.models else DEFAULT_MODELS
    if set(targets) - set(TARGETS) or set(models) - (set(DEFAULT_MODELS) | SUPPORTED_DEEP_MODELS):
        raise ValueError("Unknown target or model in requested subset")
    jobs = [(target, model, seed) for target in targets for model in models
            for seed in base["project"]["seeds"]]
    log.info("STUDY %s | %d registered runs | GPU=%s | tuned=%s",
             args.study_id, len(jobs), environment["gpu"], not args.untuned)
    status_path = study_root / "run_status.csv"
    if status_path.exists() and not args.resume:
        raise FileExistsError(f"Study {args.study_id} already has runs; use --resume or a new study ID")
    records = [] if not status_path.exists() else pd.read_csv(status_path).to_dict("records")
    started = time.perf_counter()
    for index, (target, model, seed) in enumerate(progress(jobs, "Main experiment", "run"), 1):
        output = main_root / target / model / f"seed_{seed}" / "metrics.csv"
        if args.resume and output.exists():
            log.info("[%02d/%02d] SKIP | city=%s model=%s seed=%s", index, len(jobs),
                     target, model, seed)
            continue
        cfg = base
        if model not in CLASSICAL_MODELS and not args.untuned:
            path = config_dir / f"final_{target}_{model}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"Missing tuned config {path}; run launch_tuning.py first")
            cfg = load_config(path); cfg["project"]["output_dir"] = str(main_root)
        run_started = time.perf_counter()
        log.info("[%02d/%02d] START | city=%s model=%s seed=%s", index, len(jobs),
                 target, model, seed)
        try:
            runner = run_classical if model in CLASSICAL_MODELS else train_deep
            runner(cfg, target, model, seed)
            metric = pd.read_csv(output).iloc[0]
            elapsed = time.perf_counter() - run_started
            records.append({"city": target, "model": model, "seed": seed, "status": "complete",
                            "seconds": elapsed, "macro_f1": metric.macro_f1,
                            "severe_auprc": metric.severe_auprc})
            pd.DataFrame(records).to_csv(status_path, index=False)
            log.info("[%02d/%02d] DONE  | %.1fs | macro_f1=%.5f severe_auprc=%.5f",
                     index, len(jobs), elapsed, metric.macro_f1, metric.severe_auprc)
        except Exception:
            log.error("[%02d/%02d] FAILED | city=%s model=%s seed=%s\n%s", index, len(jobs),
                      target, model, seed, traceback.format_exc())
            raise
    log.info("REPORT | aggregating, paired inference, ensembles, plots, and summary")
    aggregate_suite(base)
    reference = "rugd" if "rugd" in models else "shiftguard"
    compare_suite(main_root, reference=reference,
                  repeats=base["evaluation"]["bootstrap_repeats"])
    aggregate_seed_ensembles(main_root, base["evaluation"]["ece_bins"])
    summary = create_report(main_root)
    log.info("COMPLETE | elapsed %.1f minutes | summary=%s",
             (time.perf_counter() - started) / 60, summary)


if __name__ == "__main__":
    main()
