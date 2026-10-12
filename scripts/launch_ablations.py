"""Run registered ablations under a named study with concise logs and resumability."""
from __future__ import annotations

import argparse
import logging
import re
import shutil
import time
from pathlib import Path

import pandas as pd

from traffic_ml.config import load_config
from traffic_ml.opencity_ensemble import aggregate_seed_ensembles
from traffic_ml.opencity_experiment import aggregate_suite, train_deep
from traffic_ml.opencity_reporting import create_report
from traffic_ml.opencity_statistics import compare_suite
from traffic_ml.progress import TqdmLoggingHandler, progress

TARGETS = ("TrafficCD", "TrafficHZ", "TrafficSH")
RUGD_ABLATIONS = (
    "rugd", "rugd_no_graph", "rugd_no_topology", "rugd_no_domain",
    "rugd_no_evidence", "rugd_no_focal", "rugd_no_class_balance",
)


def main():
    parser = argparse.ArgumentParser(description="Run all registered ShiftGuard ablations.")
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.study_id):
        raise ValueError("Invalid study ID")
    study_root = Path("outputs/studies") / args.study_id
    output_root = study_root / "ablations"
    output_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("opencity-ablation"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (TqdmLoggingHandler(),
                    logging.FileHandler(study_root / "ablations.log", encoding="utf-8")):
        handler.setFormatter(formatter); logger.addHandler(handler)
    variants = RUGD_ABLATIONS
    jobs = [(target, variant, seed) for target in TARGETS for variant in variants
            for seed in load_config(args.config)["project"]["seeds"]]
    logger.info("STUDY %s | %d registered ablation runs", args.study_id, len(jobs))
    started = time.perf_counter()
    for index, (target, variant, seed) in enumerate(progress(jobs, "Ablation study", "run"), 1):
        metric_path = output_root / target / variant / f"seed_{seed}" / "metrics.csv"
        if args.resume and metric_path.exists():
            logger.info("[%02d/%02d] SKIP | city=%s variant=%s seed=%s",
                        index, len(jobs), target, variant, seed)
            continue
        tuned_name = "rugd" if args.family == "rugd" else "shiftguard"
        tuned = study_root / "configs" / f"final_{target}_{tuned_name}.yaml"
        cfg = load_config(tuned if tuned.exists() else args.config)
        cfg["project"]["output_dir"] = str(output_root)
        run_started = time.perf_counter()
        logger.info("[%02d/%02d] START | city=%s variant=%s seed=%s",
                    index, len(jobs), target, variant, seed)
        if args.family == "rugd" and variant == "rugd":
            source = study_root / "main" / target / "rugd" / f"seed_{seed}"
            if not source.exists():
                raise FileNotFoundError(f"Main RUGD reference is missing: {source}")
            shutil.copytree(source, metric_path.parent, dirs_exist_ok=True)
            logger.info("[%02d/%02d] REUSE | identical tuned main RUGD reference",
                        index, len(jobs))
        else:
            train_deep(cfg, target, variant, seed)
        metric = pd.read_csv(metric_path).iloc[0]
        logger.info("[%02d/%02d] DONE  | %.1fs | macro_f1=%.5f severe_auprc=%.5f",
                    index, len(jobs), time.perf_counter() - run_started,
                    metric.macro_f1, metric.severe_auprc)
    report_cfg = load_config(args.config); report_cfg["project"]["output_dir"] = str(output_root)
    aggregate_suite(report_cfg)
    reference = "rugd" if args.family == "rugd" else "shiftguard"
    compare_suite(output_root, reference=reference,
                  repeats=report_cfg["evaluation"]["bootstrap_repeats"])
    aggregate_seed_ensembles(output_root, report_cfg["evaluation"]["ece_bins"])
    summary = create_report(output_root)
    logger.info("COMPLETE | elapsed %.1f minutes | summary=%s",
                (time.perf_counter() - started) / 60, summary)


if __name__ == "__main__":
    main()
