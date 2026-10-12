"""Run a short real-data integration study for the final registered model suite."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODELS = ("prior,persistence,logistic,hist_gbdt,lightgbm,gru,graph_gru,rugd,"
          "uignn_inductive,staeformer_inductive,opencity_scratch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-data final-suite smoke study.")
    parser.add_argument("--study-id", default="rugd_final_smoke")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    command = [sys.executable, str(Path(__file__).with_name("launch_main_study.py")),
               "--study-id", args.study_id, "--config", "configs/smoke.yaml",
               "--targets", "TrafficSH", "--models", MODELS, "--untuned"]
    if args.resume:
        command.append("--resume")
    print("REAL-DATA SMOKE | 1 city x 11 models x 3 seeds = 33 runs", flush=True)
    subprocess.run(command, check=True)
    root = Path("outputs/studies") / args.study_id / "main"
    results = pd.read_csv(root / "results_all.csv")
    columns = ["macro_f1", "severe_auprc", "severe_recall", "nll", "ece"]
    comparison = results.groupby("model")[columns].agg(["mean", "std"])
    comparison.columns = [f"{metric}_{stat}" for metric, stat in comparison.columns]
    comparison.reset_index().to_csv(root / "smoke_comparison.csv", index=False)
    print("\nSMOKE COMPARISON (three-seed means)", flush=True)
    print(results.groupby("model")[columns].mean().sort_values(
        "severe_auprc", ascending=False).round(5).to_string(), flush=True)
    print(f"\nSMOKE COMPLETE | {root / 'smoke_comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
