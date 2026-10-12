"""Run the complete final benchmark from scratch with compute-aware tuning budgets."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CONTROLS = "gru,graph_gru"
MODERN_LIGHTWEIGHT = "rugd,uignn_inductive"
TRANSFORMERS = "staeformer_inductive,opencity_scratch"


def run(script: str, arguments: list[str]) -> None:
    command = [sys.executable, str(Path(__file__).with_name(script)), *arguments]
    print(f"\nPIPELINE STAGE | {script}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete final research pipeline.")
    parser.add_argument("--study-id", default="final_reproduction")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--baseline-trials", type=int, default=30,
                        help="Trials per recurrent control/city")
    parser.add_argument("--trials", type=int, default=20,
                        help="Trials per modern lightweight model/city")
    parser.add_argument("--transformer-trials", type=int, default=8,
                        help="Compute-matched trials per Transformer model/city")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    common = ["--study-id", args.study_id, "--config", args.config]
    resume = ["--resume"] if args.resume else []
    run("launch_tuning.py", [*common, "--trials", str(args.baseline_trials),
                             "--models", CONTROLS, *resume])
    run("launch_tuning.py", [*common, "--trials", str(args.trials),
                             "--models", MODERN_LIGHTWEIGHT, "--resume"])
    run("launch_tuning.py", [*common, "--trials", str(args.transformer_trials),
                             "--models", TRANSFORMERS, "--resume"])
    run("launch_main_study.py", [*common, *resume])
    run("launch_ablations.py", [*common, *resume])
    print(f"\nPIPELINE COMPLETE | outputs/studies/{args.study_id}", flush=True)


if __name__ == "__main__":
    main()
