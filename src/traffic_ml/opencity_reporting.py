"""Publication-oriented figures, tables, and an explicitly non-SOTA research summary."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_curve


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_report(root):
    """Build plots/tables from immutable per-seed predictions; never invent missing results."""
    root = Path(root)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(root / "results_all.csv")
    registered = ["prior", "persistence", "logistic", "hist_gbdt", "lightgbm",
                  "gru", "graph_gru", "shiftguard", "rugd",
                  "uignn_inductive", "staeformer_inductive", "opencity_scratch"]
    main = results[results.model.isin(registered)]
    summary = main.groupby("model")[["macro_f1", "severe_auprc", "nll", "ece"]].agg(["mean", "std"])
    benchmark_summary = summary.copy()
    benchmark_summary.columns = [f"{metric}_{stat}" for metric, stat in benchmark_summary.columns]
    benchmark_summary.reset_index().to_csv(root / "benchmark_summary.csv", index=False)
    order = main.groupby("model").severe_auprc.mean().sort_values().index
    means = main.groupby("model").severe_auprc.mean().loc[order]
    errors = main.groupby("model").severe_auprc.std().loc[order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(order, means, xerr=errors, color="#3274a1", alpha=.9)
    ax.set(xlabel="Severe-congestion AUPRC (mean +/- SD)", title="Leave-one-city-out performance")
    _save(fig, figures / "main_severe_auprc.pdf")

    proposed = "rugd" if (results.model == "rugd").any() else "shiftguard"
    pred_files = sorted(root.glob(f"*/{proposed}/seed_*/predictions.npz"))
    if pred_files:
        bundle = np.load(pred_files[0])
        y, p = bundle["targets"], bundle["probabilities"]
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(confusion_matrix(y, p.argmax(1), normalize="true"),
                               display_labels=["normal", "warning", "severe"]).plot(ax=ax,
                               cmap="Blues", values_format=".2f", colorbar=False)
        ax.set_title(f"{proposed} normalized confusion matrix")
        _save(fig, figures / "shiftguard_confusion.pdf")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        observed, predicted = calibration_curve((y == 2).astype(int), p[:, 2], n_bins=12)
        axes[0].plot(predicted, observed, marker="o"); axes[0].plot([0, 1], [0, 1], "--", color="grey")
        axes[0].set(xlabel="Predicted severe probability", ylabel="Observed frequency",
                    title="Severe-event calibration")
        precision, recall, _ = precision_recall_curve((y == 2).astype(int), p[:, 2])
        axes[1].plot(recall, precision); axes[1].set(xlabel="Recall", ylabel="Precision",
                                                     title="Severe-event PR curve")
        _save(fig, figures / "shiftguard_calibration_pr.pdf")

    robust_path = root / "robustness_all.csv"
    if robust_path.exists():
        robust = pd.read_csv(robust_path)
        focus_names = ["lightgbm", "gru", "graph_gru", proposed,
                       "uignn_inductive", "staeformer_inductive", "opencity_scratch"]
        focus = robust[robust.model.isin(focus_names)]
        grouped = focus.groupby(["model", "corruption", "severity"]).severe_auprc.mean().reset_index()
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (kind, part) in zip(axes, grouped.groupby("corruption")):
            for name, values in part.groupby("model"):
                ax.plot(values.severity, values.severe_auprc, marker="o", label=name)
            ax.set(title=kind, xlabel="Severity", ylabel="Severe AUPRC")
        axes[-1].legend(fontsize=8)
        _save(fig, figures / "robustness_severe_auprc.pdf")

    best = summary[("severe_auprc", "mean")].idxmax()
    text = f"""# Generated research summary

This report summarizes completed artifacts under `{root}`. It does not claim state of the art.

## Methodology and novelty

The study uses strict leave-one-city-out transfer and chronological train/calibration/validation/test
segments separated by an embargo. Congestion thresholds and normalization statistics are estimated
from source-city training periods only. RUGD combines degree-normalized graph diffusion with
local and final evidential supervision, ID-free temporal/topology encoders, source-domain reversal,
focal rare-event learning, and source-only temperature/APS calibration. Modern comparisons are explicitly adapted STAEformer, UIGNN, and
source-only OpenCity-style backbones; they are not claimed as exact published-protocol reproductions.

## Results

Across the completed city/seed runs, the highest mean severe-event AUPRC belongs to **{best}**.
See `benchmark_summary.csv` for mean +/- SD across registered seeds/cities and `results_all.csv` for every
individual run. This ranking is reported honestly even when the proposed model is not best.

## Robustness, uncertainty, and statistics

`robustness_summary.csv` covers missing sensors, Gaussian measurement noise, and temporal
misalignment for every baseline and ShiftGuard-RUGD. `paired_significance.csv` contains paired timestamp-level NLL inference.
`paired_robustness_significance.csv` contains 30/50%-missing AUPRC paired differences,
hierarchical bootstrap intervals, exact Wilcoxon tests, rank-biserial effects, Holm-adjusted
p-values, and a conservative city-level sign-flip sensitivity analysis. ECE, NLL, Brier,
uncertainty-error rank correlation, APS coverage, severe-class coverage, and mean prediction-set
size are in the main result files.

## Ablations

The named ablation output compares removal of graph diffusion, topology conditioning, domain
adaptation, evidence learning, focal modulation, and class balance.
No component claim is valid until all registered cities and seeds complete.

## Limitations and publication requirements

TrafficSH was used for architecture development; TrafficCD and TrafficHZ are the primary
confirmatory folds for RUGD v2. The public OpenCity subsets are short and restricted to three Chinese cities. Congestion is
defined by source-relative speed quantiles, not incident labels. A paper should add longer seasons,
additional geographic regions, external events/weather, stronger recent graph-transformer baselines,
compute-matched tuning, confidence intervals over more temporal blocks, and independent reproduction.
Comparisons to published papers must be protocol-aware; their reported numbers are not directly
comparable unless they use the same cities, horizon, labels, and zero-shot split.
"""
    (root / "RESEARCH_SUMMARY.md").write_text(text, encoding="utf-8")
    return root / "RESEARCH_SUMMARY.md"
