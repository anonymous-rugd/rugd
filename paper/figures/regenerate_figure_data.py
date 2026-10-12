"""Regenerate every CSV in paper/figures/data/ from the retained final-study artifacts.

Run from the repository root:
    python paper/figures/regenerate_figure_data.py

Every number this script writes is read directly from
outputs/studies/final_study/{main,ablations}/*.csv and
outputs/studies/final_study/main/TrafficCD/rugd/seed_17/predictions.npz — nothing here is
invented, estimated, or rounded favourably. This is the exact script used to produce the data
files referenced by paper/figures/*.tex.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs" / "studies" / "final_study"
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)

NAME_MAP = {
    "prior": "Prior", "persistence": "Persistence", "logistic": "Logistic",
    "hist_gbdt": "HGBDT", "lightgbm": "LightGBM", "gru": "GRU", "graph_gru": "Graph-GRU",
    "rugd": "ShiftGuard-RUGD", "uignn_inductive": "UIGNN-Inductive",
    "staeformer_inductive": "STAEformer-Inductive", "opencity_scratch": "OpenCity-Scratch",
}


def main():
    df = pd.read_csv(STUDY / "main" / "results_all.csv")
    agg = df.groupby("model").agg(mean=("severe_auprc", "mean"), sd=("severe_auprc", "std")).reset_index()
    agg["label"] = agg["model"].map(NAME_MAP)
    agg.sort_values("mean")[["label", "mean", "sd"]].to_csv(OUT / "main_severe_auprc.csv", index=False)

    ab = pd.read_csv(STUDY / "ablations" / "results_all.csv")
    abagg = ab.groupby("model").agg(mean=("severe_auprc", "mean")).reset_index()
    full = abagg.loc[abagg.model == "rugd", "mean"].values[0]
    abagg["delta"] = abagg["mean"] - full
    label_map = {
        "rugd": "Full RUGD", "rugd_no_class_balance": "w/o class balance",
        "rugd_no_domain": "w/o domain alignment", "rugd_no_evidence": "w/o evidence",
        "rugd_no_focal": "w/o focal modulation", "rugd_no_graph": "w/o graph diffusion",
        "rugd_no_topology": "w/o topology",
    }
    abagg["label"] = abagg["model"].map(label_map)
    abagg = abagg[abagg.model != "rugd"].sort_values("delta")
    abagg[["label", "delta"]].to_csv(OUT / "ablation_delta.csv", index=False)

    rob = pd.read_csv(STUDY / "main" / "robustness_all.csv")
    focus = ["rugd", "staeformer_inductive", "graph_gru", "gru", "opencity_scratch", "uignn_inductive"]
    for cond in ("missing", "noise", "shift"):
        sub = rob[rob.model.isin(focus) & (rob.corruption == cond)]
        piv = sub.groupby(["model", "severity"])["severe_auprc"].mean().reset_index()
        piv["label"] = piv["model"].map(NAME_MAP)
        piv[["label", "severity", "severe_auprc"]].to_csv(OUT / f"robustness_{cond}.csv", index=False)
        for m in focus:
            m_sub = piv[piv.model == m]
            m_sub[["severity", "severe_auprc"]].to_csv(OUT / f"robustness_{cond}_{m}.csv", index=False)

    bundle = np.load(STUDY / "main" / "TrafficCD" / "rugd" / "seed_17" / "predictions.npz")
    y, p = bundle["targets"], bundle["probabilities"]
    cm = confusion_matrix(y, p.argmax(1), normalize="true")
    labels = ["normal", "warning", "severe"]
    long_rows = [{"true": labels[i], "pred": labels[j], "value": round(float(cm[i, j]), 4)}
                 for i in range(3) for j in range(3)]
    pd.DataFrame(long_rows).to_csv(OUT / "confusion_matrix.csv", index=False)
    grid_rows = [{"x": x, "y": y_, "value": cm[y_, x]} for y_ in range(3) for x in range(3)]
    pd.DataFrame(grid_rows).to_csv(OUT / "confusion_matrix_grid.csv", index=False)

    observed, predicted = calibration_curve((y == 2).astype(int), p[:, 2], n_bins=12)
    pd.DataFrame({"predicted": predicted, "observed": observed}).to_csv(OUT / "calibration_curve.csv", index=False)

    precision, recall, _ = precision_recall_curve((y == 2).astype(int), p[:, 2])
    idx = np.linspace(0, len(precision) - 1, 40).astype(int)
    pd.DataFrame({"recall": recall[idx], "precision": precision[idx]}).to_csv(OUT / "pr_curve.csv", index=False)

    print(f"Wrote figure data to {OUT}")


if __name__ == "__main__":
    main()
