"""Generate paired block-bootstrap/Holm comparisons from suite predictions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .statistics import block_bootstrap_difference, holm_adjust, paired_score_inference


def compare_suite(root,reference="shiftguard",repeats=2000):
    root=Path(root); rows=[]
    for target_dir in [p for p in root.iterdir() if p.is_dir()]:
        ref_files=sorted((target_dir/reference).glob("seed_*/predictions.npz"))
        if not ref_files: continue
        for model_dir in [p for p in target_dir.iterdir() if p.is_dir() and p.name!=reference]:
            for ref,other in zip(ref_files,sorted(model_dir.glob("seed_*/predictions.npz"))):
                a,b=np.load(ref),np.load(other)
                if len(a["targets"])!=len(b["targets"]) or not np.array_equal(a["targets"],b["targets"]): continue
                loss_ref=-np.log(a["probabilities"][np.arange(len(a["targets"])),a["targets"]].clip(1e-9))
                loss_other=-np.log(b["probabilities"][np.arange(len(b["targets"])),b["targets"]].clip(1e-9))
                result=block_bootstrap_difference(loss_ref,loss_other,a["time_ids"],repeats,int(ref.parent.name.split("_")[-1]))
                rows.append({"target_city":target_dir.name,"baseline":model_dir.name,"seed":ref.parent.name,**result})
    frame=pd.DataFrame(rows)
    if len(frame): frame["holm_p"]=holm_adjust(frame["wilcoxon_p"].tolist())
    frame.to_csv(root / "paired_significance.csv", index=False)
    compare_robustness_scores(root, reference, max(repeats, 20000))
    return frame




def compare_robustness_scores(root, reference="shiftguard", repeats=20000):
    """Compare 30/50% missingness AUPRC using paired city-seed scores.

    Holm correction is reported across every baseline and across learned baselines.
    Prior and persistence are controls, not members of the learned-model family.
    """
    root = Path(root)
    path = root / "robustness_all.csv"
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    data = data[(data.corruption == "missing") & data.severity.isin([0.3, 0.5])]
    rows = []
    baselines = sorted(set(data.model) - {reference})
    for severity in (0.3, 0.5):
        condition = data[data.severity == severity]
        ref = condition[condition.model == reference][
            ["target_city", "seed", "severe_auprc"]
        ].rename(columns={"severe_auprc": "reference_score"})
        for baseline in baselines:
            other = condition[condition.model == baseline][
                ["target_city", "seed", "severe_auprc"]
            ].rename(columns={"severe_auprc": "baseline_score"})
            paired = ref.merge(other, on=["target_city", "seed"], validate="one_to_one")
            if paired.empty:
                continue
            result = paired_score_inference(
                paired.reference_score.to_numpy(), paired.baseline_score.to_numpy(),
                paired.target_city.to_numpy(), repeats,
                seed=20260912 + int(100 * severity),
            )
            rows.append({"corruption": "missing", "severity": severity,
                         "baseline": baseline, **result})
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["holm_p_all_baselines"] = holm_adjust(frame.wilcoxon_p.tolist())
        frame["holm_p_learned_baselines"] = np.nan
        learned = ~frame.baseline.isin(["prior", "persistence"])
        frame.loc[learned, "holm_p_learned_baselines"] = holm_adjust(
            frame.loc[learned, "wilcoxon_p"].tolist()
        )
    frame.to_csv(root / "paired_robustness_significance.csv", index=False)
    return frame
