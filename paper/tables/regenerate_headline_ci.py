"""Regenerate outputs/studies/final_study/main/headline_bootstrap_ci.csv.

Uses the SAME hierarchical city-then-seed bootstrap as the paired-comparison inference in
src/traffic_ml/statistics.py::paired_score_inference (resample the 3 cities with replacement,
then within each resampled city resample its 3 seed-level scores with replacement, concatenate,
take the mean; repeat 20,000 times; report the 2.5/97.5 percentiles) -- applied here to a single
model's metric rather than a paired difference, so the headline table's CIs are constructed with
the same methodology as every other confidence interval in this paper, not a different one.

Run from the repository root: python paper/tables/regenerate_headline_ci.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "outputs" / "studies" / "final_study"


def hierarchical_ci(values_by_city, repeats=20000, seed=0):
    rng = np.random.default_rng(seed)
    cities = list(values_by_city.keys())
    draws = np.empty(repeats)
    for d in range(repeats):
        chosen = rng.integers(0, len(cities), size=len(cities))
        sample = [rng.choice(values_by_city[cities[c]], size=len(values_by_city[cities[c]]), replace=True)
                  for c in chosen]
        draws[d] = np.concatenate(sample).mean()
    return np.percentile(draws, [2.5, 97.5])


def main():
    df = pd.read_csv(STUDY / "main" / "results_all.csv")
    rows = []
    for model, g in df.groupby("model"):
        for metric in ("macro_f1", "severe_auprc", "nll"):
            by_city = {c: sub[metric].values for c, sub in g.groupby("target_city")}
            seed = abs(hash(model + metric)) % (2**31)
            lo, hi = hierarchical_ci(by_city, seed=seed)
            rows.append({"model": model, "metric": metric, "mean": g[metric].mean(),
                         "sd": g[metric].std(), "ci_lo": lo, "ci_hi": hi, "n": len(g)})
    out = pd.DataFrame(rows)
    out.to_csv(STUDY / "main" / "headline_bootstrap_ci.csv", index=False)
    print(f"Wrote {STUDY / 'main' / 'headline_bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()
