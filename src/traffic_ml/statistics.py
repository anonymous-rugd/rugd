"""Paired, sample-level inference; never treats edges from one timestamp as independent."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon


def block_bootstrap_difference(loss_a: np.ndarray, loss_b: np.ndarray, blocks: np.ndarray,
                               repeats: int = 2000, seed: int = 0) -> dict[str, float]:
    """Bootstrap paired mean loss difference by time block to preserve spatial dependence."""
    unique = np.unique(blocks)
    per_block = np.array([(loss_a[blocks == b] - loss_b[blocks == b]).mean() for b in unique])
    rng = np.random.default_rng(seed)
    draws = rng.choice(per_block, size=(repeats, len(per_block)), replace=True).mean(1)
    lo, hi = np.quantile(draws, [.025, .975])
    try: stat, p = wilcoxon(per_block)
    except ValueError: stat, p = 0., 1.
    return {"mean_difference": float(per_block.mean()), "ci95_low": float(lo),
            "ci95_high": float(hi), "wilcoxon_statistic": float(stat), "wilcoxon_p": float(p),
            "n_time_blocks": len(unique)}


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values); order = np.argsort(p); adjusted = np.empty_like(p)
    running = 0.
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx]); adjusted[idx] = min(1., running)
    return adjusted.tolist()



def paired_score_inference(reference: np.ndarray, baseline: np.ndarray,
                           groups: np.ndarray | None = None,
                           repeats: int = 20000, seed: int = 0) -> dict[str, float]:
    """Infer paired run-score differences without treating nodes as replicates.

    Wilcoxon and rank-biserial statistics use paired city-seed scores. The
    confidence interval resamples pairs; with city labels, it uses a hierarchical
    city-then-seed bootstrap. An exact city-level sign-flip test exposes the weak
    geographic inference possible with only three held-out cities.
    """
    reference = np.asarray(reference, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if reference.shape != baseline.shape or reference.ndim != 1 or not len(reference):
        raise ValueError("reference and baseline must be non-empty paired vectors")
    difference = reference - baseline
    rng = np.random.default_rng(seed)

    if groups is None:
        draws = rng.choice(difference, size=(repeats, len(difference)), replace=True).mean(1)
        n_groups, group_wins, group_signflip_p = 0, 0, np.nan
    else:
        groups = np.asarray(groups)
        if groups.shape != difference.shape:
            raise ValueError("groups must align with paired scores")
        unique = np.unique(groups)
        grouped = [difference[groups == group] for group in unique]
        draws = np.empty(repeats, dtype=float)
        for draw in range(repeats):
            selected = rng.integers(0, len(grouped), size=len(grouped))
            sample = [rng.choice(grouped[index], size=len(grouped[index]), replace=True)
                      for index in selected]
            draws[draw] = np.concatenate(sample).mean()
        group_means = np.array([values.mean() for values in grouped])
        signs = np.array(np.meshgrid(*([[-1.0, 1.0]] * len(group_means)))).T.reshape(
            -1, len(group_means)
        )
        null = (signs * group_means).mean(1)
        observed = abs(group_means.mean())
        group_signflip_p = float(np.mean(np.abs(null) >= observed - 1e-15))
        n_groups, group_wins = len(unique), int((group_means > 0).sum())

    nonzero = difference[difference != 0]
    if len(nonzero):
        method = "exact" if len(nonzero) <= 50 and len(nonzero) == len(difference) else "auto"
        test = wilcoxon(difference, alternative="two-sided", method=method)
        ranks = rankdata(np.abs(nonzero))
        rank_biserial = float(
            (ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / ranks.sum()
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)
    else:
        statistic, p_value, rank_biserial = 0.0, 1.0, 0.0

    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_difference": float(difference.mean()),
        "sd_difference": float(difference.std(ddof=1)) if len(difference) > 1 else 0.0,
        "median_difference": float(np.median(difference)),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "wins": int((difference > 0).sum()),
        "n_pairs": len(difference),
        "rank_biserial": rank_biserial,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": p_value,
        "n_groups": n_groups,
        "group_wins": group_wins,
        "group_signflip_p": group_signflip_p,
    }
