# Generated research summary

This report summarizes completed artifacts under `outputs\studies\final_study\main`. It does not claim state of the art.

## Methodology and novelty

The study uses strict leave-one-city-out transfer and chronological train/calibration/validation/test
segments separated by an embargo. Congestion thresholds and normalization statistics are estimated
from source-city training periods only. RUGD combines degree-normalized graph diffusion with
local and final evidential supervision, ID-free temporal/topology encoders, source-domain reversal,
focal rare-event learning, and source-only temperature/APS calibration. Modern comparisons are explicitly adapted STAEformer, UIGNN, and
source-only OpenCity-style backbones; they are not claimed as exact published-protocol reproductions.

## Results

Across the completed city/seed runs, the highest mean severe-event AUPRC belongs to **rugd**.
See `paper_table_main.csv` for mean +/- SD across registered seeds/cities and `results_all.csv` for every
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
