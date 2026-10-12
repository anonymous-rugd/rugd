# OpenCity output reference

For every `outputs/studies/<study-id>/main/<target>/<model>/seed_<seed>/`:

- `provenance.json`: Python, PyTorch/CUDA, platform, and complete resolved config.
- `target_definition.json`: source-train median/IQR, class thresholds, and fit scope.
- `model.pt` or `model.joblib`: frozen deep checkpoint or classical estimator.
- `training_history.csv`: epoch loss and source-validation metrics (deep models).
- `metrics.csv`: clean target-test discrimination, rare-event, proper-score,
  calibration, conformal, uncertainty, efficiency, and size metrics.
- `predictions.npz`: probabilities, targets, uncertainty, and timestamp IDs needed to
  reproduce every metric and paired test; deep runs also retain node IDs.
- `robustness.csv`: test-only missingness, normalized-noise, and lag-roll results using
  the unchanged clean calibration parameters.

Root outputs:

- `data_manifest.json`: byte lengths and SHA-256 hashes of all six input files.
- `results_all.csv`: every clean seed result; this is the audit source of truth.
- `results_summary.csv` / `results_overall.csv`: mean and sample SD by city/model or model.
- `robustness_all.csv` / `robustness_summary.csv`: every corruption result and aggregates.
- `paired_significance.csv`: ShiftGuard-RUGD-minus-baseline paired NLL differences. Negative
  favors ShiftGuard. Intervals resample timestamps, not individual spatial nodes.
- `paired_robustness_significance.csv`: paired severe-AUPRC differences at 30% and 50%
  missingness. Positive favors ShiftGuard-RUGD. Reports mean/SD/median differences, hierarchical
  city-then-seed bootstrap intervals, win counts, rank-biserial effects, exact two-sided Wilcoxon
  p-values, Holm corrections, and an exact city-level sign-flip sensitivity test.
- `ensemble_results.csv`: fixed-three-seed predictive performance and mean/95th-percentile
  epistemic mutual information.
- `<target>/<model>/ensemble_predictions.npz`: mean probabilities, predictive entropy,
  expected entropy, and their difference (mutual information).
- `benchmark_summary.csv`: compact mean±SD table used by the generated report.
- `figures/*.pdf`: severe AUPRC, confusion, calibration/PR, and robustness figures.
- `RESEARCH_SUMMARY.md`: generated narrative that reports the observed winner without
  assuming the proposed model wins.

The ablation stage writes the same contract to `outputs/studies/<study-id>/ablations` so confirmatory
main results and component analysis are not accidentally mixed.
