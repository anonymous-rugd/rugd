# Final retained study artifacts

This directory contains the complete artifacts used by the final results report and reviewer audit.
No seed, city, configured robustness condition, or registered model result has been omitted.

## Contents

- `main/`: per-city and per-seed predictions, checkpoints, metrics, aggregate tables, figures, and statistical tests for the 11-model benchmark.
- `ablations/`: matched three-city, three-seed RUGD component ablations and aggregate analysis.
- `configs/`: frozen resolved configurations and tuned hyperparameters.
- `FINAL_STUDY_MANIFEST.json`: SHA-256 checkpoint and prediction hashes for the retained neural runs.
- `main.log`, `tuning.log`, and `ablations.log`: original execution logs.

Historical run identifiers and original output paths inside logs or per-run provenance files are retained verbatim for auditability. Current reviewer-facing entry points use the neutral `final_study` name.

Start with `docs/FINAL_RESULTS.md` for a concise interpretation and `main/aggregate/` for machine-readable final tables.