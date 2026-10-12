# OpenCity research protocol

## Data and leakage controls

The study uses only released TrafficCD, TrafficHZ, and TrafficSH speed arrays and road-network
adjacency matrices. Twelve 30-minute masked history bins predict the next bin. Zeros are treated as
missing observations. Severe/warning thresholds are source-training 10th/30th speed percentiles.

Each fold holds one city out and uses only its final chronological segment for evaluation. Source
cities provide chronological train, calibration, and validation segments separated by an embargo.
Normalization, labels, class weights, early stopping, temperature scaling, and conformal thresholds
never use the evaluated target labels. Random K-fold splitting is prohibited.

## Compared methods

The protocol runs source prior, persistence, balanced logistic regression, histogram GBDT,
LightGBM, GRU, Graph-GRU, ShiftGuard-RUGD, and inductive task adaptations of UIGNN, STAEformer, and
OpenCity. Every final method runs seeds 17, 42, and 2025. RUGD/UIGNN receive 20 trials per city;
the slower Transformer adaptations receive eight compute-matched trials. Direct RUGD ablations
remove graph transport, topology, domain alignment, evidence, focal modulation, and class balance.

## Evaluation

Primary metrics are macro-F1 and severe AUPRC. Secondary metrics include severe precision,
recall/F1/AUROC, balanced accuracy, NLL, Brier, ECE, ACE, uncertainty-error association, APS
coverage/set size, throughput, and parameters. Stress tests use 10/30/50% missingness,
normalized noise 0.1/0.3, and one/two-bin temporal rolls while reusing clean calibration.

Paired NLL differences are aggregated by timestamp, bootstrap-resampled by timestamp cluster, tested
with Wilcoxon, and Holm-corrected. Nodes within one timestamp are not independent replicates.
Headline 30%/50%-missing AUPRC comparisons use nine paired city-seed scores, exact two-sided
Wilcoxon tests, rank-biserial effects, hierarchical city-then-seed bootstrap intervals, and Holm
correction. Because seeds on one city are not new geographies, an exact sign-flip analysis over
the three city means is also reported as a conservative sensitivity check. Seeds are never
cherry-picked.

## Development and confirmation policy

The first TrafficSH smoke (rugd_smoke_v1) contained 213,377 valid node predictions, 15,924 severe
examples, and 276 usable timestamp blocks. It exposed a clean-versus-missing robustness tradeoff and
was used to design RUGD v2. TrafficSH is therefore a development/pilot city, not an untouched
confirmatory test after v2.

The final normalized-transport architecture was promoted after all three OpenCity folds had
been inspected. Accordingly, the final study is explicitly labeled exploratory development evidence.
Further architectural claims require a new external confirmation dataset or additional unseen
cities; optimization seeds do not create independent geographic replications.
