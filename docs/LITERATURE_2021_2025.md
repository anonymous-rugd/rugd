# Literature and novelty audit (2021--2025 papers only)

Last checked 2026-09-13. Older papers are intentionally excluded by the project requirement.
Published scores are methodological context, not directly comparable results.

| Work | Year | What it establishes | Gap relative to this study |
|---|---:|---|---|
| [Traffic4Cast 2021](https://proceedings.mlr.press/v176/eichenberger22a.html) | 2022 | Few-shot transfer across cities and COVID-era temporal shifts. | Grid regression rather than source-only severe-node classification with calibrated uncertainty. |
| [DeepSTUQ](https://arxiv.org/abs/2208.05875) | 2022 | Aleatoric/epistemic traffic uncertainty, ensembling, MC dropout, and calibration. | Fixed-network forecasting; no leave-one-city-out rare-event protocol. |
| [UIGNN](https://ieeexplore.ieee.org/document/10415744/) | 2023 | Uncertainty-aware graph forecasting and imputation under missing observations or locations. | Established networks rather than zero-shot severe-state classification in unseen cities. |
| [PDFormer](https://ojs.aaai.org/index.php/AAAI/article/view/25556) | 2023 | Propagation-delay-aware short/long-range traffic dependencies. | Transductive continuous forecasting on fixed nodes; no cross-city calibration. |
| [STAEformer](https://arxiv.org/abs/2308.10425) | 2023 | Strong adaptive spatiotemporal embeddings and Transformer forecasting. | Learned node embeddings are not directly transferable to unseen road graphs; rare-event uncertainty is not central. |
| [TESTAM](https://openreview.net/forum?id=N0nTk5BSvO) | 2024 | Mixture of spatial experts for dynamically changing traffic conditions. | In-situ continuous forecasting, not unseen-city severe classification. |
| [SFMGTL](https://doi.org/10.1016/j.trc.2024.104604) | 2024 | Semantic fusion, graph granularity, shared/private memory, and adversarial cross-city transfer. | Target-scarcity demand/flow setting; not strict source-only conformal rare-event evaluation. |
| [Deep-ensemble traffic uncertainty](https://doi.org/10.1109/TITS.2024.3381099) | 2024 | Predictive uncertainty for deep traffic forecasting using ensembles. | Continuous forecasting; not a compact single-model held-out-city severe classifier. |
| [OpenCity](https://arxiv.org/abs/2408.10269) | 2024/2025 | Large heterogeneous pretraining for zero-shot traffic forecasting and released data/models. | Large regression foundation model; rare severe classes and the registered corruption/calibration suite are not its focus. |
| [Robust traffic forecasting against spatial shift](https://arxiv.org/abs/2410.00373) | 2024 | Cross-year OOD benchmarks and mixture-of-graphon experts for changing spatial relations. | Continuous cross-year regression, not cross-city severe classification or conformal uncertainty. |
| [Personalized federated learning for cross-city prediction](https://www.ijcai.org/proceedings/2024/0611.pdf) | 2024 | Privacy-aware cross-city personalization. | Requires target personalization; not strict zero-shot evaluation. |
| [MDTLGCN](https://doi.org/10.1016/j.trc.2025.105065) | 2025 | Cross-city graph/domain transfer designed around missing traffic data. | Continuous flow transfer; no joint rare-class, evidential/conformal, and corruption protocol. |

## Baseline-selection rationale

Persistence and source prior expose dataset shortcuts. Balanced logistic regression tests whether
history, topology, and calendar features are linearly sufficient. Histogram GBDT and LightGBM are
strong non-neural tabular controls. A shared GRU isolates temporal encoding, while Graph-GRU adds
adjacency without domain alignment or evidential learning; neither is attributed to a recent paper.

The modern suite uses transparent task adaptations rather than relabeling old controls.
`staeformer_inductive` retains temporal/spatial attention and adaptive/time embeddings but replaces
fixed sensor identities with transferable topology. `uignn_inductive` retains bidirectional
diffusion, random-location recovery, and evidential output. `opencity_scratch` retains patch/time/graph
blocks, RMSNorm, and gated feed-forward layers and is trained from scratch on source cities because
released OpenCity pretraining includes benchmark cities. These are compute-matched adaptations to a
shared classification protocol, not exact reproductions of incompatible regression experiments.

## Exact research gap

Recent work separately studies zero-shot city transfer, missing traffic data, rare events, predictive
uncertainty, and spatial or temporal shift. This audit found no 2021--2025 study that jointly evaluates
(1) fully held-out-city classification, (2) explicitly rare severe states, (3) evidential and
source-calibrated conformal uncertainty, (4) missing/noisy/time-shift stress tests, and (5) paired
time-block-aware inference in one lightweight protocol.

## Defensible novelty statement

RUGD is a compact inductive architecture that couples a shared masked temporal encoder, transferable
topology/calendar context, degree-normalized graph diffusion, an auxiliary local evidential head,
final Dirichlet prediction, and source-domain adversarial alignment. The local head constrains useful
pre-diffusion representations while the final head quantifies predictive evidence after spatial
fusion. Class-balanced focal learning targets rare severe states; source-only temperature scaling and
adaptive prediction sets provide post-hoc calibration. Individual ingredients have precedents, so the
claim is the task-specific architecture and unified evaluation design, not invention of each part.

The final architecture was selected after exploratory development on these benchmark cities.
Consequently, the final study is strong internal evidence but not an untouched external confirmation.
Publication-quality validation still requires a preregistered frozen replication on additional cities,
independent reproduction, and broader systematic review. No universal or state-of-the-art claim is made.
