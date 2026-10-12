# ShiftGuard-RUGD: final architecture and objective

## Robust uncertainty-guided diffusion

For node i, a city-shared GRU encodes normalized speed and its observation mask over the previous
L bins into temporal state h_i. A transferable topology/calendar encoder produces s_i. A local
head emits Dirichlet concentration alpha_local_i = softplus(z_i) + 1 and receives an auxiliary
source-supervised evidential loss. This encourages uncertainty to remain grounded in each node's
own history before spatial exchange, but it does not alter graph transport.

At graph layer l, RUGD uses degree-normalized diffusion:

    m_i = sum_j A_ij h_j / max(sum_j A_ij, epsilon).

A learned residual update combines h_i and m_i. Repeating this operation provides a compact
multi-hop spatial receptive field without node IDs or city-specific embeddings. The resulting
representation is fused with topology/calendar context and passed to a final evidential
three-class head.

## Rare-event and transfer objectives

The final objective combines effective-number class weighting, focal evidential classification,
the local evidential auxiliary loss, and gradient-reversal source-city alignment. Revised RUGD does
not use corrupted-view consistency. All objectives are fitted on source cities only.

## Source-only model selection

Every deep model uses the same city-balanced source-validation objective:

    0.40 clean macro-F1 + 0.30 clean severe AUPRC
    + 0.15 missing-input macro-F1 + 0.15 missing-input severe AUPRC.

Metrics are computed per source city and averaged before combination, preventing larger graphs
from dominating selection. Target-city labels never enter tuning or checkpoint selection.

## Uncertainty and calibration

The final Dirichlet head provides expected class probabilities and vacuity. Temperature scaling
and adaptive prediction-set conformal calibration use source calibration data only. Fixed
three-seed ensembles additionally report predictive entropy and epistemic mutual information.

## Novelty boundary

GRUs, graph diffusion, evidential learning, focal loss, and domain-adversarial training each have
prior art. The candidate contribution is their compact integration as an ID-free, source-only
cross-city classifier optimized and evaluated jointly for rare severe events, calibration, and
controlled distribution shifts. Novelty is therefore protocol-and-system level rather than a claim
that any individual primitive is unprecedented.
