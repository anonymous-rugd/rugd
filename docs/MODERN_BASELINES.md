# Modern paper-derived baseline protocol

The original `gru` and `graph_gru` entries are strong recurrent controls. They are not
implementations of OpenCity, UIGNN, or STAEformer. The extension adds three separately named,
clean-room adaptations under the same three-class, source-only leave-one-city-out protocol.

| Code name | Paper mechanism retained | Required adaptation |
|---|---|---|
| `staeformer_inductive` | Input, time-of-day, day-of-week, and adaptive embeddings; temporal then spatial vanilla self-attention | Dataset-indexed node embeddings become topology-conditioned embeddings so unseen graph sizes are valid; output is a 3-class head |
| `uignn_inductive` | Three bidirectional diffusion graph convolutions, a residual hidden layer, random-location recovery, and evidential uncertainty | Continuous/NIG forecasting becomes Dirichlet classification under the common severe-event task |
| `opencity_scratch` | Temporal patch embedding, time context, graph propagation, RMSNorm, and gated feed-forward blocks | The 288-to-288 regression model becomes a 12-to-1 classifier and is trained from source data only |

Released OpenCity checkpoints are intentionally not loaded: the official pretraining configuration
lists TrafficCD and TrafficHZ among pretraining datasets, which would contaminate leave-one-city-out
tests for those cities. Therefore `opencity_scratch` tests the architecture family, not target-exposed
foundation pretraining. Published numbers are not directly comparable because the labels, horizon,
pretraining data, and split differ.

Final `rugd` uses ordinary degree-normalized transport. Graph diffusion, topology,
domain alignment, evidence, class balancing, and focal modulation are registered for matched
one-factor ablations. The final retained study requires external-city
confirmation before any broad generalization claim.
