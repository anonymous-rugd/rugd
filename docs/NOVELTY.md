# Novelty, model-selection history, and claim boundary

ShiftGuard-RUGD is a compact zero-shot cross-city severe-congestion system. Its final architecture
combines a node-shared masked-history GRU, transferable topology/calendar features,
degree-normalized multi-hop graph diffusion, local and final Dirichlet supervision,
effective-number focal learning, and source-city adversarial alignment.

None of these primitives is claimed as individually unprecedented. The candidate contribution is
the experimentally controlled integration of rare-event learning, calibrated evidential output,
inductive graph transfer, and robustness evaluation under a strict chronological held-out-city
protocol, while remaining substantially smaller than the Transformer comparison.

## Study scope

The final retained study reports all configured held-out cities and fixed seeds.
The current benchmark remains development evidence; independent confirmation requires new cities
or an external dataset.
TrafficSH also influenced earlier architecture development. No state-of-the-art or universal
generalization claim is permitted without broader geographic evaluation, faithful external
reproduction, and independent confirmation.
