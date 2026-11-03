# RLToken provenance

The `rlt_ac_fixed_v1` recipe follows RLinf's RLToken actor/critic objectives,
detached prefix reconstruction and executed-intervention reference replacement
at revision `7e90641c4135229db6d3426950cdb45cee6259a6`:

- `rlinf/models/embodiment/modules/rlt_token_transformer.py`
- `rlinf/models/embodiment/mlp_policy/rlt_mlp_policy.py`
- `rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py`
- `rlinf/algorithms/rlt/transition.py`

Nexus uses its own dimension-configurable Torch modules and explicit transport,
Replay admission and recovery contracts. Its Transformer/MLP implementation is
not a numerically identical import of those models; checkpoint keys are not
interchangeable. Comparative numerical/performance validation is separate from
CPU contract tests.

RLinf Cluster, WorkerGroup, Channel and AsyncWeightSyncMixin are imported from
the optional pinned dependency, not copied. RLinf is Copyright 2025–2026 The
RLinf Authors and licensed under Apache License 2.0; see the accompanying LICENSE.

Phase routing follows the readiness/critical-phase separation in RLinf commit
`7e90641c4135229db6d3426950cdb45cee6259a6`, specifically
`rlinf/algorithms/rlt/route.py` (`SimulatorRLTRoute`),
`rlinf/envs/sim/maniskill/maniskill_rlt_env.py`, and
`rlinf/envs/real/wrappers/episode/policy_switch.py` (inspected 2026-10-29).
Nexus independently implements per-observation routing and replay eligibility;
task predicates, keyboard handling, phase latching and episode resets remain in
the environment adapter. No RLinf environment code is vendored by this feature.
