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

## Dobot residual TD3 recipe

`rlt_td3_dual_residual_v1` implements the Explicit Residual-Q architecture and
strict Base freezing described by experiment 18 in the locally supplied
Dobot-RLT project. These Nexus modules are independently implemented; no Dobot
source is vendored or imported by production code or repository tests. The
inspected reference tree contains no LICENSE/NOTICE file; this notice does not
assign an upstream license. Original source attribution is retained here.

Reference snapshot inspected on 2026-10-28, SHA-256:

| Reference file | SHA-256 |
|---|---|
| `rlt/networks.py` | `17057a287817dba52b0c609c2f3965b4bba67a13a4e19fcf800fab55758dd397` |
| `rlt/td3.py` | `8baa38daa557767a35aa4cbe86de6be30e5073ef9380c0499e413cf523ed9693` |
| `rlt/offline.py` | `92eb36c84c3e2c0c9b47b126f55a9a062a3d6691a001f2131a82c0f2b80a4daa` |
| `rlt/replay.py` | `fbb6dbe6405e483b26cce99cbe8dfb197cbeb537db32370132fa8b4aebab65b8` |
| `scripts/pretrain_td3.py` | `e21f05d978220de35fe966185e2d1ea9095fe19ce81a26b1ac651b9930faf3e4` |
| `scripts/rlt_online_session.py` | `7e02add0dd2e9fe6f8130dc59dc377632a7416ab175ad98d73dc866ae30d2f18` |
| `docs/05_DualResidual实验.md` | `d13ff59938c34eeeefd13de0a94af0309049cca9fd8cec0ecfcdd28c1d9ad518` |

The reference path is `/share/lujunyu/Dobot-RLT`. Checkpoint parameter names
are not interchangeable. Nexus defines per-axis codec bounds, separate raw
execution/supervision records, pure-VLA Base bootstrap, explicit acceptance
gates, and durable replay budgeting. These choices are specified in
`docs/rltoken_dobot_recipe.md`; a changed task/codec is a separate experiment.
The reference's experiment-18 robot result is still pending in its experiment
notes. Neither architecture agreement nor CPU parity establishes task benefit.
