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

## RLT-in-Practice pre-clip residual TD3 recipe

`rlt_td3_residual_v1` implements the pre-clip bounded residual TD3 recipe,
legal known-success behavior anchoring, dynamic Q/BC weight schedule,
single-stage joint offline prewarm, `position_bottleneck` Token decoder mode,
and opt-in experimental switches (same-sample latent residual actor, Q-gated
success anchor, Critic hidden LayerNorm with bootstrap-only target Q clipping,
Sigmoid/BCE Twin-Q, and optional Terminal RankQ) described in the locally
inspected `RLT-in-Practice` research repository.

These Nexus modules are independently implemented; no `RLT-in-Practice` source
is vendored or imported by production code or repository tests. Checkpoint
parameter names are not interchangeable. Nexus retains its own feature/codec
contracts, durable SQLite Replay admission and recovery format
(`rltoken_td3_resume_v3`), explicit prewarm handoff checks (`rltoken_prewarm_v2`),
and three-role RLinf execution boundary. See `docs/rltoken_td3_recipe.md` for
the full specification.

Reference snapshot inspected on 2026-11-24, SHA-256:

| Reference file | SHA-256 |
|---|---|
| `research/rlt/preclip_bc_objectives.py` | `64c84da868bb1387a2987f9e6a12ca89ffaad8d3d2b814dabdc66e94a3459af4` |
| `research/rlt/preclip_residual_policy.py` | `688e84dc9b82c757867f444eb15340c96da3ccab2dcaa6f46f72f1dd72eed6bc` |
| `research/rlt/preclip_bc_workers.py` | `d2aa88bb69808e333b3fb0871067f3fbfb124ef69297bac4d62f06c46a2a8ee3` |
| `research/rlt/native_residual_anchor.py` | `bc2429c3f713a2ef2dfd8d5321403cb576c3388b1f6484936e20e02ce2515b64` |
| `research/rlt/single_sample_latent_policy.py` | `31256f127ae5d039a5818a3464ad944379e04fdfb065cde2f2fe544090f6b64e` |
| `research/rlt/latent_residual_policy.py` | `61d539525147bf3da1788532002cddf88ed2252af2fee48d00cced16c14fe534` |
| `research/rlt/latent_q_gated_anchor_worker.py` | `ebbfdc19bbaf8f5656f60aac43172e95b95e0591fd62e0912e62d217c33b0a0d` |
| `research/rlt/latent_sigmoid_bce.py` | `983659d0722437b170edacf1fdabfc5df41d91277ff6c39e54a48441919f00a0` |
| `research/rlt/success_anchor_td3_worker.py` | `30877cc6bbd1e5f52ee2affcd5e6be57f1b57df229a99c0db3384235e17b3597` |
| `research/rlt/actors/bounded_residual.py` | `22e3d9300b86035bc21e15922d65108dcc1a9d0180f36bedf6e52c1de95c8721` |
| `research/rlt/actors/latent_residual.py` | `f84480fef8b11fd8fed2ec265a6f769cb9595343c3199fdafbe0207c3e2f22d1` |
| `research/rlt/losses/success_behavior_anchor.py` | `c6329b30edd8161b788221b0cd347738f8e61aa24812fd17acf5b492f8bfd691` |
| `research/rlt/losses/q_gated_success_anchor.py` | `27d2598e5ad4d3100ea7d3fc9a7b5086348fb82cde06786dc8fcc73c1e4fea54` |
| `research/rlt/losses/terminal_rankq.py` | `d479f7c5b887bd713a7584815c4d6eee4da1688f8f67967f223ed577824c030b` |
| `research/rlt/critics/ln_target_clip.py` | `8dfc16c13d93a4b8421627df2ad2281f0eff1105f1e447bc22ebd551fdbc0bb0` |
| `research/rlt/representation/selectable_decoder.py` | `fd36dd81fd4c2ba0d8e26c825fa139030b46173674a2f183500ffad4eb042d9a` |
| `research/rlt/replay/episode_outcome_index.py` | `eef175be227e3ad3adaa4769b20f913d4e8a5e3daefb0ff33f305d6f7976984b` |
| `.bootstrap/rlinf-v0.4/rlinf/models/embodiment/mlp_policy/rlt_td3_mlp_policy.py` | `8860ba244c8f1253ec0965878e2d6214075d6f7cd71e118ab91014f5aa353ee0` |
| `.bootstrap/rlinf-v0.4/rlinf/models/embodiment/modules/rlt_token_transformer.py` | `d72364f58436af345164d9bb574d7f7edf054e5c9d1e94d367ae075b53779332` |
| `.bootstrap/rlinf-v0.4/rlinf/workers/actor/fsdp_rlt_td3_policy_worker.py` | `8030aa840b5d869c90417a442423ac7fc8b832d82c5e73701cfa0bc8398b4c08` |

CPU contract tests verify structural, gradient, and lifecycle semantics only;
GPU training throughput and real-robot/simulation task performance require
independent environment evaluation.
