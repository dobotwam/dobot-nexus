# Dobot Nexus contributor guide

This repository is an algorithm-facing VLA mapping layer. Keep the public contracts small and
stable. Under `src/dobot_nexus/`, put model-specific integration in `models/<model>/`,
algorithm-specific computation and workflows in `algorithms/<algorithm>/`, and reusable
computational components in `modules/<component>/`. Experiments follow the same ownership boundaries.

## Boundaries

- `core/`, `training/`, and `inference/` define stable repository contracts.
- `models/base.py` defines the only framework-facing model and data-adapter contracts.
- Concrete model implementations may live inside or outside this repository and are selected through
  `model.target`; do not add another repository-specific subclass protocol.
- `models/pi05/modeling_pi05.py` and `data_adapter_pi05.py` provide the reference Pi0.5 implementation.
- Each model owns its operators. Pi0.5 shares `models/pi05/torch_core/` between training and
  inference, using ordinary Python imports with no import-time compilation. Pi0 keeps its
  independent optional snapshot under `models/pi0/training/vendor/rlinf_core/`. Keep both free
  of RLinf workers, Ray, schedulers, registries, and policy wrappers; never import one model's
  implementation from another model.
- `models/pi05/data/` owns data adaptation; `training/` owns checkpoint loading and explicit
  per-model training helper compilation; `openpi_torch/` owns inference compilation and CUDA
  Graph replay. Numerical profiles belong in `torch_core/`. Pi0.5 always uses its local
  Torch core. Custom implementations are selected through `model.target`.
- External dependencies must be imported lazily and must produce actionable error messages.
- Training and inference must use the same model target and Artifact manifest.
- Training saves Torch deployment weights and model-owned assets directly from trained
  parameters, without requiring a base checkpoint during Artifact saving.
- Production Pi0.5 training and deployment are Torch-only. Keep optional reference engines,
  converters, environment tools, and their tests under Git-ignored `local_script/`;
  never import them from `src/` or `tests/`. Keep numerical profile identifiers and
  Artifact metadata stable independently of optional reference execution.

## Algorithms and shared modules

- `algorithms/<algorithm>/` owns recipes, algorithm-specific computation, labeling, training jobs,
  optimizer state, distributed execution and recovery. Algorithms may use public contracts,
  shared data readers, `modules/` and stable training utilities; do not import concrete VLA models.
- RECAP returns, advantages, annotations, Value training and checkpoint semantics belong in
  `algorithms/recap/`. Pi0.5 conditional policy integration belongs in `models/pi05/modeling_recap.py`,
  `data_adapter_recap.py`, `configuration_recap.py` and `data/recap.py`.
  Model integrations may use algorithm computation and data helpers, but not algorithm Trainers
  or distributed execution. ACP policy training uses the existing `BCTrainer` and model contracts.
- `modules/value/` owns reusable Fusion networks, input processing, distributions and losses.
  Callers own frame identities, reward/return semantics, camera order, statistics and device placement.
- `modules/memory/` owns reusable model-agnostic FrameSample memory types, relative time encoding,
  history sampling, Torch projection/attention layers, per-environment sessions, and verified
  feature storage. `modules/serving/` owns model-agnostic OpenPI and Memory WebSocket protocol
  adapters over `inference/transport/`.
- Shared `modules/` must not import `algorithms/`, concrete models under `models/`, dataset
  readers, or training/inference job orchestrators. `inference/transport/` must stay protocol-only
  and never absorb model names, camera layouts, or memory state.
- Shared dataset reading, image conversion (`resize_with_pad`), `NormStats`, and generic
  normalization/action transforms (`zscore`, `quantile`, integer-safe `delta`) belong in `data/`.
  Model tokenization, augmentation, embodiment routing, and model-specific codecs stay with the
  model. Keep package initializers lightweight and optional dependency checks local to consumers.

## Extending with new models

- Land new model families (for example ACT, GR00T, or future VLA/WM architectures) self-contained
  under `src/dobot_nexus/models/<model>/` with their own `configuration_<model>.py`,
  `modeling_<model>.py`, `data_adapter_<model>.py`, operators, licenses, and `requirements.txt`.
- Integrate solely through `BaseVLAModule` and `BaseDataAdapter` in `models/base.py` and select via
  `model.target`; do not introduce intermediate model-family base classes (e.g., `BaseDiffusionModel`
  or `BaseVLM`) or a global model registry.
- Reuse existing `data/`, `training/`, `inference/`, and `modules/` (including `modules/serving/`
  for WebSocket deployment) from day one, while keeping zero cross-imports between sibling models
  under `models/`.
- Defer new shared abstractions until at least two distinct models converge on identical,
  model-agnostic semantics; only then extract the common computation into `modules/<component>/`.

## Extending with RL algorithms

- Keep `models/base.py` as the sole model/adapter inheritance entry point; do not add a separate
  `BaseRLModel` hierarchy. Extend RL support via optional capability methods on `BaseVLAModule` /
  `BaseDataAdapter` that raise `NotImplementedError` by default (e.g., detached feature extraction
  and executed-action encoding for frozen-VLA residual RL, or log-probability evaluation for
  end-to-end policy-gradient RL), paired with typed tensor contracts under `core/`.
- Keep `BCTrainer` scoped to single-model supervised and offline-reweighted training (such as BC
  and RECAP ACP). Route multi-optimizer or online-rollout RL workflows through `algorithm.target`
  from the common `cli/train.py` entrypoint with a uniform `(config, experiment) -> TrainResult`
  Trainer contract.
- Keep algorithm-specific networks, losses, transitions, Replay/journals, learner update loops,
  weight synchronization, and recovery inside `algorithms/<algorithm>/`. Place model-specific RL
  adapters (such as feature extraction, Stage-1 wrappers, or action codecs) under
  `models/<model>/<algorithm>/`. Algorithms must construct feature providers via `model.target` or
  Artifacts without statically importing concrete models; model adapters must not import Trainers
  or distributed Workers.
- Isolate distributed RL runtimes (such as RLinf or Ray) inside lazy optional execution adapters
  loaded only when `Trainer.run()` executes. Keep Env/Rollout/Replay/Learner business logic in
  plain Python/Torch objects so unit and protocol tests run on CPU without a live cluster, and
  importing BC, deployment, or shared `modules/` never requires RL runtime dependencies.
- Keep multi-optimizer/Replay training recovery in an explicit algorithm-owned checkpoint format,
  separate from deployment. Export deployable RL policies as standard public v2 Artifacts whose
  composed policy implements `BaseVLAModule` (excluding critics and Replay) so `InferencePolicy`
  and `modules/serving/` load them transparently.
- Do not extract a shared RL execution layer (Replay/Rollout/Worker runtime) until at least two
  online RL algorithms share the same execution semantics; never place Ray/RLinf or job
  orchestration inside `modules/` or `core/`.

## Documentation

- Write for developers and users encountering the repository for the first time.
- Describe current capabilities, responsibilities, requirements, and runnable commands.
  Keep removed implementations, migration history, and before/after comparisons in Git history.
- Preserve active configuration names, dependency requirements, source attribution, and licenses.

## Checks

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Add a CPU test for every change to the stable contracts. Keep `tests/` limited to module and
contract tests that do not depend on local debug tools. Keep smoke, distributed/model parity,
benchmark scripts, and their diagnostic tests and reports under the Git-ignored `local_script/`.

Place algorithm tests in `tests/algorithms/<algorithm>/`, shared component tests in
`tests/modules/<component>/`, and model integration tests in `tests/models/<model>/`.
Test dependency isolation and saved Artifact compatibility on CPU.

## Git commits

- When asked to commit a large worktree change, organize the work into three responsibility
  groups unless the user specifies another grouping. Break each group into focused subtasks
  and commits; three groups does not mean only three commits.
- Give each commit one independently reviewable purpose within a submodule. Include its
  directly related tests, configuration, and documentation. Do not bundle unrelated features,
  broad cleanup, or large code additions merely because they share a directory.
- Split mixed-purpose files by hunk. Keep inseparable implementation and contract changes
  together, and order commits by dependency so intermediate revisions remain usable.
- Follow the existing English Conventional Commits style:
  `<type>(<scope>): <imperative summary>`, for example
  `perf(memory): reuse history projections across denoising steps`.
  Use a specific submodule scope, keep the subject within 72 characters, and describe one
  concrete behavior. Avoid vague subjects such as "update code" or lists of unrelated changes.
- Use a short body when needed to explain the reason, affected behavior, and validation.
  Report only checks actually run and distinguish final-worktree checks from checks of an
  individual commit. Do not imply that CPU tests validate GPU performance or numerical parity.
- Inspect the staged diff before every commit and stage explicit paths or hunks. Preserve
  existing user changes; do not commit ignored training outputs, weights, local benchmarks,
  diagnostic reports, credentials, or unrelated files.
- Run checks appropriate to the affected behavior and complete the repository checks above
  before handing off the batch. Report commit hashes and purposes under the three groups,
  together with validation results and any remaining uncommitted changes.
