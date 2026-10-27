<div align="center">

# Dobot Nexus

**An Open, Modular Embodied AI Platform — VLA Available Today · World Models (WM) in Design · World-Action Models (WAM) on the Roadmap**

<p>
  <a href="pyproject.toml"><img alt="Version" src="https://img.shields.io/badge/version-1.0.0-0A66C2"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green"></a>
  <a href="pyproject.toml"><img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.10%20(CI%203.11)-blue"></a>
  <a href="pyproject.toml"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.10-EE4C2C"></a>
  <a href=".github/workflows/ci.yml"><img alt="CI Workflow" src="https://img.shields.io/badge/CI-CPU%20Contracts%20%26%20Wheel-4c1"></a>
</p>

<p>
  <strong>English</strong> · <a href="README_zh.md">简体中文</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#world-models-wm--world-action-models-wam">WM / WAM</a> ·
  <a href="#documentation-hub">Docs</a> ·
  <a href="SECURITY.md">Security</a>
</p>

</div>

---

**Dobot Nexus** is a contract-driven, PyTorch-native embodied AI codebase supporting **Vision-Language-Action (VLA)** training, inference, and real-robot deployment today, while establishing the shared foundation to grow into **World Models (WM)** and **World-Action Models (WAM)**.

Instead of coupling training loops, inference servers, and robot hardware to a single monolithic model repository, Dobot Nexus separates stable repository contracts (`core/`, `training/`, `inference/`) and shared data utilities (`data/`) from model-owned operators (`models/`), algorithm workflows (`algorithms/`), and reusable neural components (`modules/`). Policy runs managed by `BCTrainer` export **Schema v2 Artifacts** so offline evaluation, HTTP/WebSocket serving, and closed-loop real-robot execution restore the same model target, dimensions, data mappings, and declared training assets.

<p align="center">
  <img src="docs/assets/architecture.png" alt="Dobot Nexus Platform Architecture" width="96%">
</p>

## Why Dobot Nexus?

- **Zero-Registry, Contract-First Extensibility**: Point `model.target` in YAML to any internal or external class/factory whose `from_config` returns a `BaseVLAModule` (owning a `BaseDataAdapter` defined in `src/dobot_nexus/models/base.py`). Model-owned operators stay strictly isolated with no cross-model imports.
- **Unified Policy Train → Artifact → Serve Loop**: `BCTrainer` exports Schema v2 Artifacts that separate portable deployment weights/assets (`deployment/`) from optional distributed resume states (`resume/`). Inference restores the model target, dimensions, data mappings, and declared training assets directly from the Artifact (external `model.target` implementations still require their Python package at deployment).
- **Multi-Paradigm VLA Stack**: Implemented in `v1.0` with CPU contract tests for **Behavior Cloning (BC)**, **Long-Horizon Memory (`Pi0.5-Memory` / `Pi0-Memory`)**, and **Offline RL / Advantage-Conditioned Policies (`RECAP / ACP` for both Pi0.5 and Pi0)** with shared reusable Value networks (`modules/value/`).
- **Numerical & Security Hardening**: Explicit BF16/FP32 precision contracts ([`docs/precision.md`](docs/precision.md)), cross-rank NaN/Inf update guards, `weights_only=True` deserialization, opt-in trusted DCP resume, loopback-by-default HTTP/WebSocket serving with token auth required on non-loopback binds, HTTP session isolation, Memory WS per-connection history, and HMAC-SHA256 Robot TCP v2 transport ([`SECURITY.md`](SECURITY.md)).
- **Real-Robot Integration**: Includes **Dobot Dual-Arm 14D** (3-process WS + TCP v2 bridge), **ATOM D 18D** (documented PC2 single-host closed-loop folding setup with Home-start, `replay_seed` seed-preserving recomputation, up to 4 bridge rows within 2° of the final command of the preceding chunk plus separate 4° handoff seam checks, and interactive terminal control), and **ATOM W 0759 18D** hardware runtime.
- **Architectural Path to WM & WAM**: Active design package ([`docs/wm/`](docs/wm/README.md)) proposing how to extend Nexus's shared config loader, `Experiment`, `Runtime`, and tracking to **Wan2.2** video world models and later action-conditioned embodied dynamics toward WAM.

---

## Model, Paradigm & Embodiment Matrix

| Family | Paradigm / Recipe | Model / Target | Training | Inference & Serving | Real Robot | Status |
| --- | --- | --- | --- | --- | --- | --- |
| **VLA** | Behavior Cloning (BC) | **Pi0.5** (`Pi05Model`) | Local / Accelerate / FSDP2, EMA, DCP | Offline, HTTP, WS, Torch inference (optional CUDA Graph) | Dual-Arm 14D, ATOM D 18D, ATOM W 0759 18D | **Implemented (`v1.0`, CPU contracts tested)** |
| **VLA** | Long-Horizon Memory | **Pi0.5-Memory** (`Pi05MemoryModel`) | Frozen vision feature store + BC loop | Per-connection session history WS (`MemorySession`) | Dual-Arm 14D | **Implemented (`v1.0`, CPU contracts tested)** |
| **VLA** | Offline RL / ACP | **Pi0.5-RECAP** (`Pi05ReCapModel`) | Offline Value (`Fusion`, separate `.pt` ckpt) + advantage labeling + ACP | Conditional policy HTTP / WS | Dual-Arm 14D | **Implemented (`v1.0`, CPU contracts tested)** |
| **VLA** | Behavior Cloning (BC) | **Pi0** (`Pi0Model`, state/action each 1–32D) | Local / Accelerate / FSDP2, EMA, DCP | Offline, HTTP | — | **Implemented (`v1.0`, CPU contracts tested)** |
| **VLA** | Long-Horizon Memory | **Pi0-Memory** (`Pi0MemoryModel`, 1–32D) | Frozen SigLIP feature store (`pi0.memory.prepare`) + BC loop | Per-session history (`MemorySession` / `pi0.memory.infer`) | — | **Implemented (`v1.0`, CPU contracts tested)** |
| **VLA** | Offline RL / ACP | **Pi0-RECAP** (`Pi0ReCapModel`, 1–32D) | Offline Value (`Fusion`, separate `.pt` ckpt) + advantage labeling + ACP | Conditional policy Offline / HTTP | — | **Implemented (`v1.0`, CPU contracts tested)** |
| **VLA** | Online RL | **Pi0.5 RLToken** | Token reconstruction, rollout, actor and twin critics | Frozen feature Artifact and actor | External environment adapter | [Implementation](docs/rltoken_implementation.md) · [Recipes](examples/configs/rltoken/) |
| **WM** | Video Generation (T2V / I2V) | **Wan2.2** (proposed `TI2V-5B` first, then `I2V-A14B`) | Proposed flow matching, LoRA / dual-expert | Proposed offline generator, then async job server | — | **In Design** ([`docs/wm/`](docs/wm/README.md)) |
| **WAM** | Action-Conditioned Dynamics & World-Action Unification | Embodied WM + Policy Direction | Proposed action-conditioned rollout (planner external) | Research direction beyond WM v1 | — | **Roadmap** |

> **Scope & verification note**: All **VLA** rows above are implemented in `v1.0` and covered by fast CPU unit/contract tests in CI (live GPU training, FSDP2 scaling, CUDA Graph speedups, and robot task success require your own GPU/hardware verification; see [`docs/robots/atom_d_inference.md`](docs/robots/atom_d_inference.md) for the documented ATOM D field setup). **WM (Wan2.2)** currently consists of design and pseudocode specifications under [`docs/wm/`](docs/wm/README.md) with no runnable WM CLI/code yet; **WAM** is a longer-term platform direction.

### Supported Data & Robot Contracts

| Layer | Supported Formats / Embodiments | Reference |
| --- | --- | --- |
| **Datasets** | Shared policy reader supports LeRobot v2.1 (Parquet embedded JPEG/PNG or Parquet + MP4, not mixed per dataset) and delegates v3 video reading to upstream LeRobot; offline RECAP requires LeRobot **v2.1** metadata; Dobot dual-arm 14D recorder PKL → LeRobot v2.1 converter (`BC`, `RECAP` success/intervention tags, `extra_fields`) | [`docs/getting_started/02_data.md`](docs/getting_started/02_data.md), [`data_processing/README.md`](data_processing/README.md) |
| **Dobot Dual-Arm 14D** | Metadata-driven Pi0.5 WS policy server (`Pi05Model` / `Pi05ReCapModel` / `Pi05MemoryModel`) + TCP v2 (`msgpack` + HMAC-SHA256) robot bridge | [`examples/real_robot/dobot_dual_arm_14d/README.md`](examples/real_robot/dobot_dual_arm_14d/README.md) |
| **ATOM D (18D)** | Single-host PC2 closed-loop bimanual T-shirt folding, Home-start, async prefetch with `replay_seed` seed-preserving recomputation (up to 4 bridge rows within 2° of the final command of the preceding chunk; separate 4° seam & consistency checks; auto-falls back to sync on older servers), interactive operator (`Enter` / `h` / `Ctrl+C`) | [`docs/robots/atom_d_inference.md`](docs/robots/atom_d_inference.md) |
| **ATOM W 0759 (18D)** | `atom18` layout contract (`examples/real_robot/shared/layouts/atom18.json`), digest-pinned SDK/URDF/gains with live PC1 position limits & calibration verified before control, `hold`/`policy` head mode | [`examples/real_robot/atom_w_0759/README.md`](examples/real_robot/atom_w_0759/README.md) |
| **ATOM DH (24D)** | Hardware field contract specification (distinct from 18D; no silent padding/truncation) | [`docs/robots/atom_dh_24d.md`](docs/robots/atom_dh_24d.md) |

---

<a id="从哪里开始"></a>
## Quickstart

### 1. Clone & Create a Model Environment

Package constraints: Python `>=3.10` (Python `3.11` used in CI) and PyTorch `2.10` (`torch>=2.10,<2.11`). For the GPU training/inference recipe below, use Linux with BF16-capable NVIDIA GPUs (CPU local execution and FP32 inference are also supported; see [`docs/environments.md`](docs/environments.md) and [`docs/robots/atom_d_inference.md`](docs/robots/atom_d_inference.md) for field-specific environments).

```bash
git clone https://github.com/fengzx99/dobot-nexus.git
cd dobot-nexus

# Create an isolated Pi0.5 environment (with HTTP serve + WebSocket extras)
python3.11 scripts/env/create_model_env.py --model pi05 --extra serve --extra websocket
source .venvs/pi05/bin/activate
python scripts/env/check_model_env.py --model pi05 --extra serve --extra websocket
```

Or install directly into an existing compatible environment using recommended extras (`pi05`, `pi0`, `accelerate`, `serve`, `websocket`, `tensorboard`, `tracking`, `legacy-data`, `dev`; `openpi` is also retained as a compatibility alias for `websocket`):

```bash
pip install -e ".[pi05,accelerate,serve,websocket]"
```

### 2. Train a Pi0.5 BC Policy & Monitor with TensorBoard

The default `examples/configs/train/pi05.yaml` expects a 14D LeRobot dataset mapped to the three required camera slots (`base`, `left_wrist`, `right_wrist`); use `model.state_dim`, `model.action_dim`, and `data.field_mapping` to remap other datasets (if either public vector exceeds the default internal width of 32, also increase `model.model_action_dim`, e.g., `64` in the shipped 34D recipe). `examples/train.sh` launches a single process for 1 GPU or `torch.distributed.run` for multiple GPUs, and the template's `runtime.launcher: auto` resolves to `local` (1 process) or `accelerate` with FSDP2 (multi-process).

```bash
# Multi-GPU (or single-GPU via --gpus 0) Pi0.5 BC training
bash examples/train.sh \
  --config examples/configs/train/pi05.yaml \
  --gpus 0,1 \
  --run-name pi05_demo \
  -- data.path=/path/to/lerobot_dataset model.checkpoint=/path/to/pi05_base_torch

# Or invoke the installed CLI directly (single process, matching run name)
dobot-nexus train --config examples/configs/train/pi05.yaml \
  exp.name=pi05_demo \
  data.path=/path/to/lerobot_dataset model.checkpoint=/path/to/pi05_base_torch
```

Training templates enable TensorBoard by default. Launch the dashboard on `127.0.0.1:6006`:

```bash
dobot-nexus tensorboard --logdir runs --port 6006
```

### 3. Run Offline Inference or Start the Loopback HTTP Policy Server

For ordinary Pi0.5 BC / RECAP and Pi0 BC / RECAP policy Artifacts, pass a run directory, `latest_deployment.txt` pointer, or `step_XXXXXXXX` directory (`Pi0.5-Memory` uses its dedicated session-history WebSocket server, and `Pi0-Memory` uses `python -m dobot_nexus.models.pi0.memory.infer --config examples/configs/infer/pi0_memory.yaml --input ... --output ...` or `MemorySession`):

```bash
# Offline action-chunk prediction (.json or .npz observation -> actions.npz)
bash examples/infer.sh \
  --model pi05 \
  --artifact runs/pi05_demo/checkpoints/latest_deployment.txt \
  --input /path/to/observation.npz \
  --output actions.npz

# Start the loopback HTTP policy server (127.0.0.1:12021; set NEXUS_POLICY_TOKEN for non-loopback binds)
bash examples/infer.sh \
  --model pi05 \
  --artifact runs/pi05_demo/checkpoints/latest_deployment.txt \
  --serve
```

Equivalent direct CLI commands (add `CUDA_VISIBLE_DEVICES=0` / `runtime.device=cuda` for wrapper parity):

```bash
dobot-nexus infer --config examples/configs/infer/pi05.yaml \
  --input /path/to/observation.npz --output actions.npz \
  model.artifact=runs/pi05_demo/checkpoints/latest_deployment.txt runtime.device=cuda

dobot-nexus serve --config examples/configs/infer/pi05.yaml \
  model.artifact=runs/pi05_demo/checkpoints/latest_deployment.txt runtime.device=cuda
```

### 4. Deploy to a Real Robot

- **Step-by-step Pi0.5 onboarding (5 guides, including the 18D T-shirt folding training guide in [§3.8](docs/getting_started/03_training.md#38-训练-18-维叠衣服任务))**: [`01_environment.md`](docs/getting_started/01_environment.md) → [`02_data.md`](docs/getting_started/02_data.md) → [`03_training.md`](docs/getting_started/03_training.md) → [`04_inference.md`](docs/getting_started/04_inference.md) → [`05_deployment.md`](docs/getting_started/05_deployment.md).
- **ATOM D 18D closed-loop folding**: [`docs/robots/atom_d_inference.md`](docs/robots/atom_d_inference.md) (`./examples/real_robot/atom_d/run_atom_d.sh`, deployment recipe: [`examples/recipes/infer/pi05_atom18.yaml`](examples/recipes/infer/pi05_atom18.yaml)).
- **Dobot Dual-Arm 14D (BC / Memory / RECAP)**: [`examples/real_robot/dobot_dual_arm_14d/README.md`](examples/real_robot/dobot_dual_arm_14d/README.md).

---

## Architecture

### Repository Layout

```text
dobot-nexus/
├── src/dobot_nexus/
│   ├── cli/                   # Unified CLI: train / infer / serve / tensorboard
│   ├── core/                  # Config resolution, contracts, Experiment, Schema v2 Artifacts
│   ├── data/                  # Shared LeRobot readers, image conversion, observation IO, stats & cache
│   ├── training/              # BCTrainer, Local/Accelerate/FSDP2 runtime, EMA, DCP resume, tracking
│   ├── inference/             # InferencePolicy, PolicySession, HTTP/WS servers, security guards
│   ├── algorithms/recap/      # Returns, advantages, offline Value training & dataset labeling
│   ├── algorithms/rltoken/    # Token, actor/critic, replay and online training
│   ├── modules/value/         # Reusable Fusion value networks, distributions & losses
│   └── models/                # BaseVLAModule/BaseDataAdapter contract + isolated models (pi05/, pi0/)
├── data_processing/           # Dobot dual-arm 14D recorder PKL -> LeRobot v2.1 converter
├── examples/                  # Train/infer YAML configs, task recipes, real-robot & ATOM runtimes
├── scripts/                   # Model venv builders/checkers and wheel packaging smoke test
├── docs/                      # Getting started, model/precision guides, robot contracts & WM design
└── tests/                     # Fast CPU contract, security, packaging & entrypoint unit tests
```

### Extending the Platform (`model.target`)

New VLA models implement `BaseVLAModule` (owning a `BaseDataAdapter`) from `src/dobot_nexus/models/base.py`—inside `src/dobot_nexus/models/<name>/` or in any external Python package—and plug into `BCTrainer`, `ArtifactManager`, and the generic `InferencePolicy` / HTTP server via YAML:

```yaml
model:
  target: my_pkg.modeling_custom:CustomVLAModel
  state_dim: 18
  action_dim: 18
  action_horizon: 50
  precision: bf16
```

Each policy checkpoint saved by `BCTrainer` under `runs/<exp_name>/checkpoints/step_XXXXXXXX/` uses **Schema v2 Artifact** separation (RECAP offline Value training writes its own separate `.pt` checkpoint):

```text
step_XXXXXXXX/
├── manifest.yaml               # model.target, specs, asset manifest, resume status (ready/disabled/pruned); model-supplied metadata (e.g., Pi0.5 numerical profile)
├── config_resolved.yaml        # Fully resolved training configuration snapshot
├── deployment/
│   ├── pytorch/full_weights.pt # Exported deployment weights
│   ├── nexus/full_weights.pt   # Optional (model-dependent; built-in Pi0 / Pi0.5 do not export this file)
│   ├── memory/vision.npz       # Optional (Pi0.5-Memory / Pi0-Memory frozen vision encoder only)
│   └── assets/                 # Conditional (created when a model declares assets such as norm_stats or tokenizer)
└── resume/                     # Optional (omitted if resume_limit=0 or pruned by retention): DCP shards, trainer v3, per-rank RNG
```

At inference time, `InferencePolicy` restores the target, dimensions, data mappings, declared assets (`OPENPI_TOKENIZER_PATH` can override the Pi0.5 tokenizer), and Pi0.5 numerical profile; inference compute/parameter precision and outer `autocast` are chosen at launch, and robot clients must match training camera geometry. See [`docs/nexus_architecture_and_integration_guide.md`](docs/nexus_architecture_and_integration_guide.md), [`docs/code_integration_and_change_rules.md`](docs/code_integration_and_change_rules.md), and [`AGENTS.md`](AGENTS.md) for strict module ownership rules.

---

## World Models (WM) & World-Action Models (WAM)

Dobot Nexus plans to expand beyond reactive VLA policies toward a unified **VLA + WM / WAM** platform. The design specification lives under [`docs/wm/`](docs/wm/README.md) (**status: design draft, not yet implemented — no WM CLI or runnable code yet**):

1. **Layer 1 — Video World Model Foundation (Wan2.2, Proposed)**:
   - Proposes WM contracts and componentized multi-part Artifacts (DiT, VAE, text encoder, LoRA / dual-expert checkpoints) while reusing Nexus's shared config loader, `Experiment`, `Runtime` (Local / Accelerate / FSDP), seeding, and TensorBoard/W&B tracking.
   - Recommends starting with **Wan2.2 TI2V-5B** (single DiT) followed by **I2V-A14B** (high/low-noise dual experts), beginning with offline generation before adding an asynchronous job server, built on a physical-timestamp video clip contract (`RGB uint8[B,T,H,W,3]`, true `timestamps` & `fps`, episode-level split isolation).
2. **Layer 2 — Action-Conditioned Dynamics & Longer-Term WAM Direction**:
   - Plans action-conditioned embodied future prediction $p(o_{t+1:t+H} \mid o_{\le t}, s_t, a_{t:t+H}, \text{prompt})$ with explicit frame-interval action alignment and offline rollout evaluation (explicitly keeping closed-loop planning/control outside first-release WM scope), establishing the building blocks for future **World-Action Models (WAM)**.

Explore the WM design package:
- [`docs/wm/README.md`](docs/wm/README.md) — Scope and status summary
- [`docs/wm/PLAN.md`](docs/wm/PLAN.md) — Proposed architecture, interfaces, phased delivery, and acceptance criteria
- [`docs/wm/PSEUDOCODE.md`](docs/wm/PSEUDOCODE.md) — Proposed data, training, generation, and Artifact call chains
- [`docs/wm/SOURCES.md`](docs/wm/SOURCES.md) — Reviewed upstream Wan2.2 file snapshots and SHA-256 log

---

## Security, Precision & Quality Verification

- **Deployment & Deserialization Security ([`SECURITY.md`](SECURITY.md))**:
  - HTTP and WebSocket servers bind to loopback (`127.0.0.1`) by default (on loopback, requests are allowed without a bearer token when `NEXUS_POLICY_TOKEN` is unset). Binding non-loopback interfaces requires a ≥32 UTF-8 byte `NEXUS_POLICY_TOKEN` (`Authorization: Bearer <token>`); `/healthz` is always public.
  - HTTP `/sessions` issues isolated random session capabilities (900-second / 15-minute idle TTL, up to 16 live sessions, 8 MiB request body limit, serialized model calls; explicit `/predict` `options.num_steps` overrides are bounded to `1–100`). WebSocket bounds incoming and outgoing messages to 32 MiB; Memory WS isolates per-connection history, whereas ordinary BC/RECAP WS shares one prediction handler and seed counter across connections.
  - Robot TCP v2 (`examples/real_robot/dobot_dual_arm_14d/_runtime/wire.py`) enforces `msgpack` + mutual nonce + sequence numbers + HMAC-SHA256 (`NEXUS_ROBOT_TOKEN`) with no pickle fallback. Authentication does not encrypt payloads—use an encrypted tunnel or VPN across untrusted networks.
  - Deployment `.pt` weights and training state v3 sidecars use `torch.load(..., weights_only=True)`; exact PyTorch DCP resume is opt-in via `train.trust_resume_checkpoint=true`. Treat YAML configs, `model.target` Python packages, Artifacts, DCP checkpoints, and recorder PKL files as trusted inputs.
- **Precision Contract ([`docs/precision.md`](docs/precision.md))**:
  - Pi0/Pi0.5 default to BF16 compute with FP32 master parameters and Adam states during training; inference defaults to BF16 parameters with explicit `model.inference_parameter_precision` and `runtime.autocast` (`auto` / `"off"`) controls.
- **CPU Contract Tests & Wheel Verification**:

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
python -m compileall -q src tests scripts examples/real_robot
python scripts/check_wheel.py
```

---

<a id="进阶参考"></a>
## Documentation Hub

| Category | Document | Scope |
| --- | --- | --- |
| **Getting Started** | [01 Environment](docs/getting_started/01_environment.md) · [02 Data](docs/getting_started/02_data.md) · [03 Training](docs/getting_started/03_training.md) · [04 Inference](docs/getting_started/04_inference.md) · [05 Deployment](docs/getting_started/05_deployment.md) | End-to-end Pi0.5 BC walkthrough, LeRobot v2.1 checks, and [18D T-shirt folding training (§3.8)](docs/getting_started/03_training.md#38-训练-18-维叠衣服任务) |
| **Models & Precision** | [Pi0.5 Training & Deploy](docs/pi05_training.md) · [Pi0 Training & Infer](docs/pi0_training.md) · [Precision Matrix](docs/precision.md) · [Environments](docs/environments.md) | Data semantics, normalization, FSDP2, resume, compilation, dtype boundaries, and venvs |
| **Algorithms & Memory** | [Pi0.5 Memory](docs/pi05_memory.md) · [RECAP / ACP](docs/recap/README.md) · [Data Conversion](data_processing/README.md) | History feature store, offline Value + advantage labeling, and Dobot 14D PKL → LeRobot conversion |
| **World Models (WM/WAM)** | [WM Overview](docs/wm/README.md) · [WM Plan](docs/wm/PLAN.md) · [Pseudocode](docs/wm/PSEUDOCODE.md) · [Sources](docs/wm/SOURCES.md) | Proposed WM/Wan2.2 architecture draft and embodied dynamics design |
| **Real-Robot & Recipes** | [Real-Robot Index](examples/real_robot/README.md) · [ATOM D 18D Inference](docs/robots/atom_d_inference.md) · [ATOM W 0759](examples/real_robot/atom_w_0759/README.md) · [ATOM DH 24D](docs/robots/atom_dh_24d.md) · [Dual-Arm 14D](examples/real_robot/dobot_dual_arm_14d/README.md) · [Task Recipes](examples/recipes/README.md) · [ATOM Deployment Recipe](examples/recipes/infer/pi05_atom18.yaml) · [All Examples](examples/README.md) | Robot launchers, hardware contracts, USB/Button/KBD training recipes, and ATOM 18D deployment recipe |
| **Engineering & Governance** | [Architecture Guide](docs/nexus_architecture_and_integration_guide.md) · [Integration Rules](docs/code_integration_and_change_rules.md) · [AGENTS.md](AGENTS.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md) · [Tests Guide](tests/README.md) | Module boundaries, contribution workflow, security boundaries, and CPU contract tests |

---

## Roadmap

<p align="center">
  <img src="docs/assets/capability-map.png" alt="Dobot Nexus Capability and Evolution Roadmap" width="96%">
</p>

- **`v1.0` (Implemented)**: Pi0.5 (BC / Memory / RECAP) and Pi0 (BC / Memory / RECAP, 1–32D) training & inference, Schema v2 Artifacts, TensorBoard/W&B tracking, security & precision hardening, CPU contract & wheel CI workflow, Dual-Arm 14D & ATOM 18D real-robot runtimes.
- **Near-Term VLA Enhancements**: Configurable `zscore`/`quantile` state-action normalization for Pi0 (tracked in [`TODO.md`](TODO.md)) and formal converter support for raw 18D frame-wise collection data.
- **Next Milestone — WM & WAM Direction**: Implement core WM contracts and proposed Wan2.2 (`TI2V-5B` / `I2V-A14B`) training & generation pipeline, add action-conditioned embodied dynamics prediction, and evolve toward unified World-Action Models (WAM).

## License & Provenance

Dobot Nexus is licensed under the **[Apache License 2.0](LICENSE)**.

- Third-party and Dobot provenance notices are recorded in [`NOTICE`](NOTICE), [`src/dobot_nexus/algorithms/recap/NOTICE.md`](src/dobot_nexus/algorithms/recap/NOTICE.md), and [`examples/real_robot/atom_common/NOTICE.md`](examples/real_robot/atom_common/NOTICE.md).
- Model-owned Apache-2.0 operator adaptations are attributed in [`src/dobot_nexus/models/pi05/torch_core/NOTICE.md`](src/dobot_nexus/models/pi05/torch_core/NOTICE.md) and [`src/dobot_nexus/models/pi0/training/vendor/rlinf_core/NOTICE.md`](src/dobot_nexus/models/pi0/training/vendor/rlinf_core/NOTICE.md).
- Downloaded pretrained weights, datasets, hardware SDKs, and external Python dependencies remain subject to their respective licenses.
