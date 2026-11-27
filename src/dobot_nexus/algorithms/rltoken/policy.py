from __future__ import annotations

import secrets
from dataclasses import replace
from pathlib import Path

import torch

from dobot_nexus.core.artifacts import resolve_artifact
from dobot_nexus.core.contracts import PolicySpec, TrainOutput
from dobot_nexus.models.base import BaseDataAdapter, BaseVLAModule

from .actor_critic import build_actor, build_critic
from .artifacts import (
    _feature_spec,
    _link_feature_bundle,
    _read_feature_manifest,
    _read_retained_source_manifest,
    _restore_deployment_config,
    _retained_source,
    _source_export_restrictions,
    _verify_retained_feature_bundle,
    load_provider,
    write_model_bundle,
)
from .configuration import HeadConfig
from .identity import content_digest
from .losses import actor_loss, critic_loss
from .routing import PolicySwitch, routed_actions


def sampling_seed(seed, stream):
    # Stable named streams independent of Python's process-randomized hash().
    if seed is None:
        return None
    return (int(seed) + {"reference": 0x45D9F3B, "actor": 0x119DE1F3}[stream]) % (2**63 - 1)


def _provider_tensors(provider):
    yield from provider.named_parameters(remove_duplicate=False)
    for name, buffer in provider.named_buffers(remove_duplicate=False):
        module_name, _, local_name = name.rpartition(".")
        module = provider.get_submodule(module_name)
        if local_name not in module._non_persistent_buffers_set:
            yield name, buffer


def _provider_fingerprint(provider, spec, key):
    """Hash every state byte on its own device, independent of placement and Torch versions.

    Two keyed universal hashes per 1 KiB block detect even untracked .data writes.
    Only block checksums cross devices. Integer arithmetic wraps identically on CPU/CUDA.
    """
    fingerprints = {}
    keys = {}
    for name, tensor in _provider_tensors(provider):
        if tensor.device not in keys:
            keys[tensor.device] = key.to(tensor.device)
        weights = keys[tensor.device]
        width = weights.shape[1]
        flat = tensor.detach().reshape(-1)
        if not flat.numel():
            empty = torch.empty((0, weights.shape[0]), dtype=torch.int64, device="cpu")
            fingerprints[name] = (str(tensor.dtype), tuple(tensor.shape), empty)
            continue
        if not flat.is_contiguous() or flat.stride() != (1,):
            unit = torch.empty(flat.shape, dtype=flat.dtype, device=flat.device)
            flat = unit.copy_(flat)
        raw = flat.view(torch.uint8)
        checksums = []
        for chunk in raw.split(1024 * width):
            padding = -chunk.numel() % width
            if padding:
                chunk = torch.nn.functional.pad(chunk, (0, padding))
            blocks = chunk.reshape(-1, width)
            checksums.append(torch.stack([(blocks * w).sum(dim=1) for w in weights], dim=1))
        fingerprints[name] = (str(tensor.dtype), tuple(tensor.shape), torch.cat(checksums).cpu())
    return content_digest((spec, fingerprints))


class ReplayAdapter(BaseDataAdapter):
    def encode_train_batch(self, raw_batch, device):
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }

    def encode_observation(self, observation):
        raise RuntimeError("Learner has no VLA; observation encoding belongs to Rollout.")

    def decode_actions(self, actions, *, aux=None):
        raise RuntimeError("Learner has no action codec; decoding belongs to Rollout.")


class RLTokenPolicy(BaseVLAModule):
    """Concrete composition of a frozen feature provider and a replaceable small Actor."""

    _source_allowed_assets = None
    _source_mode_ceilings = None
    _source_extra_mode_ceilings = None

    def __init__(self, config, *, mode, retained_feature_root=None):
        super().__init__()
        if mode not in {"train", "infer"}:
            raise ValueError("RLTokenPolicy mode must be train or infer.")
        self.config = config
        enclosing_digest = None
        if mode == "infer" and config.model.get("artifact"):
            self.config, enclosing_digest = _restore_deployment_config(config)
        self.mode = mode
        self.policy_switch = PolicySwitch.from_config(self.config)
        self.head_config = HeadConfig(**self.config.model.rltoken)
        configured = self.config.model.features.artifact
        root = resolve_artifact(configured)
        retained = retained_feature_root
        if retained is None:
            retained = (
                configured if enclosing_digest is not None else _retained_source(configured, root)
            )
        if retained is not None and Path(retained).resolve() != root:
            raise ValueError("Retained feature directory redirects away from the loaded provider.")
        if retained is not None and enclosing_digest is None:
            manifest, digest = _read_retained_source_manifest(root)
        else:
            manifest, digest = _read_feature_manifest(root)
        if enclosing_digest is not None and enclosing_digest != digest:
            raise ValueError("Deployment feature manifest checksum mismatch.")
        self.feature_spec, metadata = _feature_spec(manifest), manifest["model"]
        if (
            self.head_config.chunk_size != self.config.model.action_horizon
            or self.feature_spec.action_dim != self.config.model.action_dim
        ):
            raise ValueError("Actor chunk/action dimensions disagree with public model dimensions.")
        if metadata["state_dim"] != self.config.model.state_dim:
            raise ValueError("Raw robot state_dim must match the feature provider.")
        self._policy_spec = PolicySpec(
            "rltoken",
            "rlt_td3_residual_v1" if self.head_config.residual is not None else "rlt_ac_v1",
            tuple(metadata["camera_names"]),
            self.config.model.state_dim,
            self.feature_spec.action_dim,
            self.head_config.chunk_size,
            tuple(metadata.get("image_size", (224, 224))),
        )
        residual = self.head_config.residual is not None
        with torch.random.fork_rng(devices=[], enabled=residual):
            if residual:
                algorithm = self.config.get("algorithm", {})
                learning = algorithm.get("learning", {}) if isinstance(algorithm, dict) else {}
                torch.random.default_generator.manual_seed(learning.get("seed", 0))
            self.actor = build_actor(self.feature_spec, self.head_config)
            if mode == "train":
                self.critic = build_critic(self.feature_spec, self.head_config)
        if mode == "train":
            self.adapter = ReplayAdapter()
        else:
            self._retained_root = None
            (self._source_allowed_assets, self._source_mode_ceilings,
             self._source_extra_mode_ceilings) = _source_export_restrictions(
                 root, manifest, self.feature_spec
             )
            files = {}
            if retained is not None:
                files = _verify_retained_feature_bundle(
                    retained, self.feature_spec, require_sealed=retained_feature_root is not None,
                    manifest_digest=digest,
                )
                if all(not f.identity.mode & 0o222 for f in files.values()):
                    self._retained_files = files
                    self._retained_root = root
            self.provider, self.provider_config, loaded_spec = load_provider(
                root, _verified_files=(root, files), expected_manifest_sha256=digest,
            )
            if loaded_spec != self.feature_spec:
                raise ValueError("Loaded provider differs from the selected feature spec.")
            self.provider.eval().requires_grad_(False)
            if self._retained_root is not None:
                generator = torch.Generator().manual_seed(secrets.randbits(63))
                self._fingerprint_key = torch.randint(
                    2**62, (2, 1024), generator=generator, device="cpu"
                )
                self._frozen_fingerprint = _provider_fingerprint(
                    self.provider, self.feature_spec, self._fingerprint_key
                )
            self.adapter = self.provider.adapter
            checkpoint = self.config.model.get("checkpoint")
            if checkpoint:
                self.actor.load_state_dict(
                    torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
                )
            execute = self.config.get("inference", {}).get(
                "execute_steps", self.head_config.chunk_size
            )
            if execute != self.head_config.chunk_size:
                raise ValueError(
                    "RLToken fixed chunk requires inference.execute_steps == chunk_size."
                )

    @classmethod
    def from_config(cls, config, *, mode):
        return cls(config, mode=mode)

    @property
    def policy_spec(self):
        return self._policy_spec

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "provider"):
            self.provider.eval()
        return self

    def build_train_dataloader(self, config):
        raise RuntimeError(
            "RLTokenPolicy uses Replay; select RLTokenTrainer with algorithm.target."
        )

    def training_step(self, batch):
        if self.mode != "train":
            raise RuntimeError("training_step requires a learner policy.")
        if self.head_config.residual is not None:
            raise RuntimeError("Residual TD3 updates belong to TD3Learner.")
        if batch["update"] == "critic":
            loss = critic_loss(
                self.critic(batch["latent"], batch["state"],
                            batch.get("critic_actions", batch["executed"])), batch["target"]
            )
            return TrainOutput(loss, {"critic_loss": loss.detach()})
        if batch["update"] == "actor":
            loss, metrics = actor_loss(
                self.actor,
                self.critic,
                batch,
                batch["learning_config"],
                generator=batch.get("generator"),
            )
            return TrainOutput(loss, metrics)
        raise ValueError("RLToken batch.update must be actor or critic.")

    def warmup_control(self):
        return {"rlt_switch_flags": [True]}

    @torch.no_grad()
    def predict_action_chunk(self, observation, options):
        if self.mode != "infer":
            raise RuntimeError("Learner policy does not load the frozen VLA.")
        phase, enabled, record = self.policy_switch.select(observation)
        base_seed = (
            options.seed if options.seed is not None else (0 if options.deterministic else None)
        )
        reference_steps = options.num_steps
        if reference_steps is None:
            algorithm = self.config.get("algorithm", {})
            reference_steps = algorithm.get("reference_num_steps") if isinstance(algorithm, dict) else None
            if reference_steps is None and self.head_config.residual is not None:
                reference_steps = 10
        reference_options = replace(options, seed=sampling_seed(base_seed, "reference"),
                                    num_steps=reference_steps)
        packet = self.provider.extract_rl_features(observation, reference_options).validate(
            self.feature_spec, chunk_size=self.head_config.chunk_size
        )
        device = next(self.actor.parameters()).device
        seed = sampling_seed(options.seed, "actor")
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        actions = routed_actions(self.actor,
            packet.latent.to(device),
            packet.state.to(device),
            packet.reference_actions.to(device),
            enabled, self.head_config.chunk_size,
            deterministic=options.deterministic,
            generator=generator,
        )
        output = self.adapter.decode_actions(actions.detach().cpu(), aux=packet.action_context)
        output.aux.update(actor_switch=enabled.copy(), record_transition=record.copy(),
                          critical_phase=None if phase is None else phase.copy())
        return output

    def save_nexus_weights(self):
        return False

    def export_live_state_dict(self):
        return {
            "actor." + name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in self.actor.state_dict().items()
        }

    def export_backend_state_dict(self, state_dict):
        result = {
            key.removeprefix("actor."): value
            for key, value in state_dict.items()
            if key.startswith("actor.")
        }
        if set(result) != set(self.actor.state_dict()):
            raise ValueError("Deployment export requires a complete Actor state.")
        return result

    def export_deployment_artifacts(self, *, source_weights, artifact_dir):
        if not hasattr(self, "provider"):
            raise RuntimeError(
                "Deploy from Rollout's loaded provider at a synchronized Actor version."
            )
        destination = artifact_dir / "deployment/features"
        retained = getattr(self, "_retained_root", None)
        if retained is None:
            write_model_bundle(
                destination, self.provider, self.provider_config,
                allowed_asset_files=self._source_allowed_assets,
                mode_ceilings=self._source_mode_ceilings,
                extra_mode_ceilings=self._source_extra_mode_ceilings,
            )
        else:
            live = _provider_fingerprint(self.provider, self.feature_spec, self._fingerprint_key)
            if live != self._frozen_fingerprint:
                raise RuntimeError("Frozen provider or feature specification drifted after load.")
            files = _verify_retained_feature_bundle(
                retained, self.feature_spec, require_sealed=True,
                manifest_digest=self._retained_files[Path("manifest.yaml")].sha256,
            )
            if files != self._retained_files:
                raise ValueError("Retained feature files changed after load.")
            _link_feature_bundle(retained, destination, files)
        return {"feature_manifest": "deployment/features/manifest.yaml"}

    def artifact_metadata(self):
        return {
            **super().artifact_metadata(),
            "rl_features": self.feature_spec.to_dict(),
            "algorithm": self.policy_spec.implementation,
        }
