from __future__ import annotations

from dataclasses import replace

import torch

from dobot_nexus.core.contracts import PolicySpec, TrainOutput
from dobot_nexus.models.base import BaseDataAdapter, BaseVLAModule

from .actor_critic import Actor, TwinQ
from .artifacts import feature_metadata, load_provider, restore_policy_config, write_model_bundle
from .configuration import HeadConfig
from .losses import actor_loss, critic_loss


def sampling_seed(seed, stream):
    # Stable named streams independent of Python's process-randomized hash().
    if seed is None:
        return None
    return (int(seed) + {"reference": 0x45D9F3B, "actor": 0x119DE1F3}[stream]) % (2**63 - 1)


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

    def __init__(self, config, *, mode):
        super().__init__()
        if mode not in {"train", "infer"}:
            raise ValueError("RLTokenPolicy mode must be train or infer.")
        self.config = restore_policy_config(config) if mode == "infer" else config
        self.mode = mode
        self.head_config = HeadConfig(**self.config.model.rltoken)
        root, self.feature_spec, metadata = feature_metadata(self.config.model.features.artifact)
        if (
            self.head_config.chunk_size != self.config.model.action_horizon
            or self.feature_spec.action_dim != self.config.model.action_dim
        ):
            raise ValueError("Actor chunk/action dimensions disagree with public model dimensions.")
        if metadata["state_dim"] != self.config.model.state_dim:
            raise ValueError("Raw robot state_dim must match the feature provider.")
        self._policy_spec = PolicySpec(
            "rltoken",
            "rlt_ac_v1",
            tuple(metadata["camera_names"]),
            self.config.model.state_dim,
            self.feature_spec.action_dim,
            self.head_config.chunk_size,
            tuple(metadata.get("image_size", (224, 224))),
        )
        self.actor = Actor(self.feature_spec, self.head_config)
        if mode == "train":
            self.critic = TwinQ(self.feature_spec, self.head_config)
            self.adapter = ReplayAdapter()
        else:
            self.provider, self.provider_config, _ = load_provider(root)
            self.provider.eval().requires_grad_(False)
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

    @torch.no_grad()
    def predict_action_chunk(self, observation, options):
        if self.mode != "infer":
            raise RuntimeError("Learner policy does not load the frozen VLA.")
        base_seed = (
            options.seed if options.seed is not None else (0 if options.deterministic else None)
        )
        reference_options = replace(options, seed=sampling_seed(base_seed, "reference"))
        packet = self.provider.extract_rl_features(observation, reference_options).validate(
            self.feature_spec, chunk_size=self.head_config.chunk_size
        )
        device = next(self.actor.parameters()).device
        seed = sampling_seed(base_seed, "actor")
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        actions = self.actor(
            packet.latent.to(device),
            packet.state.to(device),
            packet.reference_actions.to(device),
            deterministic=options.deterministic,
            generator=generator,
        )
        return self.adapter.decode_actions(actions.detach().cpu(), aux=packet.action_context)

    def save_nexus_weights(self):
        return False

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
        write_model_bundle(
            artifact_dir / "deployment/features", self.provider, self.provider_config
        )
        return {"feature_manifest": "deployment/features/manifest.yaml"}

    def artifact_metadata(self):
        return {
            **super().artifact_metadata(),
            "rl_features": self.feature_spec.to_dict(),
            "algorithm": "rlt_ac_v1",
        }
