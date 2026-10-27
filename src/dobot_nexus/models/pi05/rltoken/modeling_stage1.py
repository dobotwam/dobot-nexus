from __future__ import annotations

import math
from pathlib import Path

import torch

from dobot_nexus.algorithms.rltoken.token import (
    TokenConfig,
    TokenDecoder,
    TokenEncoder,
    reconstruction_loss,
)
from dobot_nexus.core.artifacts import resolve_artifact
from dobot_nexus.core.config import clone_config, load_config
from dobot_nexus.core.contracts import TrainOutput
from dobot_nexus.models.pi05.modeling_pi05 import Pi05Model

from .data_adapter import Pi05RLDataAdapter
from .features import extract_features, feature_spec


class Pi05RLTokenModel(Pi05Model):
    def __init__(self, config, *, mode):
        config = clone_config(config)
        if mode == "infer" and config.model.get("artifact"):
            trained = load_config(resolve_artifact(config.model.artifact) / "config_resolved.yaml")
            config.model["rltoken"] = trained.model.rltoken
        settings = config.model.get("rltoken", {})
        TokenConfig(**settings.get("token", {}))
        codec = settings.get("action_codec", {})
        if not isinstance(codec.get("low"), (list, tuple)) or not isinstance(
            codec.get("high"), (list, tuple)
        ):
            raise ValueError(
                "Set model.rltoken.action_codec.low/high to actual controller command bounds."
            )
        from dobot_nexus.core.contracts import RLFeatureSpec

        RLFeatureSpec(
            1,
            1,
            config.model.action_dim,
            config.model.action_horizon,
            "validate",
            codec.get("name", ""),
            "validate",
            codec.get("units", ""),
            settings.get("control_dt", 0),
            tuple(codec["low"]),
            tuple(codec["high"]),
        )
        if config.model.get("inference_parameter_precision", "fp32") != "fp32":
            raise ValueError(
                "RLToken v1 preserves frozen feature identity with FP32 provider parameters."
            )
        config.model["inference_parameter_precision"] = "fp32"
        alpha = settings.get("alpha_vla", 1.0)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Joint RLToken training requires alpha_vla > 0.")
        if settings.get("initialization", "bc") not in {"bc", "rltoken"}:
            raise ValueError("RLToken initialization must be bc or rltoken.")
        self._stage_state = None
        self._rl_feature_spec = None
        self._exported_feature_spec = None
        super().__init__(config, mode=mode)
        self.alpha_vla = float(alpha)
        self.token_config = TokenConfig(**settings["token"])
        self.token_encoder = TokenEncoder(self.token_config)
        self.token_decoder = TokenDecoder(self.token_config)
        if self._stage_state is not None:
            state = self._stage_state
            encoder = {
                key.removeprefix("token_encoder."): value
                for key, value in state.items()
                if key.startswith("token_encoder.")
            }
            decoder = {
                key.removeprefix("token_decoder."): value
                for key, value in state.items()
                if key.startswith("token_decoder.")
            }
            self.token_encoder.load_state_dict(encoder, strict=True)
            if mode == "train" or decoder:
                self.token_decoder.load_state_dict(decoder, strict=True)
            unknown = [
                key
                for key in state
                if not key.startswith(("core_model.", "token_encoder.", "token_decoder."))
            ]
            if unknown:
                raise ValueError(f"Unknown Stage 1 weights: {unknown[:5]}")
            self._stage_state = None
        if mode == "infer":
            # Decoder is only used in Stage 1. A full Stage 1 input was checked above.
            self.token_decoder = None
            self.eval().requires_grad_(False)

    def _load_core_checkpoint(self):
        if (
            self.mode == "train"
            and self.root_config.model.rltoken.get("initialization", "bc") == "bc"
        ):
            return super()._load_core_checkpoint()
        path = Path(self.pi05_config.checkpoint)
        state = torch.load(path, map_location="cpu", weights_only=True)
        core = {
            key.removeprefix("core_model."): value
            for key, value in state.items()
            if key.startswith("core_model.")
        }
        self.core_model.load_state_dict(core, strict=True, assign=True)
        self._stage_state = state
        return path

    def _build_adapter(self, observation_type):
        return Pi05RLDataAdapter(self.pi05_config, self.root_config, observation_type)

    def build_train_dataloader(self, config):
        loader = super().build_train_dataloader(config)
        config.data = clone_config(self.root_config).data
        config.model["inference_parameter_precision"] = "fp32"
        return loader

    def artifact_assets(self):
        assets = super().artifact_assets()
        if self.adapter.pipeline.norm_stats_path is not None:
            assets["data.norm_stats_path"] = str(self.adapter.pipeline.norm_stats_path)
        return assets

    def training_step(self, batch):
        if self.mode != "train":
            raise RuntimeError("Stage 1 training requires train mode.")
        device = batch["actions"].device
        rng = (
            torch.cuda.default_generators[device.index]
            if device.type == "cuda"
            else torch.default_generator
        )
        loss_steps, prefix, valid = self.core_model.compute_loss(
            batch["observation"],
            batch["actions"],
            train=True,
            rng=rng,
            _return_prefix=True,
            **batch.get("model_kwargs", {}),
        )
        latent = self.token_encoder(prefix, valid)
        prediction = self.token_decoder(latent, prefix, valid)
        token_loss = reconstruction_loss(prediction, prefix, valid)
        vla_loss = loss_steps.float().mean()
        return TrainOutput(
            token_loss + self.alpha_vla * vla_loss,
            {"token_loss": token_loss.detach(), "flow_loss": vla_loss.detach()},
            {"token_loss": token_loss, "flow_loss": vla_loss},
        )

    @property
    def rl_feature_spec(self):
        if self.mode == "train":
            return feature_spec(self)
        if self._rl_feature_spec is None:
            self._rl_feature_spec = feature_spec(self)
        return self._rl_feature_spec

    def extract_rl_features(self, observation, options):
        return extract_features(self, observation, options)

    def predict_action_chunk(self, observation, options):
        packet = self.extract_rl_features(observation, options)
        return self.adapter.decode_actions(packet.reference_actions, aux=packet.action_context)

    def export_backend_state_dict(self, state_dict):
        result = {}
        for key, value in state_dict.items():
            key = key.replace("_fsdp_wrapped_module.", "").removeprefix("module.")
            if not key.startswith(("core_model.", "token_encoder.", "token_decoder.")):
                raise ValueError(f"Unexpected Stage 1 state key: {key}")
            result[key] = value
        expected = set(self.state_dict())
        if set(result) != expected:
            raise ValueError("Stage 1 export requires all core and Token parameters.")
        self._exported_feature_spec = feature_spec(self, weights=result)
        return result

    def artifact_metadata(self):
        spec = self._exported_feature_spec or self.rl_feature_spec
        return {**super().artifact_metadata(), "rl_features": spec.to_dict(), "rltoken_schema": 1}
