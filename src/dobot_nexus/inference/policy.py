from __future__ import annotations

import threading

import numpy as np

from dobot_nexus.core.artifacts import artifact_weight_path, load_manifest, resolve_artifact
from dobot_nexus.core.config import clone_config, load_config
from dobot_nexus.core.contracts import ObservationBatch, PolicyOutput, SamplingOptions
from dobot_nexus.core.import_utils import build_model


def _set_config_path(config, path: str, value) -> None:
    target = config
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = value


class InferencePolicy:
    """Thread-safe stateless action-chunk policy."""

    def __init__(self, config) -> None:
        self.config = clone_config(config)
        self.artifact_dir = None
        self.manifest = None
        artifact = self.config.model.get("artifact")
        if artifact:
            self.artifact_dir = resolve_artifact(str(artifact))
            self.config.model.artifact = str(self.artifact_dir)
            self.manifest = load_manifest(self.artifact_dir)
            training_config = load_config(
                self.artifact_dir / "config_resolved.yaml", expected_task="train"
            )
            manifest_model = self.manifest.get("model", {})
            if manifest_model.get("target") != training_config.model.target:
                raise ValueError("Artifact model.target disagrees with its saved configuration.")
            self.config.model.target = training_config.model.target
            for key in ("state_dim", "action_dim", "action_horizon"):
                configured = self.config.model.get(key)
                trained = training_config.model[key]
                if manifest_model.get(key, trained) != trained:
                    raise ValueError(f"Artifact {key} disagrees with its saved configuration.")
                if configured is not None and configured != trained:
                    raise ValueError(
                        f"Inference model.{key}={configured} does not match "
                        f"artifact value {trained}."
                    )
                self.config.model[key] = trained
            restored_config = clone_config(training_config)
            if "data" in restored_config:
                self.config.data = restored_config.data
            if "backend" in restored_config.model:
                self.config.model.backend = restored_config.model.backend
            if "io" in training_config.model:
                self.config.model.io = clone_config(training_config).model.io
            if "dependencies" not in self.config and "dependencies" in training_config:
                self.config.dependencies = clone_config(training_config).dependencies
            files = self.manifest.get("files", {})
            if files.get("norm_stats"):
                self.config.data.norm_stats_path = str(
                    (self.artifact_dir / files["norm_stats"]).resolve()
                )
            # Backward compatibility for v1 Pi0.5 artifacts. New artifacts use
            # the generic model_assets map with data.tokenizer_path.
            if files.get("tokenizer"):
                self.config.setdefault("data", {})["tokenizer_path"] = str(
                    (self.artifact_dir / files["tokenizer"]).resolve()
                )
            for config_path, relative_path in files.get("model_assets", {}).items():
                _set_config_path(
                    self.config,
                    str(config_path),
                    str((self.artifact_dir / relative_path).resolve()),
                )
            self.config.model.checkpoint = str(artifact_weight_path(self.artifact_dir))
        self.model = build_model(self.config, mode="infer")
        if self.artifact_dir is not None and self.manifest is not None:
            nexus_weights = self.manifest.get("files", {}).get("nexus_weights")
            if nexus_weights:
                import torch

                state_dict = torch.load(
                    self.artifact_dir / nexus_weights,
                    map_location="cpu",
                    weights_only=True,
                )
                self.model.load_state_dict(state_dict, strict=True)
        requested_device = str(self.config.get("runtime", {}).get("device", "auto"))
        if requested_device == "auto":
            import torch

            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = requested_device
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self._lock = threading.RLock()
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def spec(self):
        return self.model.policy_spec

    @property
    def metadata(self) -> dict:
        payload = self.spec.to_dict()
        payload.update(
            {
                "device": self.device,
                "artifact": str(self.artifact_dir) if self.artifact_dir else None,
            }
        )
        return payload

    def load(self) -> None:
        """Compatibility lifecycle hook; construction performs the actual load."""
        self._loaded = True

    def predict_chunk(
        self,
        observation: ObservationBatch | dict,
        options: SamplingOptions | None = None,
    ) -> PolicyOutput:
        if not self.loaded:
            raise RuntimeError("Policy is closed.")
        canonical = ObservationBatch.from_value(observation)
        if canonical.state.shape[-1] != self.spec.state_dim:
            raise ValueError(
                f"Policy expects state_dim={self.spec.state_dim}, got {canonical.state.shape[-1]}."
            )
        missing = sorted(set(self.spec.camera_names) - set(canonical.images))
        if missing:
            raise KeyError(f"Policy observation is missing cameras: {missing}.")
        resolved_options = options or SamplingOptions(
            seed=self.config.get("inference", {}).get("seed"),
            num_steps=self.config.get("inference", {}).get("num_steps"),
        )
        with self._lock:
            output = self.model.predict_action_chunk(canonical, resolved_options)
        expected = (canonical.batch_size, self.spec.action_horizon, self.spec.action_dim)
        if output.actions.shape != expected:
            raise ValueError(
                f"Policy returned actions with shape {output.actions.shape}; expected {expected}."
            )
        return output

    def create_session(self, options=None):
        from dobot_nexus.inference.session import PolicySession, SessionOptions

        if options is None:
            options = SessionOptions(
                execute_steps=int(
                    self.config.get("inference", {}).get("execute_steps", self.spec.action_horizon)
                ),
                seed=self.config.get("inference", {}).get("seed"),
                num_steps=self.config.get("inference", {}).get("num_steps"),
            )
        return PolicySession(self, options)

    def warmup(self, observation: ObservationBatch | dict | None = None) -> PolicyOutput:
        if observation is None:
            height, width = self.spec.image_size
            observation = ObservationBatch(
                images={
                    name: np.zeros((1, height, width, 3), dtype=np.uint8)
                    for name in self.spec.camera_names
                },
                state=np.zeros((1, self.spec.state_dim), dtype=np.float32),
                prompt=["warm up"],
            )
        return self.predict_chunk(observation, SamplingOptions(seed=0))

    def close(self) -> None:
        self._loaded = False


def load_policy(config) -> InferencePolicy:
    return InferencePolicy(config)
