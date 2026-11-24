from __future__ import annotations

from dataclasses import asdict

from dobot_nexus.models.pi0.modeling_pi0 import Pi0Model

from .configuration import resolve_config


class Pi0MemoryModel(Pi0Model):
    """FrameSample assembly using the standard Pi0 lifecycle and flow objective."""

    def _resolve_config(self, config, *, mode):
        resolved, self.memory_config = resolve_config(config, mode=mode)
        return resolved

    def _configure_core(self):
        from .torch import install_memory

        super()._configure_core()
        install_memory(self.core_model, self.pi0_config.torch_dtype)

    def _load_core_checkpoint(self):
        from .checkpoint import load_memory_checkpoint

        initialization = self.root_config.model.get("memory_initialization", "base")
        if initialization not in {"base", "trained"}:
            raise ValueError("model.memory_initialization must be base or trained.")
        return load_memory_checkpoint(
            self.core_model,
            self.pi0_config.checkpoint,
            allow_initialization=self.mode == "train" and initialization == "base",
        )

    def _build_adapter(self, observation_type):
        from .data import MemoryDataAdapter

        if self.memory_config.camera not in self.pi0_config.camera_names:
            raise ValueError("memory.camera is not among the configured public cameras.")
        return MemoryDataAdapter(
            self.pi0_config, self.root_config, observation_type, self.memory_config
        )

    def _sampling_kwargs(self, encoded):
        return {**super()._sampling_kwargs(encoded), "_cache_memory": True}

    def build_train_dataloader(self, config):
        loader = super().build_train_dataloader(config)
        config.model["memory"] = asdict(self.memory_config)
        if self.root_config.model.get("memory_reference_sha256"):
            config.model["memory_reference_sha256"] = self.root_config.model.memory_reference_sha256
        config.data["normalization"] = self.root_config.data["normalization"]
        config.data["norm_stats_path"] = str(self.adapter.pipeline.norm_stats_path)
        config.data["tokenizer_path"] = str(self.adapter.pipeline.tokenizer_path)
        return loader

    def artifact_metadata(self):
        result = super().artifact_metadata()
        result["memory"] = asdict(self.memory_config)
        result["memory_reference_sha256"] = self.root_config.model.get("memory_reference_sha256")
        return result

    def export_deployment_artifacts(self, *, source_weights, artifact_dir):
        from .checkpoint import export_memory_artifacts

        del source_weights
        return export_memory_artifacts(
            artifact_dir,
            memory_config=self.memory_config,
        )
