from __future__ import annotations

from dataclasses import asdict

from dobot_nexus.models.pi05.modeling_pi05 import Pi05Model

from .configuration import resolve_config


class Pi05MemoryModel(Pi05Model):
    """FrameSample assembly using the standard Pi0.5 lifecycle and flow objective."""

    def __init__(self, config, *, mode):
        resolved, memory = resolve_config(config, mode=mode)
        self.memory_config = memory
        super().__init__(resolved, mode=mode)
        if memory.camera not in self.pi05_config.camera_names:
            raise ValueError("memory.camera is not among the configured public cameras.")

    def _configure_core(self):
        from .torch import install_memory

        super()._configure_core()
        install_memory(self.core_model, self.pi05_config.torch_dtype)

    def _load_core_checkpoint(self):
        from .checkpoint import load_memory_checkpoint

        initialization = self.root_config.model.get("memory_initialization", "base")
        if initialization not in {"base", "trained"}:
            raise ValueError("model.memory_initialization must be base or trained.")
        return load_memory_checkpoint(
            self.core_model,
            self.pi05_config.checkpoint,
            allow_initialization=self.mode == "train" and initialization == "base",
        )

    def _build_adapter(self, observation_type):
        from .data import MemoryDataAdapter

        return MemoryDataAdapter(
            self.pi05_config, self.root_config, observation_type, self.memory_config
        )

    def build_train_dataloader(self, config):
        loader = super().build_train_dataloader(config)
        config.model["memory"] = asdict(self.memory_config)
        if self.root_config.model.get("memory_reference_sha256"):
            config.model["memory_reference_sha256"] = self.root_config.model.memory_reference_sha256
        config.data["normalization"] = self.adapter.pipeline.normalization
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
