from __future__ import annotations

from dataclasses import asdict

from dobot_nexus.algorithms.recap.conditioning import NEGATIVE_TAG, POSITIVE_TAG, select_condition
from dobot_nexus.algorithms.recap.configuration import validate_policy_config
from dobot_nexus.models.pi0.modeling_pi0 import Pi0Model

from .configuration_recap import resolve_policy_config
from .data.recap import route_observation
from .data_adapter_recap import ReCapDataAdapter


class Pi0ReCapModel(Pi0Model):
    """The existing Pi0 flow objective, conditioned on offline advantage labels."""

    def _resolve_config(self, config, *, mode):
        resolved = resolve_policy_config(config, mode=mode)
        self.recap_config = validate_policy_config(resolved)
        return resolved

    def _build_adapter(self, observation_type):
        return ReCapDataAdapter(self.pi0_config, self.root_config, observation_type)

    def training_step(self, batch):
        use_condition = select_condition(
            batch["positive"], unconditional_prob=self.recap_config.unconditional_prob
        )
        output = super().training_step(
            {
                **batch,
                "observation": route_observation(batch, use_condition),
            }
        )
        output.metrics.update(
            recap_positive_fraction=batch["positive"].float().mean().detach(),
            recap_conditioned_fraction=use_condition.float().mean().detach(),
        )
        return output

    def build_train_dataloader(self, config):
        loader = super().build_train_dataloader(config)
        # The standard Trainer persists its own configuration. Construction uses
        # a clone, so publish only these resolved assets/defaults at this hook.
        config.model["recap"] = asdict(self.recap_config)
        config.data["norm_stats_path"] = str(self.adapter.pipeline.norm_stats_path)
        config.data["tokenizer_path"] = str(self.adapter.pipeline.tokenizer_path)
        config.data["normalization"] = self.root_config.data.normalization
        return loader

    def artifact_metadata(self):
        metadata = super().artifact_metadata()
        metadata["recap"] = {
            **asdict(self.recap_config),
            "positive_tag": POSITIVE_TAG,
            "negative_tag": NEGATIVE_TAG,
            "default_condition": "base",
        }
        labels = self.adapter.pipeline.labels
        if labels is not None:
            metadata["recap_annotations"] = {
                key: labels.manifest[key]
                for key in (
                    "dataset_signature",
                    "dataset_id",
                    "outcome_signature",
                    "return_scale",
                    "value_checkpoint_id",
                    "thresholds",
                    "advantage_config",
                    "dataset_type",
                )
            }
        return metadata
