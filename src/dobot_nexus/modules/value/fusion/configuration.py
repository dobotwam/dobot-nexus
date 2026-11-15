"""Dobot Fusion topology; optional model dependencies are loaded on construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValueModelConfig:
    siglip_path: Path
    gemma3_path: Path
    tokenizer_path: Path
    include_state: bool = True
    freeze_vision_backbone: bool = False
    freeze_language_backbone: bool = False
    backbone_precision: str = "bfloat16"

    def __post_init__(self):
        for key in ("include_state", "freeze_vision_backbone", "freeze_language_backbone"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"value.{key} must be boolean.")
        if self.backbone_precision not in {"float32", "bfloat16"}:
            raise ValueError("value.backbone_precision must be float32 or bfloat16.")

    def architecture(self) -> dict:
        return {
            "arch": "fusion",
            "num_bins": 201,
            "v_min": -1.0,
            "v_max": 0.0,
            "image_size": 384,
            "max_token_len": 200,
            "include_state": self.include_state,
            "max_state_dim": 32,
            "state_bins": 256,
            "fusion_dim": 512,
            "fusion_num_layers": 2,
            "fusion_num_heads": 8,
            "fusion_num_queries": 4,
            "fusion_ffn_multiplier": 4,
            "dropout": 0.1,
            "backbone_precision": self.backbone_precision,
            "head_precision": "float32",
            "vision": {
                "hidden_size": 1152,
                "image_size": 384,
                "patch_size": 14,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
            },
            "language": {
                "hidden_size": 640,
                "head_dim": 256,
                "num_hidden_layers": 18,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
            },
        }

    def validate_assets(self):
        for name in ("siglip_path", "gemma3_path", "tokenizer_path"):
            path = Path(getattr(self, name))
            if not path.is_dir():
                raise FileNotFoundError(
                    f"value.{name} requires a local HF model directory: {path}."
                )
        for name, section, model_type in (
            ("siglip_path", "vision", "siglip"),
            ("gemma3_path", "language", "gemma3_text"),
        ):
            raw = json.loads((Path(getattr(self, name)) / "config.json").read_text())
            if raw.get("model_type") != model_type:
                raise ValueError(f"value.{name} requires model_type={model_type}.")
            if section == "vision":
                raw = raw.get("vision_config", {})
            for key, expected in self.architecture()[section].items():
                if raw.get(key) != expected:
                    raise ValueError(
                        f"Incompatible value.{name}: {key}={raw.get(key)!r}, expected {expected}. "
                        "Use SigLIP so400m patch14 384 and Gemma3 270M."
                    )
