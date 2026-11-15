"""Pi0 checkpoint loading with no dependency on RLinf runtime utilities."""

# The legacy key-conversion mapping is adapted from RLinf's
# openpi_pytorch_to_openpi_rlinf converter (Copyright 2026 The RLinf Authors),
# licensed under Apache-2.0. The surrounding loader is Nexus-owned.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_CANDIDATES = (
    "deployment/pytorch/full_weights.pt",
    "backend/full_weights.pt",
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
    "model.safetensors",
)
_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_orig_mod.",
    "module.",
    "backend_model.",
    "core_model.",
    "model.",
)
_LEGACY_PREFIX = "paligemma_with_expert."


def resolve_checkpoint_file(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    for relative in _CANDIDATES:
        candidate = path / relative
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Pi0 checkpoint not found under {path}. Expected one of: " + ", ".join(_CANDIDATES)
    )


def _as_state_dict(loaded: Any) -> Mapping[str, torch.Tensor]:
    value = loaded
    for _ in range(4):
        if (
            isinstance(value, Mapping)
            and value
            and all(isinstance(item, torch.Tensor) for item in value.values())
        ):
            return value
        if isinstance(value, Mapping):
            for key in ("model", "state_dict", "module"):
                nested = value.get(key)
                if isinstance(nested, Mapping):
                    value = nested
                    break
            else:
                break
        else:
            break
    raise TypeError(f"Could not extract a tensor state dict from {type(loaded)!r}.")


def _strip_prefixes(key: str) -> str:
    while True:
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        else:
            return key


def normalize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize Nexus/RLinf/FSDP wrappers to bare Pi0 operator keys."""
    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        bare_key = _strip_prefixes(str(key))
        if bare_key in normalized:
            raise ValueError(f"Duplicate Pi0 key after prefix normalization: {bare_key!r}.")
        normalized[bare_key] = tensor
    if any(key.startswith(_LEGACY_PREFIX) for key in normalized):
        normalized = _convert_legacy_openpi(normalized)
    return normalized


def _load_file(path: Path) -> Mapping[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            import safetensors.torch
        except ImportError as exc:
            raise ImportError("Pi0 safetensors checkpoints require `safetensors`.") from exc
        return safetensors.torch.load_file(str(path), device="cpu")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    return _as_state_dict(loaded)


def load_core_state_dict(value: str | Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """Load any supported checkpoint as the bare aligned Pi0 operator state."""
    path = resolve_checkpoint_file(value)
    return path, normalize_state_dict(_load_file(path))


def load_core_checkpoint(model: torch.nn.Module, value: str | Path) -> Path:
    """Strictly load a base or v2 artifact into the bare operator model."""
    path, state_dict = load_core_state_dict(value)
    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        sample = list(state_dict)[:8]
        raise RuntimeError(
            f"Pi0 checkpoint {path} does not match the operator core. "
            f"Normalized key sample={sample}."
        ) from exc
    return path


def _convert_legacy_openpi(
    old: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert official OpenPI PyTorch keys to the aligned bare Pi0 layout.

    The mapping mirrors RLinf's Apache-2.0 checkpoint converter, but lives at
    the Nexus boundary so loading a checkpoint does not import RLinf utilities.
    """
    new: dict[str, torch.Tensor] = {}
    vision = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
    for suffix in (".weight", ".bias"):
        source = vision + "embeddings.patch_embedding" + suffix
        if source in old:
            new["img.stem" + suffix] = old[source]
    source = vision + "embeddings.position_embedding.weight"
    if source in old:
        position = old[source]
        new["img.pos_embedding"] = position.unsqueeze(0) if position.dim() == 2 else position

    for index in range(27):
        source_prefix = f"{vision}encoder.layers.{index}."
        target_prefix = f"img.encoder.layers.{index}."
        for source_name, target_name in (("layer_norm1", "norm1"), ("layer_norm2", "norm2")):
            for suffix in (".weight", ".bias"):
                source = f"{source_prefix}{source_name}{suffix}"
                if source in old:
                    new[f"{target_prefix}{target_name}{suffix}"] = old[source]
        weights = [
            old[f"{source_prefix}self_attn.{name}.weight"]
            for name in ("q_proj", "k_proj", "v_proj")
            if f"{source_prefix}self_attn.{name}.weight" in old
        ]
        biases = [
            old[f"{source_prefix}self_attn.{name}.bias"]
            for name in ("q_proj", "k_proj", "v_proj")
            if f"{source_prefix}self_attn.{name}.bias" in old
        ]
        if weights:
            new[f"{target_prefix}attn.in_proj_weight"] = torch.cat(weights, dim=0)
        if biases:
            new[f"{target_prefix}attn.in_proj_bias"] = torch.cat(biases, dim=0)
        for suffix in (".weight", ".bias"):
            source = f"{source_prefix}self_attn.out_proj{suffix}"
            if source in old:
                new[f"{target_prefix}attn.out_proj{suffix}"] = old[source]
        for name in ("fc1", "fc2"):
            for suffix in (".weight", ".bias"):
                source = f"{source_prefix}mlp.{name}{suffix}"
                if source in old:
                    new[f"{target_prefix}mlp.{name}{suffix}"] = old[source]
    for suffix in (".weight", ".bias"):
        source = vision + "post_layernorm" + suffix
        if source in old:
            new["img.encoder.norm" + suffix] = old[source]
        source = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear" + suffix
        if source in old:
            new["img.head" + suffix] = old[source]

    pali = "paligemma_with_expert.paligemma.model.language_model."
    expert = "paligemma_with_expert.gemma_expert.model."
    for index in range(18):
        target = f"llm.layers.{index}."
        for expert_index, source_root in enumerate((pali, expert)):
            source_prefix = f"{source_root}layers.{index}."
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                source = f"{source_prefix}self_attn.{projection}.weight"
                if source in old:
                    new[f"{target}attn.{projection}.{expert_index}.weight"] = old[source]
            gate = f"{source_prefix}mlp.gate_proj.weight"
            up = f"{source_prefix}mlp.up_proj.weight"
            if gate in old and up in old:
                new[f"{target}mlps.{expert_index}.w_gating"] = torch.stack(
                    [old[gate].T.contiguous(), old[up].T.contiguous()], dim=0
                )
            down = f"{source_prefix}mlp.down_proj.weight"
            if down in old:
                new[f"{target}mlps.{expert_index}.w_linear"] = old[down].T.contiguous()
        for source_name, target_name in (
            ("input_layernorm", "pre_attention_norms"),
            ("post_attention_layernorm", "pre_ffw_norms"),
        ):
            source = f"{pali}layers.{index}.{source_name}.weight"
            if source in old:
                new[f"{target}{target_name}.0.scale"] = old[source]
            # Pi0 uses ordinary RMSNorm in both experts. Pi0.5 checkpoints
            # instead store the action expert's adaptive dense projection.
            source = f"{expert}layers.{index}.{source_name}.weight"
            if source in old:
                new[f"{target}{target_name}.1.scale"] = old[source]
            for suffix in (".weight", ".bias"):
                source = f"{expert}layers.{index}.{source_name}.dense{suffix}"
                if source in old:
                    new[f"{target}{target_name}.1.ada_modulation{suffix}"] = old[source]

    source = pali + "norm.weight"
    if source in old:
        new["llm.final_norms.0.scale"] = old[source]
    source = expert + "norm.weight"
    if source in old:
        new["llm.final_norms.1.scale"] = old[source]
    for suffix in (".weight", ".bias"):
        source = expert + "norm.dense" + suffix
        if source in old:
            new["llm.final_norms.1.ada_modulation" + suffix] = old[source]
    for source in (
        "paligemma_with_expert.paligemma.lm_head.weight",
        "paligemma_with_expert.gemma_expert.lm_head.weight",
    ):
        if source in old:
            new["llm.embedder.embedding.weight"] = old[source]
            break
    for key, tensor in old.items():
        if key.startswith(
            (
                "action_in_proj",
                "action_out_proj",
                "time_mlp_",
                "state_proj",
                "action_time_mlp_",
                "pointnet.",
            )
        ):
            new[key] = tensor
    return new
