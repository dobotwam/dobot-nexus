"""Fusion RGB and state-as-text inputs, independent of datasets and target values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


def preprocess_image(image: np.ndarray, size: int = 384) -> torch.Tensor:
    from PIL import Image

    array = np.asarray(image)
    if (
        array.dtype != np.uint8
        or array.ndim != 3
        or array.shape[-1] != 3
        or min(array.shape[:2]) < 1
    ):
        raise ValueError("Value images must be non-empty HWC uint8 RGB.")
    resized = Image.fromarray(array).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0


def validate_state_stats(stats):
    lo, hi = [np.asarray(stats[k], dtype=np.float32).reshape(-1) for k in ("min", "max")]
    if (
        lo.size == 0
        or lo.shape != hi.shape
        or not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or np.any(hi < lo)
    ):
        raise ValueError("State stats require finite ordered min/max vectors.")
    return {"min": lo.tolist(), "max": hi.tolist()}


def state_tokens(state, stats) -> str:
    x = torch.as_tensor(np.asarray(state).copy(), dtype=torch.float32).flatten()
    if not torch.isfinite(x).all():
        raise ValueError("Value state must be finite.")
    lo, hi = [torch.tensor(stats[k], dtype=torch.float32) for k in ("min", "max")]
    dim = min(x.numel(), lo.numel(), hi.numel(), 32)
    ids = torch.full((32,), 127, dtype=torch.long)
    ids[:dim] = torch.round(
        ((x[:dim] - lo[:dim]) / (hi[:dim] - lo[:dim]).clamp_min(1e-6)).clamp(0, 1) * 255
    ).long()
    return " ".join(str(int(v)) for v in ids)


class FusionProcessor:
    """Encode observations with an explicit camera order and optional state tokens."""

    def __init__(self, config, *, camera_names: Sequence[str], state_stats=None, tokenizer=None):
        self.config = config
        self.camera_names = tuple(camera_names)
        if (
            not self.camera_names
            or any(not isinstance(name, str) or not name.strip() for name in self.camera_names)
            or len(set(self.camera_names)) != len(self.camera_names)
        ):
            raise ValueError("Fusion camera_names must be non-empty, distinct camera names.")
        self.state_stats = validate_state_stats(state_stats) if state_stats is not None else None
        if config.include_state and self.state_stats is None:
            raise ValueError("Fusion with state tokens requires saved min/max state stats.")
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from . import require_value_dependencies

            require_value_dependencies()
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.config.tokenizer_path), local_files_only=True
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def validate_cameras(self, images: Mapping[str, np.ndarray]) -> None:
        if set(images) != set(self.camera_names):
            raise ValueError(f"Value samples require cameras {self.camera_names}.")

    def format_prompt(self, prompt: str, state=None) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("value prompt must be a non-empty string.")
        prompt = prompt.strip().replace("_", " ")
        prompt = prompt.replace("\n", " ").strip()
        if self.config.include_state:
            if state is None:
                raise ValueError("Value sample is missing state.")
            suffix = state_tokens(state, self.state_stats)
            return f"Task: {prompt}, State: {suffix}\nValue: "
        return f"Task: {prompt}\nValue: "

    def tokenize(self, prompts: Sequence[str]) -> dict:
        """Tokenize already formatted prompts using the Fusion text recipe."""
        return self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=200
        )

    def encode_images(self, images: Sequence[Mapping[str, np.ndarray]]) -> dict:
        for sample in images:
            self.validate_cameras(sample)
        return {
            "image_tensor": torch.stack(
                [torch.stack([preprocess_image(s[c]) for c in self.camera_names]) for s in images]
            ),
            "image_attention_mask": torch.ones(
                len(images), len(self.camera_names), dtype=torch.long
            ),
        }

    def encode_observation(
        self,
        *,
        images: Sequence[Mapping[str, np.ndarray]],
        prompts: Sequence[str],
        states: Sequence[np.ndarray | None] | None = None,
    ) -> dict:
        """Build model inputs without frame identities, returns or training metadata."""
        if not images or len(images) != len(prompts):
            raise ValueError("Fusion requires non-empty, equally sized images and prompts.")
        if states is None:
            states = [None] * len(images)
        if len(states) != len(images):
            raise ValueError("Fusion states must match the observation batch size.")
        formatted = []
        for sample, prompt, state in zip(images, prompts, states, strict=True):
            self.validate_cameras(sample)
            formatted.append(self.format_prompt(prompt, state))
        tokens = self.tokenize(formatted)
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            **self.encode_images(images),
        }
