"""Local LeRobot and runtime codecs for the Pi0 operator core."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dobot_nexus.data import (
    ActionTransform,
    NormStats,
    apply_action_transform,
    invert_action_transform,
    load_norm_stats,
    normalize,
    unnormalize,
)
from dobot_nexus.data.lerobot import (
    LeRobotNormStatsSpec,
    resolve_or_compute_lerobot_norm_stats,
)
from dobot_nexus.data.lerobot.source import build_source

LOGGER = logging.getLogger(__name__)
CORE_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
PUBLIC_CAMERA_TO_CORE = {
    "base": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}
_PI0_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ResolvedDataConfig:
    dataset_root: str
    format_version: str
    fps: int
    image_field_mapping: dict[str, str]
    state_key: str
    action_key: str
    language_key: str | None
    action_transform: str
    tokenizer_path: str
    norm_stats_path: str


@dataclass(frozen=True)
class EncodedObservation:
    observation: Any
    raw_state: np.ndarray
    model_kwargs: dict[str, Any] = field(default_factory=dict)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


def _stats_candidates(path: Path) -> list[Path]:
    if not path.is_dir():
        return [path]
    return [
        path / "norm_stats.json",
        path / "meta" / "stats.json",
        path / "meta" / "norm_stats.json",
        path / "stats.json",
    ]


def resolve_norm_stats_path(
    dataset_path: str | Path | None,
    explicit: str | Path | None = None,
) -> Path:
    """Resolve an explicitly selected stats file or directory."""

    candidates: list[Path] = []
    selected = explicit if explicit is not None else dataset_path
    if selected:
        candidates.extend(_stats_candidates(Path(selected).expanduser()))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Pi0 normalization statistics override was not found. "
        "Omit data.norm_stats_path during training to calculate it automatically. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_stats(
    path: str | Path,
    *,
    state_key: str = "observation.state",
    action_key: str = "action",
) -> tuple[dict[str, NormStats], Path]:
    stats_path = Path(path).expanduser()
    if stats_path.is_dir():
        stats_path = resolve_norm_stats_path(stats_path)
    raw = load_norm_stats(stats_path)
    result: dict[str, NormStats] = {}
    source_keys = {
        "state": ("state", state_key, "observation.state"),
        "actions": ("actions", action_key, "action"),
    }
    for name, aliases in source_keys.items():
        item = next(
            (raw.get(key) for key in dict.fromkeys(aliases) if raw.get(key) is not None),
            None,
        )
        if item is None:
            raise ValueError(
                f"{stats_path} must contain mean/std statistics for one of {aliases}."
            )
        result[name] = item
    return result, stats_path.resolve()


def resolve_tokenizer_path(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(_PI0_ROOT / "assets" / "pi0_tokenizer.model")
    for candidate in candidates:
        if candidate.is_dir():
            for filename in ("pi0_tokenizer.model", "paligemma_tokenizer.model"):
                nested = candidate / filename
                if nested.is_file():
                    return nested.resolve()
        elif candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Pi0 tokenizer was not found in the model package. "
        "Set data.tokenizer_path for an explicit custom override. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def _pad_last_dim(value: np.ndarray, target: int) -> np.ndarray:
    if value.shape[-1] > target:
        raise ValueError(f"Cannot fit dimension {value.shape[-1]} into Pi0 dimension {target}.")
    if value.shape[-1] == target:
        return value
    widths = [(0, 0)] * value.ndim
    widths[-1] = (0, target - value.shape[-1])
    return np.pad(value, widths)


def _convert_image(value: Any) -> np.ndarray:
    image = _as_numpy(value)
    if image.ndim != 3:
        raise ValueError(f"Pi0 image must be 3-D, got {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Pi0 image must be RGB, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        maximum = float(np.nanmax(image)) if image.size else 0.0
        scale = 255.0 if maximum <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _resize_with_pad(image: np.ndarray, size: int = 224) -> np.ndarray:
    if image.shape[:2] == (size, size):
        return image
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pi0 image preprocessing requires Pillow.") from exc
    source = Image.fromarray(image)
    width, height = source.size
    ratio = max(width / size, height / size)
    resized_width = max(1, int(width / ratio))
    resized_height = max(1, int(height / ratio))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = source.resize((resized_width, resized_height), resample=resampling)
    padded = Image.new("RGB", (size, size), 0)
    padded.paste(resized, ((size - resized_width) // 2, (size - resized_height) // 2))
    return np.asarray(padded, dtype=np.uint8)


def normalize_prompt(prompt: str) -> str:
    return prompt.strip().replace("_", " ").replace("\n", " ")


class _Tokenizer:
    def __init__(self, path: Path, max_length: int) -> None:
        self.path = path
        self.max_length = max_length
        self._processor: Any | None = None

    def _get_processor(self):
        if self._processor is None:
            try:
                import sentencepiece
            except ImportError as exc:
                raise ImportError("Pi0 prompt encoding requires sentencepiece.") from exc
            self._processor = sentencepiece.SentencePieceProcessor(
                model_proto=self.path.read_bytes()
            )
        return self._processor

    def encode(self, prompt: str, *, strict: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Encode the original Pi0 text prefix without discretizing robot state."""
        cleaned = normalize_prompt(prompt)
        processor = self._get_processor()
        tokens = list(processor.encode(cleaned, add_bos=True))
        tokens.extend(processor.encode("\n"))
        if len(tokens) > self.max_length:
            if strict:
                raise ValueError(
                    f"Pi0 prompt has {len(tokens)} tokens, exceeding max_token_len={self.max_length}."
                )
            LOGGER.warning(
                "Pi0 prompt has %d tokens; truncating to %d.", len(tokens), self.max_length
            )
        tokens = tokens[: self.max_length]
        mask = [True] * len(tokens)
        padding = self.max_length - len(tokens)
        tokens.extend([0] * padding)
        mask.extend([False] * padding)
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)


def _field_mapping(config: Any) -> tuple[dict[str, str], str, str, str | None]:
    mapping = config.get("data", {}).get("field_mapping", {})
    raw_images = dict(mapping.get("images", {}))
    images: dict[str, str] = {}
    for configured_name, source in raw_images.items():
        core_name = PUBLIC_CAMERA_TO_CORE.get(str(configured_name), str(configured_name))
        images[core_name] = str(source)
    if set(images) != set(CORE_IMAGE_KEYS):
        raise ValueError(
            "data.field_mapping.images must map base, left_wrist, and right_wrist; "
            f"got {sorted(raw_images)}."
        )
    state = str(mapping.get("state", "observation.state"))
    action = str(mapping.get("action", "action"))
    raw_language = mapping.get("language", "task")
    language = str(raw_language) if raw_language is not None else None
    return images, state, action, language


def _action_transform(config: Any, action_dim: int) -> ActionTransform:
    action = config.get("data", {}).get("transforms", {}).get("action", {})
    return ActionTransform.create(
        str(action.get("type", "absolute")),
        action.get("delta_mask"),
        action_dim=action_dim,
    )


def _norm_stats_cache_root(config: Any) -> Path:
    data = config.get("data", {})
    explicit = data.get("norm_stats_cache_dir")
    if explicit:
        return Path(str(explicit)).expanduser().resolve()
    exp = config.get("exp", {})
    resolved_run_dir = exp.get("run_dir")
    if resolved_run_dir:
        run_dir = Path(str(resolved_run_dir)).expanduser()
    else:
        output_dir = Path(str(exp.get("output_dir", "runs"))).expanduser()
        model_name = config.get("model", {}).get("name", "model")
        default_name = f"{model_name}_{config.get('task', 'train')}"
        run_dir = output_dir / str(exp.get("name", default_name))
    return (run_dir / "data").resolve()


class Pi0Pipeline:
    """One preprocessing contract shared by BC training and inference."""

    def __init__(self, root_config: Any, model_config: Any) -> None:
        self.root_config = root_config
        self.model_config = model_config
        data = root_config.get("data", {})
        self.image_fields, self.state_key, self.action_key, self.language_key = _field_mapping(
            root_config
        )
        self.action_transform = _action_transform(root_config, model_config.action_dim)
        self.delta_mask = self.action_transform.delta_mask
        task = str(root_config.get("task", "")).lower()
        self.stats: dict[str, NormStats] | None = None
        self.norm_stats_path: Path | None = None
        explicit_stats = data.get("norm_stats_path")
        if explicit_stats:
            self._set_stats(resolve_norm_stats_path(None, explicit=explicit_stats))
        elif task != "train":
            raise FileNotFoundError(
                "Pi0 inference requires the normalization statistics stored in its Artifact."
            )
        self.tokenizer_path = resolve_tokenizer_path(data.get("tokenizer_path"))
        self.tokenizer = _Tokenizer(self.tokenizer_path, model_config.max_token_len)
        if task == "train":
            data["tokenizer_path"] = str(self.tokenizer_path)
        self.default_prompt = data.get("default_prompt")
        if self.delta_mask.any() and model_config.state_dim != model_config.action_dim:
            raise ValueError("Delta actions require model.state_dim == model.action_dim.")

    def _set_stats(self, path: str | Path) -> None:
        self.stats, self.norm_stats_path = _load_stats(
            path,
            state_key=self.state_key,
            action_key=self.action_key,
        )
        for name, dimension in (
            ("state", self.model_config.state_dim),
            ("actions", self.model_config.action_dim),
        ):
            stats = self.stats[name]
            if stats.mean.shape[-1] < dimension or stats.std.shape[-1] < dimension:
                raise ValueError(f"{name} norm stats do not cover dimension {dimension}.")
        if str(self.root_config.get("task", "")).lower() == "train":
            self.root_config.setdefault("data", {})["norm_stats_path"] = str(
                self.norm_stats_path
            )

    def _resolved_stats(self) -> dict[str, NormStats]:
        if self.stats is None:
            raise RuntimeError("Pi0 normalization statistics have not been resolved yet.")
        return self.stats

    def _prompt(self, frame: Mapping[str, Any]) -> str:
        prompt = frame.get("prompt")
        if prompt is None and self.language_key is not None:
            prompt = frame.get(self.language_key)
        if prompt is None:
            prompt = frame.get("task", self.default_prompt)
        if prompt is None:
            raise ValueError("No task prompt was found and data.default_prompt is not set.")
        if isinstance(prompt, bytes):
            return prompt.decode("utf-8")
        if not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
        if not prompt.strip():
            raise ValueError("Pi0 prompt must be non-empty.")
        return prompt

    def tokenize_prompt(self, prompt: str, *, strict: bool = False):
        return self.tokenizer.encode(prompt, strict=strict)

    def collate_items(self, items):
        return collate_items(items)

    def transform_frame(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        required = [*self.image_fields.values(), self.state_key, self.action_key]
        missing = [key for key in required if key not in frame]
        if missing:
            raise KeyError(f"LeRobot frame is missing configured fields: {missing}.")
        state = _as_numpy(frame[self.state_key]).astype(np.float32, copy=True)
        actions = _as_numpy(frame[self.action_key]).astype(np.float32, copy=True)
        if state.shape != (self.model_config.state_dim,):
            raise ValueError(f"Expected state [{self.model_config.state_dim}], got {state.shape}.")
        expected_action_tail = self.model_config.action_dim
        if actions.ndim != 2 or actions.shape[-1] != expected_action_tail:
            raise ValueError(f"Expected actions [H,{expected_action_tail}], got {actions.shape}.")
        if actions.shape[0] != self.model_config.action_horizon:
            raise ValueError(
                f"Expected action horizon {self.model_config.action_horizon}, "
                f"got {actions.shape[0]}."
            )
        actions = apply_action_transform(actions, state, self.action_transform)
        stats = self._resolved_stats()
        normalized_state = normalize(state, stats["state"])
        normalized_actions = normalize(actions, stats["actions"])
        tokens, token_mask = self.tokenize_prompt(self._prompt(frame))
        return {
            "image": {
                destination: _resize_with_pad(_convert_image(frame[source]))
                for destination, source in self.image_fields.items()
            },
            "image_mask": dict.fromkeys(CORE_IMAGE_KEYS, np.True_),
            "state": _pad_last_dim(normalized_state, self.model_config.model_action_dim),
            "actions": _pad_last_dim(normalized_actions, self.model_config.model_action_dim),
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
        }

    def transform_runtime_numpy(self, observation: Any) -> tuple[dict[str, Any], np.ndarray]:
        raw_state = np.asarray(observation.state, dtype=np.float32)
        normalized_state = normalize(raw_state, self._resolved_stats()["state"])
        images: dict[str, list[np.ndarray]] = {key: [] for key in CORE_IMAGE_KEYS}
        tokens: list[np.ndarray] = []
        token_masks: list[np.ndarray] = []
        for batch_index, prompt in enumerate(observation.prompt):
            token, token_mask = self.tokenize_prompt(prompt)
            tokens.append(token)
            token_masks.append(token_mask)
            for public_name, core_name in self.model_config.camera_mapping.items():
                images[core_name].append(
                    _resize_with_pad(_convert_image(observation.images[public_name][batch_index]))
                )
        batch_size = observation.batch_size
        return (
            {
                "image": {
                    key: np.ascontiguousarray(np.stack(value), dtype=np.uint8)
                    for key, value in images.items()
                },
                "image_mask": {key: np.ones(batch_size, dtype=np.bool_) for key in CORE_IMAGE_KEYS},
                "state": np.ascontiguousarray(
                    _pad_last_dim(normalized_state, self.model_config.model_action_dim),
                    dtype=np.float32,
                ),
                "tokenized_prompt": np.ascontiguousarray(np.stack(tokens), dtype=np.int32),
                "tokenized_prompt_mask": np.ascontiguousarray(
                    np.stack(token_masks), dtype=np.bool_
                ),
            },
            raw_state,
        )

    def transform_runtime(self, observation: Any) -> tuple[dict[str, Any], np.ndarray]:
        encoded, raw_state = self.transform_runtime_numpy(observation)
        return (
            {
                "image": {key: torch.from_numpy(value) for key, value in encoded["image"].items()},
                "image_mask": {
                    key: torch.from_numpy(value) for key, value in encoded["image_mask"].items()
                },
                "state": torch.from_numpy(encoded["state"]),
                "tokenized_prompt": torch.from_numpy(encoded["tokenized_prompt"]).long(),
                "tokenized_prompt_mask": torch.from_numpy(encoded["tokenized_prompt_mask"]),
            },
            raw_state,
        )

    def decode_actions(self, value: Any, raw_state: np.ndarray) -> np.ndarray:
        actions = _as_numpy(value)
        if actions.ndim == 2:
            actions = actions[None]
        if actions.ndim != 3:
            raise ValueError(f"Pi0 actions must be [B,H,A], got {actions.shape}.")
        actions = actions[
            :, : self.model_config.action_horizon, : self.model_config.action_dim
        ].astype(np.float64, copy=False)
        actions = unnormalize(actions, self._resolved_stats()["actions"])
        actions = invert_action_transform(actions, raw_state, self.action_transform)
        return np.ascontiguousarray(actions.astype(np.float32))

    def build_dataloader(self):
        data = self.root_config.data
        experiment = self.root_config.get("exp", {})
        experiment_name = experiment.get("run_dir") or experiment.get("name")
        progress_logger = (
            logging.getLogger(f"dobot_nexus.{experiment_name}")
            if experiment_name
            else LOGGER
        )
        dataset = build_source(
            data_path=str(data.path),
            action_key=self.action_key,
            language_key=self.language_key,
            action_horizon=self.model_config.action_horizon,
            format_version_name=str(data.get("format_version", "auto")),
            fps=int(data.fps) if data.get("fps") is not None else None,
            tolerance_s=float(data.get("tolerance_s", 1e-4)),
        )
        if self.stats is None:
            spec = LeRobotNormStatsSpec(
                state_key=self.state_key,
                action_key=self.action_key,
                state_dim=self.model_config.state_dim,
                action_dim=self.model_config.action_dim,
                action_horizon=self.model_config.action_horizon,
                action_transform=self.action_transform,
            )
            self._set_stats(
                resolve_or_compute_lerobot_norm_stats(
                    dataset.dataset_root,
                    spec,
                    cache_root=_norm_stats_cache_root(self.root_config),
                    logger=progress_logger,
                )
            )
        transformed = _TransformedDataset(dataset, self)
        seed = int(self.root_config.get("exp", {}).get("seed", 0))
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            transformed,
            batch_size=int(data.get("batch_size", 1)),
            shuffle=True,
            num_workers=int(data.get("num_workers", 0)),
            persistent_workers=int(data.get("num_workers", 0)) > 0,
            collate_fn=self.collate_items,
            drop_last=True,
            pin_memory=True,
            generator=generator,
        )
        resolved = ResolvedDataConfig(
            dataset_root=str(dataset.dataset_root),
            format_version=dataset.detected_format_version,
            fps=dataset.resolved_fps,
            image_field_mapping=dict(self.image_fields),
            state_key=self.state_key,
            action_key=self.action_key,
            language_key=self.language_key,
            action_transform=self.action_transform.kind,
            tokenizer_path=str(self.tokenizer_path),
            norm_stats_path=str(self.norm_stats_path),
        )
        return loader, resolved


class _TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, source: torch.utils.data.Dataset, pipeline: Pi0Pipeline) -> None:
        self.source = source
        self.pipeline = pipeline

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.pipeline.transform_frame(self.source[index])


def collate_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty Pi0 batch.")
    return {
        "image": {
            key: torch.from_numpy(np.stack([np.asarray(item["image"][key]) for item in items]))
            for key in CORE_IMAGE_KEYS
        },
        "image_mask": {
            key: torch.from_numpy(
                np.stack([np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items])
            )
            for key in CORE_IMAGE_KEYS
        },
        "state": torch.from_numpy(
            np.stack([np.asarray(item["state"], dtype=np.float32) for item in items])
        ),
        "tokenized_prompt": torch.from_numpy(
            np.stack([np.asarray(item["tokenized_prompt"], dtype=np.int64) for item in items])
        ).long(),
        "tokenized_prompt_mask": torch.from_numpy(
            np.stack([np.asarray(item["tokenized_prompt_mask"], dtype=np.bool_) for item in items])
        ),
        "actions": torch.from_numpy(
            np.stack([np.asarray(item["actions"], dtype=np.float32) for item in items])
        ),
    }
