"""Local LeRobot and runtime codecs for the Pi0.5 operator core."""

from __future__ import annotations

import logging
import os
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
    validate_norm_stats,
)
from dobot_nexus.data.images import convert_image as _convert_image
from dobot_nexus.data.images import resize_with_pad as _resize_with_pad
from dobot_nexus.data.lerobot import (
    LeRobotNormStatsSpec,
    resolve_or_compute_lerobot_norm_stats,
)

from .source import build_source

LOGGER = logging.getLogger(__name__)
CORE_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
PUBLIC_CAMERA_TO_CORE = {
    "base": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}
_PI05_ROOT = Path(__file__).resolve().parents[1]


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
    normalization: str


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
        "Pi0.5 normalization statistics override was not found. "
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
    environment = os.getenv("OPENPI_TOKENIZER_PATH")
    if environment:
        candidates.append(Path(environment).expanduser())
    candidates.append(_PI05_ROOT / "assets" / "pi05_tokenizer.model")
    cache = Path(os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    candidates.extend(
        [
            cache / "paligemma_tokenizer.model",
            cache / "big_vision" / "paligemma_tokenizer.model",
        ]
    )
    for candidate in candidates:
        if candidate.is_dir():
            for filename in ("pi05_tokenizer.model", "paligemma_tokenizer.model"):
                nested = candidate / filename
                if nested.is_file():
                    return nested.resolve()
        elif candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Pi0.5 tokenizer was not found in the repository or OpenPI cache. "
        "Set OPENPI_TOKENIZER_PATH or data.tokenizer_path only for a custom override. "
        f"Checked: {', '.join(str(path) for path in candidates)}"
    )


def _pad_last_dim(value: np.ndarray, target: int) -> np.ndarray:
    if value.shape[-1] > target:
        raise ValueError(f"Cannot fit dimension {value.shape[-1]} into Pi0.5 dimension {target}.")
    if value.shape[-1] == target:
        return value
    widths = [(0, 0)] * value.ndim
    widths[-1] = (0, target - value.shape[-1])
    return np.pad(value, widths)


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
                raise ImportError("Pi0.5 prompt encoding requires sentencepiece.") from exc
            self._processor = sentencepiece.SentencePieceProcessor(
                model_proto=self.path.read_bytes()
            )
        return self._processor

    def encode(
        self, prompt: str, normalized_state: np.ndarray, *, strict: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        bins = np.linspace(-1, 1, 257)[:-1]
        discrete = np.digitize(normalized_state, bins=bins) - 1
        state_text = " ".join(map(str, discrete))
        text = f"Task: {cleaned}, State: {state_text};\nAction: "
        tokens = list(self._get_processor().encode(text, add_bos=True))
        if len(tokens) > self.max_length:
            if strict:
                raise ValueError(
                    f"Pi0.5 prompt has {len(tokens)} tokens, exceeding {self.max_length}; "
                    "RECAP requires the complete task, state and condition."
                )
            LOGGER.warning(
                "Pi0.5 prompt has %d tokens; truncating to %d.", len(tokens), self.max_length
            )
        tokens = tokens[: self.max_length]
        mask = [True] * len(tokens)
        padding = self.max_length - len(tokens)
        tokens.extend([0] * padding)
        mask.extend([False] * padding)
        return np.asarray(tokens, dtype=np.int64), np.asarray(mask, dtype=np.bool_)


def dataset_image_fields(raw_images: Mapping[str, str]) -> dict[str, str]:
    """Resolve dataset columns by the core slots populated during training."""
    return {
        PUBLIC_CAMERA_TO_CORE.get(str(name), str(name)): str(column)
        for name, column in raw_images.items()
    }


def _field_mapping(config: Any) -> tuple[dict[str, str], str, str, str | None]:
    mapping = config.get("data", {}).get("field_mapping", {})
    raw_images = dict(mapping.get("images", {}))
    images = dataset_image_fields(raw_images)
    if not images or not set(images).issubset(CORE_IMAGE_KEYS) or len(images) != len(raw_images):
        raise ValueError(
            "data.field_mapping.images must map a nonempty subset of base, left_wrist, "
            "and right_wrist without duplicate slots; "
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


class Pi05Pipeline:
    """One preprocessing contract shared by BC training and inference."""

    def __init__(self, root_config: Any, model_config: Any) -> None:
        self.root_config = root_config
        self.model_config = model_config
        data = root_config.get("data", {})
        self.normalization = data.get("normalization", "zscore")
        if isinstance(self.normalization, Mapping):
            # Deployed v1 Artifacts record the codec type and version together.
            # Resolve locally without mutating the saved inference configuration.
            if (set(self.normalization) != {"type", "version"}
                    or type(self.normalization["version"]) is not int
                    or self.normalization["version"] != 1):
                raise ValueError("data.normalization mapping requires type and version: 1.")
            self.normalization = self.normalization["type"]
        if not isinstance(self.normalization, str) or self.normalization not in {"zscore", "quantile"}:
            raise ValueError("data.normalization must be zscore or quantile.")
        self.image_fields, self.state_key, self.action_key, self.language_key = _field_mapping(
            root_config
        )
        camera_mapping = getattr(model_config, "camera_mapping", None)
        if camera_mapping is not None and set(self.image_fields) != set(camera_mapping.values()):
            raise ValueError("Training image fields and model.io.cameras must use the same slots.")
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
                "Pi0.5 inference requires the normalization statistics stored in its Artifact."
            )
        legacy_tokenizer = (
            root_config.get("model", {}).get("backend", {}).get("paligemma_tokenizer")
        )
        deployment_tokenizer = (
            os.getenv("OPENPI_TOKENIZER_PATH")
            if str(root_config.get("task", "")).lower() == "infer"
            else None
        )
        self.tokenizer_path = resolve_tokenizer_path(
            deployment_tokenizer or data.get("tokenizer_path") or legacy_tokenizer
        )
        self.tokenizer = _Tokenizer(self.tokenizer_path, model_config.max_token_len)
        if task == "train":
            data["tokenizer_path"] = str(self.tokenizer_path)
            data["normalization"] = self.normalization
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
            validate_norm_stats(self.stats[name], dimension, mode=self.normalization, name=name)
        if str(self.root_config.get("task", "")).lower() == "train":
            self.root_config.setdefault("data", {})["norm_stats_path"] = str(
                self.norm_stats_path
            )

    def _resolved_stats(self) -> dict[str, NormStats]:
        if self.stats is None:
            raise RuntimeError("Pi0.5 normalization statistics have not been resolved yet.")
        return self.stats

    def normalize(self, value: np.ndarray, name: str) -> np.ndarray:
        return normalize(value, self._resolved_stats()[name], mode=self.normalization)

    def unnormalize(self, value: np.ndarray, name: str) -> np.ndarray:
        return unnormalize(value, self._resolved_stats()[name], mode=self.normalization)

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
            raise ValueError("Pi0.5 prompt must be non-empty.")
        return prompt

    def tokenize_prompt(
        self, prompt: str, normalized_state: np.ndarray, *, strict: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.tokenizer.encode(prompt, normalized_state, strict=strict)

    def collate_items(self, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
        normalized_state = self.normalize(state, "state")
        normalized_actions = self.normalize(actions, "actions")
        tokens, token_mask = self.tokenize_prompt(self._prompt(frame), normalized_state)
        return {
            "image": {
                slot: (_resize_with_pad(_convert_image(frame[self.image_fields[slot]]))
                       if slot in self.image_fields else np.zeros((224, 224, 3), np.uint8))
                for slot in CORE_IMAGE_KEYS
            },
            "image_mask": {slot: np.bool_(slot in self.image_fields) for slot in CORE_IMAGE_KEYS},
            "state": _pad_last_dim(normalized_state, self.model_config.model_action_dim),
            "actions": _pad_last_dim(normalized_actions, self.model_config.model_action_dim),
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
        }

    def transform_runtime_numpy(self, observation: Any) -> tuple[dict[str, Any], np.ndarray]:
        raw_state = np.asarray(observation.state, dtype=np.float32)
        if raw_state.ndim != 2 or raw_state.shape[-1] != self.model_config.state_dim:
            raise ValueError(
                f"Expected state [B,{self.model_config.state_dim}], got {raw_state.shape}."
            )
        normalized_state = self.normalize(raw_state, "state")
        images: dict[str, list[np.ndarray]] = {key: [] for key in CORE_IMAGE_KEYS}
        tokens: list[np.ndarray] = []
        token_masks: list[np.ndarray] = []
        for batch_index, prompt in enumerate(observation.prompt):
            token, token_mask = self.tokenize_prompt(prompt, normalized_state[batch_index])
            tokens.append(token)
            token_masks.append(token_mask)
            for public_name, core_name in self.model_config.camera_mapping.items():
                images[core_name].append(
                    _resize_with_pad(_convert_image(observation.images[public_name][batch_index]))
                )
        batch_size = observation.batch_size
        active_slots = set(self.model_config.camera_mapping.values())
        return (
            {
                "image": {
                    key: (np.ascontiguousarray(np.stack(value), dtype=np.uint8)
                          if value else np.zeros((batch_size, 224, 224, 3), np.uint8))
                    for key, value in images.items()
                },
                "image_mask": {
                    key: np.full(batch_size, key in active_slots, dtype=np.bool_)
                    for key in CORE_IMAGE_KEYS
                },
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
            raise ValueError(f"Pi0.5 actions must be [B,H,A], got {actions.shape}.")
        if actions.shape[2] < self.model_config.action_dim:
            raise ValueError(
                "Pi0.5 output does not cover configured action_dim: "
                f"expected at least {self.model_config.action_dim}, got {actions.shape}."
            )
        actions = actions[
            :, : self.model_config.action_horizon, : self.model_config.action_dim
        ].astype(np.float64, copy=False)
        actions = self.unnormalize(actions, "actions")
        actions = invert_action_transform(actions, raw_state, self.action_transform)
        return np.ascontiguousarray(actions.astype(np.float32))

    def build_source(self):
        data = self.root_config.data
        return build_source(
            data_path=str(data.path),
            action_key=self.action_key,
            language_key=self.language_key,
            action_horizon=self.model_config.action_horizon,
            format_version_name=str(data.get("format_version", "auto")),
            fps=int(data.fps) if data.get("fps") is not None else None,
            tolerance_s=float(data.get("tolerance_s", 1e-4)),
        )

    def build_dataloader(self):
        data = self.root_config.data
        workers = int(data.get("num_workers", 0))
        prefetch = data.get("prefetch_factor", 2)
        if type(prefetch) is not int or prefetch <= 0:
            raise ValueError("data.prefetch_factor must be a positive integer.")
        drop_last = data.get("drop_last", True)
        if type(drop_last) is not bool:
            raise ValueError("data.drop_last must be a boolean.")
        experiment = self.root_config.get("exp", {})
        experiment_name = experiment.get("run_dir") or experiment.get("name")
        progress_logger = (
            logging.getLogger(f"dobot_nexus.{experiment_name}")
            if experiment_name
            else LOGGER
        )
        dataset = self.build_source()
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
        if data.get("episode_ids") is not None:
            from dobot_nexus.data.lerobot.dataset import episode_indices
            transformed = torch.utils.data.Subset(transformed, episode_indices(dataset, data.episode_ids))
        seed = int(self.root_config.get("exp", {}).get("seed", 0))
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            transformed,
            batch_size=int(data.get("batch_size", 1)),
            shuffle=True,
            num_workers=workers,
            persistent_workers=workers > 0,
            prefetch_factor=prefetch if workers > 0 else None,
            collate_fn=self.collate_items,
            drop_last=drop_last,
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
            normalization=self.normalization,
        )
        return loader, resolved


class _TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, source: torch.utils.data.Dataset, pipeline: Pi05Pipeline) -> None:
        self.source = source
        self.pipeline = pipeline

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.pipeline.transform_frame(self.source[index])


def collate_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("Cannot collate an empty Pi0.5 batch.")
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
