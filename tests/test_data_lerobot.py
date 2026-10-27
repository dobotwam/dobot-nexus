from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from dobot_nexus.core.config import Config
from dobot_nexus.data import ActionTransform, NormStats, load_norm_stats
from dobot_nexus.data.lerobot.norm_stats import (
    LeRobotNormStatsSpec,
    calculate_lerobot_norm_stats,
    calculate_norm_stats_from_episodes,
    resolve_or_compute_lerobot_norm_stats,
)
from dobot_nexus.models.pi05.data import Pi05Pipeline


def _spec(*, action_transform: str = "delta") -> LeRobotNormStatsSpec:
    return LeRobotNormStatsSpec(
        state_key="observation.state",
        action_key="action",
        state_dim=2,
        action_dim=2,
        action_horizon=2,
        action_transform=ActionTransform.create(
            action_transform,
            [True, False] if action_transform == "delta" else None,
            action_dim=2,
        ),
    )


class LeRobotNormStatsTest(unittest.TestCase):
    def test_action_chunks_tail_padding_and_delta_match_training_semantics(self) -> None:
        states = np.asarray([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        actions = np.asarray([[1.0, 100.0], [3.0, 200.0], [6.0, 300.0]])
        stats = calculate_norm_stats_from_episodes(
            iter([(0, states, actions, np.arange(3))]),
            _spec(),
        )

        query = np.minimum(np.arange(3)[:, None] + np.arange(2), 2)
        expected_actions = actions[query].copy()
        expected_actions[..., 0] -= states[:, None, 0]
        flattened = expected_actions.reshape(-1, 2)
        np.testing.assert_allclose(stats["state"].mean, states.mean(axis=0))
        np.testing.assert_allclose(stats["state"].std, states.std(axis=0))
        np.testing.assert_allclose(stats["actions"].mean, flattened.mean(axis=0))
        np.testing.assert_allclose(stats["actions"].std, flattened.std(axis=0))

    def test_absolute_actions_do_not_subtract_state(self) -> None:
        states = np.asarray([[10.0, 20.0], [30.0, 40.0]])
        actions = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        stats = calculate_norm_stats_from_episodes(
            iter([(0, states, actions, np.arange(2))]),
            _spec(action_transform="absolute"),
        )
        expected = actions[np.asarray([[0, 1], [1, 1]])].reshape(-1, 2)
        np.testing.assert_allclose(stats["actions"].mean, expected.mean(axis=0))
        np.testing.assert_allclose(stats["actions"].std, expected.std(axis=0))

    def test_non_contiguous_episode_frames_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "contiguous"):
            calculate_norm_stats_from_episodes(
                iter(
                    [
                        (
                            3,
                            np.zeros((2, 2)),
                            np.zeros((2, 2)),
                            np.asarray([0, 2]),
                        )
                    ]
                ),
                _spec(),
            )

    def test_stats_loader_accepts_wrapped_openpi_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "norm_stats.json"
            payload = {
                "norm_stats": {
                    "state": {"mean": [1.0], "std": [2.0], "q01": [0.0], "q99": [3.0]},
                    "actions": {"mean": [4.0], "std": [5.0]},
                }
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            stats = load_norm_stats(path)
        self.assertIsInstance(stats["state"], NormStats)
        self.assertEqual(stats["state"].q99.tolist(), [3.0])

    def test_pi05_pipeline_records_automatically_generated_stats(self) -> None:
        class FakeSource(torch.utils.data.Dataset):
            def __init__(self, root: Path) -> None:
                self.dataset_root = root
                self.detected_format_version = "v3.0"
                self.resolved_fps = 10

            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> dict[str, object]:
                raise AssertionError(
                    f"DataLoader should not read index {index} during construction"
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "cache" / "norm_stats.json"
            generated.parent.mkdir()
            generated.write_text(
                json.dumps(
                    {
                        "norm_stats": {
                            "state": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                            "actions": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
                        }
                    }
                ),
                encoding="utf-8",
            )
            data = Config(
                {
                    "path": str(root / "dataset"),
                    "batch_size": 1,
                    "num_workers": 0,
                    "field_mapping": {
                        "images": {
                            "base": "base_image",
                            "left_wrist": "left_image",
                            "right_wrist": "right_image",
                        },
                        "state": "observation.state",
                        "action": "action",
                        "language": "task",
                    },
                    "transforms": {"action": {"type": "absolute"}},
                }
            )
            config = Config(
                {
                    "task": "train",
                    "data": data,
                    "exp": Config(
                        {
                            "name": "pi05_test",
                            "output_dir": str(root / "runs"),
                            "run_dir": str(root / "runs" / "pi05_test"),
                            "seed": 3,
                        }
                    ),
                    "model": Config({"backend": Config()}),
                }
            )
            model_config = mock.Mock(
                max_token_len=8,
                state_dim=2,
                action_dim=2,
                action_horizon=2,
                model_action_dim=4,
            )
            with (
                mock.patch(
                    "dobot_nexus.models.pi05.data.pipeline.build_source",
                    return_value=FakeSource(root / "dataset"),
                ),
                mock.patch(
                    "dobot_nexus.models.pi05.data.pipeline.resolve_or_compute_lerobot_norm_stats",
                    return_value=generated,
                ) as resolve_stats,
            ):
                pipeline = Pi05Pipeline(config, model_config)
                _, resolved = pipeline.build_dataloader()

            self.assertEqual(config.data.norm_stats_path, str(generated.resolve()))
            self.assertEqual(resolved.norm_stats_path, str(generated.resolve()))
            self.assertEqual(
                resolve_stats.call_args.kwargs["cache_root"],
                (root / "runs" / "pi05_test" / "data").resolve(),
            )
            self.assertEqual(
                resolve_stats.call_args.kwargs["logger"].name,
                f"dobot_nexus.{config.exp.run_dir}",
            )
            request = resolve_stats.call_args.args[1]
            self.assertEqual(request.action_horizon, 2)
            self.assertEqual(request.action_transform.kind, "absolute")


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is optional")
class LeRobotParquetNormStatsTest(unittest.TestCase):
    def _dataset(self, root: Path, version: str) -> Path:
        import pyarrow as pa
        import pyarrow.parquet as parquet

        dataset = root / f"lerobot_{version.replace('.', '_')}"
        (dataset / "meta").mkdir(parents=True)
        (dataset / "data" / "chunk-000").mkdir(parents=True)
        (dataset / "meta" / "info.json").write_text(
            json.dumps({"codebase_version": version, "fps": 10}),
            encoding="utf-8",
        )
        # Deliberately wrong raw LeRobot stats: transformed/chunked stats must win.
        (dataset / "meta" / "stats.json").write_text(
            json.dumps(
                {
                    "observation.state": {"mean": [999.0, 999.0], "std": [1.0, 1.0]},
                    "action": {"mean": [999.0, 999.0], "std": [1.0, 1.0]},
                }
            ),
            encoding="utf-8",
        )
        table = pa.table(
            {
                "observation.state": pa.array(
                    [[0.0, 10.0], [1.0, 20.0], [5.0, 50.0]],
                    type=pa.list_(pa.float64(), 2),
                ),
                "action": pa.array(
                    [[1.0, 100.0], [3.0, 200.0], [7.0, 700.0]],
                    type=pa.list_(pa.float64(), 2),
                ),
                "episode_index": pa.array([0, 0, 1], type=pa.int64()),
                "frame_index": pa.array([0, 1, 0], type=pa.int64()),
            }
        )
        parquet.write_table(table, dataset / "data" / "chunk-000" / "file-000.parquet")
        return dataset

    def test_v21_and_v3_parquet_have_identical_transformed_stats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for version in ("v2.1", "v3.0"):
                with self.subTest(version=version):
                    dataset = self._dataset(Path(directory), version)
                    stats = calculate_lerobot_norm_stats(dataset, _spec())
                    np.testing.assert_allclose(stats["state"].mean, [2.0, 80.0 / 3.0])
                    np.testing.assert_allclose(stats["actions"].mean, [2.0, 350.0])
                    self.assertNotEqual(stats["actions"].mean.tolist(), [999.0, 999.0])

    def test_generated_cache_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self._dataset(root, "v3.0")
            cache = root / "cache"
            first = resolve_or_compute_lerobot_norm_stats(dataset, _spec(), cache_root=cache)
            self.assertTrue(first.is_file())
            with mock.patch(
                "dobot_nexus.data.lerobot.norm_stats.calculate_lerobot_norm_stats",
                side_effect=AssertionError("cache miss"),
            ):
                second = resolve_or_compute_lerobot_norm_stats(
                    dataset, _spec(), cache_root=cache
                )
            self.assertEqual(first, second)

    def test_cache_key_changes_with_preprocessing_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self._dataset(root, "v3.0")
            cache = root / "cache"
            absolute = _spec(action_transform="absolute")
            delta = _spec(action_transform="delta")
            absolute_path = resolve_or_compute_lerobot_norm_stats(
                dataset, absolute, cache_root=cache
            )
            delta_path = resolve_or_compute_lerobot_norm_stats(
                dataset, delta, cache_root=cache
            )
            self.assertNotEqual(absolute_path, delta_path)

    def test_meta_stats_is_not_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(Path(directory), "v2.1")
            (dataset / "meta" / "stats.json").unlink()
            stats = calculate_lerobot_norm_stats(dataset, _spec())
            np.testing.assert_allclose(stats["actions"].mean, [2.0, 350.0])


@unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets is optional")
class LeRobotImageSourceTest(unittest.TestCase):
    def test_embedded_images_and_actions_remain_inside_episode(self) -> None:
        import io
        from datasets import Dataset, Features, Image, Sequence, Value
        from PIL import Image as PILImage
        from dobot_nexus.data.lerobot.dataset import LeRobotSource

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "data").mkdir()
            info = {"codebase_version": "v2.1", "fps": 10,
                    "features": {"image": {"dtype": "image"}}}
            (root / "meta/info.json").write_text(json.dumps(info))
            (root / "meta/episodes.jsonl").write_text(
                '{"episode_index": 0, "length": 2}\n'
                '{"episode_index": 1, "length": 1}\n')
            (root / "meta/tasks.jsonl").write_text(
                '{"task_index": 0, "task": "insert the peg in the hole"}\n')
            encoded = io.BytesIO()
            PILImage.new("RGB", (4, 3), (19, 23, 47)).save(encoded, format="PNG")
            dataset = Dataset.from_dict({
                "image": [{"bytes": encoded.getvalue(), "path": None}] * 3,
                "action": [[0.1], [0.2], [0.9]],
                "episode_index": [0, 0, 1], "task_index": [0, 0, 0],
            }, features=Features({"image": Image(),
                "action": Sequence(Value("float32")),
                "episode_index": Value("int64"), "task_index": Value("int64")}))
            dataset.to_parquet(str(root / "data/episode.parquet"))
            source = LeRobotSource(data_path=str(root), action_key="action",
                language_key="task", action_horizon=3, format_version_name="v2.1",
                fps=None, tolerance_s=1e-4)
            item = source[0]
            np.testing.assert_allclose(item["action"], [[0.1], [0.2], [0.2]])
            np.testing.assert_allclose(source[1]["action"], [[0.2]] * 3)
            np.testing.assert_allclose(source[2]["action"], [[0.9]] * 3)
            self.assertEqual(item["prompt"], "insert the peg in the hole")
            np.testing.assert_array_equal(np.asarray(item["image"])[0, 0], [19, 23, 47])
            self.assertEqual(source.dataset.action_dataset.column_names, ["action"])
            self.assertEqual(source.detected_format_version, "v2.1")


if __name__ == "__main__":
    unittest.main()
