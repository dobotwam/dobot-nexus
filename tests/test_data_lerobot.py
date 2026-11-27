from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
                camera_mapping={
                    "base": "base_0_rgb",
                    "left_wrist": "left_wrist_0_rgb",
                    "right_wrist": "right_wrist_0_rgb",
                },
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


class _ParquetTable:
    """Minimal datasets indexing surface over real Arrow shards for CPU-only installs."""

    def __init__(self, table):
        self.table = table
        self.column_names = table.column_names

    @classmethod
    def load(cls, format_name, *, data_dir, split):
        import pyarrow as pa
        import pyarrow.parquet as pq
        return cls(pa.concat_tables([pq.read_table(p) for p in sorted(Path(data_dir).glob("*.parquet"))]))

    def select_columns(self, names):
        return type(self)(self.table.select(names))

    def __len__(self):
        return self.table.num_rows

    def __iter__(self):
        if "image" in self.column_names:
            raise AssertionError("Episode bound verification accessed a camera column")
        return iter(self.table.to_pylist())

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.table[index].to_pylist()
        if isinstance(index, list):
            return self.table.take(index).to_pydict()
        return self.table.slice(index, 1).to_pylist()[0]


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is optional")
class OfflineParquetEpisodesTest(unittest.TestCase):
    def check_shards(self, directory):
        from dobot_nexus.algorithms.rltoken.lerobot_offline import read_episodes
        from dobot_nexus.data.lerobot.dataset import LocalLeRobotV21Dataset, LeRobotSource, episode_indices
        from tests.test_inline_source import dataset

        root = Path(directory)
        dataset(root, reverse_meta=True, split=True)
        source = LeRobotSource(data_path=str(root), action_key="action", language_key=None,
                               action_horizon=3, format_version_name="auto", fps=None, tolerance_s=1e-4)
        self.assertIsInstance(source.dataset, LocalLeRobotV21Dataset)
        self.assertEqual(source.dataset.episode_bounds, {0: (0, 2), 1: (2, 4)})
        self.assertEqual(episode_indices(source, [1]), [2, 3])
        np.testing.assert_array_equal(source[1]["action"], [[1]] * 3)
        np.testing.assert_array_equal(source[2]["action"], [[102], [103], [103]])
        labels = {"1": {"origin": "demo", "success": True, "terminated": True},
                  "0": {"origin": "vla", "success": False, "terminated": True}}
        path = root / "labels.json"
        path.write_text(json.dumps(labels))
        episodes = list(read_episodes(Config(path=str(root), episode_labels=str(path), images={"base": "image"},
            state_key="state", action_key="action", last_frame_is_final_observation=True)))
        self.assertEqual([e.episode_id for e in episodes], list(labels))
        for episode in episodes:
            index = int(episode.episode_id)
            label = labels[episode.episode_id]
            self.assertEqual([obs.state.item() for obs in episode.observations], [2*index, 2*index + 1])
            self.assertEqual(episode.commands.tolist(), [[102 * index]])
            self.assertEqual(episode.rewards.tolist(), [float(label["success"])])
            self.assertEqual((episode.origin, episode.success), (label["origin"], label["success"]))
        # Upstream readers without precomputed bounds still expose the true episode column.
        source.dataset.episode_bounds = None
        self.assertEqual(episode_indices(source, [1]), [2, 3])

    def test_reversed_metadata_multi_shard_arrow(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            "sys.modules", datasets=SimpleNamespace(load_dataset=_ParquetTable.load)
        ):
            self.check_shards(tmp)

    def test_fallback_episode_index_scans_each_physical_table_once(self):
        from dobot_nexus.data.lerobot.dataset import episode_indices

        table = mock.MagicMock()
        table.__getitem__.return_value = [1, 0, 1, 0]
        source = SimpleNamespace(dataset=SimpleNamespace(hf_dataset=table))
        self.assertEqual(episode_indices(source, [0]), [1, 3])
        self.assertEqual(episode_indices(source, [1]), [0, 2])
        self.assertEqual(episode_indices(source, [0, 1]), [0, 1, 2, 3])
        table.__getitem__.assert_called_once_with("episode_index")

    def local_dataset(self, root):
        from dobot_nexus.data.lerobot.dataset import LocalLeRobotV21Dataset
        return LocalLeRobotV21Dataset(root=root, info={"features": {"image": {"dtype": "image"}}},
                                      action_key="action", action_horizon=1, tolerance_s=1e-4)

    def test_nested_hf_media_are_disabled_before_arrow_rows_are_indexed(self):
        import pyarrow as pa
        from tests.test_inline_source import dataset

        @dataclass(frozen=True)
        class Image:
            decode: bool = True

        class NestedTable(_ParquetTable):
            def cast_column(self, key, feature):
                result = NestedTable(self.table)
                result.features = dict(self.features, **{key: feature})
                return result

            def __getitem__(self, index):
                if "unused" in self.column_names:
                    frame = self.features["unused"].features["frames"].feature[0]
                    if frame.decode:
                        raise AssertionError("HF attempted to decode an unused nested image")
                return super().__getitem__(index)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            outside = str(root.parent / "external.png")
            cell = {"frames": [{"bytes": None, "path": outside}]}
            rows = [dict(row, unused=cell) for row in dataset(root)]
            table = NestedTable(pa.Table.from_pylist(rows))
            table.features = {"unused": SimpleNamespace(features={"frames": SimpleNamespace(feature=[Image()])})}
            hf = SimpleNamespace(load_dataset=mock.Mock(return_value=table), Image=Image)
            with mock.patch.dict("sys.modules", datasets=hf):
                reader = self.local_dataset(root)
                self.assertEqual(reader[0]["unused"], cell)
                # A nested decoder without a switch must fail before fetching a row.
                table.features = {"unused": {"frames": [SimpleNamespace(decode_example=lambda value: value)]}}
                with self.assertRaisesRegex(ValueError, "Cannot disable HF decoding"):
                    self.local_dataset(root)

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets is optional")
    def test_nested_huggingface_image_path_stays_encoded(self):
        import datasets
        import pyarrow.parquet as pq
        from tests.test_inline_source import dataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            rows = dataset(root)
            outside = str(root.parent / "external.png")
            cell = {"frames": [{"bytes": None, "path": outside}]}
            features = datasets.Dataset.from_list(rows).features
            features["unused"] = {"frames": datasets.Sequence(datasets.Image())}
            hf = datasets.Dataset.from_list([dict(row, unused=cell) for row in rows], features=features)
            # Keep the path-only Arrow storage and its real HF feature metadata.
            pq.write_table(hf.data.table, root / "data/000.parquet")
            with mock.patch.object(datasets.Image, "decode_example",
                                   side_effect=AssertionError("unexpected external image decode")):
                self.assertEqual(self.local_dataset(root)[0]["unused"], cell)

    def test_hf_path_only_images_go_through_contained_reader(self):
        import pyarrow as pa
        from dobot_nexus.algorithms.rltoken.lerobot_offline import read_episodes
        from tests.test_inline_source import dataset

        @dataclass
        class Image:
            decode: bool = True

        class ImageTable(_ParquetTable):
            def __init__(self, table):
                super().__init__(table)
                self.features = {"image": Image()} if "image" in self.column_names else {}

            def cast_column(self, name, feature):
                result = ImageTable(self.table)
                result.features[name] = feature
                return result

            def __getitem__(self, index):
                if "image" in self.features and self.features["image"].decode:
                    raise AssertionError("HF accessed an image path with automatic decoding enabled")
                return super().__getitem__(index)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            root.mkdir()
            rows = dataset(root)
            image = root / "frame.png"
            image.write_bytes(rows[0]["image"]["bytes"])
            (root / "linked.png").symlink_to(image)
            labels = root / "labels.json"
            labels.write_text('{"0":{"origin":"demo","success":true,"terminated":true}}')
            cfg = Config(path=str(root), episode_labels=str(labels), images={"base": "image"},
                         state_key="state", action_key="action", last_frame_is_final_observation=True)
            for name in ("frame.png", "linked.png", str(image), "../frame.png"):
                table = ImageTable(pa.Table.from_pylist([
                    dict(row, image={"path": name, "bytes": None}) for row in rows
                ]))
                hf = SimpleNamespace(load_dataset=mock.Mock(return_value=table), Image=Image)
                with self.subTest(path=name), mock.patch.dict("sys.modules", datasets=hf):
                    if name == "frame.png":
                        episode, = read_episodes(cfg)
                        self.assertEqual(episode.observations[0].images["base"].shape, (1, 2, 2, 3))
                    else:
                        with self.assertRaisesRegex(ValueError, "dataset root"):
                            list(read_episodes(cfg))

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets is optional")
    def test_reversed_metadata_multi_shard_huggingface(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.check_shards(tmp)

    def test_rejects_inconsistent_episode_and_frame_indices(self):
        import pyarrow as pa
        from dobot_nexus.data.lerobot.dataset import LocalLeRobotV21Dataset
        from tests.test_inline_source import dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = dataset(root)
            for episodes, frames in (([0, 1, 0, 1], [0, 0, 1, 1]),
                                     ([0, 0, 1, 1], [0, 0, 0, 1]),
                                     ([0, 0, 0, 1], [0, 1, 2, 0]),
                                     ([0, 0, 2, 2], [0, 1, 0, 1])):
                table = pa.Table.from_pylist([dict(row, episode_index=e, frame_index=f)
                                              for row, e, f in zip(rows, episodes, frames)])
                with self.subTest(episodes=episodes, frames=frames), mock.patch.dict(
                    "sys.modules", datasets=SimpleNamespace(load_dataset=lambda *a, **k: _ParquetTable(table))
                ), self.assertRaises(ValueError):
                    LocalLeRobotV21Dataset(root=root, info={"features": {"image": {"dtype": "image"}}},
                                           action_key="action", action_horizon=1, tolerance_s=1e-4)


class DatasetImageTest(unittest.TestCase):
    def test_image_uint8_converts_bf16_chw_with_width_four(self):
        from dobot_nexus.data.images import image_uint8
        rgb = torch.zeros(3, 8, 4, dtype=torch.bfloat16)
        rgb[1] = 1
        result = image_uint8(rgb)
        self.assertEqual(result.shape, (8, 4, 3))
        np.testing.assert_array_equal(result[3, 1], [0, 255, 0])

    def test_image_paths_must_be_local_dataset_files(self):
        from dobot_nexus.data.images import image_uint8
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            root.mkdir()
            inside = root / "inside.png"
            Image.new("RGB", (3, 2)).save(inside)
            self.assertEqual(image_uint8({"path": inside.name}, root=root).shape, (2, 3, 3))
            outside = root.parent / "outside.png"
            outside.write_bytes(inside.read_bytes())
            (root / "escape").symlink_to(outside)
            (root / "dir-link").symlink_to(root, target_is_directory=True)
            for name in (str(inside), str(outside), "../outside.png", "escape",
                         "dir-link/inside.png", ".", "missing"):
                with self.subTest(path=name), self.assertRaisesRegex(ValueError, "dataset root"):
                    image_uint8({"path": name}, root=root)


if __name__ == "__main__":
    unittest.main()
