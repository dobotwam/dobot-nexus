from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dobot_nexus.algorithms.recap.annotations import (
    file_digest,
    read_policy_labels,
    read_returns,
    write_labels,
)
from dobot_nexus.algorithms.recap.configuration import OutcomeSource
from dobot_nexus.algorithms.recap.data import (
    build_frame_index,
    read_episode_outcomes,
    read_interventions,
)
from dobot_nexus.algorithms.recap.labeling import run_returns
from dobot_nexus.algorithms.recap.targets import label_advantages

from .fakes import offline_config


class DataTest(unittest.TestCase):
    def test_mixed_provenance_roundtrip_preserves_unknown_and_human_override(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            path = sorted((config.dataset.path / "data").rglob("*.parquet"))[0]
            table = pq.read_table(path)
            table = table.set_column(
                table.schema.get_field_index("is_human"), "is_human",
                pa.array([None, True, False, None, False], type=pa.bool_()),
            )
            pq.write_table(table, path)
            index = build_frame_index(config.dataset)
            run_returns(config)
            returned = read_returns(config.annotation_dir, expected_index=index)
            self.assertEqual(
                returned.interventions.tolist(), [None, True, False, None, False, False, True, True]
            )
            self.assertEqual(
                returned.manifest["source_counts"], {"human": 3, "model": 3, "unknown": 2}
            )
            labels = label_advantages(
                np.arange(8), index.task_indices, positive_ratio=0.3,
                interventions=returned.interventions,
            )
            write_labels(
                config.annotation_dir, index=index, predicted_values=np.full(8, -0.5),
                labels=labels, value_checkpoint_id="mixed-provenance", config=config,
            )
            restored = read_policy_labels(config.annotation_dir, expected_index=index)
            np.testing.assert_array_equal(
                restored.positive, [False, True, False, False, False, True, True, True]
            )

    def test_fps_override_cannot_retime_returns_or_action_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            with self.assertRaisesRegex(ValueError, "resample"):
                build_frame_index(dataclasses.replace(config.dataset, fps=20))

    def test_split_episode_and_source_order_are_indexed_without_video_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory), order=(8, 3), split_first=True)
            index = build_frame_index(config.dataset)
            self.assertEqual(index.source_indices.tolist(), [3, 4, 5, 6, 7, 0, 1, 2])
            self.assertEqual(index.episode_offsets.tolist(), [0, 5, 8])
            with mock.patch(
                "dobot_nexus.algorithms.recap.value.dataset.build_source", side_effect=AssertionError
            ):
                run_returns(config)
            returned = read_returns(config.annotation_dir, expected_index=index)
            np.testing.assert_allclose(
                returned.returns, [-0.4, -0.3, -0.2, -0.1, 0, -0.7, -0.6, -0.5]
            )
            np.testing.assert_array_equal(
                returned.interventions, [False, True, True, False, False, False, True, True]
            )
            self.assertEqual(returned.scale.denominator(0), 10)
            with self.assertRaises(FileExistsError):
                run_returns(config)

    def test_unknown_provenance_is_not_model_and_success_is_not_inferred_from_human(self):
        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            index = build_frame_index(config.dataset)
            self.assertIsNone(read_interventions(config.dataset, index, field=None))
            with self.assertRaises(ValueError):
                read_interventions(config.dataset, index, field="missing")
            with self.assertRaises(ValueError):
                read_episode_outcomes(config.dataset, index, dataset_type="rollout", source=None)
            with self.assertRaises(ValueError):
                read_episode_outcomes(
                    config.dataset, index, dataset_type="sft", source=config.outcome
                )
            config = dataclasses.replace(config, intervention_field=None)
            run_returns(config)
            self.assertIsNone(
                read_returns(config.annotation_dir, expected_index=index).interventions
            )

    def test_explicit_all_success_jsonl_is_rollout_and_outcome_does_not_change_data_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = offline_config(root)
            original = build_frame_index(config.dataset)
            outcomes = root / "outcomes.jsonl"
            outcomes.write_text(
                '{"episode_index":3,"success":true}\n{"episode_index":8,"success":true}\n'
            )
            config = dataclasses.replace(
                config, outcome=OutcomeSource("episode_jsonl", path=outcomes)
            )
            run_returns(config)
            returned = read_returns(config.annotation_dir, expected_index=original)
            np.testing.assert_allclose(returned.returns, [-0.4, -0.3, -0.2, -0.1, 0, -0.2, -0.1, 0])
            self.assertEqual(returned.manifest["dataset_type"], "rollout")
            self.assertEqual(
                build_frame_index(config.dataset).dataset_signature, original.dataset_signature
            )
            for content in (
                '{"episode_index":3,"success":"false"}\n',
                '{"episode_index":3,"success":true}\n',
                '{"episode_index":3,"success":true}\n{"episode_index":3,"success":true}\n',
            ):
                outcomes.write_text(content)
                with self.assertRaises(ValueError):
                    read_episode_outcomes(
                        config.dataset, original, dataset_type="rollout", source=config.outcome
                    )

    def test_frame_gaps_and_duplicate_rows_fail(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            file = sorted((config.dataset.path / "data").rglob("*.parquet"))[0]
            rows = pq.read_table(file).to_pylist()
            for frame_index in (0, 4):
                rows[1]["frame_index"] = frame_index
                pq.write_table(pa.Table.from_pylist(rows), file)
                with self.assertRaisesRegex(ValueError, "layout"):
                    build_frame_index(config.dataset)

    def test_labels_task_threshold_keys_checksum_and_human_override(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            run_returns(config)
            index = build_frame_index(config.dataset)
            labels = label_advantages(
                np.arange(8) - 4,
                index.task_indices,
                positive_ratio=0.3,
                interventions=read_returns(
                    config.annotation_dir, expected_index=index
                ).interventions,
            )
            write_labels(
                config.annotation_dir,
                index=index,
                predicted_values=np.full(8, -0.5),
                labels=labels,
                value_checkpoint_id="test-model",
                config=config,
            )
            loaded = read_policy_labels(config.annotation_dir, expected_index=index)
            np.testing.assert_array_equal(
                loaded.positive, [False, True, True, False, False, True, True, True]
            )
            # Human rows 1 and 2 override negative advantages.
            self.assertTrue(loaded.positive[1])
            path = config.annotation_dir / "advantages.parquet"
            table = pq.read_table(path)
            pq.write_table(table.take([0] * 8), path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_policy_labels(config.annotation_dir, expected_index=index)
            manifest_path = config.annotation_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"]["advantages.parquet"] = file_digest(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "frame keys"):
                read_policy_labels(config.annotation_dir, expected_index=index)

    def test_incomplete_manifest_and_source_changes_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory))
            run_returns(config)
            info = config.dataset.path / "meta/info.json"
            raw = json.loads(info.read_text())
            raw["fps"] = 20
            info.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "signature"):
                read_returns(
                    config.annotation_dir, expected_index=build_frame_index(config.dataset)
                )
            manifest = config.annotation_dir / "manifest.json"
            raw = json.loads(manifest.read_text())
            raw["stages"]["returns"] = False
            manifest.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                read_returns(
                    config.annotation_dir, expected_index=build_frame_index(config.dataset)
                )

    def test_inline_image_source_preserves_rgb_and_action_chunk_boundary(self):
        import io

        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image

        from dobot_nexus.data.lerobot.source import build_source

        with tempfile.TemporaryDirectory() as directory:
            config = offline_config(Path(directory), split_first=True)
            path = config.dataset.path / "meta/info.json"
            info = json.loads(path.read_text())
            for name in config.dataset.image_fields.values():
                info["features"][name]["dtype"] = "image"
            info.pop("video_path")
            path.write_text(json.dumps(info))
            image = np.zeros((8, 12, 3), dtype=np.uint8)
            image[..., 0] = 231
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            for path in (config.dataset.path / "data").rglob("*.parquet"):
                rows = pq.read_table(path).to_pylist()
                for row in rows:
                    for name in config.dataset.image_fields.values():
                        row[name] = {"bytes": buffer.getvalue(), "path": None}
                pq.write_table(pa.Table.from_pylist(rows), path)
            build_frame_index(config.dataset)
            source = build_source(
                data_path=str(config.dataset.path),
                action_key="action",
                language_key="task",
                action_horizon=3,
            )
            np.testing.assert_array_equal(source[0][config.dataset.image_fields["base"]], image)
            np.testing.assert_array_equal(source[1]["action"][:, 0], [2, 3, 4])
            np.testing.assert_array_equal(source[4]["action"][:, 0], [5, 5, 5])
            self.assertEqual(source[5]["episode_index"], 8)
            self.assertEqual(source[0]["prompt"], "insert plug")
            from dobot_nexus.algorithms.recap.value.dataset import ReCapValueDataset

            run_returns(config)
            index = build_frame_index(config.dataset)
            returned = read_returns(config.annotation_dir, expected_index=index)
            dataset = ReCapValueDataset(config, index, returned, cache_files=3)
            self.assertEqual(len(dataset.source.cache), 3)
            for row in (0, 4, 5, 7):
                sample = dataset[row]
                self.assertEqual(sample.key, index.keys[row])
                self.assertEqual(sample.target_value, float(returned.returns[row]))
                np.testing.assert_array_equal(sample.images["base"], image)
