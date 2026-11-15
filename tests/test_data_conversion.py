from __future__ import annotations

import io
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from data_processing.convert import CAMERAS, convert, prepare_image
from data_processing.convert_recap import convert as convert_recap
from dobot_nexus.algorithms.recap.configuration import OutcomeSource, ReCapDatasetSpec
from dobot_nexus.algorithms.recap.data import build_frame_index, read_episode_outcomes
from dobot_nexus.data.lerobot.source import build_source


class ConversionTest(unittest.TestCase):
    def inputs(self, root):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[30:430, 145:545, 0] = 230
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=95)
        frames = []
        for i in range(3):
            frame = {f"{camera}_frame": buffer.getvalue() for camera in CAMERAS}
            frame.update(timestamp=100 + i * 0.1, episode_success="failure", is_human=i == 1)
            for prefix in ("", "action_"):
                for side in ("left", "right"):
                    frame[f"{prefix}{side}_dobot_arm_joints"] = np.full(7, i, np.float32)
            frames.append(frame)
        path = root / "episode.pkl"
        path.write_bytes(pickle.dumps(frames))
        return path, frames

    def test_png_geometry_and_bc_reader_preserve_chunks_and_physical_time(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self.inputs(root)
            config = {"sources": [{"path": str(path), "task": "insert plug"}]}
            output = convert(config, root / "converted")
            rows = pq.read_table(output / "data/chunk-000/episode_000000.parquet").to_pylist()
            self.assertFalse(rows[-1]["is_success"])
            self.assertEqual([r["is_human"] for r in rows], [False, True, False])
            self.assertAlmostEqual(rows[-1]["timestamp"], 0.1)
            self.assertEqual(rows[-1]["source_timestamp"], 100.2)
            source = build_source(
                data_path=output,
                action_key="action",
                language_key="task",
                action_horizon=4,
                format_version_name="auto",
            )
            np.testing.assert_array_equal(source[1]["action"][:, 0], [1, 2, 2, 2])
            for camera in CAMERAS:
                image = source[0][f"observation.images.{camera}"]
                self.assertEqual(image.shape, (224, 224, 3))
                self.assertEqual(image.dtype, np.uint8)
                self.assertGreater(image[100, 100, 0], 220)
                self.assertLess(image[100, 100, 2], 10)
                if camera == "top_camera":
                    # The 400x400 red ROI fills the square without wrist letterboxing.
                    self.assertGreater(image[10, 10, 0], 220)
                else:
                    self.assertTrue((image[:28] == 0).all())
                    self.assertTrue((image[196:] == 0).all())
            conversion = json.loads((output / "meta/conversion.json").read_text())
            self.assertFalse(conversion["resampled"])
            with self.assertRaises(FileExistsError):
                convert(config, output)
            with self.assertRaisesRegex(ValueError, "resample"):
                build_source(
                    data_path=output,
                    action_key="action",
                    language_key="task",
                    action_horizon=4,
                    fps=10,
                )

    def test_missing_outcome_fails_atomically_unknown_human_remains_unknown(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, frames = self.inputs(root)
            for frame in frames:
                del frame["episode_success"], frame["is_human"]
            path.write_bytes(pickle.dumps(frames))
            source = {"path": str(path), "task": "insert plug"}
            with self.assertRaisesRegex(ValueError, "episode_success"):
                convert({"sources": [source], "require_success": True}, root / "converted")
            self.assertFalse((root / "converted").exists())
            source["success"] = True
            output = convert({"sources": [source]}, root / "converted")
            rows = pq.read_table(output / "data/chunk-000/episode_000000.parquet").to_pylist()
            self.assertTrue(all(r["is_human"] is None for r in rows))
            with self.assertRaisesRegex(ValueError, "ROI"):
                prepare_image(frames[0]["top_camera_frame"], roi=[500, 0, 400, 400])


    def test_bc_without_labels_preserves_unknown_and_remains_readable(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, frames = self.inputs(root)
            for frame in frames:
                del frame["episode_success"], frame["is_human"]
                frame["action_left_dobot_arm_joints"] += 0.5
            path.write_bytes(pickle.dumps(frames))
            config = {"sources": [{"path": str(path), "task": "insert plug"}]}
            output = convert(config, root / "converted")
            table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
            self.assertEqual(table["is_success"].to_pylist(), [None] * 3)
            self.assertEqual(table["is_human"].to_pylist(), [None] * 3)
            source = build_source(
                data_path=output, action_key="action", language_key="task", action_horizon=2
            )
            np.testing.assert_array_equal(source[0]["action"][:, 0], [0.5, 1.5])
            np.testing.assert_array_equal(source[0]["observation.state"], np.zeros(14))
            self.assertNotIn("reward", table.column_names)
            self.assertNotIn("terminated", table.column_names)
            # BC acceptance must not turn unknown outcomes into valid RECAP rollout labels.
            spec = self.recap_spec(output)
            with self.assertRaisesRegex(ValueError, "outcome.*boolean"):
                read_episode_outcomes(
                    spec, build_frame_index(spec), dataset_type="rollout",
                    source=OutcomeSource("frame_column", field="is_success"),
                )

    @staticmethod
    def recap_spec(output):
        return ReCapDatasetSpec(
            path=output, dataset_id="converted", format_version="v2.1",
            image_fields=dict(zip(
                ("base", "left_wrist", "right_wrist"),
                (f"observation.images.{camera}" for camera in CAMERAS), strict=True,
            )),
            state_key="observation.state", action_key="action", language_key="task", fps=None,
        )

    def test_legacy_entry_keeps_strict_default_and_uses_shared_conversion(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, frames = self.inputs(root)
            config = {"sources": [{"path": str(path), "task": "insert plug"}]}
            generic = convert(config, root / "generic")
            legacy = convert_recap(config, root / "legacy")
            relative = "data/chunk-000/episode_000000.parquet"
            self.assertTrue(
                pq.read_table(generic / relative).equals(pq.read_table(legacy / relative))
            )
            spec = self.recap_spec(generic)
            self.assertEqual(read_episode_outcomes(
                spec, build_frame_index(spec), dataset_type="rollout",
                source=OutcomeSource("frame_column", field="is_success"),
            ), {0: False})
            self.assertNotIn("require_success", config)
            for frame in frames:
                del frame["episode_success"]
            path.write_bytes(pickle.dumps(frames))
            with self.assertRaisesRegex(ValueError, "episode_success"):
                convert_recap(config, root / "unknown")
            self.assertFalse((root / "unknown").exists())

    def test_episode_outcomes_are_consistent_and_never_inferred_from_human(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, frames = self.inputs(root)
            config = {"sources": [{"path": str(path), "task": "insert plug"}]}
            frames[-1]["episode_success"] = None
            path.write_bytes(pickle.dumps(frames))
            output = convert(config, root / "known")
            table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
            self.assertEqual(table["is_success"].to_pylist(), [False] * 3)
            for required in (False, True):
                for source_conflict in (False, True):
                    with self.subTest(required=required, source_conflict=source_conflict):
                        frames[-1]["episode_success"] = None if source_conflict else True
                        path.write_bytes(pickle.dumps(frames))
                        source = dict(config["sources"][0])
                        if source_conflict:
                            source["success"] = True
                        with self.assertRaisesRegex(ValueError, "conflicting episode success"):
                            convert(
                                {"sources": [source], "require_success": required}, root / "bad"
                            )
                        self.assertFalse((root / "bad").exists())

    def test_recorded_extra_fields_roundtrip_without_inventing_transition_labels(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, frames = self.inputs(root)
            for i, frame in enumerate(frames):
                frame.update(
                    raw_reward=np.float32(i / 2), terminal=np.bool_(False),
                    truncated=i == 2, policy_version="policy-v1", chunk_step=np.int64(i),
                    log_prob=np.float64(-i / 10),
                )
            frames[1].pop("policy_version")
            path.write_bytes(pickle.dumps(frames))
            extras = {
                "reward": {"source": "raw_reward", "dtype": "float32", "required": True},
                "terminated": {"source": "terminal", "dtype": "bool", "required": True},
                "truncated": {"dtype": "bool", "required": True},
                "policy_version": {"dtype": "string"},
                "chunk_step": {"dtype": "int64"},
                "behavior_log_prob": {"source": "log_prob", "dtype": "float64"},
                "unrecorded": {"dtype": "float32"},
            }
            config = {
                "sources": [{"path": str(path), "task": "insert plug"}], "extra_fields": extras
            }
            output = convert(config, root / "converted")
            table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
            self.assertEqual(table["reward"].to_pylist(), [0.0, 0.5, 1.0])
            self.assertEqual(table["terminated"].to_pylist(), [False] * 3)
            self.assertEqual(table["truncated"].to_pylist(), [False, False, True])
            self.assertEqual(table["policy_version"].to_pylist(), ["policy-v1", None, "policy-v1"])
            self.assertEqual(table["chunk_step"].to_pylist(), [0, 1, 2])
            self.assertEqual(table["unrecorded"].to_pylist(), [None] * 3)
            self.assertEqual(table["behavior_log_prob"].to_pylist(), [0.0, -0.1, -0.2])
            info = json.loads((output / "meta/info.json").read_text())
            for name, definition in extras.items():
                self.assertEqual(info["features"][name]["dtype"], definition["dtype"])
            provenance = json.loads((output / "meta/sources.jsonl").read_text())
            self.assertEqual(provenance["unknown_extra_frames"]["policy_version"], 1)
            self.assertEqual(provenance["unknown_extra_frames"]["unrecorded"], 3)
            source = build_source(
                data_path=output, action_key="action", language_key="task", action_horizon=2
            )
            self.assertEqual(source[2]["reward"], 1.0)
            self.assertFalse(source[2]["terminated"])

    def test_invalid_or_missing_extra_fields_fail_before_publishing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, frames = self.inputs(root)
            config = {"sources": [{"path": str(path), "task": "insert plug"}]}
            invalid = [
                ("float32", float("nan")), ("float32", float("inf")),
                ("float32", 1e100), ("float32", True), ("float32", "0.5"),
                ("bool", "false"), ("bool", 0), ("int64", 1.5),
                ("int64", True), ("int64", 2**63), ("string", 123),
                ("float32", [1.0]), ("float32", None),
            ]
            for dtype, value in invalid:
                with self.subTest(dtype=dtype, value=value):
                    frames[-1]["extra"] = value
                    config["extra_fields"] = {"extra": {"dtype": dtype, "required": True}}
                    # Earlier frames are valid so validation reaches the bad value.
                    for frame in frames[:-1]:
                        frame["extra"] = {"float32": 0.0, "int64": 0, "bool": False,
                                          "string": "ok"}[dtype]
                    path.write_bytes(pickle.dumps(frames))
                    with self.assertRaisesRegex(ValueError, "[Ee]xtra field"):
                        convert(config, root / "bad")
                    self.assertFalse((root / "bad").exists())
            for name in ("action", "is_success", "timestamp", "observation.images.extra", "task"):
                with self.subTest(name=name):
                    config["extra_fields"] = {name: {"dtype": "float32"}}
                    with self.assertRaisesRegex(ValueError, "conflicts"):
                        convert(config, root / "bad")
            for definition in ({"dtype": "unknown"}, {"dtype": "bool", "required": "false"}):
                config["extra_fields"] = {"extra": definition}
                with self.assertRaises(ValueError):
                    convert(config, root / "bad")


if __name__ == "__main__":
    unittest.main()
