from __future__ import annotations

import hashlib
import io
import json
import mmap
import os
import pickle
import tempfile
import tracemalloc
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dobot_nexus.core.contracts import PolicyOutput
from dobot_nexus.inference.session import SessionOptions
from dobot_nexus.modules.memory import feature_store
from dobot_nexus.modules.memory.feature_store import FrameFeatureStore, feature_spec, write_episode
from dobot_nexus.modules.memory.position import position_embeddings
from dobot_nexus.modules.memory.runtime import HistoryBuffer, MemorySession
from dobot_nexus.modules.memory.sampling import assemble_memory, select_frames
from dobot_nexus.modules.memory.types import (
    MemoryConfig,
    MemoryObservationBatch,
    MemoryReference,
    MemorySnapshot,
)
from tests.modules import model_imports


def observation(*env_ids):
    return {
        "state": np.zeros((len(env_ids), 1), np.float32),
        "images": {"base": np.zeros((len(env_ids), 2, 4, 3), np.uint8)},
        "prompt": ["pick"] * len(env_ids),
        "env_ids": list(env_ids),
    }


class MemoryModuleTest(unittest.TestCase):
    def test_model_import_boundary(self):
        self.assertEqual(model_imports("memory"), [])

    def test_shared_single_frame_history_matches_the_store_for_both_schemas(self):
        tokens = np.stack([np.full((16, 2048), step, np.float32) for step in range(5)])
        policy = SimpleNamespace(spec=SimpleNamespace(action_horizon=1))
        for version, fields in (
            (1, {}),
            (2, {"image_field": "rgb", "language_field": None}),
        ):
            for window, start in ((None, 1), (2, 3)):
                with (
                    self.subTest(version=version, window=window),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    config = MemoryConfig(budget=16, vision_sha256="a" * 64, window_frames=window)
                    write_episode(
                        directory, 0, tokens, [0, 1, 1, 1, 1], config=config, dataset_id="s",
                        **fields,
                    )
                    store = FrameFeatureStore(directory, config, **fields)
                    self.addCleanup(store.close)
                    online = HistoryBuffer(config)
                    session = MemorySession(policy, np.stack, config, SessionOptions(1))
                    for step in range(1, 5):
                        online.append(tokens[step], step)
                    session.push_frames(tokens[1:], range(1, 5), env_id="a", prompt="pick")
                    prepared = session.prepare(observation("a"), step_indices=[4]).memory
                    saved = store.lookup(0, 4)
                    np.testing.assert_array_equal(saved.features, start)
                    np.testing.assert_array_equal(saved.positions, position_embeddings([0])[0])
                    current = online.snapshot()
                    np.testing.assert_array_equal(current.features, saved.features)
                    np.testing.assert_array_equal(current.positions, saved.positions)
                    np.testing.assert_array_equal(prepared.features[0], saved.features)
                    np.testing.assert_array_equal(prepared.positions[0], saved.positions)

    def test_rollback_preserves_action_chunk_when_policy_reuses_output_storage(self):
        shared_chunk = np.zeros((1, 2, 1), np.float32)
        calls = []

        def predict(obs, options):
            env = obs.env_ids[0]
            calls.append(env)
            if env == "c":
                raise RuntimeError("env c failed")
            shared_chunk[0, :, 0] = {"a": [1, 2], "b": [8, 9]}[env]
            return PolicyOutput(shared_chunk)

        policy = SimpleNamespace(spec=SimpleNamespace(action_horizon=2), predict_chunk=predict)
        session = MemorySession(
            policy,
            lambda frames: np.zeros((len(frames), 16, 2048)),
            MemoryConfig(budget=16),
            SessionOptions(2),
        )
        np.testing.assert_array_equal(session.act(observation("a")), [[1]])
        with self.assertRaisesRegex(RuntimeError, "env c failed"):
            session.act(observation("a", "b", "c"))
        np.testing.assert_array_equal(shared_chunk[0, :, 0], [8, 9])
        self.assertEqual(set(session.buffers), {"a"})
        self.assertEqual(session.buffers["a"].last_step, 0)
        np.testing.assert_array_equal(session.act(observation("a")), [[2]])
        self.assertEqual(calls, ["a", "b", "c"])

    def test_batch_ingestion_quota_precedes_encoding_across_environments(self):
        both = observation("a", "b")
        for method in ("prepare", "act", "predict_chunk"):
            with self.subTest(method=method):
                policy = Mock()
                policy.spec = SimpleNamespace(action_horizon=2)
                encoder = Mock(side_effect=lambda frames: np.zeros((len(frames), 16, 2048)))
                session = MemorySession(
                    policy, encoder, MemoryConfig(budget=16), SessionOptions(1), max_batch_frames=1
                )
                with self.assertRaisesRegex(ValueError, "History batch exceeds frame limit"):
                    getattr(session, method)(both)
                encoder.assert_not_called()
                policy.predict_chunk.assert_not_called()
                self.assertFalse(session.buffers)
                self.assertFalse(session.actions)
                self.assertFalse(session.prompts)
                session.prepare(observation("a"), step_indices=[0])
                session.prepare(both, step_indices=[0, 0])
                self.assertEqual(encoder.call_count, 2)  # Reused A, ingested B.
                changed = {**both, "prompt": ["new task", "pick"]}
                with self.assertRaisesRegex(ValueError, "frame limit"):
                    getattr(session, method)(changed, step_indices=[0, 1])
                self.assertEqual(encoder.call_count, 2)
                self.assertEqual(session.prompts, {"a": "pick", "b": "pick"})

    def test_relative_positions_sampling_and_padding(self):
        config = MemoryConfig.from_dict({"budget": 48})
        indices, offsets = select_frames(12, config.sampled_frames, segment_start=8)
        np.testing.assert_array_equal(indices, [8, 10, 12])
        np.testing.assert_array_equal(offsets, [0, 2, 4])
        for window, start in [(None, 8), (2, 11)]:
            indices, offsets = select_frames(12, 1, segment_start=8, window_frames=window)
            np.testing.assert_array_equal(indices, [start])
            np.testing.assert_array_equal(offsets, [0])
        positions = position_embeddings([0, 2])
        np.testing.assert_array_equal(positions[0, :, :128], 0)
        np.testing.assert_array_equal(positions[0, :, 128:256], 1)
        np.testing.assert_array_equal(positions[0, :, 256:], positions[1, :, 256:])
        self.assertFalse(np.array_equal(positions[0, 0, 256:], positions[0, 1, 256:]))
        snapshot = assemble_memory(np.ones((2, 16, 2048)), [0, 2], config)
        np.testing.assert_array_equal(snapshot.positions[:32], positions.reshape(32, 768))
        np.testing.assert_array_equal(snapshot.features[32:], 0)
        np.testing.assert_array_equal(snapshot.mask, np.arange(48) < 32)
        batch = MemorySnapshot.stack([snapshot, snapshot])
        self.assertIs(batch.validate(config, batch_size=2), batch)
        with self.assertRaisesRegex(ValueError, "divisible"):
            MemoryConfig(budget=17)

    def test_read_and_extend_existing_store_versions(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        columns = {"image_field": "top.rgb", "language_field": "instruction"}
        features = np.ones((1, 16, 2048), np.float32)
        for version, fields, incompatible in [(1, {}, columns), (2, columns, {})]:
            with self.subTest(version=version), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = root / "existing.npy"
                np.save(path, features)
                # Construct the previous on-disk format independently of the writer.
                spec = {
                    "schema_version": version,
                    "vision_sha256": "a" * 64,
                    "preprocessing": "openpi_jax_float_resize_v1",
                    "camera": "base",
                    "tokens_per_image": 16,
                    "image_dim": 2048,
                    "storage_dtype": "float32",
                    "vision_parameter_dtype": "float32",
                    "vision_compute_dtype": "bfloat16",
                    "pool": "spatial_mean_16x16_to_4x4",
                    **fields,
                }
                entry = {
                    "file": path.name,
                    "frames": 1,
                    "segment_starts": [0],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                manifest = {"dataset_id": "set", "feature_spec": spec, "episodes": {"0": entry}}
                (root / "manifest.json").write_text(json.dumps(manifest))
                self.assertEqual(feature_spec(config, **fields), spec)
                write_episode(root, 1, 2 * features, [0], config=config, dataset_id="set", **fields)
                store = FrameFeatureStore(root, config, dataset_id="set", **fields)
                for episode in (0, 1):
                    ref = MemoryReference(episode_index=episode, frame_index=0)
                    np.testing.assert_array_equal(store.lookup(*ref).features, episode + 1)
                if version == 1:
                    # A worker serialized by the v1 reader has no verification cache.
                    del store._verified
                del store.single_frame  # Both legacy readers serialized no selection override.
                worker = pickle.loads(pickle.dumps(store))
                self.assertFalse(worker._arrays)
                self.assertEqual(worker.single_frame, "oldest")
                np.testing.assert_array_equal(worker.lookup(0, 0).features, 1)
                with self.assertRaisesRegex(ValueError, "schema"):
                    FrameFeatureStore(root, config, **incompatible)
                with self.assertRaisesRegex(ValueError, "different dataset"):
                    FrameFeatureStore(root, config, dataset_id="other", **fields)
                with self.assertRaisesRegex(ValueError, "identity"):
                    write_episode(
                        root, 1, features, [0], config=config, dataset_id="set", **incompatible
                    )
                if fields:
                    with self.assertRaisesRegex(ValueError, "dataset fields"):
                        FrameFeatureStore(root, config, **{**fields, "language_field": "task"})

    def test_partial_column_identity_is_rejected_at_every_entry_point(self):
        config = MemoryConfig(vision_sha256="a" * 64)
        self.assertEqual(
            feature_spec(config, image_field=None, language_field=None),
            feature_spec(config, schema_version=1),
        )
        with self.assertRaisesRegex(ValueError, "Schema v1"):
            feature_spec(config, schema_version=1, image_field="rgb", language_field=None)
        with self.assertRaisesRegex(ValueError, "image_field and language_field"):
            feature_spec(config, schema_version=2, language_field=None)
        with self.assertRaisesRegex(ValueError, "image_field and language_field"):
            feature_spec(config, schema_version=2, image_field="rgb")
        with tempfile.TemporaryDirectory() as directory:
            for field in ("image_field", "language_field"):
                kwargs = {field: "column"}
                with self.assertRaisesRegex(ValueError, "image_field and language_field"):
                    feature_spec(config, **kwargs)
                with self.assertRaisesRegex(ValueError, "image_field and language_field"):
                    FrameFeatureStore(directory, config, **kwargs)
                with self.assertRaisesRegex(ValueError, "image_field and language_field"):
                    write_episode(
                        directory,
                        0,
                        np.ones((1, 16, 2048)),
                        [0],
                        config=config,
                        dataset_id="set",
                        **kwargs,
                    )
            self.assertFalse(list(Path(directory).iterdir()))

    def test_private_mappings_stream_to_disk_and_close_on_eviction_or_close(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            for episode, count in ((0, 96), (1, 1)):
                write_episode(
                    directory, episode, np.full((count, 16, 2048), episode, np.float32),
                    np.zeros(count), config=config, dataset_id="test",
                )
            store = FrameFeatureStore(directory, config, max_cached_episodes=1)
            self.addCleanup(store.close)
            # The 12 MiB episode must stay out of the Python heap even during verification.
            tracing = tracemalloc.is_tracing()
            if not tracing:
                tracemalloc.start()
            try:
                baseline = tracemalloc.get_traced_memory()[0]
                tracemalloc.reset_peak()
                first = store.lookup(0, 0)
                peak = tracemalloc.get_traced_memory()[1] - baseline
            finally:
                if not tracing:
                    tracemalloc.stop()
            self.assertLess(peak, 3 * 1024 * 1024)
            array = store._arrays["0"]
            self.assertIsInstance(array.base, mmap.mmap)
            self.assertFalse(array.flags.writeable)
            second = store.lookup(1, 0)
            self.assertTrue(array.base.closed)
            mapping = store._arrays["1"].base
            store.close()
            self.assertTrue(mapping.closed)
            np.testing.assert_array_equal(first.features, 0)
            np.testing.assert_array_equal(second.features, 1)
            np.testing.assert_array_equal(store.lookup(0, 0).features, first.features)

    def test_failed_episode_write_cleans_partial_npy_and_preserves_published_episode(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        features = np.zeros((1, 16, 2048), np.float32)

        def fail_save(handle, *args, **kwargs):
            handle.write(b"partial npy")
            raise OSError("save failed")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root, 0, features, [0], config=config, dataset_id="s")
            published = set(root.glob("*.npy"))
            manifest = (root / "manifest.json").read_bytes()
            for module, name, failure in (
                (np, "save", fail_save),
                (feature_store, "content_sha256", OSError("hash failed")),
                (os, "replace", OSError("rename failed")),
            ):
                with self.subTest(stage=name), patch.object(module, name, side_effect=failure):
                    with self.assertRaisesRegex(OSError, "failed"):
                        write_episode(root, 0, features + 1, [0], config=config, dataset_id="s")
                self.assertEqual(set(root.glob("*.npy")), published)
                self.assertEqual((root / "manifest.json").read_bytes(), manifest)
                store = FrameFeatureStore(root, config)
                try:
                    np.testing.assert_array_equal(store.lookup(0, 0).features, 0)
                finally:
                    store.close()

    def test_growth_after_cache_reservation_rejects_valid_npy_before_first_scratch_write(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for episode in (0, 1):
                write_episode(root, episode, np.ones((1, 16, 2048)), [0],
                              config=config, dataset_id="s")
            entries = json.loads((root / "manifest.json").read_text())["episodes"]
            path = root / entries["1"]["file"]
            data = path.read_bytes()
            path.write_bytes(data[:128])
            store = FrameFeatureStore(root, config, max_cached_bytes=len(data) + 128)
            self.addCleanup(store.close)
            store.lookup(0, 0)
            snapshot = io.BytesIO()

            def grow_after_reservation():
                # Restore a valid NPY with the correct digest on the same inode.
                path.write_bytes(data)
                return snapshot

            with (
                patch.object(feature_store.tempfile, "TemporaryFile", grow_after_reservation),
                patch.object(snapshot, "write", wraps=snapshot.write) as write,
                self.assertRaisesRegex(ValueError, "file size"),
            ):
                store.lookup(1, 0)
            write.assert_not_called()
            self.assertEqual(list(store._arrays), ["0"])
            self.assertEqual(store._cached_bytes, len(data))
            self.assertTrue(snapshot.closed)

    def test_snapshot_writes_are_bounded_even_if_the_episode_grows_while_copying(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            write_episode(directory, 0, np.ones((1, 16, 2048)), [0], config=config, dataset_id="s")
            store = FrameFeatureStore(directory, config)
            path = Path(directory) / store.manifest["episodes"]["0"]["file"]
            expected = path.stat().st_size

            class CountingSnapshot(io.BytesIO):
                def __init__(self, grow_source):
                    super().__init__()
                    self.grow_source = grow_source
                    self.written = 0

                def write(self, data):
                    if self.grow_source and self.written == 0:
                        os.truncate(path, expected + 4 * 1024 * 1024)
                    self.written += len(data)
                    return super().write(data)

            for late in (False, True):
                os.truncate(path, expected if late else expected + 4 * 1024 * 1024)
                snapshot = CountingSnapshot(late)
                with self.subTest(growing=late), patch.object(
                    feature_store.tempfile, "TemporaryFile", return_value=snapshot
                ), self.assertRaisesRegex(ValueError, "trailing data|file size"):
                    store.lookup(0, 0)
                self.assertEqual(snapshot.written, expected if late else 0)
                self.assertFalse(store._arrays)

    def test_cache_byte_budget_evicts_lru_before_allocating_and_bypasses_large_episodes(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            for episode, frames in enumerate((1, 1, 1, 3)):
                write_episode(
                    directory, episode, np.full((frames, 16, 2048), episode, np.float32),
                    np.zeros(frames), config=config, dataset_id="s",
                )
            size = next(Path(directory).glob("episode_000000_*.npy")).stat().st_size
            budget = 2 * size
            store = FrameFeatureStore(directory, config, max_cached_bytes=budget)
            self.addCleanup(store.close)
            mappings = set()
            allocations = []
            create = tempfile.TemporaryFile

            def retained_bytes():
                return sum(len(m) for m in mappings if not m.closed) + sum(
                    sum(map(len, index.cached.values())) for index in store._indexes.values()
                )

            def allocate():
                # Observe the unlinked files while the next snapshot has yet to be created.
                open_bytes = retained_bytes()
                self.assertEqual(store._cached_bytes, open_bytes)
                allocations.append((tuple(store._arrays), open_bytes))
                return create()

            with patch.object(tempfile, "TemporaryFile", side_effect=allocate):
                for episode in (0, 1, 0, 2, 3, 3, 0):
                    np.testing.assert_array_equal(store.lookup(episode, 0).features, episode)
                    if episode == 3:
                        self.assertNotIn("3", store._arrays)
                    elif str(episode) in store._arrays:
                        mappings.add(store._arrays[str(episode)].base)
                    total = retained_bytes()
                    self.assertEqual(store._cached_bytes, total)
                    self.assertLessEqual(total, budget)
            self.assertEqual(allocations, [
                ((), 0), (("0",), size), (("0",), size),
            ])
            worker = pickle.loads(pickle.dumps(store))
            self.assertEqual(worker.max_cached_bytes, budget)
            self.assertEqual(worker._cached_bytes, 0)
            self.assertFalse(worker._indexes)
            store.close()
            self.assertEqual(store._cached_bytes, 0)
            self.assertFalse(store._indexes)
            self.assertTrue(all(m.closed for m in mappings))

            unlimited = FrameFeatureStore(directory, config, max_cached_bytes=None)
            self.addCleanup(unlimited.close)
            for episode in range(4):
                unlimited.lookup(episode, 0)
            self.assertEqual(len(unlimited._arrays), 4)
            for invalid in (0, -1, True, 2.5):
                with (
                    self.subTest(limit=invalid),
                    self.assertRaisesRegex(ValueError, "max_cached_bytes"),
                ):
                    FrameFeatureStore(directory, config, max_cached_bytes=invalid)

    def test_oversized_npy_header_is_rejected_after_its_length_prefix(self):
        source = io.BytesIO(np.lib.format.magic(2, 0) + (2**30).to_bytes(4, "little") + b"x" * 256)
        snapshot = io.BytesIO()
        with self.assertRaisesRegex(ValueError, "header length"):
            feature_store._episode_digest(source, snapshot, 1)
        self.assertEqual(source.tell(), 12)
        self.assertEqual(snapshot.tell(), 0)

    def test_snapshot_checks_layout_before_copying_and_preserves_fortran_order(self):
        features = np.asfortranarray(np.arange(16 * 2048, dtype=np.float32).reshape(1, 16, 2048))
        for version in ((1, 0), (2, 0)):
            with tempfile.TemporaryFile() as source:
                np.lib.format.write_array(source, features, version=version)
                source.seek(0)
                digest = feature_store._episode_digest(source)
                source.seek(0)
                mapped, index = feature_store._verified_snapshot(
                    source, {"frames": 1, "sha256": digest}, feature_store._episode_digest,
                    feature_store._map_episode, os.fstat(source.fileno()).st_size,
                )
                try:
                    np.testing.assert_array_equal(mapped, features)
                    self.assertTrue(mapped.flags.f_contiguous)
                    self.assertEqual(index.digests, hashlib.sha256(features.tobytes("F")).digest())
                finally:
                    mapped.base.close()
        for invalid in (features[:, :, :4], features.astype(np.float64)):
            with tempfile.TemporaryFile() as source, io.BytesIO() as snapshot:
                np.save(source, invalid)
                source.seek(0)
                with self.assertRaisesRegex(ValueError, "layout"):
                    feature_store._episode_digest(source, snapshot, 1)
                self.assertEqual(snapshot.tell(), 0)

    def test_v1_mapping_rejects_replacements_and_paths_outside_store(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            write_episode(root, 0, np.ones((1, 16, 2048)), [0], config=config, dataset_id="set")
            store = FrameFeatureStore(root, config)
            entry = store.manifest["episodes"]["0"]
            path = root / entry["file"]
            replacement = root / "replacement.npy"
            np.save(replacement, np.zeros((1, 16, 2048), np.float32))
            map_descriptor = feature_store._map_episode
            mappings = []

            def replace_before_map(handle, frames):
                replacement.replace(path)
                result = map_descriptor(handle, frames)
                mappings.append(result.base)
                np.testing.assert_array_equal(result, 1)  # Still the verified descriptor.
                return result

            with patch.object(feature_store, "_map_episode", side_effect=replace_before_map):
                with self.assertRaisesRegex(ValueError, "changed"):
                    store.lookup(0, 0)
            self.assertFalse(store._arrays)
            self.assertFalse(store._verified)
            self.assertTrue(mappings[0].closed)
            with self.assertRaisesRegex(ValueError, "Corrupt"):
                store.lookup(0, 0)
            outside = path.rename(root.parent / "outside.npy")
            (root / "link.npy").symlink_to(outside)
            for filename in ("../outside.npy", "link.npy"):
                entry["file"] = filename
                with self.assertRaisesRegex(ValueError, "inside the feature store"):
                    store.lookup(0, 0)

    def test_buffer_and_failed_batched_act_preserve_last_committed_state(self):
        config = MemoryConfig(budget=16, window_frames=2)
        buffer = HistoryBuffer(config, single_frame="current")
        for step in range(3):
            buffer.append(np.full((16, 2048), step, np.float32), step)
        with self.assertRaisesRegex(ValueError, "finite"):
            buffer.append(np.full((16, 2048), np.nan, np.float32), 3)
        self.assertEqual(buffer.last_step, 2)
        np.testing.assert_array_equal(buffer.snapshot().features, 2)
        np.testing.assert_array_equal(buffer.snapshot().positions, position_embeddings([1])[0])
        self.assertEqual(len(buffer.tokens), 2)
        policy = Mock()
        policy.spec = SimpleNamespace(action_horizon=2)
        policy.predict_chunk.return_value = PolicyOutput(np.array([[[1.0], [2.0]]]))
        memory = MemorySession(
            policy,
            lambda frames: np.ones((len(frames), 16, 2048)),
            config,
            SessionOptions(2, seed=3),
        )
        single = {
            "state": [0.0],
            "images": {"base": np.zeros((2, 4, 3), np.uint8)},
            "prompt": "pick",
            "env_ids": ["a"],
        }
        memory.reset("a", seed=19)
        np.testing.assert_array_equal(memory.act(single), [[1]])
        prepared, options = policy.predict_chunk.call_args.args
        self.assertIsInstance(prepared, MemoryObservationBatch)
        self.assertEqual(options.seed, 19)
        policy.predict_chunk.side_effect = RuntimeError("prediction failed")
        batch = {
            "state": [[0.0], [0.0]],
            "images": {"base": np.zeros((2, 2, 4, 3), np.uint8)},
            "prompt": ["pick", "place"],
            "env_ids": ["a", "b"],
        }
        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            memory.act(batch)
        self.assertEqual(set(memory.buffers), {"a"})
        self.assertEqual(memory.buffers["a"].last_step, 0)
        self.assertEqual(memory.actions["a"].remaining_steps, 1)
        np.testing.assert_array_equal(memory.act(single), [[2]])


@unittest.skipUnless(find_spec("torch"), "requires PyTorch")
class MemoryTorchTest(unittest.TestCase):
    def test_eager_and_injected_operators_match(self):
        import torch
        from torch.nn import functional as F

        from dobot_nexus.modules.memory.torch_layers import (
            MemoryAttention,
            MemoryModulation,
            MemoryTokenEncoder,
        )

        def rope(x, *, positions):
            return x  # Identity rotation isolates the injected operator calls.

        def rms_inverse(x):
            return torch.reciprocal(torch.sqrt(x.float().square().mean(-1, keepdim=True) + 1e-6))

        ops = SimpleNamespace(
            linear=Mock(wraps=F.linear),
            silu=Mock(wraps=F.silu),
            rope=Mock(wraps=rope),
            rope_angles=Mock(side_effect=lambda positions, dim: positions),
            rms_inverse=Mock(wraps=rms_inverse),
            einsum=Mock(wraps=torch.einsum),
            softmax=Mock(wraps=torch.softmax),
            add=Mock(wraps=torch.add),
        )
        with torch.no_grad():
            encoder = MemoryTokenEncoder(torch.float32)
            image, positions = torch.randn(1, 2, 2048), torch.randn(1, 2, 768)
            memory = encoder(image, positions)
            self.assertEqual(memory.shape, (1, 2, 1024))
            torch.testing.assert_close(encoder(image, positions, inference_ops=ops), memory)
            modulation = MemoryModulation(rope)
            self.assertIsInstance(modulation.attention, MemoryAttention)
            x = torch.randn(1, 3, 1024)
            mask = torch.tensor([[True, False]])
            kv = modulation.attention.prepare_memory(memory)
            for layer in (modulation.attention, modulation):
                expected = layer(x, memory, mask)
                actual = layer(x, memory, mask, cached_kv=kv, inference_ops=ops)
                self.assertEqual(actual.shape, x.shape)
                torch.testing.assert_close(actual, expected)
            for operator in vars(ops).values():
                operator.assert_called()
