"""Immutable episode identity and the LRU of verified private snapshots."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from dobot_nexus.models.pi0.memory import feature_store
from dobot_nexus.models.pi0.memory.configuration import MemoryConfig
from dobot_nexus.models.pi0.memory.feature_store import FrameFeatureStore, write_episode


class FeatureStoreTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "features"
        self.config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        self.fields = dict(dataset_id="set", image_field="camera.top", language_field="task")
        for episode in range(2):
            features = np.full((1, 16, 2048), episode, np.float32)
            write_episode(self.root, episode, features, [0], config=self.config, **self.fields)

    def store(self):
        return FrameFeatureStore(self.root, self.config, **self.fields, max_cached_episodes=1)

    def test_eviction_reuses_frame_digests_and_replacement_requires_full_verification(self):
        store = self.store()
        with mock.patch.object(
            feature_store, "_episode_digest", wraps=feature_store._episode_digest
        ) as digest:
            for episode in (0, 1, 0):
                result = store.lookup(episode, 0)
                self.assertTrue((result.features == episode).all())
            self.assertEqual(digest.call_count, 2)
            path = self.root / store.manifest["episodes"]["0"]["file"]
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            store.lookup(0, 0)
            self.assertEqual(digest.call_count, 3)
            with replacement.open("wb") as handle:
                np.save(handle, np.full((1, 16, 2048), -1, np.float32))
            replacement.replace(path)
            with self.assertRaisesRegex(ValueError, "Corrupt memory features"):
                store.lookup(0, 0)
            self.assertEqual(digest.call_count, 4)

    def test_private_snapshot_survives_in_place_write_and_eviction_rechecks_bytes(self):
        store = self.store()
        store.lookup(0, 0)
        snapshot = store._arrays["0"]
        self.assertFalse(snapshot.flags.writeable)
        path = self.root / store.manifest["episodes"]["0"]["file"]
        original = path.stat()
        with path.open("r+b") as handle:
            handle.seek(-4, os.SEEK_END)
            handle.write(np.float32(7.0).tobytes())
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        np.testing.assert_array_equal(snapshot, 0)
        store.lookup(1, 0)
        self.assertNotIn("0", store._verified)
        with self.assertRaisesRegex(ValueError, "Corrupt memory features"):
            store.lookup(0, 0)

    def test_in_place_modification_after_eviction_is_rehashed(self):
        store = self.store()
        store.lookup(0, 0)
        store.lookup(1, 0)
        path = self.root / store.manifest["episodes"]["0"]["file"]
        before = path.stat()
        with path.open("r+b") as handle:
            handle.seek(-4, os.SEEK_END)
            handle.write(np.float32(99).tobytes())
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "Corrupt memory features"):
            store.lookup(0, 0)
        self.assertFalse(store._arrays)
        self.assertFalse(store._verified)

    def test_previous_mapping_is_closed_before_allocating_a_new_snapshot(self):
        store = self.store()
        self.addCleanup(store.close)
        first = store.lookup(0, 0)
        mapping = store._arrays["0"].base
        create = tempfile.TemporaryFile

        def check_slot(*args, **kwargs):
            self.assertTrue(mapping.closed)
            self.assertFalse(store._arrays)
            self.assertFalse(store._verified)
            return create(*args, **kwargs)

        with mock.patch.object(tempfile, "TemporaryFile", side_effect=check_slot) as allocate:
            second = store.lookup(1, 0)
            allocate.assert_called_once()
        np.testing.assert_array_equal(first.features, 0)
        np.testing.assert_array_equal(second.features, 1)
        self.assertEqual(list(store._arrays), ["1"])

    def test_manifest_file_must_resolve_inside_root(self):
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["episodes"]["0"]
        outside = self.root.parent / "outside.npy"
        (self.root / entry["file"]).replace(outside)
        (self.root / "linked.npy").symlink_to(outside)
        for name in ("../outside.npy", "linked.npy", str(outside), "."):
            with self.subTest(name=name):
                entry["file"] = name
                manifest_path.write_text(json.dumps(manifest))
                store = self.store()
                with (
                    mock.patch.object(feature_store, "_episode_digest") as digest,
                    self.assertRaisesRegex(ValueError, "inside the feature store"),
                ):
                    store.lookup(0, 0)
                digest.assert_not_called()

    def test_replacement_during_verification_or_mapping_is_never_published(self):
        for operation in ("_episode_digest", "_map_episode"):
            with self.subTest(operation=operation):
                store = self.store()
                path = self.root / store.manifest["episodes"]["0"]["file"]
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(path.read_bytes())
                original = getattr(feature_store, operation)

                def replace_after(*args, original=original, replacement=replacement, path=path,
                                  **kwargs):
                    result = original(*args, **kwargs)
                    replacement.replace(path)
                    return result

                with mock.patch.object(feature_store, operation, side_effect=replace_after):
                    with self.assertRaisesRegex(ValueError, "changed"):
                        store.lookup(0, 0)
                self.assertFalse(store._arrays)
                self.assertFalse(store._verified)
                self.assertTrue((store.lookup(0, 0).features == 0).all())
