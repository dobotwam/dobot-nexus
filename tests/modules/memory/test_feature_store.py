from __future__ import annotations

import errno
import fcntl
import json
import os
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, call, patch

import numpy as np

from dobot_nexus.modules.memory import feature_store
from dobot_nexus.modules.memory.feature_store import FrameFeatureStore, write_episode
from dobot_nexus.modules.memory.types import MemoryConfig


class FeatureStorePublicationTest(unittest.TestCase):
    def test_retry_hashes_raw_bytes_of_a_same_size_malformed_npy(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        features = np.ones((1, 16, 2048), np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root, 0, features, [0], config=config, dataset_id="set")
            path, = root.glob("*.npy")
            valid = path.read_bytes()
            self.assertEqual(valid[:8], np.lib.format.magic(1, 0))
            size = int.from_bytes(valid[8:10], "little")
            # An unterminated multiline literal can raise tokenize.TokenError in the NPY parser.
            path.write_bytes(valid[:10] + b"{".ljust(size - 1) + b"\n" + valid[10 + size:])
            with patch.object(feature_store, "_episode_header", side_effect=AssertionError):
                write_episode(root, 0, features, [0], config=config, dataset_id="set")
            self.assertEqual(path.read_bytes(), valid)

    def test_retry_limits_raw_hash_reads_if_existing_blob_grows(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        features = np.ones((9, 16, 2048), np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root, 0, features, [0] * len(features), config=config, dataset_id="set")
            blob, = root.glob("*.npy")
            size = blob.stat().st_size
            open_episode = feature_store._open_episode
            reader = Mock()

            @contextmanager
            def grow_after_open(path):
                with open_episode(path) as (handle, opened):
                    with path.open("ab") as writer:
                        writer.write(b"trailing data")
                    reader.read.side_effect = handle.read
                    reader.fileno.return_value = handle.fileno()
                    yield reader, opened

            with (patch.object(feature_store, "_open_episode", grow_after_open),
                  self.assertRaisesRegex(ValueError, "changed during verification")):
                write_episode(
                    root, 0, features, [0] * len(features), config=config, dataset_id="set"
                )
            sizes = [call.args[0] for call in reader.read.call_args_list]
            self.assertTrue(all(0 < n <= 1024 * 1024 for n in sizes))
            self.assertEqual(sum(sizes), size)

    def test_directory_sync_failures_preserve_the_visible_manifest_and_its_blob(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        features = np.ones((1, 16, 2048), np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root, 0, features, [0], config=config, dataset_id="set")
            original = set(root.glob("*.npy"))
            manifest = root / "manifest.json"
            previous = manifest.read_bytes()
            for fail_at in (2, 3):  # Store creation barrier, then blob and manifest publication.
                root_syncs = []

                def fail_root_sync(path, target=fail_at, events=root_syncs):
                    if path == root.resolve():
                        events.append(path)
                        if len(events) == target:
                            raise OSError("directory sync")

                with (patch.object(feature_store, "_fsync_dir", side_effect=fail_root_sync),
                      self.assertRaisesRegex(OSError, "directory sync")):
                    write_episode(root, 1, features, [0], config=config, dataset_id="set")
                if fail_at == 2:
                    self.assertEqual(manifest.read_bytes(), previous)
                    self.assertEqual(set(root.glob("*.npy")), original)
                else:
                    store = FrameFeatureStore(root, config)
                    self.addCleanup(store.close)
                    np.testing.assert_array_equal(store.lookup(1, 0).features, 1)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "requires POSIX FIFOs")
    def test_lookup_rejects_fifo_without_a_writer(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root, 0, np.ones((1, 16, 2048)), [0], config=config, dataset_id="set")
            store = FrameFeatureStore(root, config)
            self.addCleanup(store.close)
            path = root / store.manifest["episodes"]["0"]["file"]
            path.unlink()
            os.mkfifo(path)

            def stalled(*args):
                raise TimeoutError("Episode lookup blocked on a FIFO.")

            previous = signal.signal(signal.SIGALRM, stalled)
            try:
                signal.alarm(3)
                with self.assertRaisesRegex(ValueError, "regular file"):
                    store.lookup(0, 0)
                manifest = root / "manifest.json"
                manifest.unlink()
                os.mkfifo(manifest)
                with self.assertRaisesRegex(ValueError, "regular file"):
                    FrameFeatureStore(root, config)
                with self.assertRaisesRegex(ValueError, "regular file"):
                    write_episode(root, 1, np.ones((1, 16, 2048)), [0],
                                  config=config, dataset_id="set")
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
            self.assertFalse(store._arrays)

    @unittest.skipUnless(hasattr(os, "O_DIRECTORY"), "requires directory descriptors")
    def test_file_and_directory_sync_order_makes_manifest_references_durable(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        events = []
        fsync, replace = os.fsync, os.replace

        def record_sync(fd):
            events.append(("sync", os.fstat(fd).st_ino))
            fsync(fd)

        def record_replace(src, dst):
            replace(src, dst)
            events.append(("publish", dst.stat().st_ino))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new" / "store"
            features = np.ones((1, 16, 2048))
            with (patch.object(os, "fsync", side_effect=record_sync),
                  patch.object(os, "replace", side_effect=record_replace)):
                write_episode(root, 0, features, [0], config=config, dataset_id="set")
            blob = next(root.glob("*.npy")).stat().st_ino
            manifest = (root / "manifest.json").stat().st_ino
            dir_sync = ("sync", root.stat().st_ino)
            publish_blob, publish_manifest = events.index(("publish", blob)), events.index(
                ("publish", manifest)
            )
            self.assertIn(("sync", blob), events[:publish_blob])
            self.assertIn(("sync", manifest), events[:publish_manifest])
            self.assertIn(dir_sync, events[publish_blob + 1:publish_manifest])
            self.assertIn(dir_sync, events[publish_manifest + 1:])
            for parent in (Path(directory), root.parent):
                self.assertIn(("sync", parent.stat().st_ino), events[:publish_manifest])
            events.clear()
            with (patch.object(os, "fsync", side_effect=record_sync),
                  patch.object(os, "replace", side_effect=record_replace)):
                write_episode(root, 0, features, [0], config=config, dataset_id="set")
            self.assertFalse(any(action == "publish" for action, _ in events))
            self.assertLess(events.index(("sync", blob)), events.index(("sync", manifest)))
            self.assertEqual(events[-1], dir_sync)

    def test_retry_syncs_parent_even_after_a_failed_mkdir_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            root = base / "new" / "store"
            config = MemoryConfig(budget=16, vision_sha256="a" * 64)

            def publish():
                write_episode(root, 0, np.ones((1, 16, 2048)), [0], config=config, dataset_id="set")

            def fail_base(path):
                if path == base:
                    raise OSError(errno.EIO, "parent sync")

            with (patch.object(feature_store, "_fsync_dir", side_effect=fail_base),
                  self.assertRaisesRegex(OSError, "parent sync")):
                publish()
            self.assertTrue(root.is_dir())
            self.assertFalse((root / "manifest.json").exists())
            with patch.object(feature_store, "_fsync_dir", wraps=feature_store._fsync_dir) as sync:
                publish()
            self.assertEqual(sync.call_args_list[:3], [call(root), call(root.parent), call(base)])

    def test_blob_reuse_requires_successful_fsync_of_the_verified_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.npy"
            path.write_bytes(b"verified content")
            digest = feature_store.content_sha256(path)
            original = path.stat()
            modes = []

            def sync(fd):
                self.assertTrue(os.path.samestat(original, os.fstat(fd)))
                mode = fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE
                modes.append(mode)
                if mode == os.O_RDONLY:
                    raise OSError(errno.EBADF, "read-only descriptor")

            with patch.object(os, "fsync", side_effect=sync):
                self.assertTrue(feature_store._matches_episode(path, digest, original.st_size))
            self.assertEqual(modes, [os.O_RDONLY, os.O_RDWR])
            for code in (errno.EBADF, errno.EINVAL, errno.EACCES, errno.EROFS, errno.EIO):
                with (
                    self.subTest(code=code),
                    patch.object(os, "fsync", side_effect=OSError(code, "sync failed")) as attempt,
                    self.assertRaises(OSError),
                ):
                    feature_store._matches_episode(path, digest, original.st_size)
                self.assertEqual(attempt.call_count, 1 if code in (errno.EROFS, errno.EIO) else 2)

            def swap_before_reopen(fd):
                replacement = path.with_suffix(".new")
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
                raise OSError(errno.EBADF, "read-only descriptor")

            with (patch.object(os, "fsync", side_effect=swap_before_reopen) as attempt,
                  self.assertRaisesRegex(ValueError, "changed while reopening")):
                feature_store._matches_episode(path, digest, original.st_size)
            attempt.assert_called_once()

    def test_oversized_episodes_verify_the_entire_file_without_using_scratch(self):
        config = MemoryConfig(budget=48, vision_sha256="a" * 64)
        data = np.arange(19 * 16 * 2048, dtype=np.float32).reshape(19, 16, 2048)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for episode, features in enumerate((data, np.asfortranarray(data))):
                write_episode(root, episode, features, [0] * len(data),
                              config=config, dataset_id="set")
            store = FrameFeatureStore(root, config, max_cached_bytes=1)
            self.addCleanup(store.close)
            read = feature_store._FrameChunks.read

            def swap_after_read(index, handle, indices):
                selected = read(index, handle, indices)
                path = next(p for p in root.glob("*.npy")
                            if os.path.samestat(p.stat(), os.fstat(handle.fileno())))
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
                return selected

            with (patch.object(tempfile, "TemporaryFile", side_effect=AssertionError),
                  patch.object(tempfile, "NamedTemporaryFile", side_effect=AssertionError)):
                for episode in (0, 1):
                    saved = store.lookup(episode, 17)
                    np.testing.assert_array_equal(saved.features.reshape(3, 16, 2048),
                                                  data[[0, 8, 17]])
                    self.assertEqual(store._cached_bytes, 0)
                    self.assertFalse(store._arrays)
                    path = root / store.manifest["episodes"][str(episode)]["file"]
                    with (patch.object(feature_store._FrameChunks, "read", swap_after_read),
                          self.assertRaisesRegex(ValueError, "changed during verification")):
                        store.lookup(episode, 17)
                    with path.open("r+b") as f:
                        f.seek(-4, os.SEEK_END)
                        f.write(np.float32(-10).tobytes())  # An unrequested value in both layouts.
                    with self.assertRaisesRegex(ValueError, "Corrupt memory features"):
                        store.lookup(episode, 17)

    @unittest.skipUnless(hasattr(os, "O_DIRECTORY"), "requires directory descriptors")
    def test_directory_sync_tolerates_unsupported_operations_and_propagates_io_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            for code in (errno.EINVAL, errno.EBADF):
                with patch.object(os, "fsync", side_effect=OSError(code, "unsupported")):
                    feature_store._fsync_dir(directory)
            with (patch.object(os, "fsync", side_effect=OSError(errno.EIO, "I/O error")),
                  self.assertRaises(OSError) as error):
                feature_store._fsync_dir(directory)
            self.assertEqual(error.exception.errno, errno.EIO)

    def test_oversized_episode_reads_only_authenticated_frames_after_the_first_pass(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        frames = np.arange(19 * 16 * 2048, dtype=np.float32).reshape(19, 16, 2048)
        frame_size = frames[0].nbytes
        open_file, pread = feature_store._open_episode, os.pread
        read_sizes = []

        @contextmanager
        def counted_open(path):
            with open_file(path) as (handle, opened):
                def read(n):
                    raw = handle.read(n)
                    read_sizes.append(len(raw))
                    return raw

                def readinto(buffer):
                    size = handle.readinto(buffer)
                    read_sizes.append(size)
                    return size

                reader = Mock(wraps=handle)
                reader.read, reader.readinto = read, readinto
                yield reader, opened

        def counted_pread(fd, n, offset):
            raw = pread(fd, n, offset)
            read_sizes.append(len(raw))
            return raw

        for fortran in (False, True):
            with self.subTest(fortran=fortran), tempfile.TemporaryDirectory() as directory:
                data = np.asfortranarray(frames) if fortran else frames
                write_episode(directory, 0, data, [0] * len(data), config=config, dataset_id="set")
                store = FrameFeatureStore(directory, config, max_cached_bytes=frame_size,
                                          single_frame="current")
                self.addCleanup(store.close)
                path = Path(directory) / store.manifest["episodes"]["0"]["file"]
                size = path.stat().st_size
                header_size = size - frames.nbytes
                with (patch.object(feature_store, "_open_episode", counted_open),
                      patch.object(os, "pread", counted_pread)):
                    for index, expected in ((1, size), (1, 0), (2, frame_size + header_size)):
                        read_sizes.clear()
                        saved = store.lookup(0, index)
                        np.testing.assert_array_equal(saved.features, frames[index])
                        self.assertEqual(sum(read_sizes), expected)
                        self.assertEqual(store._cached_bytes, frame_size)
                    # Frame 1 was evicted. Its digest must reject a bad value on a partial read.
                    with path.open("r+b") as handle:
                        handle.seek(header_size + (4 if fortran else frame_size))
                        handle.write(np.float32(-1).tobytes())
                    read_sizes.clear()
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch for frame 1"):
                        store.lookup(0, 1)
                    self.assertEqual(sum(read_sizes), frame_size + header_size)

    def test_shuffled_revisits_authenticate_frames_without_recreating_episode_snapshots(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        frames = np.arange(4 * 16 * 2048, dtype=np.float32).reshape(4, 16, 2048)
        with tempfile.TemporaryDirectory() as directory:
            for episode in range(3):
                data = frames + episode
                write_episode(directory, episode, np.asfortranarray(data) if episode == 1 else data,
                              [0] * len(data), config=config, dataset_id="set")
            store = FrameFeatureStore(directory, config, max_cached_episodes=1,
                                      max_cached_bytes=2 * frames.nbytes, single_frame="current")
            self.addCleanup(store.close)
            with patch.object(tempfile, "TemporaryFile", wraps=tempfile.TemporaryFile) as allocate:
                for episode in range(3):
                    store.lookup(episode, 0)
                self.assertEqual(allocate.call_count, 3)
                read = feature_store._FrameChunks.read
                with patch.object(feature_store._FrameChunks, "read", autospec=True,
                                  side_effect=read) as partial_read:
                    for episode, frame in ((0, 3), (2, 1), (1, 2), (0, 0)):
                        saved = store.lookup(episode, frame)
                        np.testing.assert_array_equal(saved.features, frames[frame] + episode)
                        np.testing.assert_array_equal(partial_read.call_args.args[-1], [frame])
                    self.assertEqual(partial_read.call_count, 4)
                path = Path(directory) / store.manifest["episodes"]["2"]["file"]
                with path.open("r+b") as handle:
                    handle.seek(-4, os.SEEK_END)
                    handle.write(np.float32(-1).tobytes())
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch for frame 3"):
                    store.lookup(2, 3)
                self.assertEqual(allocate.call_count, 3)

    def test_retry_replaces_same_size_blob_with_invalid_header_or_payload(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        features = np.ones((1, 16, 2048), np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_episode(root, 0, features, [0], config=config, dataset_id="set")
            path, = root.glob("*.npy")
            valid = path.read_bytes()
            for byte in (0, len(valid) - 1):
                with self.subTest(byte=byte):
                    altered = bytearray(valid)
                    altered[byte] ^= 1
                    path.write_bytes(altered)
                    damaged_inode = path.stat().st_ino
                    write_episode(root, 0, features, [0], config=config, dataset_id="set")
                    self.assertNotEqual(path.stat().st_ino, damaged_inode)
                    self.assertEqual(path.read_bytes(), valid)

    def test_manifest_publication_ignores_temp_symlink_and_rolls_back_failed_writes(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        features = np.ones((1, 16, 2048), np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "store"
            root.mkdir()
            target = root.parent / "sentinel"
            target.write_text("untouched")
            legacy_temp = root / ".manifest.tmp"
            legacy_temp.symlink_to(target)
            manifest = root / "manifest.json"
            replace = os.replace
            fsync = os.fsync

            def fail_manifest_fsync(fd):
                inode = os.fstat(fd).st_ino
                if any(p.stat().st_ino == inode for p in root.glob(".manifest.*/*.tmp")):
                    raise OSError("manifest fsync failed")
                fsync(fd)

            def fail_manifest_replace(source, destination):
                if destination == manifest:
                    raise OSError("manifest replace failed")
                replace(source, destination)

            def publish(episode, value):
                write_episode(root, episode, features * value, [0], config=config, dataset_id="set")

            def blobs():
                return {
                    p.name: (p.stat().st_ino, p.stat().st_mode, p.read_bytes())
                    for p in root.glob("*.npy")
                }

            for have_previous in (False, True):
                if have_previous:
                    publish(0, 1)
                    next(root.glob("*.npy")).chmod(0o440)
                    manifest.chmod(0o640)
                previous = manifest.read_bytes() if have_previous else None
                existing_files = blobs()
                failures = [
                    (feature_store.json, "dumps", OSError("serialize failed")),
                    (feature_store.os, "fsync", fail_manifest_fsync),
                ]
                if have_previous:
                    failures.append((feature_store.os, "replace", fail_manifest_replace))
                for module, method, failure in failures:
                    for episode, value in ((1, 2), (0, 2)):
                        with (
                            self.subTest(previous=have_previous, stage=method, episode=episode,
                                         value=value),
                            patch.object(module, method, side_effect=failure),
                            self.assertRaisesRegex(OSError, "failed"),
                        ):
                            publish(episode, value)
                        self.assertEqual(blobs(), existing_files)
                        self.assertEqual(manifest.read_bytes() if manifest.exists() else None,
                                         previous)
                        self.assertEqual(list(root.glob(".manifest.*.tmp")), [])
            self.assertTrue(legacy_temp.is_symlink())
            self.assertEqual(target.read_text(), "untouched")
            original = manifest.stat()
            publish(0, 1)
            self.assertEqual(blobs(), existing_files)
            current = manifest.stat()
            self.assertEqual((current.st_ino, current.st_gid, stat.S_IMODE(current.st_mode)),
                             (original.st_ino, original.st_gid, 0o640))
            publish(0, 2)
            store = FrameFeatureStore(root, config)
            self.addCleanup(store.close)
            np.testing.assert_array_equal(store.lookup(0, 0).features, 2)

    def test_uncertain_manifest_rename_keeps_referenced_or_unverifiable_blob(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        replace = os.replace
        for committed in (False, True):
            with self.subTest(committed=committed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = root / "manifest.json"
                features = np.full((1, 16, 2048), 2, dtype=np.float32)
                if committed:
                    write_episode(root, 0, features, [0], config=config, dataset_id="set")

                def interrupt(src, dst, publish=committed, path=manifest):
                    if dst != path or publish:
                        replace(src, dst)
                    if dst == path:
                        raise KeyboardInterrupt

                with patch.object(os, "replace", interrupt), self.assertRaises(KeyboardInterrupt):
                    write_episode(root, 1, features, [0], config=config, dataset_id="set")
                blob, = root.glob("episode_000001_*.npy")
                if not committed:
                    self.assertFalse(manifest.exists())
                    inode = blob.stat().st_ino
                    write_episode(root, 1, features, [0], config=config, dataset_id="set")
                    self.assertEqual(blob.stat().st_ino, inode)
                store = FrameFeatureStore(root, config)
                try:
                    self.assertEqual(store.manifest["episodes"]["1"]["file"], blob.name)
                    np.testing.assert_array_equal(store.lookup(1, 0).features, features[0])
                finally:
                    store.close()

    def test_semantically_identical_retry_preserves_manifest_formatting_and_inode(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        features = np.zeros((1, 16, 2048))
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            write_episode(directory, 0, features, [0], config=config, dataset_id="set")
            formatted = (" \n" + json.dumps(json.loads(manifest.read_bytes())) + "\n\n").encode()
            manifest.write_bytes(formatted)
            inode = manifest.stat().st_ino
            with patch.object(json, "dumps", side_effect=AssertionError("unexpected dump")):
                write_episode(directory, 0, features, [0], config=config, dataset_id="set")
            self.assertEqual(manifest.stat().st_ino, inode)
            self.assertEqual(manifest.read_bytes(), formatted)

    def test_retry_repairs_noninteger_frames_and_segment_offsets(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            features = np.zeros((1, 16, 2048))
            write_episode(directory, 0, features, [0], config=config, dataset_id="set")
            for field, invalid in (("frames", 1.0), ("frames", True), ("segment_starts", [False])):
                data = json.loads(manifest.read_bytes())
                data["episodes"]["0"][field] = invalid
                manifest.write_text(json.dumps(data))
                old = manifest.stat()
                write_episode(directory, 0, features, [0], config=config, dataset_id="set")
                self.assertFalse(os.path.samestat(old, manifest.stat()))
                entry = json.loads(manifest.read_bytes())["episodes"]["0"]
                self.assertIs(type(entry["frames"]), int)
                self.assertIs(type(entry["segment_starts"][0]), int)

    def test_manifest_is_staged_in_a_private_directory_then_cleaned_up(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            features = np.ones((1, 16, 2048))
            write_episode(root, 0, features, [0], config=config, dataset_id="set")
            manifest.chmod(0o644)
            directories = []
            replace = os.replace

            def check_stage(src, dst):
                if dst == manifest:
                    directories.append(src.parent)
                    self.assertEqual(src.parent.parent, root)
                    self.assertEqual(stat.S_IMODE(src.parent.stat().st_mode), 0o700)
                    self.assertEqual(stat.S_IMODE(src.stat().st_mode), 0o644)
                    if len(directories) == 1:
                        raise OSError("rename failed")
                replace(src, dst)

            with patch.object(os, "replace", check_stage):
                with self.assertRaisesRegex(OSError, "rename failed"):
                    write_episode(root, 1, features, [0], config=config, dataset_id="set")
                self.assertFalse(directories[0].exists())
                write_episode(root, 1, features, [0], config=config, dataset_id="set")
            self.assertEqual(len(directories), 2)
            self.assertFalse(directories[1].exists())

    def test_uncertain_rename_keeps_blob_referenced_by_hardlink(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            alias = root / "alias.npy"
            replace = os.replace

            def publish_alias(src, dst):
                if dst == manifest:
                    live = json.loads(src.read_bytes())
                    entry = live["episodes"]["0"]
                    os.link(root / entry["file"], alias)
                    entry["file"] = alias.name
                    src.write_text(json.dumps(live))
                    replace(src, dst)
                    raise KeyboardInterrupt
                replace(src, dst)

            with patch.object(os, "replace", publish_alias), self.assertRaises(KeyboardInterrupt):
                write_episode(root, 0, np.ones((1, 16, 2048)), [0], config=config, dataset_id="set")
            blob, = root.glob("episode_*.npy")
            self.assertTrue(blob.samefile(alias))
            alias.unlink()
            self.assertFalse(feature_store._manifest_excludes_blob(
                manifest, blob.name, allow_missing=True,
            ))

    def test_changed_manifest_preserves_group_mode_and_access_acl(self):
        config = MemoryConfig(budget=16, vision_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def publish(episode):
                write_episode(root, episode, np.ones((1, 16, 2048)), [0],
                              config=config, dataset_id="set")

            publish(0)
            manifest = root / "manifest.json"
            gid = manifest.stat().st_gid
            other_group = next((group for group in os.getgroups() if group != gid), gid)
            os.chown(manifest, -1, other_group)
            manifest.chmod(0o640)
            if sys.platform == "darwin":
                subprocess.run(["chmod", "+a", "everyone allow read", manifest], check=True)

                def read_acl():
                    return subprocess.check_output(["ls", "-le", manifest]).splitlines()[1:]
            else:
                # Linux POSIX ACL xattr with an additional named user and a read-only mask.
                entries = ((1, 6, 0xffffffff), (2, 4, os.getuid() + 1), (4, 4, 0xffffffff),
                           (16, 4, 0xffffffff), (32, 0, 0xffffffff))
                acl = struct.pack("<I", 2) + b"".join(struct.pack("<HHI", *e) for e in entries)
                try:
                    os.setxattr(manifest, "system.posix_acl_access", acl)
                except OSError as exc:
                    if exc.errno == errno.ENOTSUP:
                        self.skipTest("Filesystem does not support POSIX access ACLs.")
                    raise

                def read_acl():
                    return os.getxattr(manifest, "system.posix_acl_access")

            access = read_acl()
            self.assertTrue(access)
            previous = manifest.stat()
            with (patch.object(os, "fchown", wraps=os.fchown) as chown,
                  patch.object(feature_store, "_copy_access_acl",
                               wraps=feature_store._copy_access_acl) as copy_acl):
                publish(0)
                copy_acl.assert_not_called()
                self.assertEqual(manifest.stat().st_ino, previous.st_ino)
                publish(1)
                copy_acl.assert_called_once()
                if other_group != gid:
                    self.assertEqual(chown.call_args.args[1:], (previous.st_uid, other_group))
            current = manifest.stat()
            self.assertNotEqual(current.st_ino, previous.st_ino)
            self.assertEqual((stat.S_IMODE(current.st_mode), current.st_gid), (0o640, other_group))
            self.assertEqual(read_acl(), access)
            operations = [(feature_store, "_copy_access_acl")]
            if other_group != gid:
                operations.append((os, "fchown"))
            for module, operation in operations:
                with (patch.object(module, operation, side_effect=PermissionError),
                      self.assertRaisesRegex(OSError, "preserve manifest permissions")):
                    publish(2)
                self.assertEqual(manifest.stat().st_ino, current.st_ino)
                self.assertFalse(list(root.glob("episode_000002*")))

    def test_linux_manifest_acl_copies_descriptor_bytes_or_removes_an_inherited_acl(self):
        names = ("system.posix_acl_access", "system.nfs4_acl")
        with (patch.object(sys, "platform", "linux"),
              patch.object(os, "getxattr", create=True, return_value=b"acl") as get,
              patch.object(os, "setxattr", create=True) as set_acl,
              patch.object(os, "removexattr", create=True) as remove):
            feature_store._copy_access_acl(7, 8)
            self.assertEqual(get.call_args_list, [call(7, name) for name in names])
            self.assertEqual(set_acl.call_args_list, [call(8, name, b"acl") for name in names])
            remove.assert_not_called()
            get.side_effect = OSError(errno.ENODATA, "no access ACL")
            feature_store._copy_access_acl(7, 8)
            self.assertEqual(remove.call_args_list, [call(8, name) for name in names])
            remove.side_effect = PermissionError
            with self.assertRaises(PermissionError):
                feature_store._copy_access_acl(7, 8)

    def test_linux_nfs4_acl_is_preserved_when_posix_acls_are_unsupported(self):
        with (patch.object(sys, "platform", "linux"),
              patch.object(os, "getxattr", create=True) as get,
              patch.object(os, "setxattr", create=True) as set_acl):
            unsupported = OSError(errno.ENOTSUP, "POSIX ACLs unsupported")
            get.side_effect = [unsupported, b"nfs4 ACL"]
            feature_store._copy_access_acl(7, 8)
            self.assertEqual(get.call_args_list, [call(7, "system.posix_acl_access"),
                                                 call(7, "system.nfs4_acl")])
            set_acl.assert_called_once_with(8, "system.nfs4_acl", b"nfs4 ACL")
            get.side_effect = [unsupported, PermissionError("read ACL")]
            with self.assertRaisesRegex(PermissionError, "read ACL"):
                feature_store._copy_access_acl(7, 8)
            get.side_effect = [unsupported, b"nfs4 ACL"]
            set_acl.side_effect = PermissionError("write ACL")
            with self.assertRaisesRegex(PermissionError, "write ACL"):
                feature_store._copy_access_acl(7, 8)

    @unittest.skipUnless(sys.platform == "darwin", "requires Darwin ACLs")
    def test_darwin_copies_acl_when_python_omits_the_copyfile_acl_constant(self):
        import posix

        with tempfile.TemporaryDirectory() as directory:
            src, dst = (Path(directory) / name for name in ("src", "dst"))
            with src.open("w+b") as source, dst.open("w+b") as destination:
                subprocess.run(["chmod", "+a", "everyone allow read", src], check=True)
                with patch.dict(vars(posix)):
                    vars(posix).pop("_COPYFILE_ACL", None)
                    feature_store._copy_access_acl(source.fileno(), destination.fileno())
            acl = subprocess.check_output(["ls", "-le", src]).splitlines()[1:]
            self.assertTrue(acl)
            self.assertEqual(subprocess.check_output(["ls", "-le", dst]).splitlines()[1:], acl)

    def test_manifest_owner_and_group_are_set_before_restoring_full_mode(self):
        source = Mock(st_uid=1001, st_gid=2001, st_mode=0o102640)
        temp = Mock(st_uid=1000, st_gid=2000, st_mode=0o100600)
        operations = Mock()
        with (patch.object(os, "fstat", side_effect=[source, temp, source]),
              patch.object(os, "fchown", operations.fchown),
              patch.object(os, "fchmod", operations.fchmod),
              patch.object(feature_store, "_copy_access_acl", operations.acl)):
            feature_store._preserve_manifest_access(7, 8)
        self.assertEqual(operations.mock_calls, [
            call.fchown(8, 1001, 2001), call.fchmod(8, 0o2640), call.acl(7, 8),
        ])
