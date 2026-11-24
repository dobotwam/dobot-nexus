"""Content-addressed frozen features, with atomic per-episode publication."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import mmap
import os
import stat
import sys
import tempfile
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .sampling import (
    DEFAULT_HISTORY_SINGLE_FRAME,
    SingleFrame,
    assemble_memory,
    select_frames,
    validate_single_frame,
)
from .types import MemoryConfig, MemorySnapshot

_UNSET = object()
_MAX_HEADER_BYTES = 64 * 1024
DEFAULT_MAX_CACHED_BYTES = 512 * 1024 * 1024


def _positive_byte_limit(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def configured_cache_bytes(data):
    return _positive_byte_limit(
        data.get("memory_cache_bytes", DEFAULT_MAX_CACHED_BYTES), "data.memory_cache_bytes"
    )


def content_sha256(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Memory asset not found: {path}")
    digest = hashlib.sha256()
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not paths:
        raise ValueError(f"Memory asset is empty: {path}")
    for item in paths:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode() + b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def feature_spec(
    config: MemoryConfig,
    *,
    image_field: str | None = None,
    language_field=_UNSET,
    schema_version: int | None = None,
) -> dict[str, Any]:
    """Fields identifying the encoder and, in schema v2, its dataset columns."""
    has_columns = image_field is not None or (
        language_field is not _UNSET and language_field is not None
    )
    if schema_version is None:
        schema_version = 2 if has_columns else 1
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError(f"Unsupported feature store schema_version: {schema_version!r}.")
    if schema_version == 1:
        if has_columns:
            raise ValueError("Schema v1 does not store image_field or language_field.")
    elif not isinstance(image_field, str) or not (
        language_field is None or isinstance(language_field, str)
    ):
        raise ValueError(
            "Schema v2 requires explicit image_field and language_field "
            "(a camera column and a language column or None)."
        )
    spec = {
        "schema_version": schema_version,
        "vision_sha256": config.vision_sha256,
        "preprocessing": config.preprocessing,
        "camera": config.camera,
        "tokens_per_image": 16,
        "image_dim": 2048,
        "storage_dtype": "float32",
        "vision_parameter_dtype": "float32",
        "vision_compute_dtype": "bfloat16",
        "pool": "spatial_mean_16x16_to_4x4",
    }
    if schema_version == 2:
        spec.update(image_field=image_field, language_field=language_field)
    return spec


def write_episode(
    root: str | Path,
    episode: int,
    features: ArrayLike,
    task_ids: ArrayLike,
    *,
    config: MemoryConfig,
    dataset_id: str,
    image_field: str | None = None,
    language_field=_UNSET,
    schema_version: int | None = None,
) -> None:
    """A new dataset/encoder identity requires a different store directory."""
    identity = {
        "dataset_id": dataset_id,
        "feature_spec": feature_spec(
            config,
            image_field=image_field,
            language_field=language_field,
            schema_version=schema_version,
        ),
    }
    root = Path(root)
    features = np.asarray(features, np.float32)
    tasks = np.asarray(task_ids, np.int64)
    if (
        features.shape != (len(tasks), 16, 2048)
        or not len(tasks)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Episode features/tasks must be finite aligned nonempty arrays.")
    if not dataset_id or not config.vision_sha256:
        raise ValueError("Feature stores require dataset and vision content identities.")
    from dobot_nexus.data.cache import file_lock

    _mkdir_durable(root)
    with file_lock(root / ".write.lock"), ExitStack() as stack:
        manifest_path = root / "manifest.json"
        try:
            manifest_file, _ = stack.enter_context(_open_regular_file(manifest_path))
        except FileNotFoundError:
            manifest_file = None
        manifest = json.load(manifest_file) if manifest_file is not None else {
            **identity, "episodes": {},
        }
        if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in identity.items()):
            raise ValueError(
                "Stale feature store identity; build a new store "
                "instead of relabeling old features."
            )
        tmp = tempfile.NamedTemporaryFile(dir=root, suffix=".npy", delete=False)
        temporary = Path(tmp.name)
        try:
            with tmp:
                np.save(tmp, features, allow_pickle=False)
                tmp.flush()
                os.fsync(tmp.fileno())
            digest = content_sha256(temporary)
            filename = f"episode_{int(episode):06d}_{digest}.npy"
            entry = {
                "file": filename,
                "frames": len(tasks),
                "sha256": digest,
                "segment_starts": [0] + (np.flatnonzero(tasks[1:] != tasks[:-1]) + 1).tolist(),
            }
            key = str(int(episode))
            staged = None
            if not _same_episode_entry(manifest["episodes"].get(key), entry):
                updated = {**manifest, "episodes": {**manifest["episodes"], key: entry}}
                payload = (json.dumps(updated, sort_keys=True, indent=2) + "\n").encode("utf-8")
                staged = stack.enter_context(
                    _staged_manifest(manifest_path, payload, manifest_file)
                )
            target = root / filename
            existing = os.path.lexists(target)
            rename_attempted = False
            try:
                if not existing or not _matches_episode(target, digest, temporary.stat().st_size):
                    os.replace(temporary, target)
                _fsync_dir(root)
                if staged is None:
                    _fsync_readonly(manifest_path, manifest_file.fileno())
                else:
                    rename_attempted = True
                    os.replace(staged, manifest_path)
            except BaseException:
                allow_missing = not rename_attempted and manifest_file is None
                if not existing and not temporary.exists() and _manifest_excludes_blob(
                    manifest_path, filename, allow_missing=allow_missing
                ):
                    target.unlink(missing_ok=True)
                    _fsync_dir(root)
                raise
            _fsync_dir(root)
        finally:
            temporary.unlink(missing_ok=True)


def _same_episode_entry(previous, entry):
    # JSON numbers compare equal across int, float and bool in Python.
    return (previous == entry and all(type(previous[k]) is type(v) for k, v in entry.items())
            and all(type(step) is int for step in previous["segment_starts"]))


def _mkdir_durable(directory):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for ancestor in (directory, *directory.parents):
        try:
            _fsync_dir(ancestor)
        except OSError as exc:
            if ancestor != directory and exc.errno in (errno.EACCES, errno.EPERM):
                break
            raise


def _fsync_dir(directory):
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.EBADF, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise


def _fsync_readonly(path, fd):
    try:
        os.fsync(fd)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.EBADF, errno.EACCES):
            raise
        with _open_regular_file(path, os.O_RDWR) as (writable, opened):
            if not os.path.samestat(opened, os.fstat(fd)):
                raise ValueError(f"File changed while reopening it for fsync: {path}") from exc
            os.fsync(writable.fileno())


def _manifest_excludes_blob(path, filename, *, allow_missing):
    """An interrupted rename may already be visible. Only discard a provably unused blob."""
    try:
        with _open_regular_file(path) as (handle, _):
            episodes = json.load(handle)["episodes"]
    except FileNotFoundError:
        return allow_missing
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False
    try:
        target = (path.parent / filename).stat()
        return isinstance(episodes, dict) and all(
            not os.path.samestat(target, (path.parent / entry["file"]).stat())
            for entry in episodes.values()
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _copy_access_acl(source, destination):
    if sys.platform == "darwin":
        # CPython's descriptor-based binding to fcopyfile(3) also clears an inherited ACL.
        import posix

        copyfile = getattr(posix, "_fcopyfile", None)
        if copyfile is None:
            raise OSError("Cannot preserve the manifest ACL without Darwin fcopyfile support.")
        copyfile(source, destination, getattr(posix, "_COPYFILE_ACL", 1))
        return
    if not sys.platform.startswith("linux"):
        raise OSError(f"Cannot preserve manifest ACLs on {sys.platform}.")
    for name in ("system.posix_acl_access", "system.nfs4_acl"):
        try:
            acl = os.getxattr(source, name)
        except OSError as exc:
            if exc.errno == errno.ENOTSUP:
                continue
            if exc.errno != errno.ENODATA:
                raise
            try:
                os.removexattr(destination, name)  # Clear an inherited ACL.
            except OSError as remove_error:
                if remove_error.errno not in (errno.ENODATA, errno.ENOTSUP):
                    raise
        else:
            os.setxattr(destination, name, acl)


def _preserve_manifest_access(source, destination):
    existing, tmp = os.fstat(source), os.fstat(destination)
    owner = (existing.st_uid, existing.st_gid)
    mode = stat.S_IMODE(existing.st_mode)
    if (tmp.st_uid, tmp.st_gid) != owner:
        os.fchown(destination, *owner)
    os.fchmod(destination, mode)
    _copy_access_acl(source, destination)
    actual = os.fstat(destination)
    if (actual.st_uid, actual.st_gid) != owner or stat.S_IMODE(actual.st_mode) != mode:
        raise PermissionError("Manifest owner, group or mode differs after copying permissions.")


@contextmanager
def _staged_manifest(manifest_path, payload, previous):
    with tempfile.TemporaryDirectory(dir=manifest_path.parent, prefix=".manifest.") as directory:
        os.chmod(directory, 0o700)
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".tmp", delete=False) as tmp:
            tmp.write(payload)
            tmp.flush()
            if previous is not None:
                try:
                    _preserve_manifest_access(previous.fileno(), tmp.fileno())
                except OSError as exc:
                    raise OSError(
                        f"Cannot preserve manifest permissions, group or ACL: {exc}"
                    ) from exc
            os.fsync(tmp.fileno())
        yield Path(tmp.name)


def _file_identity(path, stat):
    return path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _episode_header(handle, frames):
    magic = handle.read(8)
    if magic == np.lib.format.magic(1, 0):
        width, parse = 2, np.lib.format.read_array_header_1_0
    elif magic == np.lib.format.magic(2, 0):
        width, parse = 4, np.lib.format.read_array_header_2_0
    else:
        raise ValueError("Corrupt memory features: invalid or unsupported NPY magic.")
    length = handle.read(width)
    size = int.from_bytes(length, "little")
    if len(length) != width or not 0 < size <= _MAX_HEADER_BYTES:
        raise ValueError("Corrupt memory features: invalid NPY header length.")
    # Parse only these bounded bytes. NumPy otherwise reads the declared header
    # length from the descriptor before it applies its own header-size check.
    header = handle.read(size)
    if len(header) != size:
        raise ValueError("Corrupt memory features: truncated NPY header.")
    shape, fortran, dtype = parse(io.BytesIO(length + header), max_header_size=_MAX_HEADER_BYTES)
    if type(frames) is not int or frames <= 0 or shape != (frames, 16, 2048) or dtype != np.float32:
        raise ValueError("Feature array layout disagrees with its manifest.")
    return magic + length + header, shape, fortran, dtype


def _episode_chunks(handle, layout, reserved_bytes):
    header, (frames, tokens, dim), _, dtype = layout
    remaining = frames * tokens * dim * dtype.itemsize
    expected_bytes = len(header) + remaining
    if reserved_bytes is not None and reserved_bytes != expected_bytes:
        raise ValueError("Corrupt memory features: NPY layout disagrees with reserved file size.")
    if os.fstat(handle.fileno()).st_size != expected_bytes:
        raise ValueError("Corrupt memory features: file size disagrees with the NPY layout.")
    yield header
    buffer = memoryview(bytearray(min(remaining, 1024 * 1024)))
    while remaining:
        block = buffer[:remaining]
        if handle.readinto(block) != len(block):
            raise ValueError("Corrupt memory features: truncated NPY payload.")
        remaining -= len(block)
        yield block
    if handle.read(1):
        raise ValueError("Corrupt memory features: trailing data after NPY payload.")


def _episode_digest(handle, destination=None, frames=None, reserved_bytes=None, *, chunks=None):
    digest = hashlib.sha256()
    if chunks is None:
        chunks = (
            iter(lambda: handle.read(1024 * 1024), b"") if frames is None
            else _episode_chunks(handle, _episode_header(handle, frames), reserved_bytes)
        )
    for chunk in chunks:
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
    return digest.hexdigest()


@contextmanager
def _open_regular_file(path, access=os.O_RDONLY):
    flags = access | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"Feature store path must be a regular file: {path}")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            yield handle, opened
    finally:
        os.close(fd)


_open_episode = _open_regular_file


def _matches_episode(path, expected_digest, expected_bytes):
    previous = path.lstat()
    if not stat.S_ISREG(previous.st_mode) or previous.st_size != expected_bytes:
        return False
    identity = _file_identity(path, previous)
    with _open_episode(path) as (handle, opened):
        if _file_identity(path, opened) != identity:
            raise ValueError("Episode changed while opening it for reuse.")
        digest = hashlib.sha256()
        remaining = expected_bytes
        while remaining:
            chunk = handle.read(min(remaining, 1024 * 1024))
            if not chunk:
                break
            digest.update(chunk)
            remaining -= len(chunk)
        matches = remaining == 0 and digest.hexdigest() == expected_digest
        if matches:
            _fsync_readonly(path, handle.fileno())
        if (_file_identity(path, os.fstat(handle.fileno())) != identity
                or _file_identity(path, path.lstat()) != identity):
            raise ValueError("Episode changed during verification for reuse.")
        return matches


def _map_episode(handle, frames):
    """Interpret the verified private NPY snapshot without reopening the episode path."""
    header, shape, fortran, dtype = _episode_header(handle, frames)
    mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    return np.ndarray(shape, dtype, mapping, len(header), order="F" if fortran else "C")


def _verified_snapshot(handle, entry, digest, decode, reserved_bytes):
    # Only the verified private file is mapped. Its pages can be reclaimed by the OS.
    with tempfile.TemporaryFile() as snapshot:
        index, _ = _stream_selected_frames(
            handle, entry, reserved_bytes, (), destination=snapshot, digest=digest
        )
        snapshot.seek(0)
        return decode(snapshot, entry["frames"]), index


_FRAME_BYTES = 16 * 2048 * np.dtype(np.float32).itemsize


@dataclass
class _FrameChunks:
    """One digest per logical frame, including strided frames in Fortran-order NPY files."""

    header: bytes
    frames: int
    fortran: bool
    digests: bytes
    cached: OrderedDict = field(default_factory=OrderedDict)

    def read(self, handle, indices):
        missing = set(indices).difference(self.cached)
        result = {i: self.cached[i] for i in indices if i not in missing}
        if missing and handle.read(len(self.header)) != self.header:
            raise ValueError("Corrupt memory features: NPY header changed.")
        for i in missing:
            if self.fortran:
                # Each feature column contains all frames. pread avoids reading ahead into
                # unrelated frames and reads from the same descriptor verified by the caller.
                start = len(self.header) + 4 * i
                raw = b"".join(os.pread(handle.fileno(), 4, start + column * self.frames * 4)
                               for column in range(16 * 2048))
            else:
                handle.seek(len(self.header) + i * _FRAME_BYTES)
                raw = handle.read(_FRAME_BYTES)
            if hashlib.sha256(raw).digest() != self.digests[i * 32:(i + 1) * 32]:
                raise ValueError(f"Corrupt memory features: SHA-256 mismatch for frame {i}.")
            result[i] = raw
        return result


def _stream_selected_frames(handle, entry, reserved_bytes, indices, *,
                            destination=None, digest=_episode_digest):
    """Authenticate the frame index and requested bytes together on the first scan."""
    layout = _episode_header(handle, entry["frames"])
    _, (frames, _, _), fortran, dtype = layout
    chunks = _episode_chunks(handle, layout, reserved_bytes)
    header = next(chunks)  # Validate the reserved size before allocating per-frame hashers.
    selected = {i: bytearray() for i in indices}
    hashes = bytearray()
    # In C order each frame is contiguous. In Fortran order its digest spans multiple blocks.
    hashers = [hashlib.sha256() for _ in range(frames)] if fortran else []

    def indexed_chunks():
        yield header
        offset = 0
        for block in chunks:
            if fortran:
                values = np.frombuffer(block, dtype)
                for j in range(min(frames, len(values))):
                    frame = (offset // dtype.itemsize + j) % frames
                    fragment = values[j::frames].tobytes()
                    hashers[frame].update(fragment)
                    if frame in selected:
                        selected[frame].extend(fragment)
            else:
                for j in range(0, len(block), _FRAME_BYTES):
                    raw = block[j:j + _FRAME_BYTES]
                    hashes.extend(hashlib.sha256(raw).digest())
                    frame = (offset + j) // _FRAME_BYTES
                    if frame in selected:
                        selected[frame].extend(raw)
            offset += len(block)
            yield block

    if digest(handle, destination, chunks=indexed_chunks()) != entry["sha256"]:
        raise ValueError(f"Corrupt memory features: {entry.get('file', 'SHA-256 mismatch')}")
    digests = b"".join(h.digest() for h in hashers) if fortran else bytes(hashes)
    index = _FrameChunks(header, frames, fortran, digests)
    return index, {i: bytes(v) for i, v in selected.items()}


def _check_episode_identity(handle, location, identity):
    if (_file_identity(identity[0], os.fstat(handle.fileno())) != identity
            or _file_identity(location.resolve(), location.stat()) != identity):
        raise ValueError("Episode feature file changed during verification or mapping.")


class FrameFeatureStore:
    """Cache verified snapshots by LRU and stream selected frames for oversized episodes."""

    _default_single_frame = DEFAULT_HISTORY_SINGLE_FRAME

    def __init__(
        self,
        root: str | Path,
        config: MemoryConfig,
        *,
        image_field: str | None = None,
        language_field=_UNSET,
        schema_version: int | None = None,
        allow_legacy_v1: bool = False,
        single_frame: SingleFrame | None = None,
        dataset_id: str | None = None,
        max_cached_episodes: int = 8,
        max_cached_bytes: int | None = DEFAULT_MAX_CACHED_BYTES,
    ) -> None:
        if type(max_cached_episodes) is not int or max_cached_episodes <= 0:
            raise ValueError("max_cached_episodes must be a positive integer.")
        self.max_cached_episodes = max_cached_episodes
        if max_cached_bytes is not None:
            _positive_byte_limit(max_cached_bytes, "max_cached_bytes")
        self.max_cached_bytes = max_cached_bytes
        self.root, self.config = Path(root).resolve(), config
        expected = feature_spec(
            config,
            image_field=image_field,
            language_field=language_field,
            schema_version=schema_version,
        )
        with _open_regular_file(self.root / "manifest.json") as (handle, _):
            self.manifest = json.load(handle)
        accepted = [expected]
        if allow_legacy_v1 and schema_version is None and expected["schema_version"] == 2:
            accepted.append(feature_spec(config, schema_version=1))
        actual = self.manifest.get("feature_spec") if isinstance(self.manifest, dict) else None
        if actual not in accepted:
            version = actual.get("schema_version") if isinstance(actual, dict) else None
            raise ValueError(
                f"Feature store (schema v{version}) does not match the requested "
                f"schema v{expected['schema_version']}, encoder, preprocessing or dataset fields. "
                "Please rebuild the feature store."
            )
        if dataset_id is not None and self.manifest.get("dataset_id") != dataset_id:
            raise ValueError("Feature store belongs to a different dataset.")
        self.single_frame = (
            self._default_single_frame
            if single_frame is None
            else validate_single_frame(single_frame)
        )
        self._arrays = OrderedDict()
        self._cached_bytes = 0
        self._verified = OrderedDict()
        self._indexes = {}
        self._evicted_indexes = OrderedDict()
        self._lock = RLock()

    def __getstate__(self):
        # Workers verify their own private snapshots on first access.
        state = dict(self.__dict__, _arrays={}, _verified={}, _indexes={}, _evicted_indexes={},
                     _cached_bytes=0)
        del state["_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Older v1 workers serialized a relative root and no verification cache.
        self.root = self.root.resolve()
        self.single_frame = state.get("single_frame", self._default_single_frame)
        self.max_cached_bytes = state.get("max_cached_bytes", DEFAULT_MAX_CACHED_BYTES)
        self._arrays = OrderedDict()
        self._cached_bytes = 0
        self._verified = OrderedDict()
        self._indexes = {}
        self._evicted_indexes = OrderedDict()
        self._lock = RLock()

    def close(self) -> None:
        """Release cached mappings. The next lookup verifies the episode again."""
        with self._lock:
            for array in self._arrays.values():
                array.base.close()
            self._arrays.clear()
            self._verified.clear()
            self._indexes.clear()
            self._evicted_indexes.clear()
            self._cached_bytes = 0

    def _read_episode(self, handle, entry, reserved_bytes):
        return _verified_snapshot(handle, entry, _episode_digest, _map_episode, reserved_bytes)

    def _evict(self, key):
        if key in self._arrays:
            mapping = self._arrays.pop(key).base
            self._cached_bytes -= len(mapping)
            mapping.close()
        index = self._indexes.pop(key)
        self._clear_frames(index)
        identity = self._verified.pop(key)
        # Retain only the header and 32 bytes per frame in this independent metadata LRU.
        self._evicted_indexes[key] = identity, index
        self._evicted_indexes.move_to_end(key)
        if len(self._evicted_indexes) > 1024:
            self._evicted_indexes.popitem(last=False)

    def _clear_frames(self, index):
        self._cached_bytes -= sum(map(len, index.cached.values()))
        index.cached.clear()

    def _reserve_bytes(self, size):
        if self.max_cached_bytes is None or self._cached_bytes + size <= self.max_cached_bytes:
            return
        # Evict payloads in episode and then frame LRU order.
        for key in list(self._verified):
            if self._cached_bytes + size <= self.max_cached_bytes:
                break
            if key in self._arrays:
                self._evict(key)
            else:
                cache = self._indexes[key].cached
                while cache and self._cached_bytes + size > self.max_cached_bytes:
                    self._cached_bytes -= len(cache.popitem(last=False)[1])

    def _indexed_frames(self, key, entry, handle, location, identity, indices, index):
        if index is None:
            index, selected = _stream_selected_frames(handle, entry, identity[3], indices)
        else:
            selected = index.read(handle, indices)
        _check_episode_identity(handle, location, identity)
        self._evicted_indexes.pop(key, None)
        self._indexes[key] = index
        self._verified[key] = identity
        self._verified.move_to_end(key)
        if self.max_cached_bytes is None or _FRAME_BYTES <= self.max_cached_bytes:
            for i in indices:
                if i not in index.cached:
                    self._reserve_bytes(_FRAME_BYTES)
                    index.cached[i] = selected[i]
                    self._cached_bytes += _FRAME_BYTES
                index.cached.move_to_end(i)
        order = "F" if index.fortran else "C"
        return np.stack([
            np.frombuffer(selected[i], np.float32).reshape(16, 2048, order=order) for i in indices
        ])

    def _frames(self, key, entry, indices):
        location = self.root / entry["file"]
        path = location.resolve()
        if self.root not in path.parents:
            raise ValueError("Episode feature file must be inside the feature store.")
        identity = _file_identity(path, path.stat())
        if key in self._arrays and self._verified[key] == identity:
            self._verified.move_to_end(key)
            return self._arrays[key][indices]
        with _open_episode(path) as (handle, opened):
            if _file_identity(path, opened) != identity:
                raise ValueError("Episode feature file changed while opening.")
            previous = self._verified.get(key)
            if previous is not None and previous != identity:
                self._evict(key)
            index = self._indexes.get(key)
            if key in self._evicted_indexes:
                previous, saved_index = self._evicted_indexes[key]
                if previous[:-1] == identity[:-1]:  # Same file and size. Recheck selected bytes.
                    index = saved_index
                else:
                    del self._evicted_indexes[key]
            if key not in self._verified and len(self._verified) >= self.max_cached_episodes:
                self._evict(next(iter(self._verified)))
            oversized = self.max_cached_bytes is not None and opened.st_size > self.max_cached_bytes
            if index is not None or oversized:
                return self._indexed_frames(key, entry, handle, location, identity, indices, index)
            self._reserve_bytes(opened.st_size)
            array, index = self._read_episode(handle, entry, opened.st_size)
            try:
                _check_episode_identity(handle, location, identity)
            except BaseException:
                array.base.close()
                raise
        self._verified[key] = identity
        self._indexes[key] = index
        self._arrays[key] = array
        self._cached_bytes += len(array.base)
        return array[indices]

    def lookup(self, episode: int, frame: int) -> MemorySnapshot:
        key = str(int(episode))
        entry = self.manifest["episodes"][key]
        frame = int(frame)
        if not 0 <= frame < entry["frames"]:
            raise IndexError("History frame is outside the episode.")
        starts = entry["segment_starts"]
        start = starts[np.searchsorted(starts, frame, side="right") - 1]
        if self.config.window_frames is None and frame - start >= self.config.max_frames:
            raise ValueError(
                "Global history capacity exceeded; training and runtime must share the same limit."
            )
        indices, positions = select_frames(
            frame,
            self.config.sampled_frames,
            segment_start=start,
            window_frames=self.config.window_frames,
            single_frame=self.single_frame,
        )
        with self._lock:
            # Advanced indexing copies the frames before another lookup can evict the mapping.
            features = self._frames(key, entry, indices)
        return assemble_memory(features, positions, self.config)
