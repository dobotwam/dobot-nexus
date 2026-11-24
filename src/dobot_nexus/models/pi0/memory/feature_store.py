"""Model-specific feature-store schema and FrameSample selection defaults."""

from __future__ import annotations

from dobot_nexus.modules.memory import feature_store as _shared

from .sampling import select_frames as select_frames

OrderedDict = _shared.OrderedDict
Path = _shared.Path
_episode_digest = _shared._episode_digest
_map_episode = _shared._map_episode
_file_identity = _shared._file_identity
assemble_memory = _shared.assemble_memory
content_sha256 = _shared.content_sha256
hashlib = _shared.hashlib
json = _shared.json
np = _shared.np
os = _shared.os
tempfile = _shared.tempfile


def feature_spec(config, *, image_field, language_field):
    return _shared.feature_spec(
        config, image_field=image_field, language_field=language_field, schema_version=2
    )


def write_episode(
    root, episode, features, task_ids, *, config, dataset_id, image_field, language_field
):
    _shared.write_episode(
        root,
        episode,
        features,
        task_ids,
        config=config,
        dataset_id=dataset_id,
        image_field=image_field,
        language_field=language_field,
        schema_version=2,
    )


class FrameFeatureStore(_shared.FrameFeatureStore):
    _default_single_frame = "current"

    def __init__(
        self,
        root,
        config,
        *,
        image_field,
        language_field,
        dataset_id=None,
        max_cached_episodes=8,
        max_cached_bytes=_shared.DEFAULT_MAX_CACHED_BYTES,
        single_frame=_default_single_frame,
    ):
        super().__init__(
            root,
            config,
            image_field=image_field,
            language_field=language_field,
            dataset_id=dataset_id,
            max_cached_episodes=max_cached_episodes,
            max_cached_bytes=max_cached_bytes,
            schema_version=2,
            single_frame=single_frame,
        )

    def _read_episode(self, handle, entry, reserved_bytes):
        # Preserve the legacy verification hooks at this module's import path.
        return _shared._verified_snapshot(
            handle, entry, _episode_digest, _map_episode, reserved_bytes
        )
