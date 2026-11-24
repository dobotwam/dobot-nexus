"""Model-specific feature-store schema and FrameSample selection defaults."""

from __future__ import annotations

from collections.abc import Mapping

from dobot_nexus.modules.memory import feature_store as _shared

from .sampling import select_frames as select_frames

OrderedDict = _shared.OrderedDict
Path = _shared.Path
assemble_memory = _shared.assemble_memory
content_sha256 = _shared.content_sha256
feature_spec = _shared.feature_spec
write_episode = _shared.write_episode
hashlib = _shared.hashlib
json = _shared.json
np = _shared.np
os = _shared.os
tempfile = _shared.tempfile


def history_image_field(pipeline, camera):
    """Return the dataset column selected for the Artifact's history camera."""
    cameras = pipeline.model_config.camera_mapping
    if camera not in cameras:
        raise ValueError(f"Unknown memory camera: {camera}")
    return pipeline.image_fields[cameras[camera]]


class FrameFeatureStore(_shared.FrameFeatureStore):
    _default_single_frame = "start"
    _legacy_error = (
        "Pi0.5 schema v1 has no producer column provenance. Please rebuild the feature store "
        "or confirm its original image_field and language_field in data.memory_legacy_columns "
        "(confirmed_legacy_columns for direct readers). "
        "Both columns must match the requested recipe."
    )

    def __init__(self, root, config, *, confirmed_legacy_columns=None, **kwargs):
        kwargs.setdefault("allow_legacy_v1", True)
        super().__init__(root, config, **kwargs)
        if self.manifest["feature_spec"]["schema_version"] == 1:
            columns = confirmed_legacy_columns
            if (
                not isinstance(columns, Mapping)
                or columns.keys() != {"image_field", "language_field"}
                or not isinstance(columns["image_field"], str)
                or not columns["image_field"]
                or not isinstance(columns["language_field"], (str, type(None)))
                or any(key in kwargs and kwargs[key] != value for key, value in columns.items())
            ):
                raise ValueError(self._legacy_error)
            self._confirmed_legacy_columns = dict(columns)

    def __setstate__(self, state):
        if (state["manifest"]["feature_spec"]["schema_version"] == 1
                and not state.get("_confirmed_legacy_columns")):
            raise ValueError(self._legacy_error)
        super().__setstate__(state)
