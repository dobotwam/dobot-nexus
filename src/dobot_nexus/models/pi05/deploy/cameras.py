"""Resolve OpenPI wire cameras against the public names of a Pi0.5 Artifact."""

from __future__ import annotations

from collections.abc import Mapping

from dobot_nexus.models.pi05.data.pipeline import PUBLIC_CAMERA_TO_CORE, dataset_image_fields
from dobot_nexus.modules.serving.openpi import OPENPI_TO_NEXUS_IMAGES

_WIRE_ALIASES = {"head": "top_camera", "left_wrist": "left_camera", "right_wrist": "right_camera"}


def artifact_camera_mapping(policy, camera_mapping=None):
    model = getattr(policy, "model", None)
    # InferencePolicy restores config.data from the Artifact's resolved training recipe.
    config = getattr(policy, "config", getattr(model, "root_config", {}))
    if not isinstance(config, Mapping):
        config = {}
    if camera_mapping is None:
        camera_mapping = config.get("server", {}).get("camera_mapping")
    model_config = getattr(model, "pi05_config", None)
    cameras = getattr(model_config, "camera_mapping", PUBLIC_CAMERA_TO_CORE)
    if len(cameras) != len(PUBLIC_CAMERA_TO_CORE) or set(cameras.values()) != set(
        PUBLIC_CAMERA_TO_CORE.values()
    ):
        raise ValueError("Artifact cameras must populate each Pi0.5 camera slot exactly once.")
    public_names = policy.spec.camera_names
    if model_config is not None and (len(public_names) != len(cameras)
                                     or set(public_names) != set(cameras)):
        raise ValueError("Artifact public cameras must match the model camera slots.")
    if camera_mapping is not None:
        return camera_mapping
    fields = config.get("data", {}).get("field_mapping", {})
    if "images" not in fields and cameras == PUBLIC_CAMERA_TO_CORE:
        return OPENPI_TO_NEXUS_IMAGES
    raw = fields.get("images")
    if isinstance(raw, Mapping) and len(raw) == len(cameras):
        columns = dataset_image_fields(raw)
        if set(columns) == set(cameras.values()):
            wire = {}
            for public, slot in cameras.items():
                name = columns[slot].rsplit(".", 1)[-1]
                wire[_WIRE_ALIASES.get(name, name)] = public
            if wire.keys() == OPENPI_TO_NEXUS_IMAGES.keys():
                return wire
    raise ValueError(
        "Saved dataset image columns do not identify all OpenPI cameras for these model "
        "slots. Set server.camera_mapping explicitly (wire name to public camera name)."
    )
