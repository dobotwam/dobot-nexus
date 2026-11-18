"""Robot layout negotiation for the existing Nexus TCP/WebSocket bridge.

This module validates and reorders transport data only. It never imports a
hardware driver or converts one robot's kinematics into another robot's layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return value


def _names(value, label):
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(x, str) or not x for x in value)
    ):
        raise ValueError(f"{label} must contain nonempty field names")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicate field names")
    return tuple(value)


def _array(value, label):
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise ValueError(f"{label} must contain real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        data = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError(f"{label} contains non-finite values")
    return data


@dataclass(frozen=True)
class RobotLayout:
    name: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    camera_map: dict[str, str]

    @classmethod
    def from_dict(cls, data):
        if data.get("schema_version") != 1:
            raise ValueError("Robot layout requires schema_version=1")
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Robot layout requires a name")
        states = _names(data.get("state_names"), "state_names")
        actions = _names(data.get("action_names"), "action_names")
        cameras = data.get("camera_map")
        if (
            not isinstance(cameras, dict)
            or not cameras
            or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v
                for k, v in cameras.items()
            )
        ):
            raise ValueError("camera_map must map policy image keys to robot image keys")
        for key, names in [("state_dim", states), ("action_dim", actions)]:
            if key in data and _positive_int(data[key], key) != len(names):
                raise ValueError(f"{key} does not match its field-name list")
        return cls(name, states, actions, dict(cameras))

    def to_schema(self):
        return {
            "schema_version": 1,
            "name": self.name,
            "state_dim": len(self.state_names),
            "action_dim": len(self.action_names),
            "state_names": list(self.state_names),
            "action_names": list(self.action_names),
            "image_names": sorted(set(self.camera_map.values())),
        }


def load_robot_layout(value="dual6"):
    if value in ("dual6", "atom18"):
        path = Path(__file__).resolve().parents[1] / "layouts" / (value + ".json")
    else:
        path = Path(value).expanduser()
    return RobotLayout.from_dict(json.loads(path.read_text()))


def _metadata_dimension(metadata, field_name, schema_value=None, fallback=None):
    spec = metadata.get("policy_spec", {})
    if not isinstance(spec, dict):
        raise ValueError("policy_spec must be an object")
    candidates = []
    for source, obj in [("policy_spec", spec), ("metadata", metadata)]:
        if field_name in obj:
            candidates.append((source, _positive_int(obj[field_name], source + "." + field_name)))
    if schema_value is not None:
        candidates.append(("wire schema", _positive_int(schema_value, field_name)))
    if not candidates:
        if fallback is None:
            return None
        return fallback
    if len({v for _, v in candidates}) != 1:
        raise ValueError(f"Conflicting {field_name} declarations: {candidates}")
    return candidates[0][1]


@dataclass
class PolicyRobotContract:
    layout: RobotLayout
    state_dim: int
    action_dim: int
    action_horizon: int | None
    image_shapes: dict[str, tuple[int, int, int]]
    state_indices: tuple[int, ...] = field(default=())
    action_indices: tuple[int, ...] = field(default=())
    robot_state_names: tuple[str, ...] = field(default=())

    @classmethod
    def from_metadata(cls, metadata, layout):
        inputs = metadata.get("input_schema", {})
        outputs = metadata.get("output_schema", {})
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise ValueError("Policy input_schema and output_schema must be objects")
        state_shape = inputs.get("state")
        action_shape = outputs.get("actions")
        if state_shape is not None and (
            not isinstance(state_shape, (list, tuple)) or len(state_shape) != 1
        ):
            raise ValueError("input_schema.state must have shape [state_dim]")
        if action_shape is not None and (
            not isinstance(action_shape, (list, tuple)) or len(action_shape) != 2
        ):
            raise ValueError("output_schema.actions must have shape [horizon, action_dim]")
        # Explicit layout preserves legacy services that omit dimension metadata.
        state_dim = _metadata_dimension(
            metadata, "state_dim", state_shape[0] if state_shape else None, len(layout.state_names)
        )
        action_dim = _metadata_dimension(
            metadata,
            "action_dim",
            action_shape[1] if action_shape else None,
            len(layout.action_names),
        )
        horizon = _metadata_dimension(
            metadata, "action_horizon", action_shape[0] if action_shape else None
        )
        if (state_dim, action_dim) != (len(layout.state_names), len(layout.action_names)):
            raise ValueError(
                f"Policy expects state/action={state_dim}/{action_dim}; "
                f"robot layout {layout.name!r} declares "
                f"{len(layout.state_names)}/{len(layout.action_names)}"
            )
        declared_images = inputs.get("images")
        if declared_images is not None and (
            not isinstance(declared_images, dict) or set(declared_images) != set(layout.camera_map)
        ):
            raise ValueError("Policy image keys do not match robot layout camera_map")
        size = metadata.get("policy_spec", {}).get("image_size", [224, 224])
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("Policy image_size must be [height, width]")
        height, width = (_positive_int(v, "image_size") for v in size)
        shapes = {}
        for name in layout.camera_map:
            shape = (declared_images or {}).get(name, [3, height, width])
            if not isinstance(shape, (list, tuple)) or len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"Policy image {name!r} must declare CHW RGB shape")
            shapes[name] = tuple(_positive_int(v, "image shape") for v in shape)
        return cls(
            layout,
            state_dim,
            action_dim,
            horizon,
            shapes,
            tuple(range(state_dim)),
            tuple(range(action_dim)),
            layout.state_names,
        )

    def validate_execute_steps(self, count):
        _positive_int(count, "actions_per_response")
        if self.action_horizon is not None and count > self.action_horizon:
            raise ValueError(
                f"actions_per_response={count} exceeds policy horizon={self.action_horizon}"
            )

    def bind_robot_hello(self, hello):
        schema = hello.get("robot_schema")
        if schema is None:
            if self.layout != load_robot_layout("dual6") or (self.state_dim, self.action_dim) != (
                14,
                14,
            ):
                raise ValueError(
                    "Custom robot layout requires robot hello.robot_schema with dimensions "
                    "and field names; the legacy robot server owns a 14-D layout"
                )
            return
        if not isinstance(schema, dict) or schema.get("schema_version") != 1:
            raise ValueError("Unsupported robot hello.robot_schema")
        states = _names(schema.get("state_names"), "robot state_names")
        actions = _names(schema.get("action_names"), "robot action_names")
        if (
            _positive_int(schema.get("state_dim"), "robot state_dim"),
            _positive_int(schema.get("action_dim"), "robot action_dim"),
        ) != (self.state_dim, self.action_dim):
            raise ValueError("Robot handshake dimensions do not match the policy/layout")
        if (
            len(states) != self.state_dim
            or len(actions) != self.action_dim
            or set(states) != set(self.layout.state_names)
            or set(actions) != set(self.layout.action_names)
        ):
            raise ValueError(
                "Robot handshake field names do not match the configured policy layout"
            )
        images = schema.get("image_names")
        if images is not None and not set(self.layout.camera_map.values()).issubset(
            _names(images, "robot image_names")
        ):
            raise ValueError("Robot handshake is missing required camera sources")
        self.state_indices = tuple(states.index(name) for name in self.layout.state_names)
        self.action_indices = tuple(self.layout.action_names.index(name) for name in actions)
        self.robot_state_names = states

    def warmup_observation(self, task):
        return {
            "state": np.zeros(self.state_dim, dtype=np.float32),
            "images": {
                name: np.zeros(shape, dtype=np.uint8) for name, shape in self.image_shapes.items()
            },
            "prompt": task,
        }

    def observation(self, message, task):
        state = _array(message["state"], "state")
        if state.shape != (self.state_dim,):
            raise ValueError(f"Expected state shape ({self.state_dim},), got {state.shape}")
        if (
            "state_names" in message
            and _names(message["state_names"], "observation state_names") != self.robot_state_names
        ):
            raise ValueError("Observation field order changed after robot handshake")
        images = {}
        for policy_name, robot_name in self.layout.camera_map.items():
            if robot_name not in message["images"]:
                raise ValueError("Missing robot image: " + robot_name)
            image = np.asarray(message["images"][robot_name])
            if image.dtype != np.uint8:
                raise ValueError(f"Image {robot_name!r} must use uint8 RGB")
            expected = self.image_shapes[policy_name]
            if image.shape == (expected[1], expected[2], 3):
                image = image.transpose(2, 0, 1)
            if image.shape != expected:
                raise ValueError(
                    f"Image {robot_name!r} must match policy shape {expected}, got {image.shape}"
                )
            images[policy_name] = np.ascontiguousarray(image)
        return {
            "state": np.ascontiguousarray(state[list(self.state_indices)]),
            "images": images,
            "prompt": task,
        }

    def actions(self, value, count):
        self.validate_execute_steps(count)
        actions = _array(value, "actions")
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected actions [H,{self.action_dim}], got {actions.shape}")
        if self.action_horizon is not None and actions.shape[0] != self.action_horizon:
            raise ValueError(
                f"Policy returned horizon {actions.shape[0]}, expected {self.action_horizon}"
            )
        if actions.shape[0] < count:
            raise ValueError("Policy returned fewer steps than actions_per_response")
        return np.ascontiguousarray(actions[:count, list(self.action_indices)])
