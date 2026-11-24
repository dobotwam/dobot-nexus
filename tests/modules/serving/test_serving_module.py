from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from dobot_nexus.core.contracts import ObservationBatch, PolicyOutput, PolicySpec
from dobot_nexus.modules.memory.types import MemoryConfig
from dobot_nexus.modules.serving.memory_ws import (
    MemoryConnection,
    MemoryWebsocketServer,
    configured_memory_limits,
)
from dobot_nexus.modules.serving.openpi import (
    OPENPI_PROTOCOL_NAME,
    NexusOpenPiPolicy,
    build_nexus_observation,
    build_openpi_metadata,
)
from tests.modules import model_imports

MAPPING = {"overhead": "tool"}
SPEC = PolicySpec("stub", "cpu", ("tool",), 4, 2, 3, image_size=(2, 5))


class StubPolicy:
    spec = SPEC
    metadata = SPEC.to_dict()
    artifact_dir = None
    manifest = {"model": {"operator_revision": "fixture"}}

    def __init__(self):
        self.calls = []

    def predict_chunk(self, observation, options):
        observation = ObservationBatch.from_value(observation)
        self.calls.append((observation, options))
        actions = np.full((observation.batch_size, 3, 2), options.seed, np.float32)
        return PolicyOutput(actions, timing={"model_ms": 1.0})


def wire_observation():
    return {
        "images": {"overhead": np.arange(30, dtype=np.uint8).reshape(3, 2, 5)},
        "state": np.zeros(4, np.float32),
    }


class ServingModuleTest(unittest.TestCase):
    def test_model_import_boundary(self):
        self.assertEqual(model_imports("serving"), [])

    def test_every_adapter_rejects_empty_or_duplicate_camera_mapping(self):
        policy = StubPolicy()
        memory_args = (policy, np.stack, MemoryConfig(camera="tool"))
        for mapping in ({}, {"first": "tool", "second": "tool"}):
            for factory, args, kwargs in (
                (build_nexus_observation, (wire_observation(),), {}),
                (build_openpi_metadata, (policy,), {}),
                (NexusOpenPiPolicy, (policy,), {}),
                (MemoryConnection, memory_args, dict(execute_steps=1)),
                (
                    MemoryWebsocketServer,
                    memory_args,
                    dict(execute_steps=1, num_steps=1, host="localhost", port=0),
                ),
            ):
                with self.subTest(mapping=mapping, factory=factory.__name__):
                    with self.assertRaisesRegex(ValueError, "camera_mapping"):
                        factory(*args, **kwargs, camera_mapping=mapping)

    def test_swapped_wire_and_native_cameras_require_explicit_protocol(self):
        policy = StubPolicy()
        policy.spec = PolicySpec("stub", "cpu", ("a", "b"), 4, 2, 3, image_size=(2, 5))
        server = MemoryWebsocketServer(
            policy,
            lambda frames: np.ones((len(frames), 16, 2048)),
            MemoryConfig(camera="a"),
            execute_steps=1,
            num_steps=1,
            seed=5,
            camera_mapping={"a": "b", "b": "a"},
            host="localhost",
            port=0,
            warmup=True,
        )
        server.prepare()
        self.assertEqual(len(policy.calls), 1)
        with server.open_connection(None) as infer:
            images = {
                key: np.full((2, 5, 3), value, np.uint8) for key, value in (("a", 1), ("b", 2))
            }
            request = {"images": images, "prompt": "pick", "state": np.zeros(4)}
            for state in (np.zeros(4), np.zeros((1, 4))):
                with self.assertRaisesRegex(ValueError, "Ambiguous camera names"):
                    infer({**request, "state": state})
            self.assertEqual(len(policy.calls), 1)
            self.assertEqual(infer({**request, "protocol": "nexus"})["actions"].shape, (1, 3, 2))
            np.testing.assert_array_equal(policy.calls[-1][0].images["a"], 1)
            self.assertEqual(infer({**request, "protocol": "openpi"})["actions"].shape, (3, 2))
            np.testing.assert_array_equal(policy.calls[-1][0].images["a"], 2)

    def test_memory_setup_validates_destination_names_and_history_camera(self):
        policy = StubPolicy()
        for factory, kwargs in (
            (MemoryConnection, {}),
            (MemoryWebsocketServer, {"num_steps": 1, "host": "localhost", "port": 0}),
        ):
            for mapping, camera, error in (
                ({"overhead": "toool"}, "tool", "camera_mapping destinations"),
                ({"overhead": "tool", "side": "other"}, "tool", "camera_mapping destinations"),
                (MAPPING, "missing", "History camera"),
            ):
                with self.subTest(factory=factory, mapping=mapping, camera=camera):
                    with self.assertRaisesRegex(ValueError, error):
                        factory(
                            policy, np.stack, MemoryConfig(camera=camera), execute_steps=1,
                            camera_mapping=mapping, **kwargs,
                        )

    def test_declared_protocol_checks_camera_names_before_using_session(self):
        policy = StubPolicy()
        encoder = Mock(side_effect=lambda frames: np.zeros((len(frames), 16, 2048), np.float32))
        connection = MemoryConnection(
            policy, encoder, MemoryConfig(camera="tool", budget=16), execute_steps=1,
            camera_mapping=MAPPING, default_prompt="load", seed=5,
        )
        wire = wire_observation()
        native = {**wire, "images": {"tool": wire["images"]["overhead"].transpose(1, 2, 0)}}
        for act in (True, False):
            for protocol, observation in (
                ("nexus", wire), ("openpi", native), (OPENPI_PROTOCOL_NAME, native),
                ("nexus", {**native, "images": {}}),
                ("openpi", {**wire, "images": {**wire["images"], **native["images"]}}),
                ("typo", wire), (None, native),
            ):
                with self.subTest(protocol=protocol, act=act):
                    with self.assertRaisesRegex(ValueError, "(?i)protocol"):
                        connection.infer({**observation, "protocol": protocol, "act": act})
        self.assertFalse(policy.calls)
        self.assertFalse(connection.session.buffers)
        encoder.assert_not_called()
        for protocol, observation, shape in (
            ("nexus", native, (1, 3, 2)), (OPENPI_PROTOCOL_NAME, wire, (3, 2)),
        ):
            response = connection.infer({**observation, "protocol": protocol})
            self.assertEqual(response["actions"].shape, shape)

    def test_metadata_and_adapter_reject_unknown_destination_camera(self):
        for factory in (build_openpi_metadata, NexusOpenPiPolicy):
            with (
                self.subTest(factory=factory),
                self.assertRaisesRegex(ValueError, "Artifact cameras"),
            ):
                factory(StubPolicy(), camera_mapping={"wire": "typo"})

    def test_custom_camera_metadata_and_replay_seed(self):
        policy = StubPolicy()
        wire = wire_observation()
        converted = build_nexus_observation(
            wire,
            image_size=SPEC.image_size,
            state_dim=4,
            default_prompt="load",
            camera_mapping=MAPPING,
        )
        np.testing.assert_array_equal(
            converted["images"]["tool"], wire["images"]["overhead"].transpose(1, 2, 0)
        )
        metadata = build_openpi_metadata(policy, camera_mapping=MAPPING)
        self.assertEqual(metadata["protocol"], OPENPI_PROTOCOL_NAME)
        self.assertEqual(metadata["operator_revision"], "fixture")
        self.assertEqual(metadata["policy_spec"], SPEC.to_dict())
        self.assertEqual(metadata["input_schema"]["images"], {"overhead": [3, 2, 5]})
        self.assertEqual(metadata["output_schema"]["actions"], [3, 2])
        with self.assertRaisesRegex(ValueError, "Artifact cameras"):
            NexusOpenPiPolicy(policy)
        adapter = NexusOpenPiPolicy(policy, camera_mapping=MAPPING, seed=5, default_prompt="load")
        for extra, expected in [({}, 5), ({"replay_seed": 15}, 15), ({}, 6)]:
            response = adapter.infer({**wire, **extra})
            self.assertEqual(response["actions"].shape, (3, 2))
            np.testing.assert_array_equal(response["actions"], expected)
            self.assertEqual(policy.calls[-1][0].prompt, ["load"])
            self.assertEqual(policy.calls[-1][1].seed, expected)
        with self.assertRaisesRegex(ValueError, "replay_seed"):
            adapter.infer({**wire, "replay_seed": True})

    def test_server_single_frame_current_selects_latest_observation(self):
        policy = StubPolicy()

        def encode(frames):
            return np.stack([np.full((16, 2048), frame.mean(), np.float32) for frame in frames])

        server = MemoryWebsocketServer(
            policy, encode, MemoryConfig(camera="tool", budget=16), single_frame="current",
            camera_mapping=MAPPING, execute_steps=1, num_steps=1, host="localhost", port=0, seed=1,
        )
        with server.open_connection(None) as infer:
            for step in (0, 1):
                image = np.full((3, 2, 5), step, np.uint8)
                infer({**wire_observation(), "images": {"overhead": image},
                       "step_index": step, "prompt": "pick"})
        np.testing.assert_array_equal(policy.calls[-1][0].memory.features, 1)

    def test_memory_connections_encode_dense_frames_and_isolate_state(self):
        policy = StubPolicy()
        limits = configured_memory_limits(SimpleNamespace(server={"memory_max_environments": 2}))

        def encode(frames):
            return np.stack([np.full((16, 2048), frame.mean(), np.float32) for frame in frames])

        server = MemoryWebsocketServer(
            policy,
            encode,
            MemoryConfig(camera="tool", budget=32),
            camera_mapping=MAPPING,
            execute_steps=2,
            num_steps=4,
            host="localhost",
            port=0,
            seed=5,
            default_prompt="load",
            **limits,
        )
        self.assertEqual(limits, {"max_environments": 2, "max_history_bytes": 256 * 1024**2})
        self.assertEqual(server.prepare()["history_camera"], "tool")
        wire = wire_observation()
        with server.open_connection(None) as first, server.open_connection(None) as second:
            self.assertEqual(
                first(
                    {
                        "add_buffer": True,
                        "history_wire_camera": "overhead",
                        "env_id": "arm",
                        "step_indices": [7],
                        "history_frames": [np.full((2, 5, 3), 77, np.uint8)],
                    }
                ),
                {"add_buffer_finished": True},
            )
            result = first({**wire, "env_id": "arm", "step_index": 8})
            self.assertEqual(result["actions"].shape, (3, 2))
            history, options = policy.calls[-1]
            self.assertEqual(options.num_steps, 4)
            np.testing.assert_array_equal(history.memory.features[0, :16], 77)
            self.assertTrue(history.memory.mask.all())
            native = {"images": {"tool": np.zeros((1, 2, 5, 3), np.uint8)}, "state": [[0.0] * 4]}
            self.assertEqual(first({**native, "env_id": "arm"})["actions"].shape, (1, 3, 2))
            self.assertEqual(
                first({"reset": True, "env_id": "arm", "seed": 18}), {"reset_finished": True}
            )
            step = first({**wire, "env_id": "arm", "act": True})
            self.assertEqual(step["actions"].shape, (2,))
            np.testing.assert_array_equal(step["actions"], 18)
            calls = len(policy.calls)
            first({**wire, "env_id": "arm", "act": True})
            self.assertEqual(len(policy.calls), calls)  # Consume the buffered action.
            self.assertEqual(second(wire)["seed"], 5)
            np.testing.assert_array_equal(policy.calls[-1][0].memory.mask.sum(), 16)
            with self.assertRaisesRegex(ValueError, "camera names"):
                second({**wire, "images": {**wire["images"], **native["images"]}})

    def test_history_upload_acknowledges_the_advertised_wire_camera(self):
        for camera, wire, legacy in (
            ("base", "top_camera", True),
            ("base", "left_camera", False),
            ("front", "top_camera", False),
        ):
            with self.subTest(camera=camera, wire=wire):
                policy = StubPolicy()
                policy.spec = PolicySpec("stub", "cpu", (camera,), 4, 2, 3, image_size=(2, 5))
                policy.model = SimpleNamespace(inference_metadata={"history_wire_camera": "old"})
                encoder = Mock(return_value=np.full((1, 16, 2048), 77, np.float32))
                server = MemoryWebsocketServer(
                    policy, encoder, MemoryConfig(camera=camera, budget=16), execute_steps=1,
                    num_steps=1, host="localhost", port=0, seed=5, camera_mapping={wire: camera},
                )
                metadata = server.prepare()
                self.assertEqual(metadata["history_camera"], camera)
                self.assertEqual(metadata["history_wire_camera"], wire)
                frame = np.full((2, 5, 3), 77, np.uint8)
                upload = {"add_buffer": True, "history_frames": [frame],
                          "step_indices": [7], "prompt": "pick"}
                with server.open_connection(None) as infer:
                    with self.assertRaisesRegex(ValueError, "history_wire_camera"):
                        infer({**upload, "history_wire_camera": "right_camera"})
                    if not legacy:
                        with self.assertRaisesRegex(ValueError, "history_wire_camera"):
                            infer(upload)
                        upload["history_wire_camera"] = metadata["history_wire_camera"]
                    encoder.assert_not_called()
                    self.assertEqual(infer(upload), {"add_buffer_finished": True})
                    infer({"images": {wire: np.zeros_like(frame)}, "state": np.zeros(4),
                           "step_index": 7, "prompt": "pick"})
                    encoder.assert_called_once()
                    np.testing.assert_array_equal(policy.calls[-1][0].memory.features[0], 77)
