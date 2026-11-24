"""Camera slot routing at the programmatic and CLI Pi0.5 serving entry points."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import yaml

from dobot_nexus.core.contracts import ObservationBatch
from dobot_nexus.models.pi05.data.pipeline import Pi05Pipeline
from dobot_nexus.models.pi05.deploy import memory_server, server
from dobot_nexus.models.pi05.memory.configuration import MemoryConfig
from examples.real_robot.dobot_dual_arm_14d._runtime.memory import HISTORY_PROTOCOL, MemoryBridge
from tests.models.pi05.test_deploy import FakePolicy, openpi_observation


class CameraRoutingTest(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.policy.model = SimpleNamespace(pi05_config=SimpleNamespace(camera_mapping={
            "base": "left_wrist_0_rgb",
            "left_wrist": "base_0_rgb",
            "right_wrist": "right_wrist_0_rgb",
        }))
        self.wire = {
            "top_camera": "left_wrist", "left_camera": "base", "right_camera": "right_wrist",
        }
        self.columns = {
            "base": "observation.images.top_camera", "left_wrist": "observation.images.left_camera",
            "right_wrist": "observation.images.right_camera",
        }
        self.policy.config = {"task": "train", "data": {"field_mapping": {"images": self.columns}}}
        self.observation = openpi_observation()

    def test_kbd_pick_full34_recipe_resolves_head_and_wrist_columns(self):
        recipes = Path(__file__).resolve().parents[3] / "examples/recipes"
        recipe = recipes / "pi05_kbd_pick_0916_full34.yaml"
        self.policy.config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        del self.policy.model.pi05_config.camera_mapping  # The recipe uses the default slots.
        adapter = server.NexusOpenPiPolicy(self.policy)
        self.assertEqual(adapter.camera_mapping, {
            "top_camera": "base", "left_camera": "left_wrist", "right_camera": "right_wrist",
        })

    def test_dataset_columns_and_public_slots_route_identical_training_and_serving_pixels(self):
        self.columns.update(base="observation.images.right_camera",
                            left_wrist="observation.images.top_camera",
                            right_wrist="observation.images.left_camera")
        model = self.policy.model.pi05_config
        model.state_dim = model.action_dim = model.model_action_dim = 14
        model.action_horizon, model.max_token_len = 1, 200
        pipeline = Pi05Pipeline(self.policy.config, model)
        frame = {
            "observation.state": self.observation["state"], "action": np.zeros((1, 14)),
            "task": "pick",
            **{f"observation.images.{wire}": image.transpose(1, 2, 0)
               for wire, image in self.observation["images"].items()},
        }
        adapter = server.NexusOpenPiPolicy(self.policy)
        adapter.infer(self.observation)
        observation = ObservationBatch.from_value(self.policy.calls[-1][0])
        with (patch.object(pipeline, "normalize", side_effect=lambda value, _: value),
              patch.object(pipeline, "tokenize_prompt", return_value=([1], [True]))):
            training = pipeline.transform_frame(frame)["image"]
            serving = pipeline.transform_runtime_numpy(observation)[0]["image"]
        for slot, wire in (("base_0_rgb", "right_camera"), ("left_wrist_0_rgb", "top_camera"),
                           ("right_wrist_0_rgb", "left_camera")):
            np.testing.assert_array_equal(training[slot], frame[f"observation.images.{wire}"])
            np.testing.assert_array_equal(serving[slot][0], training[slot])

    def test_ambiguous_dataset_columns_require_an_override_with_permuted_slots(self):
        for column in ("custom", self.columns["right_wrist"]):
            with patch.dict(self.columns, base=column):
                with self.assertRaisesRegex(ValueError, "server.camera_mapping"):
                    server.NexusOpenPiPolicy(self.policy)
                with patch.dict(self.policy.config, server={"camera_mapping": self.wire}):
                    adapter = server.NexusOpenPiPolicy(self.policy)
                    self.assertEqual(adapter.camera_mapping, self.wire)
        self.policy.config = {}
        with self.assertRaisesRegex(ValueError, "server.camera_mapping"):
            server.build_openpi_metadata(self.policy)
        adapter = server.NexusOpenPiPolicy(self.policy, camera_mapping=self.wire)
        self.assertEqual(adapter.camera_mapping, self.wire)

    def test_ambiguous_saved_columns_require_override_even_with_default_slots(self):
        del self.policy.model.pi05_config.camera_mapping  # Use the default model slots.
        fields = self.policy.config["data"]["field_mapping"]
        for images in ({"base": "observation.images.left_camera", "left_wrist": "custom",
                        "right_wrist": "observation.images.right_camera"}, {}, None):
            with patch.dict(fields, images=images):
                with self.assertRaisesRegex(ValueError, "server.camera_mapping"):
                    server.NexusOpenPiPolicy(self.policy)
                with patch.dict(self.policy.config, server={"camera_mapping": self.wire}):
                    self.assertEqual(server.NexusOpenPiPolicy(self.policy).camera_mapping,
                                     self.wire)
        del fields["images"]
        self.assertEqual(server.NexusOpenPiPolicy(self.policy).camera_mapping,
                         server.OPENPI_TO_NEXUS_IMAGES)

    def test_memory_constructors_route_swapped_slots_and_advertise_history_camera(self):
        self.policy.metadata = {}
        memory = MemoryConfig(camera="base", budget=16)
        encode = Mock(return_value=np.zeros((1, 16, 2048), np.float32))
        connection = memory_server.MemoryConnection(
            self.policy, encode, memory, execute_steps=1, seed=4,
        )
        websocket = memory_server.MemoryWebsocketServer(
            self.policy, encode, memory, execute_steps=1, host="localhost", port=0, num_steps=1,
            seed=4,
        )
        self.assertEqual(websocket.metadata["history_wire_camera"], "left_camera")
        with websocket.open_connection(None) as serve:
            for infer in (connection.infer, serve):
                infer(self.observation)
                np.testing.assert_array_equal(encode.call_args.args[0][0], 1)
                batch = self.policy.calls[-1][0]
                for wire, public in self.wire.items():
                    expected = self.observation["images"][wire].transpose(1, 2, 0)
                    np.testing.assert_array_equal(batch.images[public][0], expected)

    def test_explicit_wire_mapping_cannot_hide_duplicate_model_slots(self):
        slots = self.policy.model.pi05_config.camera_mapping
        slots["front"] = slots["base"]
        self.policy.spec = replace(self.policy.spec, camera_names=tuple(slots))
        with self.assertRaisesRegex(ValueError, "slot exactly once"):
            server.NexusOpenPiPolicy(self.policy, camera_mapping={**self.wire, "extra": "front"})
        del slots["front"]
        with self.assertRaisesRegex(ValueError, "public cameras"):
            server.NexusOpenPiPolicy(self.policy, camera_mapping={**self.wire, "extra": "front"})

    def robot_message(self):
        images = {k: v.transpose(1, 2, 0) for k, v in self.observation["images"].items()}
        packet = {"protocol": HISTORY_PROTOCOL, "fps": 30, "episode_id": "trial",
                  "frame_index": 0, "step_indices": [0], "timestamps": [0],
                  "frames": [images["top_camera"]]}
        return {**self.observation, "cursor": 0, "history": packet, "images": images}

    def test_bridge_rejects_legacy_nonbase_history_before_warmup_reset_or_upload(self):
        metadata = {"use_history": True, "history_camera": "left_wrist",
                    **self.policy.spec.to_dict()}
        client = Mock()
        client.get_server_metadata.return_value = metadata
        bridge = MemoryBridge()
        with self.assertRaisesRegex(ValueError, "base history only"):
            bridge.warmup(client, "pick", use_history=True)
        with self.assertRaisesRegex(ValueError, "base history only"):
            bridge(client, self.robot_message(), "pick", metadata)
        client.infer.assert_not_called()

    def test_bridge_warmup_and_prediction_use_openpi_with_identity_camera_mapping(self):
        cameras = {wire: slot for wire, slot in zip(
            self.observation["images"], ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
            strict=True,
        )}
        self.policy.model.pi05_config.camera_mapping = cameras
        self.policy.spec = replace(self.policy.spec, camera_names=tuple(cameras))
        self.policy.metadata = self.policy.spec.to_dict()
        websocket = memory_server.MemoryWebsocketServer(
            self.policy, Mock(return_value=np.ones((1, 16, 2048), np.float32)),
            MemoryConfig(camera="top_camera", budget=16), execute_steps=1, host="localhost",
            port=0, num_steps=1, seed=4, camera_mapping={name: name for name in cameras},
        )
        with websocket.open_connection(None) as infer:
            client = SimpleNamespace(infer=Mock(wraps=infer),
                                     get_server_metadata=lambda: websocket.metadata)
            bridge = MemoryBridge()
            bridge.warmup(client, "pick", use_history=True)
            out = bridge(client, self.robot_message(), "pick", websocket.metadata)
            requests = [call.args[0] for call in client.infer.call_args_list]
            self.assertEqual([r["protocol"] for r in requests if "images" in r],
                             ["openpi", "openpi"])
            self.assertEqual(out["actions"].shape, (self.policy.spec.action_horizon, 14))
            self.assertEqual(out["history_ack"]["frame_index"], 0)

    def test_robot_bridge_uploads_top_camera_history_for_a_renamed_public_camera(self):
        slots = {"front": "base_0_rgb", "left_wrist": "left_wrist_0_rgb",
                 "right_wrist": "right_wrist_0_rgb"}
        self.policy.model.pi05_config.camera_mapping = slots
        self.policy.spec = replace(self.policy.spec, camera_names=tuple(slots))
        self.policy.metadata = self.policy.spec.to_dict()
        encoder = Mock(return_value=np.ones((1, 16, 2048), np.float32))
        websocket = memory_server.MemoryWebsocketServer(
            self.policy, encoder, MemoryConfig(camera="front", budget=16), execute_steps=1,
            host="localhost", port=0, num_steps=1, seed=4,
        )
        message = self.robot_message()
        with websocket.open_connection(None) as infer:
            client = SimpleNamespace(infer=Mock(wraps=infer))
            metadata = websocket.metadata
            with self.assertRaisesRegex(ValueError, "top_camera history only"):
                MemoryBridge()(client, message, "pick", {**metadata, "history_wire_camera": "left"})
            client.infer.assert_not_called()
            out = MemoryBridge()(client, message, "pick", metadata)
            self.assertEqual(metadata["history_camera"], "front")
            self.assertEqual(client.infer.call_args_list[1].args[0]["history_wire_camera"],
                             "top_camera")
            self.assertEqual(out["history_ack"], {"episode_id": "trial", "frame_index": 0})
            np.testing.assert_array_equal(encoder.call_args.args[0][0],
                                          message["images"]["top_camera"])

    def test_bc_and_recap_adapter_metadata_and_launchers_use_artifact_slots(self):
        def check_routing(infer):
            infer(self.observation)
            images = self.policy.calls[-1][0]["images"]
            for wire, public in self.wire.items():
                expected = self.observation["images"][wire].transpose(1, 2, 0)
                np.testing.assert_array_equal(images[public], expected)

        check_routing(server.NexusOpenPiPolicy(self.policy).infer)
        with patch.object(
            server, "_build_openpi_metadata", wraps=server._build_openpi_metadata
        ) as metadata:
            expected_metadata = server.build_openpi_metadata(self.policy)
            metadata.assert_called_once_with(self.policy, camera_mapping=self.wire)
            for target in ("modeling_pi05:Pi05Model", "recap.modeling_recap:Pi05ReCapModel"):
                with (
                    self.subTest(target=target),
                    tempfile.TemporaryDirectory() as directory,
                    patch.dict("os.environ", {}),
                    patch("dobot_nexus.inference.load_policy", return_value=self.policy),
                    patch.object(server, "WebsocketPolicyServer") as transport,
                ):
                    (Path(directory) / "manifest.yaml").write_text(yaml.safe_dump({
                        "schema_version": 2, "model": {
                            "target": f"dobot_nexus.models.pi05.{target}",
                            "state_dim": 14, "action_dim": 14,
                        },
                    }))
                    metadata.reset_mock()
                    server.main(["--artifact", directory, "--no-warmup"])
                    metadata.assert_called_once_with(self.policy, camera_mapping=self.wire)
                    settings = transport.call_args.kwargs
                    self.assertEqual(settings["metadata"], expected_metadata)
                    with settings["open_connection"](None) as infer:
                        check_routing(infer)
                    transport.return_value.serve_forever.assert_called_once()
