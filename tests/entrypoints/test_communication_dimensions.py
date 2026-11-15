from __future__ import annotations

import os
import asyncio
import socket
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dobot_nexus.core.contracts import PolicySpec
from dobot_nexus.inference.transport.websocket import WebsocketPolicyServer
from dobot_nexus.models.pi05.deploy import server as deployment
from examples.pi05_real_robot._runtime import memory, dobot_dual_arm_14d as bridge
from examples.pi05_real_robot._runtime.dimensions import CommunicationSpec


def observation(state_dim):
    return {"type": "observation", "cursor": 7,
            "state": np.arange(state_dim, dtype=np.float32),
            "images": {name: np.zeros((224, 224, 3), np.uint8) for name in bridge.IMAGE_NAMES}}


class CommunicationDimensionsTest(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict(os.environ, {"NEXUS_ROBOT_TOKEN": "x" * 32})
        patch.start()
        self.addCleanup(patch.stop)
    def test_metadata_supports_legacy_nested_and_schema_dimensions(self):
        self.assertEqual(CommunicationSpec.from_metadata({}), CommunicationSpec(14, 14))
        for state_dim, action_dim in ((14, 14), (18, 18), (34, 34), (12, 7)):
            values = {"state_dim": state_dim, "action_dim": action_dim, "action_horizon": 5}
            for metadata in (values, {"policy_spec": values},
                             {"input_schema": {"state": [state_dim]},
                              "output_schema": {"actions": [5, action_dim]}}):
                with self.subTest(metadata=metadata):
                    self.assertEqual(CommunicationSpec.from_metadata(metadata),
                                     CommunicationSpec(state_dim, action_dim, 5))

    def test_invalid_or_conflicting_metadata_is_rejected(self):
        for metadata in (
            {"state_dim": 18},
            {"state_dim": 18, "action_dim": 18, "policy_spec": {"state_dim": 14}},
            {"state_dim": 18, "action_dim": 18, "input_schema": {"state": [14]}},
            {"action_horizon": 5, "output_schema": {"actions": [6, 18]}, "state_dim": 18},
            {"input_schema": {"state": [1, 18]}},
            {"state_dim": True, "action_dim": 18},
            {"state_dim": 0, "action_dim": 18},
            {"state_dim": -1, "action_dim": 18},
            {"state_dim": 18.5, "action_dim": 18},
        ):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                CommunicationSpec.from_metadata(metadata)

    def test_robot_handshake_requires_matching_dimensions(self):
        CommunicationSpec(14, 14).validate_robot({"protocol": bridge.PROTOCOL_VERSION})
        CommunicationSpec(18, 7).validate_robot({"state_dim": 18, "action_dim": 7})
        for hello in ({}, {"state_dim": 18, "action_dim": 14}):
            with self.subTest(hello=hello), self.assertRaisesRegex(ValueError, "do not match policy"):
                CommunicationSpec(18, 18).validate_robot(hello)
        runtime = object.__new__(bridge.RobotRuntimeEnv)
        self.assertEqual(runtime.communication_spec["state_dim"], 14)
        self.assertEqual(runtime.communication_spec["action_dim"], 14)
        self.assertEqual(len(runtime.communication_spec["action_names"]), 14)

    def test_custom_state_is_preserved_and_invalid_state_rejected(self):
        message = observation(18)
        result = bridge.build_openpi_obs(message, "pick", state_dim=18)
        np.testing.assert_array_equal(result["state"], message["state"])
        for state in (np.zeros(14), np.zeros((1, 18)), np.full(18, np.nan)):
            with self.subTest(shape=state.shape), self.assertRaisesRegex(ValueError, "finite state"):
                bridge.build_openpi_obs({**message, "state": state}, "pick", state_dim=18)
            with self.assertRaises(ValueError):
                CommunicationSpec(18, 18).validate_state(state)

    def test_actions_require_full_width_horizon_and_finite_values(self):
        spec = CommunicationSpec(18, 7, 5)
        expected = np.arange(35, dtype=np.float32).reshape(5, 7)
        np.testing.assert_array_equal(spec.validate_actions(expected), expected)
        for actions in (np.zeros((5, 14)), np.zeros((4, 7)), np.zeros((0, 7)),
                        np.zeros(7), np.full((5, 7), np.inf)):
            with self.subTest(shape=actions.shape), self.assertRaises(ValueError):
                spec.validate_actions(actions)

    def test_dimension_mismatch_stops_before_robot_observation_or_actions(self):
        policy = mock.Mock()
        policy.get_server_metadata.return_value = {"state_dim": 18, "action_dim": 18}
        with (
            mock.patch("dobot_nexus.inference.transport.client.WebsocketPolicyClient", return_value=policy),
            mock.patch.object(bridge, "ClientHealthServer"),
            mock.patch.object(bridge, "warmup_openpi_policy"),
            mock.patch.object(bridge.socket, "create_connection"),
            mock.patch.object(bridge, "authenticate", side_effect=lambda s, **kw: s),
            mock.patch.object(bridge, "recv_message", return_value={"protocol": bridge.PROTOCOL_VERSION}),
            mock.patch.object(bridge, "send_message") as send,
        ):
            with self.assertRaisesRegex(ValueError, "do not match policy"):
                bridge.run_openpi_client(bridge.parse_args(["openpi-client"]))
            send.assert_not_called()
            policy.infer.assert_not_called()
            policy.close.assert_called_once()

    def test_memory_warmup_and_dense_inference_use_custom_dimensions(self):
        policy = mock.Mock()
        policy.infer.side_effect = [{"actions": np.zeros((5, 18))}, {"reset_finished": True}]
        memory.MemoryBridge.warmup(policy, "pick", use_history=True, state_dim=18)
        self.assertEqual(policy.infer.call_args_list[0].args[0]["state"].shape, (18,))
        policy.infer.assert_called_with({"reset": True})

        dense = memory.MemoryBridge()
        dense.episode_id, dense.last_frame, dense.last_cursor = "episode", 0, 7
        dense.prompt, dense.awaiting_observation = "pick", True
        dense.last_image = np.zeros((224, 224, 3), np.uint8)
        message = {**observation(18), "history": {
            "protocol": memory.HISTORY_PROTOCOL, "uploaded": True, "episode_id": "episode",
            "frame_index": 0, "fps": 30.0}}
        policy.infer.side_effect = None
        policy.infer.return_value = {"actions": np.zeros((5, 18), np.float32)}
        result = dense(policy, message, "pick", {
            "state_dim": 18, "action_dim": 18, "action_horizon": 5, "use_history": True})
        self.assertEqual(policy.infer.call_args.args[0]["state"].shape, (18,))
        self.assertEqual(policy.infer.call_args.args[0]["step_index"], 0)
        self.assertEqual(result["actions"].shape, (5, 18))
        self.assertEqual(result["history_ack"]["frame_index"], 0)


class CommunicationLoopbackTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patch = mock.patch.dict(os.environ, {"NEXUS_ROBOT_TOKEN": "x" * 32})
        patch.start()
        self.addCleanup(patch.stop)
    async def test_real_websocket_and_robot_tcp_preserve_custom_dimensions(self):
        # CPU stub replaces only model prediction. Both wire codecs and the actual
        # deployment adapter, warmup and bridge loop run without hardware.
        for state_dim, action_dim in ((14, 14), (18, 18), (34, 34), (12, 7)):
            for mode in ("standard", "memory"):
                with self.subTest(state=state_dim, action=action_dim, mode=mode):
                    await self._roundtrip(state_dim, action_dim, mode)

    async def _roundtrip(self, state_dim, action_dim, mode):
        expected = np.arange(5 * action_dim, dtype=np.float32).reshape(1, 5, action_dim)
        states = []

        def predict(value, options):
            states.append(value["state"].copy())
            return SimpleNamespace(actions=expected.copy(), timing={})

        spec = PolicySpec(model_name="pi05", implementation="cpu-stub",
                          camera_names=("base", "left_wrist", "right_wrist"),
                          state_dim=state_dim, action_dim=action_dim, action_horizon=5)
        policy = SimpleNamespace(spec=spec, model=SimpleNamespace(), manifest=None,
                                 artifact_dir=None, predict_chunk=predict)
        args = SimpleNamespace(seed=42, num_steps=10, vary_seed=False, default_prompt=None,
                               warmup=False, host="127.0.0.1", port=0)
        with (mock.patch.object(deployment, "_load_policy", return_value=policy),
              mock.patch.object(deployment, "WebsocketPolicyServer") as constructor):
            deployment.run_server(args)
            server_options = constructor.call_args.kwargs
        metadata = server_options["metadata"]
        self.assertEqual(metadata["state_dim"], state_dim)
        self.assertEqual(metadata["action_dim"], action_dim)
        self.assertEqual(metadata["input_schema"]["state"], [state_dim])
        self.assertEqual(metadata["output_schema"]["actions"], [5, action_dim])
        CommunicationSpec.from_metadata(metadata)

        with socket.socket() as robot:
            robot.bind(("127.0.0.1", 0))
            robot.listen(1)
            robot.settimeout(10)

            def exchange():
                connection, _ = robot.accept()
                with connection:
                    connection = bridge.authenticate(connection, server=True)
                    # Exercise old dimensionless robot handshakes for 14D too.
                    hello = {"type": "hello", "protocol": bridge.PROTOCOL_VERSION}
                    if state_dim != 14 or action_dim != 14:
                        hello.update(state_dim=state_dim, action_dim=action_dim)
                    bridge.send_message(connection, hello)
                    bridge.send_message(connection, observation(state_dim))
                    reply = bridge.recv_message(connection)
                    self.assertEqual(reply["cursor"], 7)
                    np.testing.assert_array_equal(reply["actions"], expected[0, :3])

            async with WebsocketPolicyServer(**server_options).start() as listener:
                port = listener.sockets[0].getsockname()[1]
                client_args = bridge.parse_args([
                    "openpi-client", "--policy-port", str(port), "--robot-host", "127.0.0.1",
                    "--robot-port", str(robot.getsockname()[1]), "--actions-per-response", "3"])
                dense = memory.MemoryBridge()
                options = {} if mode == "standard" else {
                    "observation_handler": dense, "warmup_handler": dense.warmup,
                    "message_handler": dense.handle_message}
                with mock.patch.object(bridge, "ClientHealthServer"):
                    await asyncio.gather(asyncio.to_thread(exchange), asyncio.to_thread(
                        bridge.run_openpi_client, client_args, **options))
        self.assertEqual(len(states), 2)
        np.testing.assert_array_equal(states[0], np.zeros(state_dim))
        np.testing.assert_array_equal(states[1], np.arange(state_dim))


if __name__ == "__main__":
    unittest.main()
