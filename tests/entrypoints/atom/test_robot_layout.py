from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.real_robot.shared.robot_contract import (
    PolicyRobotContract,
    RobotLayout,
    load_robot_layout,
)


def metadata(state=18, action=18, horizon=50, cameras=None):
    cameras = cameras or {
        "top_camera": [3, 4, 5],
        "left_camera": [3, 4, 5],
        "right_camera": [3, 4, 5],
    }
    return {
        "policy_spec": {
            "state_dim": state,
            "action_dim": action,
            "action_horizon": horizon,
            "image_size": [4, 5],
        },
        "input_schema": {"state": [state], "images": cameras},
        "output_schema": {"actions": [horizon, action]},
    }


class RobotContractTests(unittest.TestCase):
    def setUp(self):
        self.layout = load_robot_layout("atom18")
        self.contract = PolicyRobotContract.from_metadata(metadata(), self.layout)

    def observation(self, dimension=18):
        return {
            "type": "observation",
            "cursor": 0,
            "state": np.arange(dimension, dtype=np.float32),
            "images": {
                name: np.zeros((4, 5, 3), dtype=np.uint8)
                for name in self.layout.camera_map.values()
            },
        }

    def test_nested_nexus_policy_spec_and_all_eighteen_action_columns(self):
        self.contract.bind_robot_hello({"robot_schema": self.layout.to_schema()})
        observation = self.contract.observation(self.observation(), "Fold the T-shirt.")
        np.testing.assert_array_equal(observation["state"], np.arange(18))
        self.assertEqual(observation["images"]["top_camera"].shape, (3, 4, 5))
        actions = np.arange(50 * 18, dtype=np.float32).reshape(50, 18)
        np.testing.assert_array_equal(self.contract.actions(actions, 8), actions[:8])
        self.assertEqual(self.contract.warmup_observation("fold")["state"].shape, (18,))

    def test_legacy_fourteen_dimensional_service_and_handshake_remain_valid(self):
        layout = load_robot_layout("dual6")
        contract = PolicyRobotContract.from_metadata({}, layout)
        contract.bind_robot_hello({"protocol": "dual_env_socket_v2"})
        self.assertEqual(contract.state_dim, 14)
        self.assertEqual(contract.action_dim, 14)
        self.assertIsNone(contract.action_horizon)
        np.testing.assert_array_equal(contract.actions(np.ones((8, 14)), 8), np.ones((8, 14)))

    def test_arbitrary_asymmetric_state_action_sizes_and_camera_names(self):
        config = {
            "schema_version": 1,
            "name": "custom_robot",
            "state_names": [f"s{i}" for i in range(20)],
            "action_names": [f"a{i}" for i in range(12)],
            "camera_map": {"front_rgb": "camera_2"},
        }
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "layout.json"
            file.write_text(json.dumps(config))
            layout = load_robot_layout(str(file))
        contract = PolicyRobotContract.from_metadata(
            metadata(20, 12, 7, {"front_rgb": [3, 4, 5]}), layout
        )
        contract.bind_robot_hello({"robot_schema": layout.to_schema()})
        observation = contract.observation(
            {"state": np.zeros(20), "images": {"camera_2": np.zeros((3, 4, 5), dtype=np.uint8)}},
            "task",
        )
        self.assertEqual(observation["state"].shape, (20,))
        self.assertEqual(set(observation["images"]), {"front_rgb"})
        self.assertEqual(contract.actions(np.zeros((7, 12)), 5).shape, (5, 12))

    def test_robot_field_names_reorder_observations_and_actions_independently(self):
        schema = self.layout.to_schema()
        schema["state_names"] = schema["state_names"][::-1]
        schema["action_names"] = schema["action_names"][1:] + schema["action_names"][:1]
        self.contract.bind_robot_hello({"robot_schema": schema})
        observation = self.observation()
        np.testing.assert_array_equal(
            self.contract.observation(observation, "fold")["state"], np.arange(18)[::-1]
        )
        actions = np.tile(np.arange(18), (50, 1))
        np.testing.assert_array_equal(
            self.contract.actions(actions, 1)[0], list(range(1, 18)) + [0]
        )

    def test_reject_conflicting_top_level_nested_and_wire_dimensions(self):
        for mutate in [
            lambda m: m.update(state_dim=14),
            lambda m: m["input_schema"].update(state=[14]),
            lambda m: m["output_schema"].update(actions=[49, 18]),
            lambda m: m["policy_spec"].update(state_dim=True),
        ]:
            data = metadata()
            mutate(data)
            with self.assertRaises(ValueError):
                PolicyRobotContract.from_metadata(data, self.layout)

    def test_custom_layout_requires_explicit_robot_handshake(self):
        with self.assertRaisesRegex(ValueError, "hello.robot_schema"):
            self.contract.bind_robot_hello({})
        wrong = load_robot_layout("dual6").to_schema()
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.contract.bind_robot_hello({"robot_schema": wrong})

    def test_custom_fields_cannot_claim_legacy_compatibility_by_layout_name(self):
        data = {
            "schema_version": 1,
            "name": "dual6",
            "state_names": [f"other_s{i}" for i in range(14)],
            "action_names": [f"other_a{i}" for i in range(14)],
            "camera_map": dict(load_robot_layout("dual6").camera_map),
        }
        contract = PolicyRobotContract.from_metadata(metadata(14, 14), RobotLayout.from_dict(data))
        with self.assertRaisesRegex(ValueError, "hello.robot_schema"):
            contract.bind_robot_hello({})

    def test_duplicate_or_unknown_robot_fields_are_rejected(self):
        for mutation in [
            lambda s: s["state_names"].__setitem__(0, s["state_names"][1]),
            lambda s: s["action_names"].__setitem__(0, "unknown_axis"),
            lambda s: s.update(image_names=["top_camera"]),
        ]:
            schema = self.layout.to_schema()
            mutation(schema)
            with self.assertRaises(ValueError):
                self.contract.bind_robot_hello({"robot_schema": schema})

    def test_state_nan_wrong_dimension_and_changed_order_are_rejected(self):
        for change in [
            lambda m: m.update(state=np.zeros(14)),
            lambda m: m.update(state=np.full(18, np.nan)),
            lambda m: m.update(state_names=list(self.layout.state_names)[::-1]),
        ]:
            message = self.observation()
            change(message)
            with self.assertRaises(ValueError):
                self.contract.observation(message, "fold")

    def test_output_width_horizon_nonfinite_and_overflow_fail_without_truncation(self):
        for actions in [
            np.zeros((50, 14)),
            np.zeros((49, 18)),
            np.full((50, 18), np.nan),
            np.full((50, 18), 1e100),
            np.zeros((50, 18), dtype=complex),
        ]:
            with self.assertRaises(ValueError):
                self.contract.actions(actions, 8)
        for count in [0, -1, True, 51]:
            with self.assertRaises(ValueError):
                self.contract.actions(np.zeros((50, 18)), count)

    def test_image_map_shape_and_dtype_are_checked_before_inference(self):
        for change in [
            lambda m: m["images"].pop("left_camera"),
            lambda m: m["images"].update(top_camera=np.zeros((224, 224, 3), dtype=np.uint8)),
            lambda m: m["images"].update(top_camera=np.zeros((4, 5, 3), dtype=np.float32)),
        ]:
            message = self.observation()
            change(message)
            with self.assertRaises(ValueError):
                self.contract.observation(message, "fold")

    def test_frame_order_cannot_change_during_connection(self):
        schema = self.layout.to_schema()
        schema["state_names"].reverse()
        self.contract.bind_robot_hello({"robot_schema": schema})
        message = self.observation()
        message["state_names"] = list(self.layout.state_names)
        with self.assertRaisesRegex(ValueError, "order changed"):
            self.contract.observation(message, "fold")


if __name__ == "__main__":
    unittest.main()
