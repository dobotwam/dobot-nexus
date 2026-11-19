import json
import sys
import tempfile
import types
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.real_robot.atom_common.core import RuntimeFault
from examples.real_robot.atom_w_0759.native import NativeBridge, load_last_holding, sdk_effort_limits


class NativeContractTest(unittest.TestCase):
    def test_home_can_hold_uninitialized_grippers_without_weakening_joint_state_checks(self):
        fake = types.ModuleType("atom_training.dh_pc1")
        hands = {s: {"opening": 1.0, "init_state": 0} for s in ("left", "right")}
        fake.feedback_values = lambda value: hands
        session = types.SimpleNamespace(
            requested=False,
            pump=lambda **kw: None,
            check=lambda expected: None,
            q=lambda: np.zeros(17),
            last={
                "upper": types.SimpleNamespace(
                    motor_state=[types.SimpleNamespace(tau_est=0.0)] * 17
                )
            },
            hands=types.SimpleNamespace(_poll=lambda: None, _ages=lambda: [0.0, 0.0], _hands=None),
        )
        policy = NativeBridge("/sdk", "/state", execute=False)
        home = NativeBridge("/sdk", "/state", execute=False, require_initialized_grippers=False)
        policy.session = home.session = session
        with patch.dict(sys.modules, {"atom_training.dh_pc1": fake}):
            with self.assertRaises(RuntimeFault):
                policy.snapshot()
            self.assertEqual(home.snapshot()["state"].shape, (18,))
            session.check = lambda expected: (_ for _ in ()).throw(RuntimeFault("drive fault"))
            with self.assertRaises(RuntimeFault):
                home.snapshot()
        self.assertEqual(home.command_count, 0)

    def test_explicit_fresh_holding_uses_current_pose_without_old_reference_or_extra_effort(self):
        bridge = NativeBridge("/missing/sdk", "/old/state", execute=True, fresh_hold=True)
        bridge.kp = np.full(17, 200.0)
        bridge.effort = np.full(17, 40.0)
        q = np.linspace(-0.5, 1.4, 17)
        tau = np.full(17, 2.0)
        with patch("examples.real_robot.atom_w_0759.native.load_last_holding", side_effect=AssertionError):
            bridge.seed_holding(q, tau)
        np.testing.assert_allclose(bridge.reference, q)
        np.testing.assert_allclose(bridge.tau_ff[1:15], 2.0)
        np.testing.assert_allclose(bridge.tau_ff[[0, 15, 16]], 0.0)
        self.assertEqual(bridge.command_count, 0)
        self.assertIsNone(bridge.session)
        with self.assertRaises(RuntimeError):
            bridge.seed_holding(q, np.full(17, 40.01))

    def test_normal_holding_still_refuses_stale_history(self):
        bridge = NativeBridge("/missing/sdk", "/old/state", execute=True)
        bridge.kp = np.full(17, 200.0)
        bridge.effort = np.full(17, 40.0)
        with patch(
            "examples.real_robot.atom_w_0759.native.load_last_holding", side_effect=RuntimeFault("stale")
        ):
            with self.assertRaises(RuntimeFault):
                bridge.seed_holding(np.zeros(17), np.zeros(17))
        self.assertTrue(bridge.require_initialized_grippers)
        self.assertEqual(bridge.command_count, 0)

    def test_constructor_and_read_only_acquire_have_no_hardware_effect(self):
        bridge = NativeBridge("/missing/sdk", "/missing/state", execute=False)
        self.assertIsNone(bridge.session)
        with self.assertRaises(RuntimeFault):
            bridge.acquire()
        self.assertEqual(bridge.command_count, 0)

    def test_locked_waist_is_not_assigned_an_invented_urdf_limit(self):
        rows = [{"name": "torso_joint"}] + [{"name": f"arm{i}"} for i in range(16)]
        xml = ET.fromstring(
            "<robot>"
            + "".join(f'<joint name="arm{i}"><limit effort="12"/></joint>' for i in range(16))
            + "</robot>"
        )
        limits = sdk_effort_limits(xml, rows)
        self.assertTrue(np.isinf(limits[0]))
        np.testing.assert_array_equal(limits[1:], 12)
        rows[3]["name"] = "not_a_real_joint"
        with self.assertRaises(RuntimeFault):
            sdk_effort_limits(xml, rows)

    def test_read_only_reports_cannot_replace_latest_real_holding_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            real = root / "atom-runtime-20261109-120000"
            real.mkdir()
            value = {
                "commands_sent": 10,
                "upper_control_returned": True,
                "last_motor_reference": [0.0] * 17,
                "last_tau_ff": [0.0] * 17,
            }
            (real / "result.json").write_text(json.dumps(value))
            read = root / "atom-runtime-20261109-130000"
            read.mkdir()
            (read / "result.json").write_text('{"commands_sent":0}')
            holding = load_last_holding(np.zeros(17), root)
            self.assertEqual(holding["source"], str(real))
            value["upper_control_returned"] = False
            (real / "result.json").write_text(json.dumps(value))
            with self.assertRaises(RuntimeFault):
                load_last_holding(np.zeros(17), root)


if __name__ == "__main__":
    unittest.main()
