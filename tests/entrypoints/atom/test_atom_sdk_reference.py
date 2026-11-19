"""Regression contracts for native SDK action execution, without robot hardware."""

import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from examples.atom_robot.core import FixedRateChunk, RuntimeFault, actuator_targets
from examples.atom_robot.native import NativeBridge, sdk_reference_gains
from examples.atom_robot.sdk_session import sdk_reference_session


class SDKReferenceRegressionTests(unittest.TestCase):
    def test_camera_startup_retries_do_not_hide_runtime_frame_loss(self):
        from examples.atom_robot.sensors import CameraRig, CameraSpec, FrameTimeout

        fail = threading.Event()

        class Camera:
            def __init__(self, spec):
                self.count = 0

            def start(self):
                pass

            def close(self):
                pass

            def read(self):
                self.count += 1
                if self.count <= 2 or fail.is_set():
                    raise FrameTimeout("Frame did not arrive")
                time.sleep(0.005)
                return (
                    np.zeros((4, 5, 3), np.uint8),
                    self.count,
                    self.count * 10.0,
                    time.monotonic(),
                )

        spec = CameraSpec("test", "test", 5, 4, warmup_frames=2)
        rig = CameraRig((spec,), factory=Camera)
        try:
            rig.start()
            rig.wait_ready(1.0)
            self.assertEqual(rig.stats["test"]["startup_timeouts"], 2)
            fail.set()
            deadline = time.monotonic() + 1.0
            while not rig.errors and time.monotonic() < deadline:
                time.sleep(0.005)
            with self.assertRaises(RuntimeFault):
                rig.snapshot()
        finally:
            rig.close()

    def test_real_training_50_rows_have_original_cadence_and_all_fields(self):
        fixture = json.loads(
            (Path(__file__).resolve().parents[2] / "fixtures/atom_training_chunk.json").read_text()
        )
        actions = np.asarray(fixture["actions"])
        ref = np.r_[0.1, fixture["state"][:7], fixture["state"][8:15], fixture["state"][16:18]]
        points = actuator_targets(actions, ref, np.tile([-3.2, 3.2], (17, 1)))
        path = FixedRateChunk(points)
        self.assertEqual(path.duration, 2.5)
        for row in range(50):
            value, completed = path.sample(row / 20)
            self.assertEqual(path.frame_index(row / 20), row)
            self.assertEqual(completed, row)
            np.testing.assert_array_equal(value[1:8], actions[row, :7])
            np.testing.assert_array_equal(value[8:15], actions[row, 8:15])
            np.testing.assert_array_equal(value[15:17], actions[row, 16:18])
            np.testing.assert_array_equal(value[17:], actions[row, [7, 15]])
        self.assertEqual(path.sample(2.5)[1], 50)
        np.testing.assert_array_equal(path.sample(0.049)[0], points[0])

    def bridge(self):
        bridge = NativeBridge(
            "/sdk", "/validation", execute=True, allow_head_motion=True, command_law="sdk_reference"
        )
        bridge.limits = np.tile([-3.0, 3.0], (17, 1))
        bridge.kp = np.r_[200.0, [300.0] * 3, [200.0] * 4, [300.0] * 3, [200.0] * 4, [200.0] * 2]
        bridge.kd = np.r_[100.0, [15.0] * 3, [8.0] * 4, [15.0] * 3, [8.0] * 4, [8.0] * 2]
        bridge.reference = np.zeros(17)
        messages = []
        bridge.session = NS(
            pump=lambda: None,
            check=lambda ownership: None,
            q=lambda: np.zeros(17),
            matches={"upper": 1},
            last_ref=None,
            last={"upper": NS(motor_state=[NS(dq=0.0) for _ in range(17)])},
            writer=NS(write=messages.append),
            dds=NS(MotorCmd_=lambda **kw: NS(**kw), UpperCmd_=lambda **kw: NS(**kw)),
        )
        bridge.comp = NS(step=lambda *a: (_ for _ in ()).throw(AssertionError("PI must not run")))
        return bridge, messages

    def test_sdk_message_preserves_large_enough_test_step_and_zero_extra_tau(self):
        bridge, messages = self.bridge()
        target = np.linspace(-0.3, 0.3, 17)
        target[0] = 0.0
        bridge.send(target)
        np.testing.assert_array_equal([m.q for m in messages[-1].motor_cmd], target)
        self.assertTrue(
            all(m.tau == 0.0 and m.dq == 0.0 and m.mode == 11 for m in messages[-1].motor_cmd)
        )
        np.testing.assert_array_equal([m.kp for m in messages[-1].motor_cmd], bridge.kp)
        np.testing.assert_array_equal([m.kd for m in messages[-1].motor_cmd], bridge.kd)
        target[1] += 0.1
        bridge.last_send = None
        bridge.send(target)
        self.assertAlmostEqual(messages[-1].motor_cmd[1].q, target[1])
        target[1] = 4.0
        with self.assertRaises(RuntimeFault):
            bridge.send(target)
        self.assertEqual(len(messages), 2)

    def test_zero_tau_startup_preserves_measured_support_by_native_pd_equivalence(self):
        bridge, _ = self.bridge()
        tau = np.linspace(-12.0, 12.0, 17)
        q = np.linspace(-0.4, 0.4, 17)
        bridge.seed_holding(q, tau)
        np.testing.assert_allclose(bridge.kp * (bridge.reference - q), tau, atol=1e-12)
        np.testing.assert_array_equal(bridge.tau_ff, np.zeros(17))

    def test_sdk_checks_keep_drive_faults_but_not_old_demo_velocity_cap(self):
        cls = sdk_reference_session(object)
        obj = cls()
        obj.cfg = {"joints": [{"limits": [-3.0, 3.0]} for _ in range(17)]}
        import time

        obj.stamp = {key: time.monotonic() for key in ("upper", "main/nodes", "amr")}

        def axis():
            return NS(
                is_virtual=False,
                servo_state=3,
                node_state=5,
                display_op_mode=11,
                error_code=0,
                pos_err_code=0,
                vel_err_code=0,
                torque_err_code=0,
            )

        main = NS(
            waist=axis(),
            left_arm=[axis() for _ in range(7)],
            right_arm=[axis() for _ in range(7)],
            head=[axis() for _ in range(2)],
            ecat2can=[NS(slave_state=1, error_code=0)],
        )
        obj.last = {
            "upper": NS(
                robot_type="ATOM W",
                fsm_id=2,
                is_upper_control=True,
                motor_state=[NS(dq=0.3) for _ in range(17)],
            ),
            "main/nodes": main,
            "amr": NS(amr_event=NS(emergency_stop_pressed=False, obstacle_detected=False)),
        }
        obj.q = lambda: [0.0] * 17
        obj.check(True)
        main.left_arm[0].torque_err_code = 1
        with self.assertRaises(RuntimeFault):
            obj.check(True)

    def test_sdk_gains_use_array_rows_including_waist_not_metadata_indices(self):
        from examples.atom_robot.core import STATE_NAMES

        names = ("torso_joint",) + STATE_NAMES[:7] + STATE_NAMES[8:15] + STATE_NAMES[16:]
        rows = [{"name": name, "index": i + 12, "kp": 200, "kd": 8} for i, name in enumerate(names)]
        rows[0].update(kp=200, kd=100)
        for i in (1, 2, 3, 8, 9, 10):
            rows[i].update(kp=300, kd=15)
        rows.append({"name": "unused_joint", "index": 20, "kp": 0, "kd": 0})
        with patch(
            "examples.atom_robot.native.checked_bytes", return_value=json.dumps(rows).encode()
        ):
            kp, kd = sdk_reference_gains("/sdk")
        np.testing.assert_array_equal(
            kp,
            [200, 300, 300, 300, 200, 200, 200, 200, 300, 300, 300, 200, 200, 200, 200, 200, 200],
        )
        np.testing.assert_array_equal(
            kd, [100, 15, 15, 15, 8, 8, 8, 8, 15, 15, 15, 8, 8, 8, 8, 8, 8]
        )
