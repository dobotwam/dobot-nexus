"""SDK-reference feedback checks, without the old demo motion envelope."""

import math
import time

from ..atom_common.core import RuntimeFault


def sdk_reference_session(base_type):
    """Retain DDS ownership/lifecycle, SDK drive faults and live PC1 limits."""

    class SDKReferenceSession(base_type):
        def check(self, ownership=None):
            for name, age in (("upper", 0.12), ("main/nodes", 0.7), ("amr", 0.5)):
                if name not in self.last or time.monotonic() - self.stamp[name] > age:
                    raise RuntimeFault("Missing/stale live feedback: " + name)
            upper, main, amr = (self.last[n] for n in ("upper", "main/nodes", "amr"))
            value = upper.robot_type
            robot = value if isinstance(value, str) else "".join(value)
            if robot.rstrip("\0").strip() != "ATOM W" or int(upper.fsm_id) != 2:
                raise RuntimeFault("Unexpected robot model or FSM")
            if ownership is not None and bool(upper.is_upper_control) != ownership:
                raise RuntimeFault("Unexpected upper-control ownership")
            axes = [main.waist, *main.left_arm, *main.right_arm, *main.head]
            for i, axis in enumerate(axes):
                if (
                    axis.is_virtual
                    or axis.servo_state != 3
                    or axis.node_state != 5
                    or axis.display_op_mode != 11
                    or any(
                        getattr(axis, field)
                        for field in (
                            "error_code",
                            "pos_err_code",
                            "vel_err_code",
                            "torque_err_code",
                        )
                    )
                ):
                    raise RuntimeFault("SDK axis drive is not ready: " + str(i))
            for i, slave in enumerate(main.ecat2can):
                if slave.slave_state != 1 or slave.error_code != 0:
                    raise RuntimeFault("SDK ECAT2CAN slave error: " + str(i))
            if amr.amr_event.emergency_stop_pressed or amr.amr_event.obstacle_detected:
                raise RuntimeFault("Controller emergency stop or obstacle signal")
            q = self.q()
            if len(q) != 17 or not all(math.isfinite(v) for v in q):
                raise RuntimeFault("Invalid upper feedback")
            if not all(math.isfinite(m.dq) for m in upper.motor_state):
                raise RuntimeFault("Invalid upper velocity feedback")
            for i, value in enumerate(q):
                lo, hi = self.cfg["joints"][i]["limits"]
                if not lo <= value <= hi:
                    raise RuntimeFault("Feedback exceeds active controller limits: " + str(i))

        def verify_held_pose(self, target):
            # Native PD references are not measured positions. Their static
            # offset must not be mistaken for a failed ownership handover.
            self.pump(exclusive=False)
            self.check(False)

    return SDKReferenceSession
