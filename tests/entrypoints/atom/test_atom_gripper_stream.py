import queue
import sys
import threading
import unittest
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from examples.real_robot.atom_w_0759 import workers
from examples.real_robot.atom_common.core import RuntimeFault


class GripperStreamTests(unittest.TestCase):
    def test_poll_uses_available_stream_without_a_blocking_fresh_read(self):
        hands = NS(
            _poll=lambda: None,
            _ages=lambda: (0.01, 0.02),
            _hands={"sample": 1},
            config=NS(max_state_age_s=0.25),
            read=lambda: (_ for _ in ()).throw(AssertionError("Blocking read in streaming loop")),
        )
        self.assertEqual(workers.poll_gripper_state(hands, lambda v: v)["hands"], {"sample": 1})

    def test_feedback_freshness_is_not_relaxed_to_avoid_action_deadlines(self):
        for ages in ((0.26, 0.01), (0.01, 0.26), (float("nan"), 0.01), (-0.01, 0.01)):
            hands = NS(
                _poll=lambda: None, _ages=lambda ages=ages: ages, config=NS(max_state_age_s=0.25)
            )
            with self.subTest(ages=ages), self.assertRaises(RuntimeFault):
                workers.poll_gripper_state(hands, lambda v: v)

    def test_all_50_fifo_rows_are_consumed_with_only_one_startup_read(self):
        clock = NS(value=100.0)
        fake_time = NS(
            monotonic=lambda: clock.value,
            sleep=lambda seconds: setattr(clock, "value", clock.value + seconds),
        )
        stop = threading.Event()
        values = {side: {"init_state": 1, "grip_state": 1} for side in ("left", "right")}
        reads = []

        def fresh_read():
            reads.append(1)
            if len(reads) > 1:
                raise AssertionError("Per-row fresh read stalls the action queue")
            return {"hands": values}

        hands = NS(
            config=NS(max_state_age_s=0.25),
            open=lambda: None,
            close=lambda: None,
            read=fresh_read,
            _poll=lambda: None,
            _ages=lambda: (0.0, 0.0),
            _hands=values,
        )
        module = ModuleType("atom_training.dh_pc1")
        module.DHConfig = lambda **kwargs: NS(**kwargs)
        module.DualPC1DH = lambda cfg: hands
        module.feedback_values = lambda v: v
        for name in ("command_payload", "validate_send_state", "dds_command"):
            setattr(module, name, lambda *a, **k: None)
        parent = ModuleType("atom_training")
        parent.__path__ = []
        events = []
        sink = NS(put=lambda event, **kw: events.append(event), put_nowait=events.append)

        class Targets:
            index = 0

            def get(self, timeout):
                if self.index == 50:
                    stop.set()
                    raise queue.Empty
                deadline = 100.0 + self.index / 20
                clock.value = max(clock.value, deadline)
                row = {
                    "sequence": 1,
                    "row": self.index,
                    "deadline": deadline,
                    "time": deadline,
                    "opening": np.array([self.index / 50, 1 - self.index / 50]),
                    "is_hold": False,
                }
                self.index += 1
                return row

        with (
            patch.dict(sys.modules, {"atom_training": parent, "atom_training.dh_pc1": module}),
            patch.object(workers, "time", fake_time),
            patch("examples.real_robot.atom_w_0759.native.checked_bytes", lambda *a: b""),
        ):
            workers.gripper_worker(
                {"sdk_root": "/sdk", "execute": False, "action_fps": 20.0, "horizon": 50},
                Targets(),
                stop,
                threading.Event(),
                NS(value=0.0),
                sink,
            )
        closed = next(v for v in events if v["kind"] == "closed")
        self.assertEqual(len(reads), 1)
        self.assertEqual(closed["action_rows"], 50)
        self.assertEqual(closed["writes"], 0)
        self.assertFalse([v for v in events if v["kind"] == "fault"])


if __name__ == "__main__":
    unittest.main()
