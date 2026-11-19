import hashlib
import json
import queue
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from examples.atom_robot import workers
from examples.atom_robot.core import RuntimeFault
from examples.pi05_real_robot._runtime.robot_contract import PolicyRobotContract, load_robot_layout


class ObservationPipelineTests(unittest.TestCase):
    def contract(self):
        metadata = {
            "policy_spec": {"state_dim": 18, "action_dim": 18, "action_horizon": 50},
            "input_schema": {
                "state": [18],
                "images": {k: [3, 224, 224] for k in ("top_camera", "left_camera", "right_camera")},
            },
            "output_schema": {"actions": [50, 18]},
        }
        layout = load_robot_layout("atom18")
        c = PolicyRobotContract.from_metadata(metadata, layout)
        c.bind_robot_hello({"robot_schema": layout.to_schema()})
        return c

    def test_observation_waits_for_both_robot_state_and_every_camera_after_boundary(self):
        clock = NS(value=10.15)
        calls = []
        states = iter([{"time": 9.99}, {"time": 10.12}, {"time": 10.14}])
        images = {"left_camera": np.zeros((224, 224, 3), np.uint8)}

        class Rig:
            def snapshot(self, *a):
                i = len(calls)
                calls.append(i)
                return images, {"left_camera": {"received_monotonic": [10.11, 9.98, 10.13][i]}}

        with (
            patch.object(workers, "drain_latest", lambda *a, **k: next(states)),
            patch.object(
                workers,
                "time",
                NS(
                    monotonic=lambda: clock.value,
                    sleep=lambda s: setattr(clock, "value", clock.value + 0.02),
                ),
            ),
        ):
            observation, _, _, source = workers.fresh_observation(
                None,
                Rig(),
                {"camera_max_age_s": 0.35, "camera_max_skew_s": 0.15},
                NS(value=10.0),
                threading.Event(),
            )
        self.assertEqual(len(calls), 3)
        self.assertEqual(observation["time"], 10.14)
        self.assertEqual(source, 10.13)

    def test_archive_keeps_every_input_output_and_latest_without_overwriting_first(self):
        c = self.contract()
        images = {
            k: np.full((224, 224, 3), n, np.uint8) for n, k in enumerate(c.layout.camera_map, 1)
        }
        packet = c.observation({"state": np.arange(18), "images": images}, "fold")
        with tempfile.TemporaryDirectory() as d:
            first = workers.archive_policy_exchange(
                d, 1, packet, np.ones((50, 18)), {"policy_seed": 42}
            )
            packet["state"] = packet["state"] + 1
            second = workers.archive_policy_exchange(
                d, 2, packet, np.zeros((50, 18)), {"policy_seed": 43}
            )
            with np.load(first, allow_pickle=False) as values:
                np.testing.assert_array_equal(values["state"], np.arange(18))
                np.testing.assert_array_equal(values["actions"], np.ones((50, 18)))
            with np.load(Path(d) / "latest-observation.npz", allow_pickle=False) as values:
                np.testing.assert_array_equal(values["state"], np.arange(18) + 1)
                np.testing.assert_array_equal(values["image.left_camera"], images["left_camera"])
            sidecar = json.loads(Path(second).with_suffix(".json").read_text())
            self.assertEqual(sidecar["policy_seed"], 43)
            self.assertEqual(
                sidecar["wire_image_sha256"]["left_camera"],
                hashlib.sha256(packet["images"]["left_camera"].tobytes()).hexdigest(),
            )


class LiveGripperDispatchTests(unittest.TestCase):
    def run_stream(
        self, targets, *, late=None, write_delay=0.0, mutate=None, startup_empty_reads=0
    ):
        clock = NS(value=100.0)
        fake_time = NS(
            monotonic=lambda: clock.value, sleep=lambda n: setattr(clock, "value", clock.value + n)
        )
        stop = threading.Event()
        armed = threading.Event()
        armed.set()
        values = {s: {"init_state": 1, "grip_state": 1, "opening": 1.0} for s in ("left", "right")}
        writes = []
        incoming = []
        events = []
        checks = []
        late = late or {}

        class Writer:
            def __init__(self, *a, **kw):
                pass

            def write(self, message):
                writes.append(message)
                clock.value += write_delay
                message.sample_info = NS(valid_data=True, publication_handle=7)
                incoming.append(message)

        class Reader:
            def __init__(self, *a, **kw):
                pass

            def take(self, n):
                result = incoming[:]
                incoming.clear()
                return result

        cfg = NS(max_state_age_s=0.25, require_execute=lambda execute: None)
        hands = NS(
            config=cfg,
            open=lambda: None,
            close=lambda: None,
            read=lambda: {"hands": values},
            _poll=lambda: None,
            _ages=lambda: (0.0, 0.0),
            _hands=values,
            _upper=object(),
            _participant=object(),
            matched=2,
            _discover=lambda required: None,
            _get_type=lambda name: (
                object,
                {"DHGripperSingleCmd_": object, "DHGripperInitCmd": object},
            ),
            _own=lambda obj: obj,
        )
        sdk = ModuleType("atom_training.dh_pc1")
        sdk.DHConfig = lambda **kw: cfg
        sdk.DualPC1DH = lambda cfg: hands
        sdk.feedback_values = lambda v: {s: dict(a) for s, a in v.items()}
        sdk.command_payload = lambda **kw: kw
        sdk.validate_send_state = lambda *a, **kw: checks.append(kw)
        sdk.dds_command = lambda p, *a: NS(
            left=NS(position=p["left"]), right=NS(position=p["right"])
        )
        parent = ModuleType("atom_training")
        parent.__path__ = []
        names = {
            "atom_training": parent,
            "atom_training.dh_pc1": sdk,
            "cyclonedds": ModuleType("cyclonedds"),
        }
        names["cyclonedds"].__path__ = []
        for name in ("core", "pub", "sub", "topic", "qos"):
            names["cyclonedds." + name] = ModuleType("cyclonedds." + name)
        names["cyclonedds.core"].Listener = lambda **kw: object()
        names["cyclonedds.pub"].DataWriter = Writer
        names["cyclonedds.sub"].DataReader = Reader
        names["cyclonedds.topic"].Topic = lambda *a: object()
        names["cyclonedds.qos"].Qos = lambda *a: a
        names["cyclonedds.qos"].Policy = NS(
            Reliability=NS(Reliable=lambda **kw: None, BestEffort=None),
            History=NS(KeepLast=lambda n: None),
        )

        class Targets:
            index = 0
            base = None
            empty = startup_empty_reads

            def get(self, timeout):
                if self.empty:
                    self.empty -= 1
                    clock.value += timeout
                    raise queue.Empty
                if self.index == len(targets):
                    stop.set()
                    raise queue.Empty
                if self.base is None:
                    self.base = clock.value
                seq, row, opening = targets[self.index][:3]
                deadline = self.base + self.index / 20
                clock.value = max(clock.value, deadline + late.get(self.index, 0.0))
                if mutate:
                    mutate(self.index, values, incoming)
                result = {
                    "sequence": seq,
                    "row": row,
                    "deadline": deadline,
                    "time": deadline,
                    "horizon": 50,
                    "opening": np.array(opening),
                    "is_hold": False,
                }
                if len(targets[self.index]) > 3:
                    result.update(targets[self.index][3])
                self.index += 1
                return result

        sink = NS(put=lambda e, **kw: events.append(e), put_nowait=events.append)
        with (
            patch.dict(sys.modules, names),
            patch.object(workers, "time", fake_time),
            patch("examples.atom_robot.native.checked_bytes", lambda *a: b""),
        ):
            workers.gripper_worker(
                {"sdk_root": "/sdk", "execute": True, "action_fps": 20.0, "horizon": 50},
                Targets(),
                stop,
                armed,
                NS(value=0.0),
                sink,
            )
        return writes, events, checks

    def faults(self, events):
        return [e for e in events if e["kind"] == "fault"]

    def test_identical_rows_are_acknowledged_without_repeated_hardware_commands(self):
        writes, events, checks = self.run_stream([(1, n, [1.0, 1.0]) for n in range(50)])
        self.assertEqual(len(writes), 1)
        self.assertEqual(len(checks), 50)
        rows = [e for e in events if e["kind"] == "gripper_row"]
        self.assertEqual(len(rows), 50)
        self.assertEqual(sum(e["transmitted"] for e in rows), 1)
        self.assertTrue(all(e["sent_at"] is None and e["already_held"] for e in rows[1:]))
        self.assertFalse(self.faults(events))

    def test_waiting_for_first_target_does_not_flood_startup_telemetry(self):
        _, events, _ = self.run_stream([(1, 0, [1.0, 1.0])], startup_empty_reads=50)
        first = next(e for e in events if e["kind"] == "gripper_row")["deadline"]
        self.assertTrue(
            all(e["observed_at"] >= first for e in events if e["kind"] == "gripper_feedback")
        )
        self.assertFalse(self.faults(events))

    def test_right_then_left_stages_survive_250_fifo_rows_and_horizon_boundaries(self):
        opening = (
            [[1.0, 1.0]] * 50
            + [[1.0, 1.0]] * 30
            + [[1.0, 0.0]] * 70
            + [[1.0, 1.0]] * 50
            + [[0.0, 1.0]] * 50
        )
        rows = [(i // 50 + 1, i % 50, v) for i, v in enumerate(opening)]
        writes, events, _ = self.run_stream(rows)
        self.assertEqual(
            [(w.left.position, w.right.position) for w in writes],
            [(1.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )
        self.assertEqual(
            [(e["sequence"], e["row"]) for e in events if e["kind"] == "gripper_row"],
            [(s, r) for s, r, _ in rows],
        )
        self.assertFalse(self.faults(events))

    def test_late_already_held_row_is_covered_and_not_retransmitted(self):
        writes, events, _ = self.run_stream([(1, n, [1.0, 1.0]) for n in range(50)], late={1: 0.06})
        self.assertEqual(len(writes), 1)
        self.assertFalse(self.faults(events))
        self.assertEqual(
            next(e for e in events if e["kind"] == "closed")["late_rows_already_held"], 1
        )

    def test_changed_late_target_is_rejected_before_write(self):
        writes, events, _ = self.run_stream(
            [(1, 0, [1.0, 1.0]), (1, 1, [0.0, 1.0])], late={1: 0.06}
        )
        self.assertEqual(len(writes), 1)
        self.assertIn("changed target missed", self.faults(events)[0]["error"])

    def test_a_blocked_dds_write_does_not_acknowledge_an_ontime_row(self):
        writes, events, _ = self.run_stream([(1, 0, [1.0, 1.0])], write_delay=0.06)
        self.assertEqual(len(writes), 1)
        self.assertFalse([e for e in events if e["kind"] == "gripper_row"])
        self.assertIn("DDS write exceeded", self.faults(events)[0]["error"])

    def test_missing_rows_chunks_duplicates_and_unannounced_cancellation_stop(self):
        for second in [
            (1, 2, [1.0, 1.0]),
            (1, 0, [1.0, 1.0]),
            (3, 0, [1.0, 1.0]),
            (2, 0, [1.0, 1.0]),
        ]:
            with self.subTest(second=second):
                writes, events, _ = self.run_stream([(1, 0, [1.0, 1.0]), second])
                self.assertEqual(len(writes), 1)
                self.assertIn("Missing or duplicate", self.faults(events)[0]["error"])

    def test_explicit_obstruction_cancellation_allows_next_chunk(self):
        writes, events, _ = self.run_stream(
            [(1, 0, [1.0, 1.0]), (2, 0, [1.0, 1.0], {"cancelled_previous_sequence": 1})]
        )
        self.assertEqual(len(writes), 1)
        self.assertFalse(self.faults(events))

    def test_drop_and_competing_writer_checks_remain_active_for_unchanged_targets(self):
        def drop(i, values, incoming):
            if i == 1:
                values["left"]["grip_state"] = 3

        _, events, _ = self.run_stream([(1, 0, [1.0, 1.0]), (1, 1, [1.0, 1.0])], mutate=drop)
        self.assertIn("drop reported", self.faults(events)[0]["error"])

        def foreign(i, values, incoming):
            if i == 1:
                incoming.append(NS(sample_info=NS(valid_data=True, publication_handle=99)))

        _, events, _ = self.run_stream([(1, 0, [1.0, 1.0]), (1, 1, [1.0, 1.0])], mutate=foreign)
        self.assertIn("Another gripper controller", self.faults(events)[0]["error"])


class HandoverRegressionTests(unittest.TestCase):
    def test_lost_initial_handover_is_retried_and_requires_fresh_feedback(self):
        from examples.atom_robot.native import release_owned_control

        clock = NS(value=100.0)
        writes = []
        upper = NS(is_upper_control=True)

        def write(value):
            self.assertFalse(value.flag)
            writes.append(clock.value)
            if len(writes) == 3:
                upper.is_upper_control = False

        session = NS(
            requested=True,
            released=False,
            last={"upper": upper},
            stamp={"upper": 100.0},
            switch=NS(write=write),
            dds=NS(SwitchUpperControl_=lambda **kw: NS(**kw)),
            pump=lambda **kw: session.stamp.update(upper=clock.value),
        )
        release_owned_control(
            session,
            clock=lambda: clock.value,
            sleep=lambda s: setattr(clock, "value", clock.value + s),
        )
        self.assertTrue(session.released)
        self.assertEqual(len(writes), 3)
        self.assertGreaterEqual(writes[1] - writes[0], 0.05)

    def test_no_handover_is_sent_for_control_not_acquired_by_this_session(self):
        from examples.atom_robot.native import NativeBridge, release_owned_control

        session = NS(
            requested=False,
            released=False,
            last={"upper": NS(is_upper_control=True)},
            last_ref=None,
            writer=None,
        )
        release_owned_control(session)
        bridge = NativeBridge("/sdk", "/validation")
        bridge.session = session
        self.assertFalse(bridge.result()["upper_control_returned"])
        self.assertFalse(bridge.result()["upper_control_release_confirmed"])

    def test_stale_false_ownership_feedback_does_not_confirm_release(self):
        from examples.atom_robot.native import release_owned_control

        clock = NS(value=100.0)
        session = NS(
            requested=True,
            released=False,
            last={"upper": NS(is_upper_control=False)},
            stamp={"upper": 90.0},
            switch=NS(write=lambda value: None),
            dds=NS(SwitchUpperControl_=lambda **kw: NS(**kw)),
            pump=lambda **kw: None,
        )
        with self.assertRaises(RuntimeFault):
            release_owned_control(
                session,
                clock=lambda: clock.value,
                sleep=lambda s: setattr(clock, "value", clock.value + s),
            )
        self.assertFalse(session.released)

    def test_receipt_keeps_frequency_for_the_next_live_start(self):
        from examples.atom_robot import validation, verify
        from tests.entrypoints.atom.test_atom_home_and_validation import ValidationContractTest

        report = ValidationContractTest().valid()
        fake = NS(testsRun=112, wasSuccessful=lambda: True)
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / name for name in ("one.json", "two.json")]
            for number, path in enumerate(paths):
                path.write_text(json.dumps({**report, "run_number": number}))
            with (
                patch.object(
                    verify.unittest.defaultTestLoader, "discover", return_value=unittest.TestSuite()
                ),
                patch.object(verify.unittest.TextTestRunner, "run", return_value=fake),
            ):
                self.assertEqual(
                    verify.main(
                        [
                            "--report",
                            str(paths[0]),
                            "--report",
                            str(paths[1]),
                            "--validation",
                            folder,
                        ]
                    ),
                    0,
                )
            receipt = validation.require_live_validation(
                folder, expected_horizon=20, expected_action_fps=20.0
            )
            self.assertEqual(receipt["action_fps"], 20.0)


if __name__ == "__main__":
    unittest.main()
