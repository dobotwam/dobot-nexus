"""CPU contracts for overlapping policy work with ATOM D servo rows."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dobot_nexus.inference.handlers import PredictionHandler
from examples.atom_robot import atom_d_inference as driver


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.state = np.zeros(18, np.float32)
        self.state[[7, 15]] = 1
        self.actions = np.tile(self.state, (50, 1))
        self.bounds = np.tile([-2.0, 2.0], (17, 1))
        self.upper = np.zeros(17)
        self.images = {"top_camera": np.zeros((16, 16, 3), dtype=np.uint8)}
        self.points = driver.actuator_targets(self.actions, self.upper, self.bounds)
        self.chunk = driver.ActionChunk.create(1, 0.0, 0.1, self.actions, horizon=50)
        self.result = (self.chunk, self.points, self.state, self.images, {}, 43)
        self.job = driver.PendingChunk(
            None, 46, 0.0, self.points[0], self.points[:4].copy(), (1, 1), np.ones(2)
        )

    def reason(self, age=0.2, elapsed=0.2):
        with patch.object(driver.time, "monotonic", return_value=age):
            return self.job.rejection(self.result, 4, elapsed, self.state, self.images, (1, 1), 0.7)

    def test_stale_slow_divergent_or_transient_hands_replan(self):
        self.assertIsNone(self.reason())
        self.assertEqual(self.reason(age=0.71), "stale")
        self.assertEqual(self.reason(elapsed=0.31), "timing")
        self.points[4, 17] = 0.0
        self.assertEqual(self.reason(), "hand_intent")
        self.points[4, 17] = 1.0
        self.points[2, 1] = 0.1
        self.assertEqual(self.reason(), "overlap")
        self.points[2, 1] = 0.0
        self.points[4, 1] = 0.1
        self.assertEqual(self.reason(), "bridge_excursion")
        self.job.boundary = True
        self.assertEqual(self.reason(), "seam")
        self.job.hand_changed = True
        self.assertEqual(self.reason(), "hand_feedback")

    def test_bridge_stops_before_the_first_large_motion_or_hand_command(self):
        self.assertEqual(self.job.bridge_length(self.points, 4), 4)
        self.points[6, 8] = np.deg2rad(2.1)
        self.assertEqual(self.job.bridge_length(self.points, 4), 2)
        self.points[5, 17] = 0.0
        self.assertEqual(self.job.bridge_length(self.points, 4), 1)

    def test_scene_and_measured_displacement_are_checked(self):
        other = {"top_camera": np.full((16, 16, 3), 20, np.uint8)}
        with patch.object(driver.time, "monotonic", return_value=0.2):
            self.assertEqual(
                self.job.rejection(self.result, 4, 0.2, self.state, other, (1, 1), 0.7), "scene"
            )
            moved = self.state.copy()
            moved[8] += 0.1
            self.assertEqual(
                self.job.rejection(self.result, 4, 0.2, moved, self.images, (1, 1), 0.7), "tracking"
            )

    def test_seed_retry_leaves_next_seed_unchanged(self):
        policy = Mock()
        policy.predict_chunk.return_value = SimpleNamespace(actions=self.actions, timing={})
        handler = PredictionHandler(policy, seed=42)
        first = handler({})
        self.assertEqual(handler({}, replay_seed=first["seed"])["seed"], 42)
        with self.assertRaises(ValueError):
            handler({}, replay_seed=True)
        self.assertEqual(handler({})["seed"], 43)

    def test_playback_overlaps_skips_only_sent_rows_and_retries_on_mismatch(self):
        for mismatch in (None, "early", "boundary", "closing"):
            with self.subTest(mismatch=mismatch), ExitStack() as stack:
                self.play_two_chunks(stack, mismatch)

    def play_two_chunks(self, stack, mismatch):
        folder = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        pool = stack.enter_context(ThreadPoolExecutor(1))
        report = {"rows_sent": 0, "feedback": []}
        owner = get_ident()
        calls = []
        continuing = Event()
        bridging = Event()
        clock = [1.0]
        robot = Mock()
        robot.has_error.return_value = False
        hand = SimpleNamespace(grip_state=SimpleNamespace(value=1), get_force_value=100.0)
        fb = SimpleNamespace(
            messages={"rt/hands/state": SimpleNamespace(left=hand, right=hand)},
            arrivals={"rt/hands/state": 1.0},
        )
        rig = Mock()

        def snapshot():
            self.assertEqual(get_ident(), owner)
            return self.images, {"top_camera": {"received_monotonic": clock[0]}}

        rig.snapshot.side_effect = snapshot

        def measure(_):
            self.assertEqual(get_ident(), owner)
            fb.arrivals["rt/hands/state"] = clock[0]
            return self.state.copy(), self.upper.copy(), self.upper.copy(), self.upper.copy()

        def tick(seconds):
            clock[0] += seconds

        def network(ws, state, images, replay_seed=None):
            calls.append((get_ident() == owner, replay_seed))
            if get_ident() != owner:
                event = continuing if replay_seed is None else bridging
                self.assertTrue(event.wait(2), "Servo must advance while policy work is pending")
            actions = self.actions.copy()
            if mismatch == "early" and replay_seed is None:
                actions[:4, 0] = 0.1
            actions[4:, 0] = 0.01 if replay_seed is None else 0.025
            if mismatch == "boundary" and replay_seed is not None and get_ident() != owner:
                actions[:4, 15] = 0.0
            return actions, 43

        sent = []

        def command(left, *args, send_hand):
            self.assertEqual(get_ident(), owner)
            if send_hand:
                sent.append(left[0])
                if len(sent) == 47:
                    continuing.set()
                if len(sent) == 51:
                    bridging.set()
                if len(sent) == (51 if mismatch == "closing" else 56):
                    (folder / "stop").touch()

        robot.command_joint_state.side_effect = command
        stack.enter_context(patch.object(driver.time, "monotonic", lambda: clock[0]))
        stack.enter_context(patch.object(driver.time, "sleep", tick))
        stack.enter_context(patch.object(driver, "measure", measure))
        stack.enter_context(patch.object(driver, "request_chunk", network))
        stack.enter_context(patch.object(driver, "save_view"))
        if mismatch == "closing":
            self.actions[47:, 7] = 0.0
            self.points[47:, 17] = 0.0
        initial = (
            driver.ActionChunk.create(1, 1.0, 1.0, self.actions, horizon=50),
            self.points,
            self.state,
            self.images,
            {},
            42,
        )
        status = driver.playback(
            None,
            fb,
            rig,
            self.bounds,
            robot,
            [np.ones(17)] * 4,
            folder,
            report,
            True,
            initial,
            pool,
        )
        self.assertEqual(status, "stopped_by_supervisor")
        next_chunk = report["handoffs"][1]
        self.assertEqual(report["executed_rows_by_chunk"][0], 50)
        if mismatch == "closing":
            self.assertEqual(next_chunk["source"], "sync")
            self.assertEqual(calls, [(True, None)])
        elif mismatch == "early":
            self.assertEqual(next_chunk["discarded"], "overlap")
            self.assertEqual(next_chunk["skipped_rows"], 0)
            self.assertEqual(calls, [(False, None), (True, 43)])
            self.assertEqual(sent[50], 0.0)
        else:
            self.assertEqual(next_chunk["skipped_rows"], 4)
            self.assertEqual(report["executed_rows_by_chunk"][1], 4)
            self.assertGreater(sent[50], 0.0)
            corrected = report["handoffs"][2]
            if mismatch == "boundary":
                self.assertEqual(corrected["discarded"], "hand_intent")
                self.assertEqual(corrected["skipped_rows"], 0)
                self.assertEqual(calls, [(False, None), (False, 43), (True, 43)])
            else:
                self.assertEqual(corrected["source"], "boundary")
                self.assertEqual(corrected["skipped_rows"], 4)
                self.assertEqual(calls, [(False, None), (False, 43)])
                self.assertGreater(sent[54], 0.01)
