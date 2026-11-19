import unittest

import numpy as np

from examples.real_robot.atom_common.core import (
    ActionChunk,
    BlockedDirectionGuard,
    ChunkGate,
    LoopGuard,
    PlaybackClock,
    RuntimeFault,
    Trajectory,
    actuator_targets,
    negotiate_horizon,
    require_fresh,
    state_vector,
)


class RuntimeCoreTest(unittest.TestCase):
    def test_50_step_negotiation_preserves_every_action_row_and_final_target(self):
        horizon = negotiate_horizon(50)
        actions = np.zeros((horizon, 18))
        actions[:, 0] = np.linspace(0.01, 0.5, horizon)
        actions[:, 8] = np.linspace(0.0, -0.3, horizon)
        chunk = ActionChunk.create(1, 10, 10.1, actions, horizon=horizon)
        points = actuator_targets(chunk.actions, np.zeros(17), np.tile([-2, 2], (17, 1)))
        path = Trajectory(np.zeros(19), points)
        for index, timestamp in enumerate(path.ends):
            reference, consumed = path.sample(float(timestamp))
            self.assertEqual(consumed, index + 1)
            np.testing.assert_allclose(reference, points[index])
        self.assertEqual(consumed, 50)
        self.assertAlmostEqual(reference[1], 0.5)
        with self.assertRaises(RuntimeFault):
            ActionChunk.create(1, 10, 10.1, actions[:20], horizon=horizon)

    def test_horizon_mismatch_is_rejected_without_silently_slicing(self):
        self.assertEqual(negotiate_horizon(20), 20)
        self.assertEqual(negotiate_horizon(50, 50), 50)
        for server, requested in [(50, 20), (None, None), (0, None), (True, None), (257, None)]:
            with self.assertRaises(RuntimeFault):
                negotiate_horizon(server, requested)

    def test_tracking_pause_never_accumulates_a_catch_up_jump(self):
        clock = PlaybackClock(0.0, 0.1)
        self.assertEqual(clock.advance(1.0, scale=0.0), 0.0)
        elapsed = clock.advance(1.001, scale=0.5)
        self.assertAlmostEqual(elapsed, 0.0005)
        clock.advance(10.0, scale=0.0)
        resumed = clock.advance(10.001)
        self.assertLessEqual(resumed - elapsed, 0.001 + 1e-12)

    def test_scheduler_jitter_delays_playback_without_skipping_targets_or_reference_jumps(self):
        points = np.zeros((20, 19))
        points[:, 1] = np.linspace(0.01, 0.2, 20)
        points[:, 8] = np.linspace(-0.005, -0.1, 20)
        path = Trajectory(np.zeros(19), points)
        clock = PlaybackClock(0.0, path.joint_speed)
        now = 0.0
        previous = np.zeros(19)
        consumed_rows = set()
        for n in range(10000):
            now += (0.0012, 0.0158, 0.049)[n % 3]
            reference, consumed = path.sample(clock.advance(now))
            self.assertLessEqual(
                float(np.max(abs(reference[:17] - previous[:17]))), np.deg2rad(0.04) + 1e-12
            )
            consumed_rows.add(consumed)
            previous = reference
            if consumed == 20:
                break
        self.assertEqual(consumed_rows, set(range(21)))
        np.testing.assert_allclose(reference, points[-1])
        self.assertGreater(clock.delayed, 0)
        self.assertGreater(now, path.duration)

    def test_playback_clock_rejects_backward_time_and_relaxed_step_envelopes(self):
        clock = PlaybackClock(10.0, 0.1)
        for now in [9.9, float("nan")]:
            with self.assertRaises(RuntimeFault):
                clock.advance(now)
        with self.assertRaises(RuntimeFault):
            PlaybackClock(0.0, 0.1, reference_step_deg=0.06)

    def test_state_layout_excludes_waist_and_keeps_both_grippers(self):
        state = state_vector(np.arange(17), [0.2, 0.7])
        np.testing.assert_allclose(
            state, np.r_[np.arange(1, 8), 0.2, np.arange(8, 15), 0.7, [15, 16]]
        )

    def test_horizon_and_all_waypoints_are_preserved(self):
        actions = np.zeros((20, 18))
        actions[:, 0] = np.linspace(0, 0.2, 20)
        actions[:, 7] = np.linspace(1, 0, 20)
        actions[:, 8] = np.linspace(0, -0.1, 20)
        actions[:, 15] = 1
        points = actuator_targets(actions, np.zeros(17), np.tile([-2, 2], (17, 1)))
        path = Trajectory(np.r_[np.zeros(17), 1, 1], points)
        self.assertEqual(len(path.points), 20)
        for n, t in enumerate(path.ends):
            output, consumed = path.sample(float(t))
            self.assertEqual(consumed, n + 1)
            np.testing.assert_allclose(output, points[n], atol=1e-12)
        self.assertGreater(points[-1, 1], np.pi / 180)

    def test_reference_speed_and_joint_envelope(self):
        p = np.zeros((20, 19))
        p[:, 1] = np.linspace(0, 0.3, 20)
        p[:, 17] = np.linspace(0, 1, 20)
        trajectory = Trajectory(np.zeros(19), p)
        previous = np.zeros(19)
        dt = 0.001
        for t in np.arange(dt, trajectory.duration, dt):
            current, _ = trajectory.sample(float(t))
            velocity = abs(current - previous) / dt
            self.assertLessEqual(max(velocity[:17]), 0.10001)
            self.assertLessEqual(max(velocity[17:]), 0.50001)
            previous = current

    def test_invalid_shape_nan_limits_and_timestamps_fail_closed(self):
        for value in [np.zeros((50, 18)), np.full((20, 18), np.nan)]:
            with self.assertRaises(RuntimeFault):
                ActionChunk.create(1, 10, 10.1, value)
        with self.assertRaises(RuntimeFault):
            require_fresh(10, 11, 0.2, "camera")
        with self.assertRaises(RuntimeFault):
            require_fresh(12, 11, 0.2, "camera")
        actions = np.zeros((20, 18))
        actions[3, 2] = 5
        with self.assertRaises(RuntimeFault):
            actuator_targets(actions, np.zeros(17), np.tile([-2, 2], (17, 1)))

    def test_chunk_sequence_and_age(self):
        gate = ChunkGate()
        chunk = ActionChunk.create(1, 10, 10.1, np.zeros((20, 18)))
        gate.accept(chunk, 10.2)
        with self.assertRaises(RuntimeFault):
            gate.accept(chunk, 10.3)
        with self.assertRaises(RuntimeFault):
            ChunkGate().accept(chunk, 11.0)

    def test_repeating_without_progress_stops_on_third_horizon(self):
        guard = LoopGuard()
        a = np.zeros((20, 18))
        a[:, 0] = np.sin(np.linspace(0, 2 * np.pi, 20)) * 0.02
        self.assertFalse(guard.observe(a, np.zeros(18)))
        self.assertFalse(guard.observe(a, np.zeros(18)))
        self.assertTrue(guard.observe(a, np.zeros(18)))

    def test_valid_state_progress_does_not_look_like_a_loop(self):
        guard = LoopGuard()
        a = np.zeros((20, 18))
        for n in range(5):
            state = np.zeros(18)
            state[0] = n * 0.02
            self.assertFalse(guard.observe(a, state))

    def test_two_horizon_cycle_stops_after_three_repetitions(self):
        guard = LoopGuard()
        a = np.zeros((20, 18))
        b = np.zeros((20, 18))
        b[:, 0] = 0.1
        for n in range(5):
            self.assertFalse(guard.observe(a if n % 2 == 0 else b, np.zeros(18)))
        self.assertTrue(guard.observe(b, np.zeros(18)))

    def test_obstruction_needs_next_horizon_and_stops_another_push(self):
        guard = BlockedDirectionGuard()
        guard.record_blockage(4, [0, 0, 0.7], [0, 0, 0.69])
        with self.assertRaises(RuntimeFault):
            guard.next_horizon_pushes_into_obstruction(4, [[0, 0, 0.68]])
        self.assertFalse(
            guard.next_horizon_pushes_into_obstruction(5, [[0, 0, 0.72], [0.02, 0, 0.7]])
        )
        self.assertTrue(guard.next_horizon_pushes_into_obstruction(5, [[0, 0, 0.69], [0, 0, 0.68]]))


if __name__ == "__main__":
    unittest.main()
