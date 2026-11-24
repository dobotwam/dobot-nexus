"""Request-level history and action semantics for Pi0 FrameSample serving."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dobot_nexus.core.contracts import ObservationBatch, PolicyOutput
from dobot_nexus.inference.session import SessionOptions
from dobot_nexus.models.pi0.memory.configuration import MemoryConfig
from dobot_nexus.models.pi0.memory.position import position_embeddings
from dobot_nexus.models.pi0.memory.runtime import HistoryBuffer, MemorySession


def observation(*ids, prompt="pick"):
    return ObservationBatch(
        images={"base": np.zeros((len(ids), 2, 2, 3), np.uint8)},
        state=np.zeros((len(ids), 1), np.float32),
        prompt=[prompt] * len(ids),
        env_ids=list(ids),
    )


def session():
    policy = mock.Mock(spec=["spec", "predict_chunk"])
    policy.spec = SimpleNamespace(action_horizon=3)
    policy.predict_chunk.return_value = PolicyOutput(np.array([[[1], [2], [3]]], np.float32))
    encoder = mock.Mock(side_effect=lambda frames: np.ones((len(frames), 16, 2048), np.float32))
    return MemorySession(
        policy, encoder, MemoryConfig(budget=16), SessionOptions(3), max_batch_frames=2
    )


class MemorySessionTest(unittest.TestCase):
    def test_failed_prompt_change_preserves_history_actions_and_seed(self):
        memory = session()
        memory.reset("a", seed=7)
        previous = observation("a")
        memory.act(previous, step_indices=[4])
        history = memory.buffers["a"].snapshot()
        changed = observation("a", prompt="place")
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            memory.prepare(changed, step_indices=[-1])
        with self.assertRaisesRegex(ValueError, "consecutive"):
            memory.push_frames([1, 2], [5, 7], env_id="a", prompt="place")
        memory.policy.predict_chunk.side_effect = RuntimeError("prediction failed")
        with self.assertRaisesRegex(RuntimeError, "prediction failed"):
            memory.predict_chunk(changed, step_indices=[0])
        self.assertEqual(memory.prompts, {"a": "pick"})
        self.assertEqual(memory.actions["a"].options.seed, 7)
        self.assertEqual(memory.buffers["a"].last_step, 4)
        np.testing.assert_array_equal(memory.buffers["a"].snapshot().features, history.features)
        np.testing.assert_array_equal(memory.act(previous, step_indices=[5]), [[2]])

    def test_batched_act_rolls_back_history_and_cursor_on_later_policy_error(self):
        memory = session()
        np.testing.assert_array_equal(memory.act(observation("a")), [[1]])
        memory.policy.predict_chunk.side_effect = RuntimeError("second environment failed")
        batch = observation("a", "b")
        with self.assertRaisesRegex(RuntimeError, "second environment failed"):
            memory.act(batch, step_indices=[1, 0])
        self.assertEqual(set(memory.buffers), {"a"})
        self.assertEqual(memory.buffers["a"].last_step, 0)
        self.assertEqual(memory.actions["a"].remaining_steps, 2)
        memory.policy.predict_chunk.side_effect = None
        np.testing.assert_array_equal(memory.act(batch, step_indices=[1, 0]), [[2], [1]])

    def test_quota_precedes_iteration_and_bounds_unsized_inputs(self):
        memory = session()

        class Oversized:
            def __len__(self):
                return 3

            def __iter__(self):
                raise AssertionError("Oversized batch was iterated")

        def unused():
            raise AssertionError("Other input consumed before checking sizes")
            yield

        for inputs in ((Oversized(), unused()), (unused(), Oversized())):
            with self.assertRaisesRegex(ValueError, "frame limit"):
                memory.push_frames(*inputs, prompt="pick")

        def unbounded():
            yield from range(3)
            raise AssertionError("Consumed more than max_batch_frames + 1 elements")

        for inputs in ((unbounded(), unused()), (iter([0]), unbounded())):
            with self.assertRaisesRegex(ValueError, "frame limit"):
                memory.push_frames(*inputs, prompt="pick")
        memory.encoder.assert_not_called()
        self.assertFalse(memory.buffers)

    def test_one_frame_budget_selects_current_global_and_windowed_history(self):
        for window, relative in ((None, 4), (3, 2)):
            with self.subTest(window=window):
                buffer = HistoryBuffer(MemoryConfig(budget=16, window_frames=window))
                for step in range(11, 16):
                    buffer.append(np.full((16, 2048), step, np.float32), step)
                snap = buffer.snapshot()
                np.testing.assert_array_equal(snap.features, np.full((16, 2048), 15, np.float32))
                np.testing.assert_array_equal(snap.positions, position_embeddings([relative])[0])
                self.assertTrue(snap.mask.all())
