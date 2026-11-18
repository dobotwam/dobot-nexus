import time
import unittest

import numpy as np

from examples.atom_robot.core import RuntimeFault
from examples.atom_robot.sensors import CameraRig, CameraSpec


class FakeCamera:
    def __init__(self, spec):
        self.spec = spec
        self.n = 0
        self.starts = 0
        self.stops = 0

    def start(self):
        self.starts += 1

    def read(self):
        time.sleep(0.002)
        self.n += 1
        return (
            np.full((4, 5, 3), self.n % 255, dtype=np.uint8),
            self.n,
            self.n * 2.0,
            time.monotonic(),
        )

    def close(self):
        self.stops += 1


class CameraLifecycleTest(unittest.TestCase):
    def test_many_observations_reuse_stream_and_cleanup_is_idempotent(self):
        specs = [CameraSpec(name, name, 5, 4, warmup_frames=2) for name in ["a", "b", "c"]]
        rig = CameraRig(specs, factory=FakeCamera)
        rig.start()
        rig.wait_ready(timeout=2)
        previous = {}
        for _ in range(10):
            images, metadata = rig.snapshot()
            self.assertEqual(len(images), 3)
            for name in images:
                self.assertEqual(images[name].shape, (224, 224, 3))
                self.assertGreaterEqual(metadata[name]["frame_number"], previous.get(name, 0))
                previous[name] = metadata[name]["frame_number"]
            time.sleep(0.003)
        rig.close()
        rig.close()
        self.assertTrue(all(c.starts == 1 and c.stops == 1 for c in rig.cameras))
        with self.assertRaises(RuntimeFault):
            rig.start()

    def test_stream_failure_rejects_observation(self):
        class Failed(FakeCamera):
            def read(self):
                raise RuntimeError("USB disconnected")

        rig = CameraRig([CameraSpec("a", "a", 5, 4, warmup_frames=1)], factory=Failed)
        try:
            rig.start()
            with self.assertRaisesRegex(RuntimeFault, "USB disconnected"):
                rig.wait_ready(timeout=2)
        finally:
            rig.close()

    def test_clock_adjustment_does_not_invalidate_a_new_frame(self):
        class Adjusted(FakeCamera):
            def read(self):
                image, number, stamp, host = super().read()
                return image, number, stamp if number < 3 else stamp - 1000, host

        rig = CameraRig([CameraSpec("a", "a", 5, 4, warmup_frames=5)], factory=Adjusted)
        try:
            rig.start()
            rig.wait_ready(timeout=2)
            rig.snapshot()
            self.assertEqual(rig.stats["a"]["device_clock_adjustments"], 1)
        finally:
            rig.close()

    def test_counter_reset_remains_a_fault(self):
        class Reset(FakeCamera):
            def read(self):
                image, number, stamp, host = super().read()
                return image, 10 if number == 1 else 1, stamp, host

        rig = CameraRig([CameraSpec("a", "a", 5, 4, warmup_frames=5)], factory=Reset)
        try:
            rig.start()
            with self.assertRaisesRegex(RuntimeFault, "counter reset"):
                rig.wait_ready(timeout=2)
        finally:
            rig.close()


if __name__ == "__main__":
    unittest.main()
