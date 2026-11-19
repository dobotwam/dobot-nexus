"""Compatibility import for examples.real_robot.atom_w_0759.workers."""

import sys

from examples.real_robot.atom_w_0759 import workers as _implementation

sys.modules[__name__] = _implementation
