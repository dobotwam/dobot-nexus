"""Compatibility import for examples.real_robot.atom_d.policy."""

import sys

from examples.real_robot.atom_d import policy as _implementation

sys.modules[__name__] = _implementation
