"""Compatibility import for examples.real_robot.atom_common.torque."""

import sys

from examples.real_robot.atom_common import torque as _implementation

sys.modules[__name__] = _implementation
