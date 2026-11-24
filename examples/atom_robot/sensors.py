"""Compatibility import for examples.real_robot.atom_common.sensors."""

import sys

from examples.real_robot.atom_common import sensors as _implementation

sys.modules[__name__] = _implementation
