"""Compatibility import for examples.real_robot.atom_common.core."""

import sys

from examples.real_robot.atom_common import core as _implementation

sys.modules[__name__] = _implementation
