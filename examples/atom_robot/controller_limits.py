"""Compatibility import for examples.real_robot.atom_common.controller_limits."""

import sys

from examples.real_robot.atom_common import controller_limits as _implementation

sys.modules[__name__] = _implementation
