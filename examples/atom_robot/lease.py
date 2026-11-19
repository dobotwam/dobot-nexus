"""Compatibility import for examples.real_robot.atom_common.lease."""

import sys

from examples.real_robot.atom_common import lease as _implementation

sys.modules[__name__] = _implementation
