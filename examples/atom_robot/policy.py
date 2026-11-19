"""Compatibility import for examples.real_robot.atom_common.policy."""

import sys

from examples.real_robot.atom_common import policy as _implementation

sys.modules[__name__] = _implementation
