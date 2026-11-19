"""Compatibility import for examples.real_robot.atom_d.console."""

import sys

from examples.real_robot.atom_d import console as _implementation

sys.modules[__name__] = _implementation
