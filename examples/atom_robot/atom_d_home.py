"""Compatibility import for examples.real_robot.atom_d.home."""

import sys

from examples.real_robot.atom_d import home as _implementation

sys.modules[__name__] = _implementation
