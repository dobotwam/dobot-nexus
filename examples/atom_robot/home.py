"""Compatibility import for examples.real_robot.atom_w_0759.home."""

import sys

from examples.real_robot.atom_w_0759 import home as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
