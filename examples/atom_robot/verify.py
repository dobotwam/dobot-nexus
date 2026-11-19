"""Compatibility import for examples.real_robot.atom_w_0759.verify."""

import sys

from examples.real_robot.atom_w_0759 import verify as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
