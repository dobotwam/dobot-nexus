"""Compatibility import for examples.real_robot.atom_w_0759.run."""

import sys

from examples.real_robot.atom_w_0759 import run as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
