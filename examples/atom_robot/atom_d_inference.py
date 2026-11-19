"""Compatibility import for examples.real_robot.atom_d.inference."""

import sys

from examples.real_robot.atom_d import inference as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
