# Copyright 2026 The RLinf Authors.
#
# Model-owned PyTorch Pi0.5 architecture, shared by training and inference.
#
# The implementation derives from RLinf/OpenPI; provenance, licensing and local
# adaptations are documented in NOTICE.md. Imports do not compile operators.
# Training helper compilation lives in ../training/compile.py; inference
# compilation and CUDA Graph replay live in ../openpi_torch/engine.py.
