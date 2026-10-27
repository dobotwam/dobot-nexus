# Pi0.5 Torch core: provenance and local adaptations

The architecture and operators in this directory were originally imported from
`rlinf/models/embodiment/openpi_rlinf/pi0_model` at RLinf revision
`89b0cd5fce528180559bbd50d055fb2e21a25bb5`.

Copyright 2026 The RLinf Authors. Licensed under Apache License 2.0; see `LICENSE`.
Local adaptations for Dobot/OpenPI policy training:

- The implementation lives in `models/pi05/torch_core`, shared by training
  and inference, and is imported as a normal Python package. `source.py` records
  provenance.
- `numerics.py` owns the optional OpenPI arithmetic profile. Training helper
  compilation is selected per model in `../training/compile.py` for RoPE,
  gated residual and GELU-GLU.

- `model.py` delegates training augmentation to `augmentation.py`.
- `model.py` optionally enables antialiased resize for frozen history images.
  `numerics.py` configures independent frozen SigLIP instances with FP32 stem
  and normalization parameters, BF16 dense/attention operands, and spatial
  pooling that preserves BF16 addition order. Policy vision defaults are unchanged.
- `augmentation.py` implements the geometry and HSV color operations of augmax
  0.4.1 (Apache-2.0, https://github.com/khdlr/augmax). It preserves that release's
  default hue=.1, probability=.5, and ineffective saturation assignment.
- `pi0.py` uses separate caller-derived image/noise/time random streams and FP32
  flow interpolation, targets and loss. Operator execution keeps its own dtype.
- `pi0.py` and `gemma.py` accept explicit optional history tensors and registered
  encoder/pre-FFN modules. The baseline has no additional parameters; history
  operators are owned by `models/pi05/memory`, outside this core.
- GELU approximation and preservation of projection inputs are configurable
  without changing the baseline defaults. FrameSample selects OpenPI's tanh
  GELU and FP32 timestep inputs; its Linear dtype adaptation lives in
  `numerics.py` and preserves the original weight keys. Optional RMSNorm math
  preserves the scale parameter's dtype before adding one, as in OpenPI.

The base revision above identifies the imported source, not an unmodified file
snapshot. Nexus records its training recipe separately in the resolved config.

Inference adaptations: `pi0.py` accepts optional compiled prefix/suffix callables
without changing the Euler solver. Its AR mask uses device-side fills, and
`gemma.py` uses a scalar attention-mask fill instead of allocating a CPU scalar
and copying it to CUDA. These operations allow whole-sampler CUDA Graph capture.
The optional `inference_ops` argument routes legacy/OpenPI inference through model-owned
ATen numerical boundaries and shares RoPE angles across layers. Training uses
the original operators; no parameters or checkpoint keys are added or replaced.
Inference may also supply cached per-layer history K/V and native timestep
modulations. History projections are recomputed for every request/graph replay
and reused only across that request's Euler steps; training does not use them.

RLToken adaptation: `compute_loss(..., _return_prefix=True)` also returns the
same forward's prefix hidden states and validity mask. The default flow loss,
random streams and parameter keys are unchanged. Token operators live outside
this core and detach the prefix before reconstruction.
