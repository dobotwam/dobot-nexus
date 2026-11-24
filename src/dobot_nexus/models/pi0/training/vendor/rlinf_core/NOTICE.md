# Vendored RLinf Pi model core

The Python sources in this directory are vendored from
`rlinf/models/embodiment/openpi_rlinf/pi0_model` at RLinf revision
`89b0cd5fce528180559bbd50d055fb2e21a25bb5`.

Copyright 2026 The RLinf Authors. Licensed under Apache License 2.0; see `LICENSE`.
Dobot Nexus adds optional FrameSample hooks to the action expert and flow sampler,
including per-request history K/V projection caching. The resize helper supports
antialiasing and the SigLIP MLP supports a configurable GELU approximation for
the frozen history encoder. Shared dtype conversion lives in a separate module so
the frozen encoder can import SigLIP eagerly. The loader, memory operators, checkpoint and data
adapters, training loop, and deployment integration are maintained by Dobot Nexus.
