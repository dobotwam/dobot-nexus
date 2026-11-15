# Dobot Fusion adaptation

Copyright 2026 Dobot.

`modeling_fusion.py` adapts the Fusion path in Dobot-RECAP
`values/backbone/pistar06.py` from the local reference snapshot.
Source SHA256: `85b31f3e383bb1f683c2d49103d18a855f2dfe113c2ece9e1b9cad5d31955584`.

Retained parameter names and attention ordering permit strict reference weight loading.
Adaptations: Nexus-owned configuration, lazy/local-only SiglipModel and Gemma3ForCausalLM
loading with Transformers 4.53.2, injectable encoders for CPU contract tests, and optional
encoder gradient checkpointing. No runtime import of the
reference checkout is needed.

The referenced Dobot-RECAP code originates from Dobot internal open-source development.
It is included in Dobot Nexus under the Apache License, Version 2.0; see the repository
LICENSE and NOTICE. This statement covers the Dobot code and its Nexus adaptations;
external model weights and dependencies retain their separate licenses.

Target/state/processor and training semantics follow `values/utils/acp_math.py`,
`values/utils/state_discretize.py`, `values/training/collate.py` and `values/training/train.py`.
