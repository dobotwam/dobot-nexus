"""Run Pi0 Memory inference with an isolated history and the Artifact's frozen vision model."""

from __future__ import annotations


def open_memory_session(config, *, options=None):
    from dobot_nexus.inference import load_policy

    from .modeling_memory import Pi0MemoryModel
    from .runtime import MemorySession
    from .torch_vision import TorchFrameTokenEncoder

    if not config.model.get("artifact"):
        raise ValueError("A trained Pi0MemoryModel Artifact is required.")
    policy = load_policy(config)
    if not isinstance(policy.model, Pi0MemoryModel):
        raise ValueError("Expected a Pi0MemoryModel Artifact.")
    spec = policy.model.memory_config
    encoder = TorchFrameTokenEncoder(
        spec.vision_checkpoint, device=policy.device, expected_sha256=spec.vision_sha256
    )
    return MemorySession(policy, encoder, spec, policy.create_session(options).options)


def main(argv=None):
    from dobot_nexus.cli.infer import run_inference

    run_inference(argv, load_predictor=open_memory_session)


if __name__ == "__main__":
    main()
