"""Model-independent policy handshake for the ATOM 18D hardware example."""

from examples.pi05_real_robot._runtime.robot_contract import (
    PolicyRobotContract,
    load_robot_layout,
)

from .core import RuntimeFault, negotiate_horizon


def policy_contract(metadata, requested_horizon=None):
    if metadata.get("use_history") or metadata.get("history_requires_dense_frames"):
        raise RuntimeFault("ATOM example does not supply dense Memory history")
    layout = load_robot_layout("atom18")
    contract = PolicyRobotContract.from_metadata(metadata, layout)
    contract.bind_robot_hello({"robot_schema": layout.to_schema()})
    # The camera pipeline produces padded 224px RGB. Reject a different input
    # contract before acquiring control instead of silently resizing twice.
    if any(shape != (3, 224, 224) for shape in contract.image_shapes.values()):
        raise RuntimeFault("ATOM camera preprocessing requires 224x224 RGB policy inputs")
    horizon = negotiate_horizon(contract.action_horizon, requested_horizon)
    return contract, horizon


def policy_identity(metadata):
    """Bind a validation receipt to advertised policy settings, excluding timing."""
    keys = (
        "artifact",
        "model_name",
        "implementation",
        "operator_source",
        "operator_revision",
        "policy_spec",
        "input_schema",
        "output_schema",
        "backend",
        "autocast",
        "autocast_enabled",
        "torch_compile",
        "cuda_graph",
        "seed",
        "num_steps",
    )
    return {key: metadata[key] for key in keys if key in metadata}


def preflight_policy(config):
    from dobot_nexus.inference.transport.client import WebsocketPolicyClient

    with WebsocketPolicyClient(
        host=config["policy_host"], port=config["policy_port"], timeout=config["policy_timeout_s"]
    ) as client:
        metadata = client.get_server_metadata()
    _, horizon = policy_contract(metadata, config.get("horizon"))
    return metadata, horizon
