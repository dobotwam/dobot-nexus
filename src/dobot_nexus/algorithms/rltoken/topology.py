"""RLToken role placement and transport lanes from the public Stage 2 config."""

from dobot_nexus.core.config import clone_config
from dobot_nexus.runtime.contracts import ChannelSpec, PlacementSpec, RoleSpec, TopologySpec

RLTOKEN_CHANNELS = (
    "requests",
    "responses",
    "transitions",
    "durable_acks",
    "weights",
    "weight_acks",
)


def build_rltoken_topology(config, *, run_id, epoch, task_fingerprint, resume_path,
                           expected_manifest_sha256=None, feature_root=None, expected_spec=None,
                           source_sha256=None):
    if feature_root is not None:
        config = clone_config(config)
        config.model.features.artifact = str(feature_root)
    runtime = config.runtime
    resources = runtime.resources
    kwargs = {"config": config, "run_id": run_id, "epoch": epoch}
    module = "dobot_nexus.algorithms.rltoken.roles"
    env_gpu = resources.get("env_gpu")
    roles = (
        RoleSpec(
            "rollout",
            f"{module}:RolloutRole",
            PlacementSpec("gpu", resources.rollout_gpu),
            init_kwargs={**kwargs, "expected_manifest_sha256": expected_manifest_sha256,
                         "expected_spec": expected_spec, "source_sha256": source_sha256},
        ),
        RoleSpec(
            "learner",
            f"{module}:LearnerRole",
            PlacementSpec("gpu", resources.learner_gpu),
            init_kwargs={
                **kwargs,
                "expected_spec": expected_spec,
                "task_fingerprint": task_fingerprint,
                "resume_path": resume_path,
            },
        ),
        RoleSpec(
            "env",
            f"{module}:EnvRole",
            PlacementSpec("cpu") if env_gpu is None else PlacementSpec("gpu", env_gpu),
            init_kwargs=kwargs.copy(),
        ),
    )
    limit = runtime.get("max_message_bytes", 64 * 2**20)
    return TopologySpec(
        run_prefix=f"rlt-{epoch}",
        roles=roles,
        channels=tuple(ChannelSpec(name, max_message_bytes=limit) for name in RLTOKEN_CHANNELS),
        rpc_timeout=runtime.get("rpc_timeout", 120.0),
        log_dir=str(config.exp.run_dir),
        ray_tmpdir=runtime.get("ray_tmpdir"),
    )
