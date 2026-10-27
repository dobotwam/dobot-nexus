"""Fixed-version RLinf lifecycle adapter; imported only by RLTokenTrainer.run()."""

from __future__ import annotations

import time
from types import SimpleNamespace

import ray
from rlinf.runners.async_weight_sync_mixin import AsyncWeightSyncMixin
from rlinf.scheduler import Channel, Cluster, NodePlacementStrategy, PackedPlacementStrategy

from .workers_rlinf import EnvProxyWorker, LearnerWorker, RolloutWorker

SUPPORTED_REVISION = "7e90641c4135229db6d3426950cdb45cee6259a6"


def wait_result(handle, timeout):
    deadline = time.monotonic() + timeout
    while not handle.done():
        if time.monotonic() >= deadline:
            raise TimeoutError("RLinf worker RPC deadline exceeded.")
        time.sleep(0.01)
    result = handle.wait()
    if not isinstance(result, list) or len(result) != 1:
        raise RuntimeError("RLToken v1 requires exactly one worker per role.")
    return result[0]


class _BlockingSync:
    def update_rollout_weights(self):
        receive = self.rollout.request_actor_sync_model()
        send = self.actor.sync_model_to_rollout()
        self.applied_actor_update = wait_result(receive, self.timeout)
        wait_result(send, self.timeout)


class RLinfExecution(AsyncWeightSyncMixin, _BlockingSync):
    def __init__(self, config, logger, run_id, epoch, task_fingerprint, resume_path):
        self.config, self.logger = config, logger
        self.run_id, self.epoch = run_id, epoch
        self.task_fingerprint, self.resume_path = task_fingerprint, resume_path
        self.timeout = float(config.runtime.get("rpc_timeout", 120))
        self.cfg = SimpleNamespace(actor={"sync_weight_no_wait": True})
        self.init_weight_sync_state()
        self.groups = []
        self.channels = {}
        self.applied_actor_update = -1

    def start(self):
        self.cluster = Cluster(num_nodes=1, distributed_log_dir=str(self.config.exp.run_dir))
        # Creating Cluster first establishes the RLinf namespace used by Channel.create.
        for name in (
            "requests",
            "responses",
            "transitions",
            "durable_acks",
            "weights",
            "weight_acks",
        ):
            self.channels[name] = Channel.create(f"rlt-{self.epoch}-{name}", maxsize=1)
        resources = self.config.runtime.resources
        env_gpu = resources.get("env_gpu")
        roles = (
            (
                "rollout",
                RolloutWorker,
                PackedPlacementStrategy(resources.rollout_gpu, resources.rollout_gpu),
            ),
            (
                "actor",
                LearnerWorker,
                PackedPlacementStrategy(resources.learner_gpu, resources.learner_gpu),
            ),
            (
                "env",
                EnvProxyWorker,
                NodePlacementStrategy([0])
                if env_gpu is None
                else PackedPlacementStrategy(env_gpu, env_gpu),
            ),
        )
        for name, cls, placement in roles:
            args = (self.config, self.channels, self.run_id, self.epoch)
            if name == "actor":
                args += (self.task_fingerprint, self.resume_path)
            group = cls.create_group(*args).launch(
                self.cluster,
                placement_strategy=placement,
                name=f"rlt-{self.epoch}-{name}",
                max_concurrency=1,
            )
            self.groups.append(group)
            setattr(self, name, group)
        # Start GPU initialization together; no data dependencies between roles.
        handles = {name: getattr(self, name).initialize() for name in ("rollout", "actor", "env")}
        wait_result(handles["rollout"], self.timeout)
        status = wait_result(handles["actor"], self.timeout)
        wait_result(handles["env"], self.timeout)
        self.update_rollout_weights(no_wait=False)
        return status

    def finish_sync(self):
        pending = self._pending_rollout_weight_sync
        if pending:
            self.applied_actor_update = wait_result(pending[0], self.timeout)
            wait_result(pending[1], self.timeout)
        self.drain_pending_rollout_weight_sync()

    def wait(self, handle):
        return wait_result(handle, self.timeout)

    def close(self):
        # The coordinator drains business calls first; private cleanup is isolated here.
        errors = []
        for group in reversed(self.groups):
            try:
                wait_result(group.close(), self.timeout)
            except Exception as exc:
                errors.append(exc)
                # A failed business RPC may never service another cleanup RPC.
                # Kill only actors owned by this group, never unrelated Ray jobs.
                for info in group.worker_info_list:
                    try:
                        ray.kill(info.worker, no_restart=True)
                    except Exception as cleanup_error:
                        errors.append(cleanup_error)
            else:
                try:
                    group._close()
                except Exception as cleanup_error:
                    errors.append(cleanup_error)
        self.groups.clear()
        for channel in self.channels.values():
            try:
                channel._channel_worker_group._close()
            except Exception as exc:
                errors.append(exc)
        self.channels.clear()
        if errors:
            self.logger.error("RLToken shutdown encountered: %s", errors[0])
