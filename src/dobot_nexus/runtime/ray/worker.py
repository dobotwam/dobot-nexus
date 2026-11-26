"""Static scheduler worker hosting a single algorithm-owned role."""

import os

from rlinf.scheduler import Worker

from dobot_nexus.core.import_utils import import_from_string

from ..contracts import RoleContext, invoke_role


class ManagedRoleWorker(Worker):
    def __init__(self, role_spec, channels):
        super().__init__()
        self.role_spec = role_spec
        self.channels = channels
        self.role = None
        if role_spec.placement.device_type == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    def initialize(self):
        device = "cpu"
        if self.role_spec.placement.device_type == "gpu":
            import torch

            if torch.cuda.device_count() != 1:
                raise RuntimeError("Each GPU role must see exactly one assigned CUDA device.")
            torch.cuda.set_device(0)
            device = "cuda:0"
        ctx = RoleContext(
            self.role_spec.name,
            getattr(self, "_rank", 0),
            getattr(self, "_world_size", 1),
            device,
            self.channels,
        )
        self.role = import_from_string(self.role_spec.target)(**self.role_spec.init_kwargs)
        return self.role.initialize(ctx)

    def invoke(self, method_name, *args, **kwargs):
        return invoke_role(self.role, method_name, args, kwargs)

    def close(self):
        if self.role is not None:
            self.role.close()
