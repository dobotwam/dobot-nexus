"""Single-node, single-worker roles and bounded message transport."""

from __future__ import annotations

import math
import pickle
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol, TypeVar


class ChannelEmpty(Exception):
    """The channel has no message available."""


class ChannelFull(Exception):
    """The channel already contains a message."""


@dataclass(frozen=True)
class PlacementSpec:
    device_type: Literal["cpu", "gpu"]
    gpu_id: int | None = None
    node_id: int = 0

    def __post_init__(self):
        if type(self.node_id) is not int or self.node_id != 0:
            raise ValueError("Placement requires node_id=0.")
        if self.device_type not in ("cpu", "gpu"):
            raise ValueError("device_type must be cpu or gpu.")
        if self.device_type == "cpu" and self.gpu_id is not None:
            raise ValueError("CPU placement requires gpu_id=None.")
        if self.device_type == "gpu" and (type(self.gpu_id) is not int or self.gpu_id < 0):
            raise ValueError("GPU placement requires a nonnegative integer gpu_id.")


@dataclass(frozen=True)
class RoleSpec:
    name: str
    target: str
    placement: PlacementSpec
    max_concurrency: int = 1
    init_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if type(self.max_concurrency) is not int or self.max_concurrency != 1:
            raise ValueError("Role requires max_concurrency=1.")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Role name must be nonempty.")
        if not isinstance(self.target, str) or ":" not in self.target:
            raise ValueError("Role target must use module:object.")


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    maxsize: int = 1
    max_message_bytes: int = 64 * 2**20

    def __post_init__(self):
        if type(self.maxsize) is not int or self.maxsize != 1:
            raise ValueError("Channel requires maxsize=1.")
        if type(self.max_message_bytes) is not int or self.max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be a positive integer.")


@dataclass(frozen=True)
class TopologySpec:
    run_prefix: str
    roles: tuple[RoleSpec, ...]
    channels: tuple[ChannelSpec, ...]
    rpc_timeout: float = 120.0
    log_dir: str = ""
    ray_tmpdir: str | None = None

    def __post_init__(self):
        names = [spec.name for spec in (*self.roles, *self.channels)]
        if len(set(names)) != len(names):
            raise ValueError("Role and channel names must be unique.")
        if not math.isfinite(self.rpc_timeout) or self.rpc_timeout <= 0:
            raise ValueError("rpc_timeout must be finite and positive.")


class ChannelClient(Protocol):
    def put_nowait(self, name: str, payload: Any) -> None: ...
    def try_get(self, name: str) -> Any: ...
    def recv(self, name: str, *, timeout: float) -> Any: ...


@dataclass(frozen=True)
class RoleContext:
    role_name: str
    rank: int
    world_size: int
    device: str
    channels: ChannelClient


class RoleProtocol(Protocol):
    rpc_methods: ClassVar[tuple[str, ...]]

    def initialize(self, ctx: RoleContext) -> Any: ...
    def close(self) -> None: ...


T = TypeVar("T", covariant=True)


class RpcFuture(Protocol[T]):
    @property
    def deadline(self) -> float: ...
    def done(self) -> bool: ...
    def wait(self, timeout: float | None = None) -> T: ...


class RoleGroupHandle(Protocol):
    def call(self, method: str, *args, timeout: float | None = None, **kwargs) -> RpcFuture: ...


class RuntimeSession(Protocol):
    @property
    def init_results(self) -> Mapping[str, Any]: ...
    def role(self, name: str) -> RoleGroupHandle: ...
    def wait_any(self, futures: Mapping[str, RpcFuture]) -> list[tuple[str, Any]]: ...
    def close(self) -> None: ...


def check_message_bytes(payload: Any, limit: int) -> bytes:
    """Return the checked wire image, including any backing tensor storage."""
    data = pickle.dumps(payload, protocol=5)
    if len(data) > limit:
        raise ValueError(
            f"Channel message exceeds runtime.max_message_bytes: {len(data)} > {limit}"
        )
    return data


def invoke_role(role: RoleProtocol, method: str, args, kwargs):
    if method not in role.rpc_methods:
        raise ValueError(f"Role RPC method {method!r} is not allowed.")
    return getattr(role, method)(*args, **kwargs)


def rpc_deadline(timeout: float) -> float:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("RPC timeout must be finite and positive.")
    return time.monotonic() + timeout


def wait_any(futures: Mapping[str, RpcFuture], poll_interval=0.01) -> list[tuple[str, Any]]:
    """Check the entire set for starvation before harvesting any ready result."""
    while futures:
        now = time.monotonic()
        for name, future in futures.items():
            if now >= future.deadline:
                raise TimeoutError(f"RPC {name!r} exceeded its deadline.")
        ready = [(name, future.wait()) for name, future in futures.items() if future.done()]
        if ready:
            return ready
        remaining = min(f.deadline for f in futures.values()) - time.monotonic()
        time.sleep(max(0, min(poll_interval, remaining)))
    return []
