"""Compile reusable regions and optionally replay the complete Euler sampler."""

from __future__ import annotations

import dataclasses
import functools
import threading
from collections import OrderedDict
from typing import Any

import torch


def _map_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{
            field.name: _map_tensors(getattr(value, field.name), fn)
            for field in dataclasses.fields(value)
        })
    if isinstance(value, dict):
        return {key: _map_tensors(item, fn) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_map_tensors(item, fn) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"Unsupported Pi0.5 graph input: {type(value).__name__}.")


def _signature(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), value.stride(), value.dtype, value.device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple(
            (field.name, _signature(getattr(value, field.name)))
            for field in dataclasses.fields(value)
        ))
    if isinstance(value, dict):
        return (dict, tuple((key, _signature(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(_signature(item) for item in value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return (type(value), value)
    raise TypeError(f"Unsupported Pi0.5 graph input: {type(value).__name__}.")


def _tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        children = (getattr(value, field.name) for field in dataclasses.fields(value))
    elif isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (tuple, list)):
        children = value
    else:
        return []
    return [tensor for child in children for tensor in _tensor_leaves(child)]


def _autocast_signature(device_type):
    enabled = torch.is_autocast_enabled(device_type)
    return (enabled, torch.get_autocast_dtype(device_type) if enabled else None)


@dataclasses.dataclass
class _CapturedSampler:
    graph: Any
    inputs: Any
    output: torch.Tensor
    ready: Any
    # Keep constant modulation storage alive for the lifetime of the graph.
    modulations: Any = None


class TorchPi05Engine:
    """Inference-only accelerator; the core retains ownership of all weights.

    Capture/compilation is lazy, after the model reaches its inference device.
    Graphs are bounded by an LRU of shapes/steps, never by input values. All
    observation, noise and optional memory tensors are refreshed on every call.
    Call ``clear`` before replacing parameters/storage or moving the core.
    """

    def __init__(self, core, *, compile=False, cuda_graph=False,
                 compile_mode="max-autotune-no-cudagraphs", preserve_precision=True,
                 max_graphs=2):
        if max_graphs < 1:
            raise ValueError("Pi0.5 cuda_graph_max_entries must be positive.")
        if compile_mode not in {"legacy", "openpi", "default", "max-autotune-no-cudagraphs"}:
            raise ValueError("Pi0.5 compile_mode must disable compiler-owned CUDA Graphs.")
        if compile and compile_mode in {"legacy", "openpi"} and not preserve_precision:
            raise ValueError("Pi0.5 native compilation requires compile_preserve_precision=true.")
        self.core = core
        self.compile_enabled = compile
        self.cuda_graph_enabled = cuda_graph
        self.compile_mode = compile_mode
        self.preserve_precision = preserve_precision
        self.max_graphs = max_graphs
        self._compiled = None
        self._graphs = OrderedDict()
        self._lock = threading.RLock()
        self.capture_count = 0

    @property
    def metadata(self):
        return {
            "torch_compile": self.compile_enabled,
            "torch_compile_mode": self.compile_mode if self.compile_enabled else None,
            "compile_preserve_precision": self.preserve_precision,
            "cuda_graph": self.cuda_graph_enabled,
            "cuda_graph_entries": len(self._graphs),
            "cuda_graph_captures": self.capture_count,
        }

    def clear(self):
        with self._lock:
            for entry in self._graphs.values():
                entry.ready.synchronize()
            self._graphs.clear()
            self._compiled = None

    def _legacy_ops(self):
        if (not getattr(self.core, "pi05", False)
                or getattr(self.core, "preserve_projection_inputs", False)
                or any(getattr(module, "openpi_math", False)
                       or getattr(module, "memory_modulation", None) is not None
                       for module in self.core.modules())):
            raise ValueError(
                "Pi0.5 legacy compilation requires the baseline legacy "
                "operator profile without memory modules."
            )
        if not hasattr(torch.library, "custom_op"):
            raise RuntimeError(
                "Pi0.5 legacy compilation requires torch.library.custom_op; "
                "install a recent PyTorch release (validated with 2.10)."
            )
        from . import legacy_ops

        return legacy_ops

    def _openpi_ops(self):
        if (not getattr(self.core, "pi05", False)
                or not getattr(self.core, "preserve_projection_inputs", False)):
            raise ValueError("Pi0.5 openpi compilation requires the OpenPI numerical profile.")
        if not hasattr(torch.library, "custom_op"):
            raise RuntimeError(
                "Pi0.5 openpi compilation requires torch.library.custom_op; "
                "install a recent PyTorch release (validated with 2.10)."
            )
        from . import openpi_ops

        return openpi_ops

    def _sample(self, inputs, num_steps, modulations=None):
        observation, noise, model_kwargs = inputs
        callbacks = {}
        if self.compile_enabled:
            if self._compiled is None:
                # No nested compiler-owned graphs: the outer graph owns the
                # static buffers and the complete multi-step sampling lifetime.
                options = {
                    "triton.cudagraphs": False,
                    # Preserve the BF16/FP16 rounding between fused operators.
                    "emulate_precision_casts": self.preserve_precision,
                }
                if self.compile_mode == "max-autotune-no-cudagraphs":
                    options.update(max_autotune=True, coordinate_descent_tuning=True)
                if self.compile_mode in {"legacy", "openpi"}:
                    ops = self._legacy_ops() if self.compile_mode == "legacy" else self._openpi_ops()

                    # Keep native prefix arithmetic. Compile the repeatedly used
                    # action expert, retaining ATen numerical boundaries rather
                    # than letting autotuning replace its GEMMs/reductions.
                    options.update(max_autotune=False)
                    prefix = functools.partial(self.core.build_prefix_cache,
                                               inference_ops=ops)
                    suffix = functools.partial(self.core.run_suffix,
                                               inference_ops=ops)
                    self._compiled = (prefix, torch.compile(
                        suffix, options=options, fullgraph=True, dynamic=False,
                    ))
                else:
                    self._compiled = tuple(torch.compile(
                        fn, options=options, fullgraph=True, dynamic=False,
                    ) for fn in (self.core.build_prefix_cache, self.core.run_suffix))
            callbacks = dict(zip(("_prefix_fn", "_suffix_fn"), self._compiled))
        if modulations is not None:
            suffix = callbacks.get("_suffix_fn", self.core.run_suffix)
            steps = iter(modulations)

            def with_modulations(*args, **kwargs):
                return suffix(*args, cached_modulations=next(steps), **kwargs)

            callbacks["_suffix_fn"] = with_modulations
        if getattr(self.core, "memory_encoder", None) is not None:
            callbacks["_cache_memory"] = True
        return self.core.sample_actions(
            observation, noise=noise, num_steps=num_steps, **model_kwargs, **callbacks,
        )

    @torch.inference_mode()
    def sample_actions(self, observation, *, num_steps=10, noise=None, rng=None,
                       **model_kwargs):
        if num_steps <= 0:
            raise ValueError("SamplingOptions.num_steps must be positive.")
        if not self.compile_enabled and not self.cuda_graph_enabled:
            return self.core.sample_actions(
                observation, num_steps=num_steps, noise=noise, rng=rng, **model_kwargs,
            )
        if self.core.training:
            raise RuntimeError("Pi0.5 inference acceleration requires model.eval().")
        device = observation.state.device
        if self.cuda_graph_enabled and device.type != "cuda":
            raise ValueError("Pi0.5 runtime.cuda_graph requires a CUDA device.")
        # Generate once outside capture. Warmup must not consume extra RNG, and
        # replays must receive fresh noise rather than a captured random sample.
        if noise is None:
            noise = torch.randn(
                observation.state.shape[0], self.core.action_horizon, self.core.action_dim,
                device=device, dtype=getattr(self.core, "sampling_noise_dtype", torch.float32),
                generator=rng,
            )
        inputs = (observation, noise, model_kwargs)
        with self._lock:
            if not self.cuda_graph_enabled:
                return self._sample(inputs, num_steps)
            if any(tensor.device != device for tensor in _tensor_leaves(inputs)):
                raise ValueError("All Pi0.5 CUDA Graph inputs must be on the same CUDA device.")
            # Capture bakes in autocast's operator casts and cached modulations.
            # Identical input dtypes do not imply identical internal arithmetic.
            key = (num_steps, _signature(inputs), _autocast_signature(device.type))
            with torch.cuda.device(device):
                entry = self._graphs.get(key)
                if entry is None:
                    if len(self._graphs) >= self.max_graphs:
                        _, evicted = self._graphs.popitem(last=False)
                        evicted.ready.synchronize()
                        del evicted
                    # AMP's weight cache can retain casts made during warmup
                    # outside the graph pool. Record those casts in the graph
                    # instead, so replay never references a freed cache tensor.
                    with torch.autocast(
                        device.type, enabled=torch.is_autocast_enabled(device.type),
                        dtype=torch.get_autocast_dtype(device.type), cache_enabled=False,
                    ):
                        entry = self._capture(inputs, num_steps, device)
                    self._graphs[key] = entry
                self._graphs.move_to_end(key)
                stream = torch.cuda.current_stream(device)
                stream.wait_event(entry.ready)
                for destination, source in zip(_tensor_leaves(entry.inputs), _tensor_leaves(inputs)):
                    destination.copy_(source)
                entry.graph.replay()
                # A caller may retain its result across subsequent replays.
                output = entry.output.clone()
                entry.ready.record(stream)
                return output

    def _capture(self, inputs, num_steps, device):
        static = _map_tensors(inputs, lambda tensor: tensor.detach().clone())
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        # Compile/autotune and initialize library workspaces before capture.
        with torch.cuda.stream(stream):
            # Every AdaRMSNorm projection depends on t and weights, not the
            # observation or noise. Evaluate with the native training operators
            # once per captured batch/step schedule, outside graph replay.
            modulations = None
            if getattr(self.core, "pi05", False):
                from .conditioning import prepare_modulations

                modulations = prepare_modulations(
                    self.core, static[0], static[1], num_steps,
                )
            for _ in range(3):
                self._sample(static, num_steps, modulations)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = self._sample(static, num_steps, modulations)
        ready = torch.cuda.Event()
        ready.record(stream)
        self.capture_count += 1
        return _CapturedSampler(graph, static, output, ready, modulations)
