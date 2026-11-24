from __future__ import annotations

import os
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch
from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

try:
    from accelerate.utils import DistributedType
except ImportError:
    DistributedType = None

from dobot_nexus.core.config import Config
from dobot_nexus.core.contracts import TrainOutput
from dobot_nexus.training.bc_trainer import _build_scheduler
from dobot_nexus.training.runtime import (
    AccelerateRuntime,
    LocalRuntime,
    _accelerate_precision,
    _build_fsdp_plugin,
    _register_train_output_pytree,
    build_runtime,
)


def _config(*, fsdp: bool) -> Config:
    runtime = Config({"precision": "bf16"})
    if fsdp:
        runtime.fsdp = Config(
            {
                "enabled": True,
                "version": 1,
                "sharding_strategy": "no_shard",
                "param_dtype": "bf16",
                "reduce_dtype": "fp32",
                "buffer_dtype": "fp32",
                "use_orig_params": False,
                "sync_module_states": True,
            }
        )
    return Config({"runtime": runtime, "train": Config({"grad_accum_steps": 1})})


class RuntimeTest(unittest.TestCase):
    def test_local_autocast_off_disables_inherited_context_without_casting_parameters(self):
        config = Config(runtime=Config(device="cpu", precision="bf16", autocast="off"),
                        train=Config())
        runtime = LocalRuntime(config)
        model = torch.nn.Linear(4, 4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            with runtime.autocast():
                self.assertFalse(torch.is_autocast_enabled("cpu"))
                self.assertEqual(model(torch.ones(1, 4)).dtype, torch.float32)
            self.assertTrue(torch.is_autocast_enabled("cpu"))
        self.assertEqual(model.weight.dtype, torch.float32)
        self.assertFalse(runtime.autocast_enabled)
        config.runtime.device = "cuda"
        config.runtime.autocast = "auto"
        runtime = LocalRuntime(config)
        self.assertTrue(runtime.autocast_enabled)
        with mock.patch("torch.autocast") as autocast:
            runtime.autocast()
        autocast.assert_called_once_with(device_type="cuda", dtype=torch.bfloat16)

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_non_fsdp_accelerate_can_explicitly_disable_autocast(self):
        config = _config(fsdp=False)
        config.runtime.autocast = "off"
        with mock.patch("accelerate.Accelerator", return_value=SimpleNamespace(
            device=torch.device("cpu"),
        )) as constructor:
            runtime = AccelerateRuntime(config)
        self.assertEqual(constructor.call_args.kwargs["mixed_precision"], "no")
        with torch.autocast("cpu", dtype=torch.bfloat16), runtime.autocast():
            self.assertFalse(torch.is_autocast_enabled("cpu"))

    def test_build_runtime_persists_resolved_strategy_before_checkpoint_fingerprint(self):
        from dobot_nexus.core.artifacts import resume_fingerprint
        from dobot_nexus.core.config import clone_config

        for count, launcher, fsdp in ((1, "local", False), (2, "accelerate", True)):
            config = _config(fsdp=True)
            config.train.max_steps = 2
            config.runtime.launcher = "auto"
            config.runtime.fsdp.enabled = "auto"
            expected = clone_config(config)
            expected.runtime.launcher = launcher
            expected.runtime.fsdp.enabled = fsdp
            with (
                self.subTest(count=count),
                mock.patch.dict(os.environ, {"WORLD_SIZE": str(count)}),
                mock.patch("dobot_nexus.training.runtime.LocalRuntime") as local,
                mock.patch("dobot_nexus.training.runtime.AccelerateRuntime") as accelerate,
            ):
                runtime = build_runtime(config)
                selected, unused = (local, accelerate) if count == 1 else (accelerate, local)
                self.assertIs(runtime, selected.return_value)
                selected.assert_called_once_with(config)
                unused.assert_not_called()
                self.assertEqual(config.runtime, expected.runtime)
                self.assertEqual(resume_fingerprint(config), resume_fingerprint(expected))

    def setUp(self):
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("FSDP_VERSION", None)

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_scheduler_tracks_updates_independently_of_world_size(self):
        from accelerate.scheduler import AcceleratedScheduler

        config = _config(fsdp=False)
        config.scheduler = Config({"name": "warmup_cosine", "warmup_steps": 3, "total_steps": 8})
        config.train.max_steps = 8
        with mock.patch(
            "accelerate.Accelerator", return_value=SimpleNamespace(device="cpu")
        ) as constructor:
            AccelerateRuntime(config)
        step_with_optimizer = constructor.call_args.kwargs.get(
            "step_scheduler_with_optimizer", True
        )
        self.assertEqual(constructor.call_args.kwargs.get("rng_types"), ["generator"])
        reference = None
        for world_size in (1, 2, 8):
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.AdamW([parameter], lr=0.1)
            scheduler = _build_scheduler(config, optimizer)
            wrapped = AcceleratedScheduler(
                scheduler,
                SimpleNamespace(step_was_skipped=False),
                split_batches=False,
                step_with_optimizer=step_with_optimizer,
            )
            lrs = [optimizer.param_groups[0]["lr"]]
            with mock.patch(
                "accelerate.scheduler.AcceleratorState",
                return_value=SimpleNamespace(num_processes=world_size),
            ):
                for step in range(1, 9):
                    parameter.grad = torch.ones_like(parameter)
                    optimizer.step()
                    wrapped.step()
                    self.assertEqual(scheduler.last_epoch, step)
                    lrs.append(optimizer.param_groups[0]["lr"])
            if reference is None:
                reference = lrs
            self.assertEqual(lrs, reference)
        self.assertAlmostEqual(reference[0], 0.1 / 3)

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_accumulation_carries_across_short_epochs(self):
        # Isolate Accelerate's process-wide state from other precision tests.
        import subprocess
        import sys
        import textwrap

        script = textwrap.dedent("""
            import torch
            from dobot_nexus.core.config import Config
            from dobot_nexus.training.runtime import AccelerateRuntime, LocalRuntime
            config = Config(runtime=Config(device='cpu', precision='fp32'),
                            train=Config(grad_accum_steps=3))
            results = []
            for runtime_type in (LocalRuntime, AccelerateRuntime):
                runtime = runtime_type(config)
                model = torch.nn.Linear(1, 1, bias=False)
                optimizer = torch.optim.SGD(model.parameters(), lr=0)
                loader = torch.utils.data.DataLoader(torch.ones(5, 1), batch_size=1)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
                model, optimizer, loader, scheduler = runtime.prepare(model, optimizer, loader, scheduler)
                updates = []
                for epoch in range(2):
                    for index, batch in enumerate(loader):
                        with runtime.accumulate(model):
                            runtime.backward(model(batch).sum())
                            if runtime.sync_gradients:
                                grad = runtime.unwrap_model(model).weight.grad.item()
                                updates.append((epoch * 5 + index + 1, round(grad, 6)))
                                optimizer.step()
                                optimizer.zero_grad()
                results.append(updates)
                runtime.end_training()
            assert results == [[(3, 1.0), (6, 1.0), (9, 1.0)]] * 2, results
        """)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "ACCELERATE_USE_CPU": "true"},
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_runtime_exposes_optimizer_skip_status(self):
        config = Config({"runtime": Config({"device": "cpu"}), "train": Config({})})
        self.assertFalse(LocalRuntime(config).optimizer_step_was_skipped)
        runtime = AccelerateRuntime.__new__(AccelerateRuntime)
        runtime.accelerator = SimpleNamespace(optimizer_step_was_skipped=True)
        self.assertTrue(runtime.optimizer_step_was_skipped)

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_non_fsdp_accelerate_models_keep_global_autocast(self) -> None:
        fake = SimpleNamespace(device=torch.device("cpu"))
        with mock.patch("accelerate.Accelerator", return_value=fake) as constructor:
            runtime = AccelerateRuntime(_config(fsdp=False))
        self.assertIs(runtime.accelerator, fake)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["mixed_precision"], "bf16")
        self.assertIsNone(kwargs["fsdp_plugin"])
        self.assertNotIn("kwargs_handlers", kwargs)

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_distributed_timeout_is_passed_to_process_group_initialization(self):
        config = _config(fsdp=False)
        config.runtime.distributed_timeout_seconds = 7200
        with mock.patch("accelerate.Accelerator", return_value=SimpleNamespace(device="cpu")) as build:
            AccelerateRuntime(config)
        handlers = build.call_args.kwargs["kwargs_handlers"]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0].timeout, timedelta(seconds=7200))

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_distributed_timeout_rejects_invalid_values_before_accelerator_creation(self):
        for value in (0, -1, 0.5, True, "7200"):
            with self.subTest(value=value), mock.patch("accelerate.Accelerator") as build:
                config = _config(fsdp=False)
                config.runtime.distributed_timeout_seconds = value
                with self.assertRaisesRegex(ValueError, "distributed_timeout_seconds"):
                    AccelerateRuntime(config)
                build.assert_not_called()

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_pi_fsdp_uses_explicit_mixed_precision_without_amp(self) -> None:
        fake = SimpleNamespace(
            device=torch.device("cpu"),
            distributed_type=DistributedType.FSDP,
            state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=1)),
        )
        # Accelerate 1.14 resolves the FSDP sync device while constructing the
        # plugin. Mock that hardware probe so this remains a CPU configuration
        # test; the GPU integration smoke covers the real device path.
        with (
            mock.patch("accelerate.utils.dataclasses.is_cuda_available", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("accelerate.Accelerator", return_value=fake) as constructor,
        ):
            AccelerateRuntime(_config(fsdp=True))
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["mixed_precision"], "no")
        plugin = kwargs["fsdp_plugin"]
        self.assertEqual(plugin.fsdp_version, 1)
        self.assertEqual(plugin.sharding_strategy, ShardingStrategy.NO_SHARD)
        self.assertEqual(plugin.state_dict_type, StateDictType.FULL_STATE_DICT)
        self.assertEqual(plugin.mixed_precision_policy.param_dtype, torch.bfloat16)
        self.assertEqual(plugin.mixed_precision_policy.reduce_dtype, torch.float32)
        self.assertEqual(plugin.mixed_precision_policy.buffer_dtype, torch.float32)
        self.assertFalse(plugin.use_orig_params)

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_full_shard_exposes_accelerate_wrap_and_prefetch_controls(self) -> None:
        with (
            mock.patch("accelerate.utils.dataclasses.is_cuda_available", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=0),
        ):
            plugin = _build_fsdp_plugin(
                Config(
                    {
                        "version": 1,
                        "sharding_strategy": "full_shard",
                        "auto_wrap_policy": "transformer_based_wrap",
                        "transformer_layer_cls_to_wrap": ["Block", "Encoder1DBlock"],
                        "backward_prefetch": "backward_pre",
                        "cpu_offload": False,
                        "use_orig_params": True,
                        "limit_all_gathers": True,
                    }
                )
            )
        self.assertEqual(plugin.sharding_strategy, ShardingStrategy.FULL_SHARD)
        self.assertIs(plugin.auto_wrap_policy, transformer_auto_wrap_policy)
        self.assertEqual(
            plugin.transformer_cls_names_to_wrap,
            ["Block", "Encoder1DBlock"],
        )
        self.assertEqual(plugin.backward_prefetch, BackwardPrefetch.BACKWARD_PRE)
        self.assertFalse(plugin.cpu_offload.offload_params)
        self.assertTrue(plugin.use_orig_params)
        self.assertTrue(plugin.limit_all_gathers)

    def test_orig_params_uses_joint_accelerate_prepare(self) -> None:
        runtime = AccelerateRuntime.__new__(AccelerateRuntime)
        runtime.fsdp_enabled = True
        runtime.fsdp_version = 1
        runtime.fsdp_use_orig_params = True
        runtime.accelerator = mock.Mock()
        prepared = tuple(object() for _ in range(4))
        runtime.accelerator.prepare.return_value = prepared
        model, data_loader, scheduler = (object() for _ in range(3))
        optimizer = SimpleNamespace(state={})

        result = runtime.prepare(model, optimizer, data_loader, scheduler)

        self.assertIs(result, prepared)
        runtime.accelerator.prepare.assert_called_once_with(
            model, optimizer, data_loader, scheduler
        )

    def test_precision_aliases_are_consistent(self) -> None:
        self.assertEqual(_accelerate_precision("bf16-mixed"), "bf16")
        self.assertEqual(_accelerate_precision("16-mixed"), "fp16")
        self.assertEqual(_accelerate_precision("32-true"), "no")

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_unknown_fsdp_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 1 or 2"):
            _build_fsdp_plugin(Config({"version": 3}))

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_launch_version_conflict_is_rejected_before_plugin_creation(self):
        for configured, launched in ((1, 2), (2, 1)):
            with (
                self.subTest(configured=configured, launched=launched),
                mock.patch.dict(os.environ, {"FSDP_VERSION": str(launched)}),
                self.assertRaisesRegex(ValueError, "conflicts with runtime.fsdp.version"),
            ):
                _build_fsdp_plugin(Config({"version": configured}))

    @unittest.skipIf(DistributedType is None, "accelerate is not installed")
    def test_existing_accelerate_state_cannot_silently_select_another_version(self):
        config = _config(fsdp=True)
        fake = SimpleNamespace(
            device=torch.device("cpu"), distributed_type=DistributedType.FSDP,
            state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=2)),
        )
        with (
            mock.patch("dobot_nexus.training.runtime._build_fsdp_plugin",
                       return_value=SimpleNamespace(use_orig_params=True)),
            mock.patch("accelerate.Accelerator", return_value=fake),
            self.assertRaisesRegex(RuntimeError, "selected FSDP2.*requested FSDP1"),
        ):
            AccelerateRuntime(config)


@unittest.skipUnless(
    DistributedType is not None and hasattr(torch.distributed.fsdp, "MixedPrecisionPolicy"),
    "requires Accelerate and the public PyTorch FSDP2 API",
)
class FSDP2ConfigurationTest(unittest.TestCase):
    def setUp(self):
        environment = mock.patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("FSDP_VERSION", None)

    def test_policy_and_strategy_conversion_preserve_pi_precision(self):
        from torch.distributed.fsdp import MixedPrecisionPolicy

        for strategy, reshard in (("full_shard", True), ("shard_grad_op", False)):
            with self.subTest(strategy=strategy):
                config = Config({
                    "version": 2, "sharding_strategy": strategy,
                    "auto_wrap_policy": "transformer_based_wrap",
                    "transformer_layer_cls_to_wrap": ["Block", "Encoder1DBlock"],
                    "use_orig_params": True, "sync_module_states": True,
                    "backward_prefetch": "backward_pre", "limit_all_gathers": True,
                    "forward_prefetch": False, "buffer_dtype": "fp32",
                })
                with self.assertLogs("dobot_nexus.training.runtime", level="WARNING"):
                    plugin = _build_fsdp_plugin(config)
                self.assertEqual(plugin.fsdp_version, 2)
                self.assertIs(plugin.reshard_after_forward, reshard)
                self.assertIsInstance(plugin.mixed_precision_policy, MixedPrecisionPolicy)
                self.assertEqual(plugin.mixed_precision_policy.param_dtype, torch.bfloat16)
                self.assertEqual(plugin.mixed_precision_policy.reduce_dtype, torch.float32)
                self.assertIsNone(plugin.mixed_precision_policy.output_dtype)
                self.assertIs(plugin.auto_wrap_policy, transformer_auto_wrap_policy)
                self.assertEqual(plugin.transformer_cls_names_to_wrap, ["Block", "Encoder1DBlock"])
                for field in ("use_orig_params", "sync_module_states", "forward_prefetch",
                              "backward_prefetch", "sharding_strategy"):
                    self.assertIsNone(getattr(plugin, field), field)
                self.assertFalse(plugin.activation_checkpointing)
                self.assertFalse(plugin.cpu_ram_efficient_loading)

    def test_defaults_and_explicit_reshard(self):
        plugin = _build_fsdp_plugin(Config({}))
        self.assertEqual(plugin.fsdp_version, 2)
        self.assertTrue(plugin.reshard_after_forward)
        self.assertTrue(_build_fsdp_plugin(Config({"version": 2})).reshard_after_forward)
        plugin = _build_fsdp_plugin(Config({"version": 2, "reshard_after_forward": False}))
        self.assertFalse(plugin.reshard_after_forward)

    def test_omitted_version_is_persisted_for_resume_and_artifacts(self):
        from dobot_nexus.core.artifacts import resume_fingerprint

        config = _config(fsdp=True)
        config.runtime.fsdp.pop("version")
        config.runtime.fsdp.sharding_strategy = "full_shard"
        config.scheduler = Config({"name": "constant"})
        unversioned_fingerprint = resume_fingerprint(config)
        fake = SimpleNamespace(
            device=torch.device("cpu"), distributed_type=DistributedType.FSDP,
            state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=2)),
        )
        with mock.patch("accelerate.Accelerator", return_value=fake) as constructor:
            runtime = AccelerateRuntime(config)
        self.assertEqual(runtime.fsdp_version, 2)
        self.assertEqual(config.runtime.fsdp.version, 2)
        self.assertEqual(constructor.call_args.kwargs["fsdp_plugin"].fsdp_version, 2)
        fingerprint = resume_fingerprint(config)
        self.assertNotEqual(fingerprint, unversioned_fingerprint)
        config.runtime.fsdp.version = 1
        self.assertNotEqual(fingerprint, resume_fingerprint(config))

    def test_enabled_training_templates_select_fsdp2(self):
        from pathlib import Path

        import yaml

        root = Path(__file__).resolve().parents[1] / "examples/configs/train"
        enabled = []
        for path in sorted(root.glob("*.yaml")):
            config = yaml.safe_load(path.read_text())
            fsdp = config.get("runtime", {}).get("fsdp", {})
            if fsdp.get("enabled"):
                with self.subTest(path=path.name):
                    self.assertEqual(fsdp["version"], 2)
                    self.assertEqual(_build_fsdp_plugin(Config(fsdp)).fsdp_version, 2)
                enabled.append(path.name)
        self.assertIn("pi0.yaml", enabled)
        self.assertIn("pi05.yaml", enabled)

    def test_unsupported_or_conflicting_settings_fail(self):
        cases = (
            ({"sharding_strategy": "no_shard"}, "does not support"),
            ({"sharding_strategy": "hybrid_shard"}, "does not support"),
            ({"sharding_strategy": "full_shard", "reshard_after_forward": False}, "conflicts"),
            ({"reshard_after_forward": 1}, "must be a boolean"),
            ({"forward_prefetch": True}, "forward_prefetch=true"),
            ({"backward_prefetch": "backward_post"}, "backward_post"),
            ({"buffer_dtype": "int8"}, "buffer_dtype"),
        )
        for fields, message in cases:
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, message):
                _build_fsdp_plugin(Config({"version": 2, **fields}))

    def test_missing_joint_prepare_api_has_actionable_error(self):
        with (
            mock.patch("accelerate.utils.fsdp_utils.fsdp2_prepare_model", None),
            self.assertRaisesRegex(ImportError, "torch==2.10.0 and accelerate==1.14.0"),
        ):
            _build_fsdp_plugin(Config({"version": 2}))

    def test_old_plugin_has_actionable_error(self):
        class OldPlugin:
            def __init__(self, sharding_strategy=None):
                pass

        with (
            mock.patch("accelerate.utils.FullyShardedDataParallelPlugin", OldPlugin),
            self.assertRaisesRegex(ImportError, "joint model/optimizer preparation"),
        ):
            _build_fsdp_plugin(Config({"version": 2}))

    def test_runtime_records_version_without_global_amp(self):
        config = _config(fsdp=True)
        config.runtime.fsdp.version = 2
        config.runtime.fsdp.sharding_strategy = "full_shard"
        fake = SimpleNamespace(
            device=torch.device("cpu"), distributed_type=DistributedType.FSDP,
            state=SimpleNamespace(fsdp_plugin=SimpleNamespace(fsdp_version=2)),
        )
        with mock.patch("accelerate.Accelerator", return_value=fake) as constructor:
            runtime = AccelerateRuntime(config)
        self.assertEqual(runtime.fsdp_version, 2)
        self.assertEqual(constructor.call_args.kwargs["mixed_precision"], "no")
        self.assertFalse(constructor.call_args.kwargs["step_scheduler_with_optimizer"])
        self.assertEqual(constructor.call_args.kwargs["rng_types"], ["generator"])


class FSDP2PreparationTest(unittest.TestCase):
    def _runtime(self):
        runtime = AccelerateRuntime.__new__(AccelerateRuntime)
        runtime.fsdp_enabled = True
        runtime.fsdp_version = 2
        # The v2 plugin sets this to None/False; it must not select v1's flat path.
        runtime.fsdp_use_orig_params = False
        runtime.fsdp_buffer_dtype = torch.float32
        runtime.accelerator = mock.Mock()
        runtime.accelerator.prepare.side_effect = lambda *args: args
        return runtime

    def test_joint_prepare_keeps_groups_and_frozen_parameters(self):
        runtime = self._runtime()
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        model[0].bias.requires_grad_(False)
        model.register_buffer("float_buffer", torch.ones(2))
        model.register_buffer("counter", torch.tensor(0, dtype=torch.int64))
        optimizer = torch.optim.AdamW([
            {"params": [model[0].weight], "lr": 0.01, "weight_decay": 0.1},
            {"params": list(model[1].parameters()), "lr": 0.02, "weight_decay": 0.0},
        ])
        before = [([id(p) for p in g["params"]], g["lr"], g["weight_decay"])
                  for g in optimizer.param_groups]
        loader, scheduler = object(), object()
        result = runtime.prepare(model, optimizer, loader, scheduler)
        self.assertEqual(result, (model, optimizer, loader, scheduler))
        runtime.accelerator.prepare.assert_called_once_with(model, optimizer, loader, scheduler)
        runtime.accelerator.prepare_model.assert_not_called()
        self.assertEqual(before, [([id(p) for p in g["params"]], g["lr"], g["weight_decay"])
                                  for g in optimizer.param_groups])
        self.assertFalse(model[0].bias.requires_grad)

    def test_buffer_mismatch_fails_before_any_parameter_mutation(self):
        runtime = self._runtime()
        model = torch.nn.Linear(2, 2)
        model.register_buffer("running_scale", torch.ones(2, dtype=torch.bfloat16))
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError, "running_scale.*torch.bfloat16"):
            runtime.prepare(model, optimizer, object(), object())
        runtime.accelerator.prepare.assert_not_called()
        self.assertEqual(model.running_scale.dtype, torch.bfloat16)

    def test_initialized_optimizer_is_rejected(self):
        runtime = self._runtime()
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        with self.assertRaisesRegex(RuntimeError, "before its first update"):
            runtime.prepare(model, optimizer, object(), object())
        runtime.accelerator.prepare.assert_not_called()

    def test_optimizer_full_state_uses_dcp_and_unwraps_optimizer(self):
        runtime = self._runtime()
        model, optimizer, state = object(), object(), {"state": {}, "param_groups": []}
        with mock.patch(
            "torch.distributed.checkpoint.state_dict.get_optimizer_state_dict", return_value=state
        ) as get_state:
            result = runtime.optimizer_state_dict(model, SimpleNamespace(optimizer=optimizer))
        self.assertIs(result, state)
        self.assertEqual(get_state.call_args.args, (model, optimizer))
        options = get_state.call_args.kwargs["options"]
        self.assertTrue(options.full_state_dict)
        self.assertTrue(options.cpu_offload)

    def test_train_output_pytree_preserves_all_fields_and_autograd(self):
        from torch.utils import _pytree

        _register_train_output_pytree()
        _register_train_output_pytree()
        parameter = torch.tensor(2.0, requires_grad=True)
        output = TrainOutput(parameter.square(), {"loss": parameter.detach()},
                             {"nested": [torch.ones(2), "label"], "none": None})
        leaves, spec = _pytree.tree_flatten(output)
        self.assertIs(leaves[0], output.loss)
        restored = _pytree.tree_unflatten(leaves, spec)
        self.assertIsInstance(restored, TrainOutput)
        self.assertIs(restored.loss, output.loss)
        self.assertIs(restored.metrics["loss"], output.metrics["loss"])
        self.assertIs(restored.aux["nested"][0], output.aux["nested"][0])
        self.assertEqual(restored.aux["nested"][1], "label")
        self.assertIsNone(restored.aux["none"])
        paths, _ = _pytree.tree_flatten_with_path(output)
        self.assertEqual(paths[0][0], (_pytree.GetAttrKey("loss"),))
        restored.loss.backward()
        self.assertEqual(parameter.grad.item(), 4.0)

    @unittest.skipUnless(hasattr(torch.distributed.fsdp, "FSDPModule"), "requires FSDP2")
    def test_real_fsdp2_pre_backward_hook_discovers_train_output(self):
        from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

        _register_train_output_pytree()
        hook = mock.Mock(side_effect=lambda gradient: gradient)
        state = SimpleNamespace(_pre_backward=hook)
        for _ in range(4):
            parameter = torch.tensor(2.0, requires_grad=True)
            output = TrainOutput(parameter.square(), {"loss": parameter.detach()})
            self.assertIs(FSDPState._register_pre_backward_hook(state, output), output)
            output.loss.backward()
            self.assertEqual(parameter.grad.item(), 4.0)
        self.assertEqual(hook.call_count, 4)


if __name__ == "__main__":
    unittest.main()
