import math
import os
import pickle
import sys
import unittest
from unittest.mock import Mock, patch

import torch

from dobot_nexus.runtime.config import validate_launcher_env, validate_ray_tmpdir
from dobot_nexus.runtime.contracts import (
    ChannelSpec,
    PlacementSpec,
    RoleSpec,
    TopologySpec,
    check_message_bytes,
)
from dobot_nexus.runtime.factory import create_runtime
from dobot_nexus.runtime.sync import SingleInFlightSyncCoordinator


class ContractsTest(unittest.TestCase):
    def test_rlinf_fsdp_cli_preflight_precedes_experiment_creation(self):
        from pathlib import Path
        from dobot_nexus.cli import train

        recipe = Path(__file__).resolve().parents[2] / "examples/configs/rltoken/stage2.yaml"
        with patch.object(train.Experiment, "from_config") as create:
            with self.assertRaisesRegex(ValueError, "FSDP"):
                train.main(["--config", str(recipe), "runtime.fsdp.enabled=true"])
            create.assert_not_called()

    def test_topology_rejects_unsupported_placement_and_lane_shapes(self):
        cpu = PlacementSpec("cpu")
        role = RoleSpec("role", "module:Role", cpu)
        channel = ChannelSpec("messages")
        for construct, args in [
            (PlacementSpec, ("cpu", None, 1)),
            (PlacementSpec, ("mps",)),
            (PlacementSpec, ("cpu", 0)),
            (PlacementSpec, ("gpu",)),
            (PlacementSpec, ("gpu", -1)),
            (PlacementSpec, ("gpu", True)),
            (RoleSpec, ("", "module:Role", cpu)),
            (RoleSpec, ("role", "Role", cpu)),
            (RoleSpec, ("role", "module:Role", cpu, 2)),
            (ChannelSpec, ("lane", 2)),
            (ChannelSpec, ("lane", 1, 0)),
            (ChannelSpec, ("lane", 1, 1.5)),
            (TopologySpec, ("run", (role, role), ())),
            (TopologySpec, ("run", (), (channel, channel))),
            (TopologySpec, ("run", (role,), (ChannelSpec("role"),))),
            *((TopologySpec, ("run", (), (), t)) for t in (0, -1, math.nan, math.inf)),
        ]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                construct(*args)

    def test_message_limit_counts_entire_protocol_five_image(self):
        view = torch.zeros(2**20, dtype=torch.uint8)[:1]
        payload = {"view": view}
        size = len(pickle.dumps(payload, protocol=5))
        self.assertGreater(size, 2**20)
        check_message_bytes(payload, size)
        with self.assertRaisesRegex(ValueError, "message exceeds"):
            check_message_bytes(payload, size - 1)

    def test_launcher_preflight_and_factory(self):
        config = {"runtime": {"launcher": "rlinf"}, "algorithm": {"target": "m:C"}}
        with patch.dict(os.environ, {}, clear=True):
            validate_launcher_env(config)
            with self.assertRaisesRegex(ValueError, "requires algorithm.target"):
                validate_launcher_env({"runtime": config["runtime"], "algorithm": "bc"})
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
                with patch.dict(os.environ, {key: "0"}):
                    with self.assertRaisesRegex(ValueError, "do not launch it with torchrun"):
                        validate_launcher_env(config)
                    validate_launcher_env({"runtime": {"launcher": "local"}})
        with self.assertRaisesRegex(ValueError, "launcher=rlinf"):
            create_runtime({"runtime": {"launcher": "local"}})

    def test_missing_backend_names_both_install_paths(self):
        with patch.dict(sys.modules, {"ray": None}), self.assertRaises(ImportError) as failure:
            create_runtime({"runtime": {"launcher": "rlinf"}})
        self.assertIn("src/dobot_nexus/runtime/ray/requirements.txt", str(failure.exception))
        self.assertIn("algorithms/rltoken/requirements-rlinf.txt", str(failure.exception))

    def test_socket_budget_in_utf8_includes_ray_session_and_pid(self):
        with patch("os.getpid", return_value=1234):
            validate_ray_tmpdir("/" + "a" * 37)
            with self.assertRaisesRegex(ValueError, "AF_UNIX"):
                validate_ray_tmpdir("/" + "é" * 19)
        with patch("os.getpid", return_value=1234567), self.assertRaises(ValueError):
            validate_ray_tmpdir("/" + "a" * 37)


class SyncTest(unittest.TestCase):
    def setUp(self):
        self.sync = SingleInFlightSyncCoordinator()
        self.recv = Mock(deadline=math.inf)
        self.send = Mock(deadline=math.inf)
        self.events = []
        self.recv.wait.side_effect = lambda: self.events.append("ack") or 1
        self.send.wait.side_effect = lambda: self.events.append("sent") or 2

    def receive(self):
        self.events.append("receive")
        return self.recv

    def send_weights(self):
        self.events.append("send")
        return self.send

    def test_submits_both_before_wait_and_coalesces_without_retrigger(self):
        for future in (self.recv, self.send):
            future.done.return_value = False
        self.sync.sync_async(self.receive, self.send_weights)
        self.recv.done.return_value = True
        self.sync.sync_async(self.receive, self.send_weights)
        self.assertEqual(self.events, ["receive", "send", "ack"])
        self.send.done.return_value = True
        self.sync.sync_async(self.receive, self.send_weights)
        self.assertEqual(self.events, ["receive", "send", "ack", "sent", "receive", "send"])
        self.assertEqual(self.sync.drain(), (1, 2))
        self.assertIsNone(self.sync.drain())
        self.assertEqual(self.sync.last_result, (1, 2))
        self.assertFalse(self.sync.failed)

    def test_each_failed_endpoint_is_reported_while_peer_is_pending(self):
        for failed, pending in ((self.recv, self.send), (self.send, self.recv)):
            with self.subTest(failed=failed):
                sync = SingleInFlightSyncCoordinator()
                failed.done.return_value = True
                failed.wait.side_effect = ValueError("endpoint failed")
                pending.done.return_value = False
                pending.wait.reset_mock()
                sync.sync_async(self.receive, self.send_weights)
                with self.assertRaisesRegex(ValueError, "endpoint failed"):
                    sync.poll()
                self.assertTrue(sync.failed)
                pending.wait.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, "previous synchronization failed"):
                    sync.drain()

    def test_blocking_drain_propagates_sender_failure_while_receiver_never_completes(self):
        self.recv.done.return_value = False
        self.recv.wait.side_effect = AssertionError("waited on pending receive")
        self.send.done.side_effect = [False, True]
        self.send.wait.side_effect = ValueError("failed sender")
        self.sync.sync_async(self.receive, self.send_weights)
        with self.assertRaisesRegex(ValueError, "failed sender"):
            self.sync.drain()
        self.recv.wait.assert_not_called()
        self.assertTrue(self.sync.failed)

    def test_validator_belongs_to_submitted_pair_and_partial_wait_is_cached(self):
        for harvest in ("poll", "drain"):
            sync = SingleInFlightSyncCoordinator()
            validate = Mock(side_effect=ValueError("stale pair"))
            skipped = Mock()
            self.recv.done.return_value = False
            self.send.done.return_value = True
            self.send.wait.reset_mock()
            sync.sync_async(self.receive, self.send_weights, validator=validate)
            sync.sync_async(self.receive, self.send_weights, validator=skipped)
            self.send.wait.assert_called_once_with()
            self.recv.done.return_value = True
            with self.assertRaisesRegex(ValueError, "stale pair"):
                getattr(sync, harvest)()
            self.send.wait.assert_called_once_with()
            validate.assert_called_once_with(1, 2)
            skipped.assert_not_called()
            self.assertTrue(sync.failed)
            self.assertIsNone(sync.last_result)
            with self.assertRaises(RuntimeError):
                sync.sync_blocking(self.receive, self.send_weights)

    def test_second_submit_and_either_wait_failure_poison_barrier(self):
        for failure in ("submit", "send", "recv"):
            with self.subTest(failure=failure):
                sync = SingleInFlightSyncCoordinator()
                recv, send = Mock(deadline=math.inf), Mock(deadline=math.inf)
                submit = Mock(return_value=send)
                if failure == "submit":
                    submit.side_effect = RuntimeError("submit")
                else:
                    (send if failure == "send" else recv).wait.side_effect = TimeoutError("ack")
                with self.assertRaises((RuntimeError, TimeoutError)):
                    sync.sync_blocking(Mock(return_value=recv), submit)
                self.assertTrue(sync.failed)
                self.assertIsNone(sync.last_result)
                with self.assertRaisesRegex(RuntimeError, "previous synchronization failed"):
                    sync.sync_blocking(self.receive, self.send_weights)
                with self.assertRaises(RuntimeError):
                    sync.drain()
