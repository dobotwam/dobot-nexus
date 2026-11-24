from __future__ import annotations

import sys
import unittest
from collections import OrderedDict, UserDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, dataclass
from importlib.util import find_spec
from threading import Event, Lock
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dobot_nexus.core.contracts import PolicyOutput, PolicySpec
from dobot_nexus.inference.policy import InferencePolicy
from dobot_nexus.inference.session import SessionOptions


def make_policy(predict):
    model = Mock(policy_spec=PolicySpec("stub", "cpu", ("base",), 1, 1, 2))
    model.predict_action_chunk.side_effect = predict
    with patch("dobot_nexus.inference.policy.build_model", return_value=model):
        return InferencePolicy({"model": {}, "runtime": {"device": "cpu"}})


def observation(state=1):
    return {"images": {"base": np.zeros((2, 2, 3), np.uint8)}, "state": [state], "prompt": "pick"}


def predict_with_aux(aux):
    output = PolicyOutput(np.ones((1, 2, 1)), aux=aux)
    return make_policy(lambda *_: output).predict_chunk(observation()).aux


class PolicySessionTest(unittest.TestCase):
    def test_snapshot_preserves_container_types_aliases_cycles_and_array_flags(self):
        backing = {}
        proxy = MappingProxyType(backing)
        values = []
        nested = (values,)
        values.extend([values, nested])
        data = np.arange(5)
        readonly = data[::-2]
        readonly.setflags(write=False)
        key = (1, "k")
        backing.update({key: [data, readonly], "self": proxy})
        aux = {"proxy": proxy, "data": data, "bytes": bytearray(b"x"), "nested": nested,
               "tags": {1, 2}, "frozen": frozenset({3}), "key": key,
               "scalars": (None, True, 2, 1.5, 3j, "abc", b"ab", np.float32(4))}
        aux["self"] = aux
        saved = predict_with_aux(aux)
        data[:] = 99
        aux["bytes"][0] = 0
        aux["tags"].clear()
        values.clear()
        backing.clear()

        self.assertIs(saved["self"], saved)
        self.assertIs(saved["nested"][0][0], saved["nested"][0])
        self.assertIs(saved["nested"][0][1], saved["nested"])
        self.assertIsInstance(saved["proxy"], MappingProxyType)
        self.assertIs(saved["proxy"]["self"], saved["proxy"])
        self.assertIs(next(iter(saved["proxy"])), saved["key"])
        snapshot, sliced = saved["proxy"][key]
        self.assertIs(snapshot, saved["data"])
        np.testing.assert_array_equal(snapshot, np.arange(5))
        np.testing.assert_array_equal(sliced, [4, 2, 0])
        self.assertTrue(snapshot.flags.writeable)
        self.assertFalse(sliced.flags.writeable)
        for name, expected in (("bytes", bytearray(b"x")), ("tags", {1, 2}),
                               ("frozen", frozenset({3})), ("scalars", aux["scalars"])):
            self.assertIs(type(saved[name]), type(expected))
            self.assertEqual(saved[name], expected)
        self.assertIs(saved["scalars"][-1], aux["scalars"][-1])

    def test_ordered_dict_and_frozen_dataclass_snapshot_their_fields(self):
        @dataclass(frozen=True, slots=True)
        class Record:
            values: np.ndarray
            links: list

        record = Record(np.arange(3), [])
        metadata = OrderedDict([("z", record), ("a", record.values)])
        record.links.append(metadata)
        saved = predict_with_aux({"metadata": metadata})["metadata"]
        record.values[:] = -1
        metadata.clear()
        self.assertIs(type(saved), OrderedDict)
        self.assertEqual(list(saved), ["z", "a"])
        self.assertIs(type(saved["z"]), Record)
        self.assertIs(saved["z"].links[0], saved)
        self.assertIs(saved["z"].values, saved["a"])
        np.testing.assert_array_equal(saved["z"].values, np.arange(3))
        with self.assertRaises(FrozenInstanceError):
            saved["z"].values = None

    def test_unsupported_values_and_keys_raise_type_error(self):
        class Opaque:
            pass

        structured = np.zeros(1, dtype=[("n", "i4")])
        for value in (structured, np.array([object()], dtype=object)):
            with self.subTest(dtype=value.dtype), self.assertRaisesRegex(
                TypeError, "only supports plain non-object NumPy arrays"
            ):
                predict_with_aux({"bad": value})
        for value in (memoryview(b"x"), Opaque(), object(), Lock(), self.fail, structured[0],
                      UserDict(), *(type("Custom", (t,), {})() for t in
                                    (dict, OrderedDict, list, tuple, set, frozenset))):
            with self.subTest(kind=type(value)), self.assertRaisesRegex(
                TypeError, f"Unsupported PolicyOutput.aux value type: {type(value).__name__}"
            ):
                predict_with_aux({"bad": [value]})
        with self.assertRaisesRegex(TypeError, "Unsupported PolicyOutput.aux value type: Opaque"):
            predict_with_aux({Opaque(): "bad key"})

    def test_numpy_dtype_metadata_is_rejected_on_arrays_and_scalars(self):
        array = np.zeros(2, dtype=np.dtype("i4", metadata={"mutable": []}))
        scalar = np.zeros(1, dtype=np.dtype("V4", metadata={"mutable": []}))[0]
        for value in (array, scalar):
            with (self.subTest(kind=type(value)),
                  self.assertRaisesRegex(TypeError, "dtype metadata")):
                predict_with_aux({"value": value})

    def test_numpy_subclasses_are_rejected(self):
        class Scalar(np.float32):
            pass

        class Array(np.ndarray):
            pass

        class Spoofed(np.float32):
            __module__ = "numpy"

            @property
            def dtype(self):
                raise AssertionError("Custom scalar dtype was accessed.")

        for value in (Scalar(2), Spoofed(2), np.zeros(1).view(Array), np.ma.array([1], mask=True)):
            with self.subTest(kind=type(value)), self.assertRaisesRegex(
                TypeError, f"Unsupported PolicyOutput.aux value type: {type(value).__name__}"
            ):
                predict_with_aux({"value": value})

    @unittest.skipUnless(find_spec("torch"), "requires PyTorch")
    def test_quantized_tensor_is_rejected(self):
        import torch

        value = torch.quantize_per_tensor(torch.tensor([1.0, 2.0]), 0.5, 0, torch.qint8)
        with self.assertRaisesRegex(TypeError, "quantized Torch tensors"):
            predict_with_aux({"value": value})

    @unittest.skipUnless(find_spec("torch"), "requires PyTorch")
    def test_cpu_tensor_snapshot_detaches_and_owns_reused_storage(self):
        import torch

        tensor = torch.ones(3, requires_grad=True) * 2
        with patch.object(torch.cuda, "current_stream") as stream:
            saved = predict_with_aux({"a": tensor, "b": tensor})
        stream.assert_not_called()
        tensor.detach().fill_(8)
        self.assertIs(saved["a"], saved["b"])
        self.assertFalse(saved["a"].requires_grad)
        torch.testing.assert_close(saved["a"], torch.full((3,), 2.0))

    def test_cuda_copies_finish_on_each_device_before_unlock_even_on_rejection(self):
        lock = Lock()
        events = []

        class Tensor:
            is_cuda = True

            def __init__(self, device):
                self.device = device

            def detach(self):
                return self

            def clone(self):
                events.append(("clone", self.device, lock.locked()))
                return Tensor(self.device)

        def current_stream(device):
            def synchronize():
                events.append(("sync", device, lock.locked()))
            return SimpleNamespace(synchronize=synchronize)

        first = Tensor("cuda:0")
        aux = [first, first, Tensor("cuda:1"), Tensor("cuda:0")]
        output = PolicyOutput(np.ones((1, 2, 1)), aux={"items": aux})
        policy = make_policy(lambda *_: output)
        torch = SimpleNamespace(Tensor=Tensor, cuda=SimpleNamespace(current_stream=current_stream))
        with patch.dict(sys.modules, torch=torch), patch.object(policy, "_lock", lock):
            for invalid in (False, True):
                events.clear()
                if invalid:
                    aux.append(object())
                    with self.assertRaises(TypeError):
                        policy.predict_chunk(observation())
                else:
                    policy.predict_chunk(observation())
                self.assertEqual(events[:3], [("clone", d, True) for d in
                                              ("cuda:0", "cuda:1", "cuda:0")])
                self.assertCountEqual(events[3:], [("sync", d, True) for d in ("cuda:0", "cuda:1")])
                self.assertFalse(lock.locked())

    def test_two_sessions_own_all_output_fields_before_unlock(self):
        shared = PolicyOutput(np.zeros((1, 2, 1), np.float32), valid_mask=np.zeros((1, 2), bool),
                              aux={"sample": np.zeros(1)})
        outputs = {}

        def predict(obs, options):
            n = int(obs.state[0, 0])
            shared.actions[0, :, 0] = [n, n + 1]
            shared.valid_mask[0] = [n == 1, n == 8]
            shared.timing["ms"] = n
            shared.aux["sample"][0] = n
            return shared

        policy = make_policy(predict)
        a, b = (policy.create_session(SessionOptions(2)) for _ in range(2))
        predict_chunk, lock = policy.predict_chunk, policy._lock
        a_released, b_finished = Event(), Event()

        def capture(obs, options):
            output = predict_chunk(obs, options)
            outputs[int(obs.state[0, 0])] = output
            return output

        class Gate:
            def __enter__(self):
                lock.acquire()

            def __exit__(self, *exc):
                lock.release()
                if not a_released.is_set():
                    a_released.set()
                    if not b_finished.wait(timeout=10):
                        raise TimeoutError("Session B did not finish its prediction.")

        with (patch.object(policy, "_lock", Gate()),
              patch.object(policy, "predict_chunk", side_effect=capture),
              ThreadPoolExecutor(1) as pool):
            first = pool.submit(a.act, observation(1))
            try:
                self.assertTrue(a_released.wait(timeout=10))
                np.testing.assert_array_equal(b.act(observation(8)), [[8]])
            finally:
                b_finished.set()
            np.testing.assert_array_equal(first.result(timeout=10), [[1]])
        for session, n in ((a, 1), (b, 8)):
            output = outputs[n]
            np.testing.assert_array_equal(session.act(observation(n)), [[n + 1]])
            np.testing.assert_array_equal(output.actions.ravel(), [n, n + 1])
            np.testing.assert_array_equal(output.valid_mask[0], [n == 1, n == 8])
            self.assertEqual(output.timing, {"ms": n})
            np.testing.assert_array_equal(output.aux["sample"], [n])
