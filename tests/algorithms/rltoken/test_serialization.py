import json
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from dobot_nexus.algorithms.rltoken.identity import content_digest
from dobot_nexus.algorithms.rltoken.serialization import load_payload, save_payload
from dobot_nexus.algorithms.rltoken.checkpoint import load_resume
from dobot_nexus.algorithms.rltoken.offline_data import load_cache
from dobot_nexus.algorithms.rltoken.pretraining import read_prewarm
from dobot_nexus.algorithms.rltoken.replay import Replay
from dobot_nexus.algorithms.rltoken.token_pretraining import TokenPretrainTrainer
from dobot_nexus.core.config import clone_config

from .fakes import feature_spec, transition


_callback_calls = []


def _forbidden_callback():
    _callback_calls.append(True)
    raise AssertionError("pickle executed its reduction callback")


class _ExecutableRecord:
    def __reduce__(self):
        return _forbidden_callback, ()


class RestrictedLoadTest(unittest.TestCase):
    def test_resume_journal_trust_does_not_extend_to_other_live_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            live, selected = (Path(directory) / name for name in ("live.sqlite", "selected.sqlite"))
            spec = feature_spec()
            row = transition()
            replay = Replay(live, spec, 2)
            replay.commit(row)
            replay.close()
            selected.write_bytes(live.read_bytes())

            # Same transition, but the live row now needs legacy deserialization.
            legacy = io.BytesIO()
            torch.save(row, legacy)
            connection = sqlite3.connect(live)
            with connection:
                connection.execute("UPDATE events SET payload=?", (legacy.getvalue(),))
            connection.close()
            with self.assertRaisesRegex(ValueError, "Restricted RLToken load"):
                Replay(live, spec, 2, trusted_checkpoint=True, resume_journal=selected)
            selected.write_bytes(live.read_bytes())
            restored = Replay(live, spec, 2, trusted_checkpoint=True, resume_journal=selected)
            self.addCleanup(restored.close)
            self.assertEqual(restored.watermark, 1)

    def test_numeric_arrays_record_identity_and_module_versions_round_trip(self):
        row = transition(executed_commands=np.array([[0.125, 0.5]], dtype=np.float32))
        row.features.action_context = {"__rltoken_record_v1__": "user metadata",
                                       "scale": np.float64(0.125), "offset": np.float32(0.2),
                                       "index": np.int64(7), "enabled": np.bool_(True)}
        weights = torch.nn.Linear(2, 1).state_dict()
        payload = {"row": row, "weights": weights}
        stream = io.BytesIO()
        save_payload(payload, stream)
        stream.seek(0)
        disk = torch.load(stream, map_location="cpu", weights_only=True)
        self.assertIsInstance(disk["row"], dict)
        stream.seek(0)
        loaded = load_payload(stream)
        self.assertEqual(content_digest(loaded), content_digest(payload))
        context = loaded["row"].features.action_context
        self.assertEqual([type(context[k]) for k in ("scale", "offset", "index", "enabled")],
                         [float, float, int, bool])
        self.assertEqual(loaded["weights"]._metadata, weights._metadata)

    def test_all_storage_entry_points_reject_executable_records(self):
        _callback_calls.clear()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "state.pt"
            torch.save(_ExecutableRecord(), state)
            (root / "cache.pt").write_bytes(state.read_bytes())
            (root / "manifest.json").write_text(json.dumps({"format": "rltoken_resume_v1"}))

            journal = root / "journal.sqlite"
            replay = Replay(journal, feature_spec(), 2)
            replay.commit(transition())
            replay.close()
            connection = sqlite3.connect(journal)
            with connection:
                connection.execute("UPDATE events SET payload=?", (state.read_bytes(),))
            connection.close()
            cfg = clone_config({
                "algorithm": {"token": {"train_episodes": [0], "validation_episodes": [1],
                                        "min_zero_ratio": 2, "min_shuffle_ratio": 2}},
                "train": {"max_steps": 1, "save_every": 1, "resume_from": str(state)},
                "model": {"rltoken": {"training_mode": "token_only"}, "target": "fixture:model"},
                "data": {"norm_stats_path": "fixture", "batch_size": 1},
                "exp": {}, "runtime": {"device": "cpu"},
            })

            def load_token_resume():
                trainer = TokenPretrainTrainer(cfg, SimpleNamespace(run_dir=root))
                with patch("dobot_nexus.algorithms.rltoken.token_pretraining.import_from_string") as target:
                    model = target.return_value.from_config.return_value.to.return_value
                    model.get_param_groups.return_value = [torch.nn.Parameter(torch.ones(1))]
                    trainer.run()

            for name, load in (
                ("token resume", load_token_resume),
                ("prewarm", lambda: read_prewarm(state, None, task_identity="fixture")),
                ("cache", lambda: load_cache(root, feature_spec(), chunk_size=2, reward_definition="fixture")),
                ("online resume", lambda: load_resume(root, None, task_fingerprint="fixture")),
                ("journal", lambda: Replay(journal, feature_spec(), 2)),
            ):
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "trusted_checkpoint"):
                        load()
                    self.assertEqual(_callback_calls, [])
