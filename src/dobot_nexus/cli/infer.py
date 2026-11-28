from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dobot_nexus.core.config import load_config
from dobot_nexus.core.contracts import ObservationBatch, SamplingOptions
from dobot_nexus.data.observation import observation_from_json
from dobot_nexus.data.observation import observation_from_npz as _load_npz_observation
from dobot_nexus.inference import load_policy


def _load_observation(path: str | Path) -> ObservationBatch:
    source = Path(path)
    if source.suffix == ".json":
        return observation_from_json(json.loads(source.read_text(encoding="utf-8")))
    if source.suffix == ".npz":
        return _load_npz_observation(source)
    raise ValueError("Inference input must be .json or .npz.")


def run_inference(argv: Sequence[str] | None, *, load_predictor) -> None:
    parser = argparse.ArgumentParser(description="Run standardized action-chunk inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="Observation .json or .npz file.")
    parser.add_argument("--output", default="actions.npz")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args(argv)
    config = load_config(args.config, list(args.overrides), expected_task="infer")
    policy = load_predictor(config)
    observation = _load_observation(args.input)
    output = policy.predict_chunk(
        observation,
        options=SamplingOptions(
            seed=config.get("inference", {}).get("seed"),
            num_steps=config.get("inference", {}).get("num_steps"),
        ),
    )
    np.savez_compressed(
        args.output,
        actions=output.actions,
        valid_mask=(output.valid_mask if output.valid_mask is not None else np.array([])),
    )
    print(json.dumps({"output": str(Path(args.output).resolve()), "shape": output.actions.shape}))


def main(argv: Sequence[str] | None = None) -> None:
    run_inference(argv, load_predictor=load_policy)


if __name__ == "__main__":
    main()
