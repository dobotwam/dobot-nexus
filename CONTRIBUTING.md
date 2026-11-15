# Contributing

Read [AGENTS.md](AGENTS.md) for ownership boundaries and commit conventions.
Use Python 3.11 for the CPU reference environment. The tested runtime is Torch 2.10 and
Accelerate 1.14; `requirements-ci.txt` pins direct CPU test/build dependencies. It is a
constraints file, not a complete training-environment lock. GPU drivers, model assets and
optional model dependencies must be recorded separately for training experiments.

```bash
python -m venv .venvs/test
source .venvs/test/bin/activate
python -m pip install -c requirements-ci.txt torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements-ci.txt -e '.[accelerate,serve,websocket,tensorboard]' pandas pyarrow Pillow imageio opencv-python-headless safetensors tqdm sentencepiece einops httpx setuptools wheel
python -m pip check
PYTHONPATH=src:. python -m unittest discover -s tests -v
python -m compileall -q src tests
python scripts/check_wheel.py
```

Add CPU regression coverage for changed contracts. CPU tests validate data, serialization,
state ownership and model assembly; they do not validate GPU numerical parity, throughput,
real cameras, robot motion or emergency stops. Keep GPU and hardware validation results
separate from these checks. Review [SECURITY.md](SECURITY.md) before deploying services or
loading external files. Contributions are distributed under Apache-2.0; preserve source
attribution and all third-party license notices.
