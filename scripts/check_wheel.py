#!/usr/bin/env python3
"""Build, install and smoke-test a wheel outside its source checkout."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='nexus-wheel-') as directory:
        workspace = Path(directory)
        wheel_dir = workspace/'dist'
        wheel_dir.mkdir()
        source = workspace/'source'
        source.mkdir()
        for name in ('pyproject.toml', 'README.md', 'LICENSE', 'NOTICE'):
            if (root/name).is_file():
                shutil.copy2(root/name, source/name)
        shutil.copytree(root/'src', source/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        # Build in an isolated source tree so old build/lib cannot leak removed modules.
        # CI pins the installed backend in its constraints.
        subprocess.run([sys.executable, '-c',
                        'from setuptools.build_meta import build_wheel; build_wheel('+repr(str(wheel_dir))+')'],
                       cwd=source, check=True, stdout=subprocess.PIPE, text=True)
        wheel = next(wheel_dir.glob('*.whl'))
        with zipfile.ZipFile(wheel) as archive:
            entries = archive.namelist()
            assert not any('/models/act/' in p or '/models/groot/' in p for p in entries)
            assert any(p.endswith('.dist-info/licenses/LICENSE') for p in entries)
            assert any(p.endswith('.dist-info/licenses/NOTICE') for p in entries)
            recap_notice = 'dobot_nexus/algorithms/recap/NOTICE.md'
            assert recap_notice in entries
            assert b'Copyright 2026 Dobot.' in archive.read(recap_notice)
        installed = workspace/'installed'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-compile',
                        '--target', str(installed), str(wheel)], check=True, stdout=subprocess.PIPE,
                       text=True)
        probe = r'''
import importlib.metadata, json, pathlib, sys, tempfile
from unittest.mock import patch
sys.path.insert(0, INSTALLED)
import dobot_nexus
from dobot_nexus.models.pi05.deploy.server import _load_policy, parse_args
assert pathlib.Path(dobot_nexus.__file__).is_relative_to(INSTALLED)
assert importlib.metadata.version('dobot-nexus') == dobot_nexus.__version__
assert importlib.metadata.metadata('dobot-nexus')['License-Expression'] == 'Apache-2.0'
for target in ('dobot_nexus.models.pi05.modeling_pi05:Pi05Model',
               'dobot_nexus.models.pi05.modeling_recap:Pi05ReCapModel',
               'dobot_nexus.models.pi05.memory.modeling_memory:Pi05MemoryModel'):
    with tempfile.TemporaryDirectory() as artifact:
        pathlib.Path(artifact, 'manifest.yaml').write_text(
            'schema_version: 2\nmodel:\n  target: '+target+'\n')
        args = parse_args(['--artifact', artifact, '--device', 'cpu'])
        with patch('dobot_nexus.inference.load_policy', side_effect=lambda config: config):
            config = _load_policy(args)
        assert config.model.target == target
        assert config.model.artifact == artifact
print(json.dumps({'version': dobot_nexus.__version__, 'profiles_loaded': 3,
                  'installed_module': dobot_nexus.__file__}))
'''
        subprocess.run([sys.executable, '-I', '-c', 'INSTALLED='+repr(str(installed))+'\n'+probe],
                       cwd=workspace, check=True, env={k: v for k, v in os.environ.items()
                                                     if k != 'PYTHONPATH'})
        print('Wheel installation smoke test passed.')


if __name__ == '__main__':
    main()
