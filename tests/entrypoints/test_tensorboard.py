from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from dobot_nexus.cli import main as cli
from dobot_nexus.cli import tensorboard as monitor


class TensorBoardEntrypointTest(unittest.TestCase):
    def test_dashboard_uses_real_logdir_loopback_and_foreground_lifetime(self):
        dashboard = mock.Mock()
        dashboard.main.return_value = 0
        program = SimpleNamespace(TensorBoard=mock.Mock(return_value=dashboard))
        with tempfile.TemporaryDirectory(prefix='logs with spaces ') as directory, \
             mock.patch.dict(sys.modules, {'tensorboard': SimpleNamespace(program=program)}):
            cli.main(['tensorboard', '--logdir', directory])
        dashboard.configure.assert_called_once_with(argv=[
            'tensorboard', '--logdir', directory, '--host', '127.0.0.1', '--port', '6006',
            '--reload_interval', '5'])
        dashboard.main.assert_called_once_with()

    def test_subcommand_help_and_invalid_arguments_do_not_load_tensorboard(self):
        with mock.patch.dict(sys.modules, {'tensorboard': None}):
            with redirect_stdout(io.StringIO()) as stdout, self.assertRaises(SystemExit) as caught:
                cli.main(['tensorboard', '--help'])
            self.assertEqual(caught.exception.code, 0)
            self.assertIn('--reload-interval', stdout.getvalue())
            with tempfile.TemporaryDirectory() as directory:
                for option in ('--port', '--reload-interval'):
                    with self.subTest(option=option), redirect_stderr(io.StringIO()), \
                         self.assertRaises(SystemExit) as caught:
                        monitor.main(['--logdir', directory, option, '0'])
                    self.assertEqual(caught.exception.code, 2)
                with self.assertRaisesRegex(ImportError, 'tensorboard'):
                    monitor.main(['--logdir', directory])
