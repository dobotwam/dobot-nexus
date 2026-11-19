import contextlib
import io
import unittest

from examples.atom_robot.run import config_for, parse_args


class RuntimeOptionsTest(unittest.TestCase):
    def test_horizon_is_negotiated_or_explicitly_checked_independently_of_round_count(self):
        self.assertIsNone(config_for(parse_args([]), "/report")["horizon"])
        args = parse_args(["--horizon", "50", "--max-horizons", "20"])
        self.assertEqual(config_for(args, "/report")["horizon"], 50)
        self.assertEqual(args.max_horizons, 20)

    def test_operator_supervision_is_explicit_and_preserves_episode_and_home(self):
        default = parse_args([])
        self.assertEqual(default.task_guards, "monitor")
        args = parse_args(["--execute", "--site-clear", "--task-guards", "monitor"])
        self.assertTrue(args.execute)
        self.assertTrue(args.home_on_start)
        self.assertTrue(args.return_home)
        self.assertGreater(args.max_horizons, 0)
        self.assertEqual(config_for(args, "/report")["task_guard_mode"], "monitor")

    def test_unknown_guard_modes_and_missing_site_clear_are_rejected(self):
        for argv in [["--task-guards", "off"], ["--execute", "--task-guards", "monitor"]]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    parse_args(argv)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
