"""Exercise the suite launcher and existing shell scripts from outside the repo."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SuiteLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cwd = Path(self.temporary.name) / "caller directory"
        self.cwd.mkdir()
        (self.cwd / "data root").mkdir()
        (self.cwd / "checkpoint root").mkdir()
        for name in ("rn checkpoint.pth", "vit checkpoint.pth"):
            (self.cwd / name).touch()
        self.calls = self.cwd / "calls.jsonl"
        recorder = self.cwd / "record python"
        recorder.write_text("#!/usr/bin/env python3\n" + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-u":
    args = args[1:]
with open(os.environ["ACTTA_TEST_CALLS"], "a") as f:
    f.write(json.dumps({"args": args, "cwd": str(Path.cwd()),
                        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}) + "\\n")
if args[0].startswith("evaluate") and "--output" in args:
    Path(args[args.index("--output") + 1]).write_text("{}\\n")
sys.exit(int(os.environ.get("ACTTA_TEST_EXIT", "0")))
''')
        recorder.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment.update(ACTTA_PYTHON=str(recorder), ACTTA_TEST_CALLS=str(self.calls))
        self.environment.pop("DEPTH_POINTS", None)

    def arguments(self, suite, output="new output"):
        args = [suite, "--data-root", "data root", "--output", output, "--gpu", "3"]
        if suite == "cifar":
            args += ["--checkpoint-root", "checkpoint root"]
        else:
            args += ["--resnet-checkpoint", "rn checkpoint.pth",
                     "--vit-checkpoint", "vit checkpoint.pth"]
        return args

    def invoke(self, args):
        return subprocess.run([sys.executable, str(ROOT / "tools/run_suite.py"), *args],
                              cwd=self.cwd, env=self.environment, text=True, capture_output=True)

    def records(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_suites_from_external_directory(self):
        for suite, evaluator, count in [("cifar", "evaluate.py", 12),
                                        ("imagenet-settings", "evaluate_architecture.py", 12),
                                        ("imagenet-depth", "evaluate_depth_sweep_ordered.py", 4)]:
            with self.subTest(suite=suite):
                if self.calls.exists():
                    self.calls.unlink()
                args = self.arguments(suite, f"{suite} output")
                if suite == "imagenet-depth":
                    args += ["--depths", "0", "50"]
                result = self.invoke(args)
                self.assertEqual(result.returncode, 0, result.stderr)
                records = self.records()
                runs = [r for r in records if r["args"][0] == evaluator]
                self.assertEqual(len(runs), count)
                for record in records:
                    self.assertEqual(record["cwd"], str(ROOT))
                    self.assertEqual(record["gpu"], "3")
                for record in runs:
                    command = record["args"]
                    self.assertTrue((ROOT / command[command.index("--config") + 1]).is_file())
                    self.assertTrue(Path(command[command.index("--data-dir") + 1]).is_absolute())
                    self.assertTrue(Path(command[command.index("--checkpoint") + 1]).is_absolute())
                    self.assertEqual(Path(command[command.index("--output") + 1]).parent,
                                     self.cwd / f"{suite} output")

    def test_existing_results_and_logs_are_not_overwritten(self):
        output = self.cwd / "new output"
        output.mkdir()
        original = output / "previous.log"
        original.write_text("previous experiment\n")
        result = self.invoke(self.arguments("imagenet-settings"))
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be new or empty", result.stderr)
        self.assertEqual(original.read_text(), "previous experiment\n")
        self.assertFalse(self.calls.exists())

    def test_dry_run_is_side_effect_free_and_shell_quoting_is_executable(self):
        result = self.invoke(self.arguments("imagenet-depth") + ["--depths", "0", "--dry-run"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertFalse((self.cwd / "new output").exists())
        executed = subprocess.run(["bash", "-c", result.stdout], cwd=self.cwd,
                                  env=self.environment, text=True, capture_output=True)
        self.assertEqual(executed.returncode, 0, executed.stderr)
        self.assertEqual(sum(r["args"][0] == "evaluate_depth_sweep_ordered.py"
                             for r in self.records()), 2)

    def test_script_failure_is_returned(self):
        self.environment["ACTTA_TEST_EXIT"] = "9"
        result = self.invoke(self.arguments("imagenet-settings"))
        self.assertEqual(result.returncode, 9)
        self.assertEqual(len(self.records()), 1)

    def test_invalid_depth_selection_stops_before_execution(self):
        for suite, points in [("cifar", ["0"]), ("imagenet-depth", ["50", "50"])]:
            with self.subTest(suite=suite, points=points):
                result = self.invoke(self.arguments(suite) + ["--depths", *points])
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
