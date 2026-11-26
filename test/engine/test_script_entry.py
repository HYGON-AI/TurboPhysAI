# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class ScriptEntryTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        (self.root / "demo").mkdir()
        self.write("demo/__init__.py", "")
        self.write("demo/values.py", "value = 5\n")
        self.write("demo/entry_deps.py", """
            def early(value):
                return value

            def late(value):
                return value
        """)
        self.write("demo/unused_optional.py", "raise RuntimeError('compatibility missing')\n")
        self.write("entry_optim.py", """
            from functools import wraps
            from turbo_physai import group, replace, wrap
            from turbo_physai.engine import optional_import

            def optimized(value):
                return value + 10

            def factory(original, options):
                @wraps(original)
                def wrapped(self, value):
                    return original(self, value) * 10
                return wrapped

            group("demo.compatibility", optional_import("demo.unused_optional"))
            group("demo.early", replace("demo.entry_deps.early", "entry_optim.optimized"))
            group(
                "demo.entry",
                replace("train.loop", "entry_optim.optimized", aliases=("train.alias",)),
                wrap("train.Trainer.train", "entry_optim.factory"),
                depends_on=("demo.early", "demo.compatibility"),
            )
            group(
                "demo.dependent",
                replace("demo.entry_deps.late", "entry_optim.optimized"),
                depends_on=("demo.entry",),
            )
        """)
        self.config("demo.dependent", "demo.entry", "demo.compatibility", "demo.early")
        self.source = textwrap.dedent("""
            from __future__ import annotations
            import json
            import os
            import sys
            from demo import unused_optional
            from demo.entry_deps import early
            if __package__:
                from .values import value
            else:
                from demo.values import value

            print("PRELUDE", early(value), flush=True)

            def loop(value):
                return value + 1

            alias = loop

            class Trainer:
                def train(self, value):
                    return value + 2

            def main():
                from demo.entry_deps import late
                print("RESULT", json.dumps({
                    "name": __name__,
                    "package": __package__,
                    "spec": __spec__.name if __spec__ else None,
                    "argv": sys.argv,
                    "path0": sys.path[0],
                    "file": __file__,
                    "optimized": [loop(value), alias(value), Trainer().train(value), late(value)],
                    "rank": int(os.environ.get("RANK", 0)),
                    "separate_import": "train" in sys.modules or "demo.train" in sys.modules,
                }), flush=True)
                if "--exit" in sys.argv:
                    sys.exit(3)

            if "__main__" == __name__:
                main()
        """).lstrip()
        self.write("demo/train.py", self.source)

    def write(self, name, source):
        (self.root / name).write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")

    def config(self, *groups):
        header = textwrap.dedent("""
            schema_version: turbophysai/optimization-config/v1
            kind: OptimizationConfig
            metadata: {id: demo, version: "1"}
            optimization_modules: [entry_optim]
            optimization_groups:
        """).strip()
        self.write("optimization.yaml", header + "\n" + "\n".join(f"  - id: {group}" for group in groups) + "\n")

    def launch(self, args, *, options=(), rank=None, command=None, timeout=25):
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("TURBO_PHYSAI_")}
        env.update({
            "PYTHONPATH": os.pathsep.join(map(str, (REPO_ROOT, self.root / "demo"))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "RANK": str(rank if rank is not None else 0),
            "WORLD_SIZE": "2" if rank is not None else "1",
        })
        completed = subprocess.run(
            [sys.executable, "-m", "turbo_physai.cli", "run", "--disable-numa",
             "--optimization-config", "optimization.yaml", "--log-report",
             *options, "--", *(command or [sys.executable]), *args],
            cwd=self.root, env=env, text=True, capture_output=True, timeout=timeout,
        )
        return completed

    def result(self, completed, *, status=0):
        self.assertEqual(completed.returncode, status, completed.stdout + completed.stderr)
        results = [line for line in completed.stdout.splitlines() if line.startswith("RESULT ")]
        self.assertEqual(len(results), 1, completed.stdout)
        return json.loads(results[0][len("RESULT "):])

    def test_direct_script_with_interpreter_flags_and_no_duplicate_import(self):
        run = self.launch(["-W", "default", "-X", "dev", "demo/train.py", "--demo"])
        actual = self.result(run)
        self.assertEqual(actual, {
            "name": "__main__", "package": None, "spec": None,
            "argv": ["demo/train.py", "--demo"], "path0": str(self.root / "demo"),
            "file": str(self.root / "demo/train.py"),
            "optimized": [15, 15, 70, 15], "rank": 0, "separate_import": False,
        })
        self.assertEqual(run.stdout.count("PRELUDE 15"), 1)
        self.assertIn("Summary: applied=4 skipped=0 blocked=0", run.stdout)

    def test_module_entry_preserves_relative_imports_and_module_spec(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                run = self.launch(["-m", "demo.train", "--exit"], rank=rank)
                actual = self.result(run, status=3)
                self.assertEqual(actual["package"], "demo")
                self.assertEqual(actual["spec"], "demo.train")
                self.assertEqual(actual["argv"], [str(self.root / "demo/train.py"), "--exit"])
                self.assertEqual(actual["path0"], str(self.root))
                self.assertEqual(actual["optimized"], [15, 15, 70, 15])
                self.assertEqual(actual["rank"], rank)
                self.assertFalse(actual["separate_import"])
                self.assertIn("applied=4", run.stdout)

    def test_wrapper_imports_helpers_from_the_live_top_level_entry(self):
        self.write("entry_factory.py", """
            from train import Trainer

            def factory(original, options):
                assert Trainer.train is original
                return lambda self, value: original(self, value) * 10
        """)
        catalog = self.root / "entry_optim.py"
        catalog.write_text(catalog.read_text().replace("entry_optim.factory", "entry_factory.factory"))
        run = self.launch(["-m", "demo.train"])
        actual = self.result(run)
        self.assertEqual(actual["optimized"], [15, 15, 70, 15])
        self.assertFalse(actual["separate_import"])
        self.assertEqual(run.stdout.count("PRELUDE"), 1)

    def test_script_invoked_through_symlink_keeps_its_file_and_argv(self):
        (self.root / "linked.py").symlink_to("demo/train.py")
        actual = self.result(self.launch(["linked.py"]))
        self.assertEqual(actual["file"], str(self.root / "linked.py"))
        self.assertEqual(actual["argv"], ["linked.py"])
        self.assertEqual(actual["path0"], str(self.root / "demo"))
        self.assertEqual(actual["optimized"], [15, 15, 70, 15])

    def test_disabled_entry_and_dependents_use_original_targets(self):
        run = self.launch(["-m", "demo.train"], options=("--disable-group", "demo.entry"))
        self.assertEqual(self.result(run)["optimized"], [6, 6, 7, 5])
        self.assertIn("dependency_disabled", run.stdout)
        self.assertIn("Summary: applied=2 skipped=2 blocked=0", run.stdout)

    def test_missing_guard_blocks_entry_group_and_dependents(self):
        self.write("demo/train.py", self.source.replace('if "__main__" == __name__:\n    main()', "main()"))
        run = self.launch(["-m", "demo.train"])
        self.assertEqual(self.result(run)["optimized"], [6, 6, 7, 5])
        self.assertIn("no unique top-level __main__ guard", run.stdout)
        self.assertIn("Summary: applied=2 skipped=0 blocked=2", run.stdout)

    def test_mismatched_live_alias_blocks_whole_group(self):
        self.write("demo/train.py", self.source.replace(
            'if "__main__" == __name__:', 'del alias\nalias = lambda value: value + 1\n\nif "__main__" == __name__:'
        ))
        run = self.launch(["demo/train.py"])
        self.assertEqual(self.result(run)["optimized"], [6, 6, 7, 5])
        self.assertIn("check alias.identity: status=fail", run.stdout)

    def test_failed_assignment_rolls_back_group_before_training(self):
        self.write("demo/train.py", self.source.replace("class Trainer:", """
class Locked(type):
    def __setattr__(cls, name, value):
        if name == "train" and value is not cls.train:
            raise RuntimeError("assignment rejected")
        super().__setattr__(name, value)

class Trainer(metaclass=Locked):"""))
        run = self.launch(["demo/train.py"])
        self.assertEqual(self.result(run)["optimized"], [6, 6, 7, 5])
        self.assertIn("demo.entry: status=rolled_back", run.stdout)
        self.assertIn("demo.dependent: status=not_started", run.stdout)

    def test_rollback_failure_aborts_before_the_guard_body(self):
        self.write("demo/train.py", self.source.replace("class Trainer:", """
class Locked(type):
    def __setattr__(cls, name, value):
        raise RuntimeError("assignment rejected")

class Trainer(metaclass=Locked):"""))
        run = self.launch(["demo/train.py"])
        self.assertEqual(run.returncode, 91, run.stdout + run.stderr)
        self.assertNotIn("RESULT ", run.stdout)
        self.assertIn("demo.dependent: status=not_started", run.stdout)

    def test_early_terminal_failure_reports_unreached_entry_groups(self):
        with (self.root / "demo/entry_deps.py").open("a", encoding="utf-8") as source:
            source.write(textwrap.dedent("""

                class Locked(type):
                    def __setattr__(cls, name, value):
                        raise RuntimeError("assignment rejected")

                class Extra(metaclass=Locked):
                    def fn(self, value):
                        return value
            """))
        with (self.root / "entry_optim.py").open("a", encoding="utf-8") as catalog:
            catalog.write('\ngroup("demo.terminal", wrap("demo.entry_deps.Extra.fn", "entry_optim.factory"))\n')
        self.config("demo.compatibility", "demo.terminal", "demo.early", "demo.entry", "demo.dependent")
        run = self.launch(["demo/train.py"])
        self.assertEqual(run.returncode, 91, run.stdout + run.stderr)
        self.assertNotIn("PRELUDE ", run.stdout)
        self.assertIn("demo.entry: dependencies=demo.early,demo.compatibility decision=block "
                      "reason=entrypoint_startup_failed", run.stdout)

    def test_spawned_python_process_does_not_apply_again(self):
        self.write("demo/train.py", self.source.replace(
            'if "__main__" == __name__:\n    main()',
            '''
def worker(queue):
    queue.put((__name__, loop(1)))

if "__main__" == __name__:
    import multiprocessing
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    child = context.Process(target=worker, args=(queue,))
    child.start()
    assert queue.get(timeout=10) == ("__mp_main__", 2)
    child.join(timeout=10)
    assert child.exitcode == 0
    main()''',
        ))
        # A spawned process re-imports the original program as __mp_main__.
        # It must not depend on a rank-local optional-import hook.
        self.write("demo/unused_optional.py", "")
        run = self.launch(["-m", "demo.train"])
        self.assertEqual(self.result(run)["optimized"], [15, 15, 70, 15])
        self.assertEqual(run.stdout.count("TURBO_PHYSAI_OPTIMIZATION_REPORT_BEGIN"), 1)

    def test_same_file_targets_using_different_names_conflict(self):
        with (self.root / "entry_optim.py").open("a", encoding="utf-8") as catalog:
            catalog.write('\ngroup("demo.overlap", replace("demo.train.loop", "entry_optim.optimized"))\n')
        self.config("demo.dependent", "demo.entry", "demo.compatibility", "demo.early", "demo.overlap")
        run = self.launch(["-m", "demo.train"])
        self.assertEqual(self.result(run)["optimized"], [6, 6, 7, 5])
        self.assertIn("target.group_duplicate", run.stdout)
        self.assertIn("Summary: applied=2 skipped=0 blocked=3", run.stdout)


if __name__ == "__main__":
    unittest.main()
