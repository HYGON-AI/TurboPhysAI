# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "turbo_physai/optimizations/models/flashocc/scripts/build_flashocc_ops_external.sh"
)


class BuildCleanupTests(unittest.TestCase):
    def run_build(self, scenario):
        root = Path(tempfile.mkdtemp(prefix="flashocc-cleanup-test-"))
        model_root = root / "FlashOCC"
        source_script = (
            root / "TurboPhysAI/turbo_physai/optimizations/models/flashocc/scripts"
            / SCRIPT.name
        )
        source_script.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, source_script)
        projects = model_root / "projects"
        projects.mkdir(parents=True)
        (projects / "setup.py").write_text("# test fixture\n")
        for op in ("bev_pool", "bev_pool_v2", "nearest_assign"):
            src = projects / "mmdet3d_plugin" / "ops" / op / "src"
            src.mkdir(parents=True)
            (src / "source.cu").write_text("// test fixture\n")
        sentinel = root / "do-not-remove"
        sentinel.write_text("untouched")
        temp_root = root / "owned-tmp"
        temp_root.mkdir()
        fake_bin = root / "bin"
        fake_bin.mkdir()
        commands = {
            "git": "#!/bin/sh\nexit 0\n",
            "mktemp": """#!/bin/sh
case "$2" in
  /tmp/flashocc-ops-src.*) name=src ;;
  /tmp/flashocc-ops-build.*)
    if [ "$TEST_SCENARIO" = mktemp_fail ]; then exit 72; fi
    name=build ;;
  *) exit 73 ;;
esac
mkdir "$TEST_TMP_ROOT/$name"
printf '%s\n' "$TEST_TMP_ROOT/$name"
""",
            "python": """#!/bin/sh
if [ "$1" = "-" ]; then exit 0; fi
if [ "$TEST_SCENARIO" = build_fail ]; then exit 23; fi
for op in bev_pool bev_pool_v2 nearest_assign; do
    path="$TEST_MODEL_ROOT/projects/mmdet3d_plugin/ops/$op"
    mkdir -p "$path"
    touch "$path/${op}_ext.test.so"
done
""",
        }
        for name, content in commands.items():
            executable = fake_bin / name
            executable.write_text(content)
            executable.chmod(0o755)
        env = os.environ.copy()
        env.update(
            PATH=str(fake_bin) + os.pathsep + env["PATH"],
            TEST_TMP_ROOT=str(temp_root),
            TEST_MODEL_ROOT=str(model_root),
            TEST_SCENARIO=scenario,
        )
        result = subprocess.run(
            ["bash", str(source_script)], env=env, text=True, capture_output=True
        )
        self.assertEqual(result.returncode, {
            "success": 0, "build_fail": 23, "mktemp_fail": 72
        }[scenario], result.stderr)
        self.assertFalse((temp_root / "src").exists())
        self.assertFalse((temp_root / "build").exists())
        self.assertEqual(sentinel.read_text(), "untouched")
        if scenario == "success":
            for op in ("bev_pool", "bev_pool_v2", "nearest_assign"):
                output = projects / "mmdet3d_plugin" / "ops" / op / f"{op}_ext.test.so"
                self.assertTrue(output.is_file())

    def test_success_cleans_only_owned_temporary_directories(self):
        self.run_build("success")

    def test_build_failure_cleans_owned_temporary_directories(self):
        self.run_build("build_fail")

    def test_second_mktemp_failure_cleans_first_directory(self):
        self.run_build("mktemp_fail")


if __name__ == "__main__":
    unittest.main()
