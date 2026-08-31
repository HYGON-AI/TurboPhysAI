# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KERNEL_ROOT = ROOT / "kernel"


class KernelLayoutTest(unittest.TestCase):
    def test_kernel_tree_contains_only_maintained_sources(self):
        generated_ignore_rules = {
            "kernel/**/*.hip",
            "kernel/**/*_hip.h",
            "kernel/**/*_hip.hpp",
            "kernel/**/*_hip.cpp",
            "kernel/**/*_hip.cuh",
        }
        ignore_rules = {
            line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertTrue(generated_ignore_rules.issubset(ignore_rules))

        completed = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--", "kernel"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            # Exported source trees do not contain .git metadata. The ignore
            # policy above is the available repository-layout evidence there.
            return

        tracked = [
            ROOT / raw.decode("utf-8")
            for raw in completed.stdout.split(b"\0")
            if raw
        ]
        generated = [
            path
            for path in tracked
            if path.suffix == ".hip" or "_hip." in path.name.lower()
        ]
        self.assertEqual(generated, [])

    def test_only_central_binding_defines_python_module(self):
        sources = [
            path
            for root in (KERNEL_ROOT, ROOT / "turbo_physai" / "csrc")
            for path in root.rglob("*")
            if path.suffix in {".cu", ".cpp", ".cc", ".cxx"}
        ]
        owners = [
            path
            for path in sources
            if "PYBIND11_MODULE" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(owners, [ROOT / "turbo_physai" / "csrc" / "pybind.cpp"])

    def test_central_binding_registers_bevfusion_operators(self):
        source = (ROOT / "turbo_physai" / "csrc" / "pybind.cpp").read_text(
            encoding="utf-8"
        )
        for binding in (
            "bind_bev_pool",
            "bind_voxelization",
            "bind_sparse_conv",
        ):
            self.assertRegex(source, rf"\b{re.escape(binding)}\(m\)")


if __name__ == "__main__":
    unittest.main()
