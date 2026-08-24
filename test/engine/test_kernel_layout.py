# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KERNEL_ROOT = ROOT / "kernel"


class KernelLayoutTest(unittest.TestCase):
    def test_kernel_tree_contains_only_maintained_sources(self):
        generated = [
            path
            for path in KERNEL_ROOT.rglob("*")
            if path.is_file()
            and (path.suffix == ".hip" or "_hip." in path.name.lower())
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
