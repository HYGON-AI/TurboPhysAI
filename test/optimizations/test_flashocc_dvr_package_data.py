# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import ast
from pathlib import Path


def test_dvr_sources_are_included_in_package_data():
    root = Path(__file__).resolve().parents[2]
    module = ast.parse((root / "setup.py").read_text(encoding="utf-8"))
    setup_call = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    data = ast.literal_eval(next(
        keyword.value for keyword in setup_call.keywords
        if keyword.arg == "package_data"
    ))
    package = root / "turbo_physai"
    included = {p for pattern in data["turbo_physai"] for p in package.glob(pattern)}
    for name in ("dvr.cpp", "dvr.cu"):
        assert package / "optimizations/models/flashocc/compat_sources/dvr" / name in included
