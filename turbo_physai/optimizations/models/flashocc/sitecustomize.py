# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Install FlashOCC compatibility in each Python rank started by torchrun."""

import importlib.util
import os
import sys
from pathlib import Path


if os.environ.get("TURBO_PHYSAI_BOOTSTRAP") == "1":
    model_root = os.environ.get("FLASHOCC_ROOT")
    runtime_root = os.environ.get("FLASHOCC_RUNTIME_ROOT")
    if not model_root or not runtime_root:
        raise RuntimeError("FlashOCC compatibility roots are not configured")

    module_name = "_flashocc_runtime_compatibility"
    compatibility_path = Path(__file__).with_name("compatibility.py")
    spec = importlib.util.spec_from_file_location(module_name, compatibility_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load FlashOCC compatibility: {compatibility_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.install_compatibility(model_root, runtime_root)
