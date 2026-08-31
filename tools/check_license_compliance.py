# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Check release-license artifacts and required Apache change notices."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
APACHE_FILES = {
    "test/msda_reference.py",
    "turbo_physai/optimizations/common/mmcv/msda.py",
    *{
        f"turbo_physai/optimizations/models/bevformer/{name}.py"
        for name in ("backbone", "data", "geometry_sca", "grid_mask", "mdc", "msda", "training", "tsa")
    },
    *{
        f"turbo_physai/optimizations/models/bevfusion/{name}.py"
        for name in ("backbone", "bev_pool", "depth", "gaussian", "indice", "sparse", "training", "transfusion", "transfusion_bbox_coder", "transfusion_bbox_coder_runtime", "voxel")
    },
}
APACHE_DIRECTORIES = ("kernel/bev_pool", "kernel/voxelization")



def is_source(path: Path) -> bool:
    return path.suffix in {
        ".py", ".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".sh", ".yaml", ".yml"
    }


def requires_apache_notice(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    return rel in APACHE_FILES or any(
        rel.startswith(item + "/") for item in APACHE_DIRECTORIES
    )


def requires_hygon_bsd_header(path: Path) -> bool:
    """Return whether a known Hygon-authored path needs the standard header."""
    if requires_apache_notice(path):
        return False
    rel = path.relative_to(ROOT).as_posix()
    return (
        rel.startswith("turbo_physai/csrc/")
        or rel == "turbo_physai/operators/__init__.py"
        or rel.startswith("scripts/")
        or rel.startswith("test/")
        or "/configs/" in rel and path.suffix in {".yaml", ".yml"}
    )


def main() -> int:
    required = [ROOT / "LICENSE", ROOT / "NOTICE", ROOT / "THIRD_PARTY_NOTICES.md", ROOT / "third_party/licenses/Apache-2.0.txt"]
    failures = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    for path in ROOT.rglob("*"):
        if not path.is_file() or not is_source(path) or not requires_apache_notice(path):
            continue
        text = path.read_text(errors="replace")
        if "Licensed under the Apache License, Version 2.0" not in text or "Copyright 2026 Hygon Information Technology Co., Ltd." not in text or "Modified by Hygon." not in text:
            failures.append(str(path.relative_to(ROOT)))
    for path in ROOT.rglob("*"):
        if not path.is_file() or not is_source(path) or not requires_hygon_bsd_header(path):
            continue
        text = path.read_text(errors="replace")
        if (
            "Copyright 2026 Hygon Information Technology Co., Ltd." not in text
            or "SPDX-License-Identifier: BSD-3-Clause" not in text
        ):
            failures.append(str(path.relative_to(ROOT)))
    if failures:
        print("license-compliance check failed:", *failures, sep="\n- ")
        return 1
    print("license-compliance check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
