# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Explicit build-time compatibility patches for a MapTRv2 checkout.

Compilation flags, C++ headers and dependency upper bounds cannot be changed by
the runtime recipe engine.  Call :func:`apply_build_compatibility` before
building a baseline checkout; every replacement is idempotent and fails loudly
if the expected upstream text is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class BuildPatchError(RuntimeError):
    """Raised when an upstream build file no longer matches the patch."""


@dataclass(frozen=True)
class PatchResult:
    path: str
    changed: bool


def _apply_replacements(text, replacements):
    for old, new in replacements:
        if old not in text:
            if new and new not in text:
                raise BuildPatchError(
                    f"expected build text not found: {old!r}"
                )
            continue
        already_applied = False
        if new and old in new:
            without_applied = text.replace(new, "", 1)
            already_applied = old not in without_applied
        if not already_applied:
            text = text.replace(old, new, 1)
    return text


def _replace_once(path, replacements):
    original = path.read_text(encoding="utf-8")
    try:
        text = _apply_replacements(original, replacements)
    except BuildPatchError as exc:
        raise BuildPatchError(f"{exc.args[0]} in {path}") from exc
    if text == original:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _build_compatibility_patches(repo_root):
    mmdet3d = repo_root / "mmdetection3d"
    return (
        (
            mmdet3d / "mmdet3d" / "__init__.py",
            (("mmcv_maximum_version = '1.4.0'",
              "mmcv_maximum_version = '1.6.2'"),),
        ),
        (
            mmdet3d
            / "mmdet3d"
            / "datasets"
            / "pipelines"
            / "data_augment_utils.py",
            ((
                "from numba.errors import NumbaPerformanceWarning",
                "from numba.core.errors import NumbaPerformanceWarning",
            ),),
        ),
        (
            mmdet3d / "mmdet3d" / "ops" / "bev_pool" / "bev_pool.py",
            ((
                "    x = x.permute(0, 4, 1, 2, 3).contiguous()",
                "    #x = x.permute(0, 4, 1, 2, 3).contiguous()",
            ),),
        ),
        (
            mmdet3d / "mmdet3d" / "ops" / "spconv" / "conv.py",
            tuple(
                (
                    "@CONV_LAYERS.register_module()\nclass " + class_name,
                    "@CONV_LAYERS.register_module(force=True)\nclass "
                    + class_name,
                )
                for class_name in (
                    "SparseConv2d",
                    "SparseConv3d",
                    "SparseConv4d",
                    "SparseConvTranspose2d",
                    "SparseConvTranspose3d",
                    "SparseInverseConv2d",
                    "SparseInverseConv3d",
                    "SubMConv2d",
                    "SubMConv3d",
                    "SubMConv4d",
                )
            ),
        ),
        (
            mmdet3d / "mmdet3d" / "ops" / "voxel" / "src"
            / "scatter_points_cuda.cu",
            (
                ("#ifdef __CUDA_ARCH__", "#ifdef __CUDACC__"),
                ("#if (__CUDA_ARCH__ < 200)", "#if (__CUDACC__ < 200)"),
                ("#if (__CUDA_ARCH__ < 600)", "#if (__CUDACC__ < 600)"),
            ),
        ),
        (
            mmdet3d / "requirements" / "runtime.txt",
            (
                ("networkx>=2.2,<2.3", "#networkx>=2.2,<2.3"),
                ("numba==0.48.0", "#numba==0.48.0"),
                ("numpy<1.20.0", "#numpy<1.20.0"),
                ("plyfile", "#plyfile"),
                ("scikit-image", "#scikit-image"),
                ("trimesh>=2.35.39,<2.35.40", "#trimesh>=2.35.39,<2.35.40"),
            ),
        ),
        (
            mmdet3d / "setup.py",
            (("-std=c++14", "-std=c++17"),),
        ),
    )


_POINT_OPS = (
    "ball_query/src/ball_query.cpp",
    "furthest_point_sample/src/furthest_point_sample.cpp",
    "gather_points/src/gather_points.cpp",
    "group_points/src/group_points.cpp",
    "interpolate/src/interpolate.cpp",
    "knn/src/knn.cpp",
)


def _point_op_patches(repo_root):
    root = repo_root / "mmdetection3d" / "mmdet3d" / "ops"
    patches = []
    for relative in _POINT_OPS:
        path = root / relative
        text = path.read_text(encoding="utf-8")
        replacements = []
        if "#include <THC/THC.h>" in text:
            has_aten = "#include <ATen/cuda/CUDAContext.h>" in text
            replacements.append(
                (
                    "#include <THC/THC.h>",
                    "" if has_aten else "#include <ATen/cuda/CUDAContext.h>",
                )
            )
        if "extern THCState *state;" in text:
            replacements.append(("extern THCState *state;", ""))
        patches.append((path, tuple(replacements)))
    return tuple(patches)


def force_geometric_kernel_extension(repo_root):
    """Select ``CUDAExtension`` even when the build host has no visible GPU."""

    path = (
        repo_root
        / "projects"
        / "mmdet3d_plugin"
        / "maptr"
        / "modules"
        / "ops"
        / "geometric_kernel_attn"
        / "setup.py"
    )
    return _replace_once(
        path,
        ((
            "    if torch.cuda.is_available() and CUDA_HOME is not None:",
            "    if 1:\n"
            "    # if torch.cuda.is_available() and CUDA_HOME is not None:",
        ),),
    )


def apply_build_compatibility(repo_root, *, dry_run=False):
    """Apply MapTRv2 build patches and return one result per touched file."""

    root = Path(repo_root).expanduser().resolve()
    if not (root / "mmdetection3d").is_dir():
        raise BuildPatchError(f"not a MapTRv2 checkout: {root}")
    patches = _build_compatibility_patches(root) + _point_op_patches(root)
    results = []
    if dry_run:
        for path, replacements in patches:
            text = path.read_text(encoding="utf-8")
            try:
                changed = (
                    _apply_replacements(text, replacements) != text
                )
            except BuildPatchError as exc:
                raise BuildPatchError(
                    f"{exc.args[0]} in {path}"
                ) from exc
            results.append(PatchResult(str(path.relative_to(root)), changed))
        kernel_path = (
            root
            / "projects"
            / "mmdet3d_plugin"
            / "maptr"
            / "modules"
            / "ops"
            / "geometric_kernel_attn"
            / "setup.py"
        )
        text = kernel_path.read_text(encoding="utf-8")
        replacement = (
            "    if torch.cuda.is_available() and CUDA_HOME is not None:",
            "    if 1:\n"
            "    # if torch.cuda.is_available() and CUDA_HOME is not None:",
        )
        try:
            changed = (
                _apply_replacements(text, (replacement,)) != text
            )
        except BuildPatchError as exc:
            raise BuildPatchError(
                f"{exc.args[0]} in {kernel_path}"
            ) from exc
        results.append(
            PatchResult(
                str(kernel_path.relative_to(root)),
                changed,
            )
        )
        return tuple(results)
    for path, replacements in patches:
        changed = _replace_once(path, replacements)
        results.append(PatchResult(str(path.relative_to(root)), changed))
    kernel_path = (
        root
        / "projects"
        / "mmdet3d_plugin"
        / "maptr"
        / "modules"
        / "ops"
        / "geometric_kernel_attn"
        / "setup.py"
    )
    changed = force_geometric_kernel_extension(root)
    results.append(
        PatchResult(str(kernel_path.relative_to(root)), changed)
    )
    return tuple(results)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Patch a MapTRv2 checkout for TurboPhysAI builds."
    )
    parser.add_argument("repo", help="MapTRv2 repository root")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report files that would change without writing them",
    )
    args = parser.parse_args(argv)
    for result in apply_build_compatibility(
        args.repo, dry_run=args.dry_run
    ):
        state = "would update" if result.changed else "unchanged"
        print(f"{state}: {result.path}")
    return 0


__all__ = [
    "BuildPatchError",
    "PatchResult",
    "apply_build_compatibility",
    "force_geometric_kernel_extension",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
