# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime compatibility for the Hygon FlashOCC checkout."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import linecache
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping


_NVCC_ONLY_FLAGS = {"-allow-unsupported-compiler", "--allow-unsupported-compiler"}


def order_import_roots(
    existing_paths, model_root, optimization_root
) -> list[str]:
    """Prefer official model modules while retaining runtime dependencies."""
    priority = [
        Path(model_root).expanduser().resolve(),
        (Path(model_root).expanduser().resolve() / "mmdetection3d"),
    ]
    priority_set = set(priority) | {Path(optimization_root).expanduser().resolve()}
    remaining = []
    for entry in existing_paths:
        resolved = Path(entry or os.curdir).expanduser().resolve()
        if resolved not in priority_set:
            remaining.append(entry)
    return [str(path) for path in priority] + remaining


def strip_nvcc_only_flags(options: Mapping[str, object]) -> dict[str, object]:
    """Return extension options without flags unsupported by the HCU compiler."""
    cleaned = dict(options)
    flags = cleaned.get("extra_cuda_cflags")
    if isinstance(flags, (list, tuple)):
        cleaned["extra_cuda_cflags"] = [
            flag for flag in flags if flag not in _NVCC_ONLY_FLAGS
        ]
    return cleaned


def resolve_dvr_compat_sources(
    source_paths, compat_source_root: str | os.PathLike[str]
) -> list[str]:
    """Redirect the optional DVR build to sources shipped with this package."""
    root = Path(compat_source_root).expanduser().resolve()
    resolved = []
    for source in source_paths:
        source = Path(source)
        if source.name in {"dvr.cpp", "dvr.cu"}:
            source = root / source.name
            if not source.is_file():
                raise RuntimeError(
                    "bundled dvr compatibility source missing: "
                    f"{source}"
                )
        resolved.append(str(source.resolve()))
    return resolved


class _GB18030SourceLoader(importlib.machinery.SourceFileLoader):
    def get_source(self, fullname: str) -> str:
        source = self.get_data(self.path).decode("gb18030")
        stat = os.stat(self.path)
        linecache.cache[self.path] = (
            stat.st_size,
            stat.st_mtime,
            source.splitlines(keepends=True),
            self.path,
        )
        return source

    def get_code(self, fullname: str):
        source = self.get_source(fullname)
        return self.source_to_code(source, self.path)


class LegacyEncodingSourceFinder(importlib.abc.MetaPathFinder):
    """Load the one legacy GB18030 Python source without editing the model."""

    _flashocc_finder_kind = "encoding"

    _SOURCE_PATH = Path(
        "projects/mmdet3d_plugin/models/detectors/bevdet_occ.py"
    )

    def __init__(self, model_root: str | os.PathLike[str]) -> None:
        self.model_root = Path(model_root).expanduser().resolve()

    def find_spec(self, fullname: str, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if (
            spec is None
            or spec.origin is None
            or not isinstance(spec.loader, importlib.machinery.SourceFileLoader)
        ):
            return None

        source_path = Path(spec.origin).resolve()
        try:
            relative_path = source_path.relative_to(self.model_root)
        except ValueError:
            return None
        if relative_path != self._SOURCE_PATH:
            return None

        source_bytes = source_path.read_bytes()
        try:
            source_bytes.decode("utf-8")
        except UnicodeDecodeError:
            source_bytes.decode("gb18030")
        else:
            return None

        loader = _GB18030SourceLoader(fullname, str(source_path))
        return importlib.util.spec_from_file_location(
            fullname,
            source_path,
            loader=loader,
            submodule_search_locations=spec.submodule_search_locations,
        )


class LazyDvrModule:
    """Defer the optional RayIoU extension until its rendering API is called."""

    def __init__(self, loader: Callable[[], ModuleType]) -> None:
        self._loader = loader
        self._module: ModuleType | None = None

    def __getattr__(self, name: str):
        if self._module is None:
            self._module = self._loader()
        return getattr(self._module, name)


def _compatible_dvr_loader(original_load, args, kwargs, compat_source_root: Path):
    compile_options = strip_nvcc_only_flags(kwargs)
    source_paths = compile_options.get("sources")
    if source_paths is None and args:
        source_paths = args[0]
    if not source_paths:
        raise RuntimeError("dvr compatibility build requires its source list")

    compat_sources = resolve_dvr_compat_sources(source_paths, compat_source_root)
    if "sources" in compile_options:
        compile_options["sources"] = compat_sources
        return original_load("dvr", **compile_options)

    positional = list(args)
    positional[0] = compat_sources
    return original_load("dvr", *positional, **compile_options)


def install_dvr_compatibility(optimization_root: str | os.PathLike[str]) -> None:
    import torch.utils.cpp_extension as cpp_extension

    if getattr(cpp_extension.load, "_flashocc_compat", False):
        return

    original_load = cpp_extension.load
    root = (
        Path(optimization_root).expanduser().resolve()
        / "compat_sources"
        / "dvr"
    )

    def compatible_load(name, *args, **kwargs):
        if name != "dvr":
            return original_load(name, *args, **kwargs)
        return LazyDvrModule(
            lambda: _compatible_dvr_loader(original_load, args, kwargs, root)
        )

    compatible_load._flashocc_compat = True
    cpp_extension.load = compatible_load


def install_compatibility(
    model_root: str | os.PathLike[str],
) -> None:
    model_path = Path(model_root).expanduser().resolve()
    optimization_path = Path(__file__).resolve().parent
    finders = [
        finder for finder in sys.meta_path
        if getattr(finder, "_flashocc_finder_kind", None) == "encoding"
    ]
    for finder in finders:
        if finder.model_root != model_path:
            raise RuntimeError("FlashOCC compatibility is already installed with different roots")
    sys.path[:] = order_import_roots(
        sys.path, model_path, optimization_path
    )
    if not finders:
        finder = LegacyEncodingSourceFinder(model_path)
        sys.meta_path.insert(0, finder)
    install_dvr_compatibility(optimization_path)


def install_for_turbo_run() -> bool:
    """Install compatibility before Turbo resolves FlashOCC targets."""
    if os.environ.get("TURBO_PHYSAI_BOOTSTRAP") != "1":
        return False
    model_root = os.environ.get("FLASHOCC_ROOT")
    if not model_root:
        raise RuntimeError("FlashOCC model root is not configured")
    install_compatibility(model_root)
    return True
