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


_PREBUILT_OPS = {
    "projects.mmdet3d_plugin.ops.bev_pool.bev_pool_ext": (
        "bev_pool",
        "bev_pool_ext",
    ),
    "projects.mmdet3d_plugin.ops.bev_pool_v2.bev_pool_v2_ext": (
        "bev_pool_v2",
        "bev_pool_v2_ext",
    ),
    "projects.mmdet3d_plugin.ops.nearest_assign.nearest_assign_ext": (
        "nearest_assign",
        "nearest_assign_ext",
    ),
}
_NVCC_ONLY_FLAGS = {"-allow-unsupported-compiler", "--allow-unsupported-compiler"}


def resolve_config_base_path(
    filename: str | os.PathLike[str],
    model_root: str | os.PathLike[str],
    runtime_root: str | os.PathLike[str],
) -> Path:
    """Use the runtime checkout only for absent mmdetection3d base files."""
    requested = Path(filename).expanduser().resolve()
    if requested.is_file():
        return requested

    model_dependency_root = Path(model_root).expanduser().resolve() / "mmdetection3d"
    try:
        relative_path = requested.relative_to(model_dependency_root)
    except ValueError:
        return requested

    fallback = (
        Path(runtime_root).expanduser().resolve()
        / "mmdetection3d"
        / relative_path
    )
    return fallback if fallback.is_file() else requested

def order_import_roots(
    existing_paths, model_root, runtime_root, optimization_root
) -> list[str]:
    """Prefer official model modules while retaining runtime dependencies."""
    priority = [
        Path(optimization_root).expanduser().resolve(),
        Path(model_root).expanduser().resolve(),
        Path(runtime_root).expanduser().resolve(),
    ]
    priority_set = set(priority)
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


def find_prebuilt_dvr(build_root: str | os.PathLike[str]) -> Path | None:
    """Find a previously built HCU dvr extension without rebuilding it."""
    root = Path(build_root)
    preferred = root / "py310_cpu" / "dvr" / "dvr.so"
    if preferred.is_file():
        return preferred
    candidates = sorted(root.glob("*/dvr/dvr.so"))
    return candidates[-1] if candidates else None


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


class PrebuiltExtensionFinder(importlib.abc.MetaPathFinder):
    """Resolve selected official-package imports to compatible prebuilt HCU ops."""

    def __init__(self, module_files: Mapping[str, str | os.PathLike[str]]) -> None:
        self.module_files = {
            name: Path(path) for name, path in module_files.items()
        }

    def find_spec(self, fullname: str, path=None, target=None):
        extension_path = self.module_files.get(fullname)
        if extension_path is None or not extension_path.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, extension_path)


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


def _extension_files(runtime_root: Path) -> dict[str, Path]:
    ops_root = runtime_root / "projects" / "mmdet3d_plugin" / "ops"
    result = {}
    for module_name, (package_dir, extension_name) in _PREBUILT_OPS.items():
        candidates = sorted((ops_root / package_dir).glob(f"{extension_name}*.so"))
        if candidates:
            result[module_name] = candidates[-1]
    return result


def _load_extension_module(module_name: str, extension_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, extension_path)
    if spec is None or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        raise ImportError(f"cannot load HCU extension {module_name} from {extension_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _compatible_dvr_loader(original_load, args, kwargs, compat_source_root: Path):
    import torch.utils.cpp_extension as cpp_extension

    cached = find_prebuilt_dvr(cpp_extension.get_default_build_root())
    if cached is not None:
        return _load_extension_module("dvr", cached)

    compile_options = strip_nvcc_only_flags(kwargs)
    source_paths = compile_options.get("sources")
    if source_paths is None and args:
        source_paths = args[0]
    if not source_paths:
        raise RuntimeError("dvr compatibility build requires its source list")

    compat_sources = resolve_dvr_compat_sources(source_paths, compat_source_root)
    for source in compat_sources:
        source = Path(source)
        if source.suffix == ".cu":
            source_text = source.read_text(encoding="utf-8")
            if "AT_DISPATCH_FLOATING_TYPES(sigma.scalar_type()" not in source_text:
                raise RuntimeError(
                    "dvr HCU fallback source must use Tensor.scalar_type(); "
                    f"unsupported source: {source}"
                )
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


def install_config_path_compatibility(
    model_root: str | os.PathLike[str],
    runtime_root: str | os.PathLike[str],
) -> None:
    from mmcv import Config

    if getattr(Config._file2dict, "_flashocc_compat", False):
        return

    original_file2dict = Config._file2dict
    model_path = Path(model_root).expanduser().resolve()
    runtime_path = Path(runtime_root).expanduser().resolve()

    def file2dict_with_fallback(filename, *args, **kwargs):
        resolved = resolve_config_base_path(filename, model_path, runtime_path)
        return original_file2dict(str(resolved), *args, **kwargs)

    file2dict_with_fallback._flashocc_compat = True
    Config._file2dict = staticmethod(file2dict_with_fallback)


_FLASHOCC_INFO_FILES = {
    "bevdetv2-nuscenes_infos_train.pkl",
    "bevdetv2-nuscenes_infos_val.pkl",
}


def _resolve_info_path(value, data_base: Path):
    if not isinstance(value, (str, os.PathLike)) or not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return os.path.abspath(data_base / path)


def _resolve_nuscenes_info_paths(data, data_base: Path):
    if not isinstance(data, dict):
        return data

    for info in data.get("infos", ()):
        if not isinstance(info, dict):
            continue
        for key in ("lidar_path", "occ_path"):
            if key in info:
                info[key] = _resolve_info_path(info[key], data_base)
        for camera in (info.get("cams") or {}).values():
            if "data_path" in camera:
                camera["data_path"] = _resolve_info_path(
                    camera["data_path"], data_base
                )
        for sweep in info.get("sweeps", ()):
            if "data_path" in sweep:
                sweep["data_path"] = _resolve_info_path(
                    sweep["data_path"], data_base
                )
    return data


def install_relative_nuscenes_path_compatibility(
    data_base: str | os.PathLike[str],
) -> None:
    """Resolve FlashOCC info PKL paths in memory without rewriting the PKL."""
    import mmcv

    if getattr(mmcv.load, "_flashocc_relative_paths", False):
        return

    original_load = mmcv.load
    resolved_base = Path(data_base).expanduser().resolve()
    if not resolved_base.is_dir():
        raise FileNotFoundError(
            f"FlashOCC data path base does not exist: {resolved_base}"
        )

    def load_with_flashocc_paths(filename, *args, **kwargs):
        data = original_load(filename, *args, **kwargs)
        try:
            source = getattr(filename, "name", filename)
            basename = Path(os.fspath(source)).name
        except TypeError:
            return data
        if basename in _FLASHOCC_INFO_FILES:
            return _resolve_nuscenes_info_paths(data, resolved_base)
        return data

    load_with_flashocc_paths._flashocc_relative_paths = True
    mmcv.load = load_with_flashocc_paths


def install_compatibility(
    model_root: str | os.PathLike[str],
    runtime_root: str | os.PathLike[str],
) -> None:
    model_path = Path(model_root).expanduser().resolve()
    runtime_path = Path(runtime_root).expanduser().resolve()
    optimization_path = Path(__file__).resolve().parent
    sys.path[:] = order_import_roots(
        sys.path, model_path, runtime_path, optimization_path
    )
    install_config_path_compatibility(model_path, runtime_path)

    data_base = os.environ.get(
        "FLASHOCC_DATA_BASE", str(model_path.parent / "static_data")
    )
    install_relative_nuscenes_path_compatibility(data_base)

    sys.meta_path.insert(0, LegacyEncodingSourceFinder(model_path))
    finder = PrebuiltExtensionFinder(_extension_files(runtime_path))
    sys.meta_path.insert(0, finder)
    install_dvr_compatibility(optimization_path)
