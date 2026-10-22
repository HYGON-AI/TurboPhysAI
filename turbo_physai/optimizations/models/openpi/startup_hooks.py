# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration-gated pre-import hooks for optional OpenPI optimizations."""

from __future__ import annotations

import builtins as _builtins
import os as _os
from pathlib import Path as _Path
import sys as _sys


_NUMA_GROUP = "openpi.numa"
_NUMA_PREBOUND = "OPENPI_TURBO_NUMA_PREBOUND"
_TRAIN_FUNCTIONS = ("setup_ddp", "init_logging", "train_loop")
_DIRECT_ENTRY_BRIDGE_GROUPS = frozenset(
    {
        "openpi.numa",
        "openpi.compile_stack",
        "openpi.deferred_metric_sync",
    }
)


def _enabled_groups() -> set[str]:
    if _os.environ.get("TURBO_PHYSAI_BOOTSTRAP") != "1":
        return set()
    config_path = _os.environ.get("TURBO_PHYSAI_OPTIMIZATION_CONFIG")
    if not config_path:
        return set()
    try:
        import yaml

        config = yaml.safe_load(_Path(config_path).read_text(encoding="utf-8")) or {}
    except Exception as exc:
        if "LOCAL_RANK" in _os.environ:
            raise SystemExit(
                f"[openpi-turbo] cannot read optimization config before rank startup: {exc}"
            )
        return set()
    groups = config.get("optimization_groups", []) if isinstance(config, dict) else []
    return {
        item.get("id")
        for item in groups
        if isinstance(item, dict) and item.get("enabled") is True
    }


def _bind_rank_before_torch_import(enabled_groups: set[str]) -> None:
    if (
        _NUMA_GROUP not in enabled_groups
        or "LOCAL_RANK" not in _os.environ
        or _os.environ.get(_NUMA_PREBOUND) == "1"
    ):
        return

    # sitecustomize runs in each torchrun worker before Turbo/OpenPI/Torch
    # imports. Bind here so setup_ddp never falls back to late CPU placement.
    from turbo_physai.optimizations.models.openpi.numa_topology import bind_current_rank_or_log

    result = bind_current_rank_or_log()
    if result is None:
        raise SystemExit("[openpi-turbo] pre-Torch NUMA binding failed; refusing a late bind")
    _os.environ[_NUMA_PREBOUND] = "1"


def _is_direct_train_script() -> bool:
    if not _sys.argv:
        return False
    script = _Path(_sys.argv[0])
    return script.name == "train_pytorch.py" and script.parent.name == "scripts"


def _install_direct_entry_bridge() -> None:
    if not _is_direct_train_script():
        return
    if _sys.gettrace() is not None:
        raise SystemExit(
            "[openpi-turbo] cannot synchronize patched entry functions while another trace is active"
        )

    target_file = str(_Path(_sys.argv[0]).resolve())
    patched_module = _sys.modules.get("scripts.train_pytorch")
    if patched_module is None:
        raise SystemExit(
            "[openpi-turbo] Turbo did not import scripts.train_pytorch before the direct script"
        )

    def synchronize_before_main(frame, event, arg):
        del arg
        if str(_Path(frame.f_code.co_filename).resolve()) != target_file:
            return None
        if event != "line":
            return synchronize_before_main
        namespace = frame.f_globals
        if not all(name in namespace for name in _TRAIN_FUNCTIONS):
            return synchronize_before_main

        for name in _TRAIN_FUNCTIONS:
            replacement = getattr(patched_module, name, None)
            if replacement is None:
                raise SystemExit(f"[openpi-turbo] patched module is missing {name}")
            namespace[name] = replacement

        # Disable tracing before main() starts; no training-step trace overhead.
        _sys.settrace(None)
        frame.f_trace = None
        print(
            "[openpi-turbo] direct scripts/train_pytorch.py entry uses Turbo-patched functions",
            flush=True,
        )
        return None

    _sys.settrace(synchronize_before_main)


def _wrap_turbo_activate(bootstrap) -> None:
    if getattr(bootstrap, "_openpi_direct_entry_hook_installed", False):
        return
    original_activate = bootstrap.activate

    def activate_then_bridge(*args, **kwargs):
        result = original_activate(*args, **kwargs)
        _install_direct_entry_bridge()
        return result

    bootstrap.activate = activate_then_bridge
    bootstrap._openpi_direct_entry_hook_installed = True


def _install_turbo_activate_hook() -> None:
    bootstrap = _sys.modules.get("turbo_physai.bootstrap")
    if bootstrap is not None:
        _wrap_turbo_activate(bootstrap)
        return

    original_import = _builtins.__import__

    def import_turbo_bootstrap(name, globals=None, locals=None, fromlist=(), level=0):
        try:
            module = original_import(name, globals, locals, fromlist, level)
        except BaseException:
            if name == "turbo_physai.bootstrap":
                _builtins.__import__ = original_import
            raise
        if name == "turbo_physai.bootstrap" and "activate" in fromlist:
            _builtins.__import__ = original_import
            loaded = _sys.modules.get(name)
            if loaded is not None:
                _wrap_turbo_activate(loaded)
        return module

    _builtins.__import__ = import_turbo_bootstrap


def install_pre_torch_hooks() -> None:
    """Install only hooks needed by the active optimization configuration."""
    enabled_groups = _enabled_groups()
    _bind_rank_before_torch_import(enabled_groups)

    # The official direct script has its own __main__ function objects. Copy
    # Turbo replacements into that namespace only when an enabled group wraps
    # one of those functions; baseline runs need no Turbo entry bridge.
    if enabled_groups.intersection(_DIRECT_ENTRY_BRIDGE_GROUPS):
        _install_turbo_activate_hook()
