# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Run an entry script with an engine callback before its training entry block."""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import tokenize
import uuid
from dataclasses import dataclass
from functools import lru_cache
from importlib.machinery import PathFinder, SourceFileLoader
from pathlib import Path
from types import ModuleType

from . import _python_entry_index
from ..engine.contracts import CheckResult, CheckStatus, Mechanism
from ..engine.execution.replacements.base import bind_entry_module

RUNNER_MODULE = "turbo_physai.bootstrap.script_entry"


def _find_spec(name, paths):
    """Find an on-disk module without executing its parent packages."""
    parts = name.split(".")
    spec = None
    for index in range(1, len(parts) + 1):
        spec = PathFinder.find_spec(parts[index - 1], paths)
        if spec is None:
            return None
        paths = spec.submodule_search_locations
        if index < len(parts) and paths is None:
            return None
    return spec


@dataclass(frozen=True)
class PythonEntry:
    path: Path
    argv: tuple[str, ...]
    module_name: str | None
    python_options: tuple[str, ...]
    path0: str | None


def find_entry(arguments):
    """Identify a Python rank's file or '-m' entry without importing it."""
    index = _python_entry_index(arguments)
    options = tuple(arguments[:index])
    args = tuple(arguments[index:])
    if not args or args[0] == "-" or args[0].startswith("-c"):
        return None

    safe_path = getattr(sys.flags, "safe_path", False)
    if args[0].startswith("-m"):
        module_args = (args[0][2:], *args[1:]) if args[0][2:] else args[1:]
        if not module_args:
            return None
        name = module_args[0]
        spec = _find_spec(name, sys.path)
        if spec and spec.submodule_search_locations is not None:
            name += ".__main__"
            spec = _find_spec(name, sys.path)
        if not spec or not isinstance(spec.loader, SourceFileLoader):
            return None
        return PythonEntry(
            Path(spec.origin).resolve(),
            (spec.origin, *module_args[1:]),
            name,
            options,
            None if safe_path else str(Path.cwd()),
        )

    path = Path(args[0]).resolve()
    if path.suffix != ".py" or not path.is_file():
        return None
    # '--' ends Python option parsing and must not precede the internal '-m'.
    return PythonEntry(
        path, args, None, options[:-1] if options[-1:] == ("--",) else options,
        None if safe_path else str(path.parent),
    )


def read_entry(entry):
    with tokenize.open(entry.path) as source:
        tree = ast.parse(source.read(), filename=str(entry.path))
    guards = [
        node for node in tree.body if isinstance(node, ast.If) and _is_main_guard(node.test)
    ]
    return tree, guards[0] if len(guards) == 1 else None


def _is_main_guard(test):
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    sides = (test.left, test.comparators[0])
    return any(
        isinstance(name, ast.Name) and name.id == "__name__"
        and isinstance(value, ast.Constant) and value.value == "__main__"
        for name, value in (sides, sides[::-1])
    )


@dataclass(frozen=True)
class EntryGroups:
    targets: dict[str, tuple[str, str]]
    direct: frozenset[str]
    deferred: frozenset[str]

    @property
    def module_names(self):
        return {module for module, _ in self.targets.values()}

    def target_key(self, path):
        return "__main__." + self.targets[path][1] if path in self.targets else path

    def block(self, reason):
        return {
            group_id: CheckResult("entrypoint.unavailable", CheckStatus.FAIL, detail=reason)
            for group_id in self.direct
        }


def find_entry_groups(entry, application):
    """Select entire Groups and their dependents using effective enablement."""
    paths = [entry.path0, *sys.path] if entry.path0 is not None else sys.path

    @lru_cache(maxsize=None)
    def module_file(name):
        spec = _find_spec(name, paths)
        return Path(spec.origin).resolve() if spec and isinstance(spec.loader, SourceFileLoader) else None

    targets = {}
    direct = set()
    definitions = [
        application.registry.get_group(group.id)
        for group in application.regular.optimization_groups if group.enabled
    ]
    definitions = [group for group in definitions if group is not None]
    for group in definitions:
        for member in group.members:
            spec = application.registry.get_spec(member)
            if spec is None or spec.mechanism not in {Mechanism.REPLACE, Mechanism.WRAPPER}:
                continue
            for target in (spec.target, *spec.aliases):
                parts = target.split(".")
                for index in range(len(parts) - 1, 0, -1):
                    module = ".".join(parts[:index])
                    file = module_file(module)
                    if file is None:
                        continue
                    if file == entry.path:
                        targets[target] = (module, ".".join(parts[index:]))
                        direct.add(group.group_id)
                    break

    deferred = set(direct)
    while True:
        dependents = {
            group.group_id for group in definitions
            if deferred.intersection(group.depends_on)
        }
        if dependents.issubset(deferred):
            return EntryGroups(targets, frozenset(direct), frozenset(deferred))
        deferred.update(dependents)


def reexec_entry(command, entry):
    os.execv(
        sys.executable,
        [sys.executable, *entry.python_options, "-m", RUNNER_MODULE, *command[1:]],
    )


def _entry_namespace(entry):
    module = ModuleType("__main__")
    module.__file__ = entry.argv[0] if entry.module_name else os.path.abspath(entry.argv[0])
    module.__cached__ = None
    if entry.module_name:
        spec = importlib.util.find_spec(entry.module_name)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != entry.path:
            raise RuntimeError(f"training entry changed during startup: {entry.module_name}")
        module.__spec__ = spec
        module.__package__ = spec.parent
        module.__loader__ = spec.loader
        module.__cached__ = spec.cached
    else:
        module.__loader__ = SourceFileLoader("__main__", module.__file__)
    return module


def _prepare_entry(arguments):
    from . import _add_working_directory, _new_application, _report_application

    _add_working_directory()
    entry = find_entry(arguments)
    if entry is None:
        raise RuntimeError("cannot resolve the original Python training entry")
    if entry.path0 is not None and sys.path[:1] != [entry.path0]:
        sys.path.insert(0, entry.path0)
    sys.argv[:] = entry.argv
    sys.orig_argv[:] = [sys.executable, *arguments]

    application = _new_application()
    groups = find_entry_groups(entry, application)
    tree, guard = read_entry(entry)
    application.apply_compatibility()
    if not groups.direct or guard is None or not application.compatibility_ok:
        application.apply_regular(blocked_groups=groups.block("no unique top-level __main__ guard"))
        _report_application(application)
        return entry, tree, None

    application.apply_regular(
        group_ids={group.id for group in application.regular.optimization_groups} - groups.deferred,
        target_key=groups.target_key,
    )
    if application.terminal_error:
        application.block_pending_after_failure(groups.deferred)
        _report_application(application)

    return entry, tree, (guard, application, groups)


def main():
    from . import _abort_activation, _report_application

    try:
        entry, tree, pending = _prepare_entry(sys.argv[1:])
        module = _entry_namespace(entry)
    except BaseException as error:
        _abort_activation(error)

    sys.modules["__main__"] = module
    finalize = None
    if pending is not None:
        guard, application, groups = pending
        name = "_turbo_physai_entry_" + uuid.uuid4().hex

        def finish(reason=None):
            if module.__dict__.pop(name, None) is None:
                return
            try:
                imported = groups.module_names.intersection(sys.modules)
                if imported:
                    reason = "entry module already imported as " + ", ".join(sorted(imported))
                with bind_entry_module(groups.module_names, module):
                    application.apply_regular(
                        group_ids=groups.deferred,
                        blocked_groups=groups.block(reason) if reason else None,
                        target_key=groups.target_key,
                    )
                _report_application(application)
            except BaseException as error:
                _abort_activation(error)

        module.__dict__[name] = finish
        hook = ast.Expr(ast.Call(ast.Name(name, ast.Load()), [], []))
        guard.body.insert(0, ast.copy_location(hook, guard.body[0]))
        ast.fix_missing_locations(tree)
        finalize = finish

    # Execute user code outside our startup error handler. Preserve its exceptions
    # and exit status, even if it exits before reaching the entry block.
    try:
        exec(compile(tree, module.__file__, "exec", dont_inherit=True), module.__dict__)
    finally:
        if finalize is not None:
            finalize("the __main__ guard was not reached")


if __name__ == "__main__":
    main()
