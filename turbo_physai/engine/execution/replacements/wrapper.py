# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...contracts import Mechanism, ReplacementSpec
from .base import (
    HandlerError,
    MechanismHandler,
    PreparedReplacement,
    prepare_runtime_condition,
    resolve_replacement,
    resolve_targets,
)


class WrapperHandler(MechanismHandler):
    mechanism = Mechanism.WRAPPER

    def prepare(
        self,
        spec: ReplacementSpec,
        options: Mapping[str, Any],
        *,
        import_missing: bool = True,
    ) -> PreparedReplacement:
        targets = resolve_targets(spec, import_missing=import_missing)
        wrapper = resolve_replacement(
            spec.replacement, import_missing=import_missing
        )
        if not callable(wrapper):
            raise HandlerError(f"wrapper must be callable: {spec.replacement_id}")
        replacement = wrapper(targets[0].original, options)
        if not callable(replacement):
            raise HandlerError(f"wrapper must return callable: {spec.replacement_id}")
        installed, condition = prepare_runtime_condition(
            spec,
            targets[0].original,
            replacement,
            import_missing=import_missing,
        )
        return PreparedReplacement(spec, targets, installed, replacement, condition)
