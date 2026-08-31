# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ...contracts import Mechanism, ReplacementSpec
from .base import (
    HandlerError,
    MechanismHandler,
    PreparedReplacement,
    prepare_runtime_condition,
    resolve_replacement,
    resolve_targets,
)


class ReplaceHandler(MechanismHandler):
    """Infer direct function/class replacement after real objects are available."""

    mechanism = Mechanism.REPLACE

    def prepare(
        self,
        spec: ReplacementSpec,
        options,
        *,
        import_missing: bool = True,
    ) -> PreparedReplacement:
        del options
        targets = resolve_targets(spec, import_missing=import_missing)
        replacement_value = resolve_replacement(
            spec.replacement, import_missing=import_missing
        )
        original = targets[0].original
        if isinstance(original, property):
            if not isinstance(replacement_value, property):
                raise HandlerError(
                    f"property replacement must be a property: "
                    f"{spec.replacement_id}"
                )
        elif isinstance(original, type):
            if not isinstance(replacement_value, type):
                raise HandlerError(
                    f"class replacement must be a class: {spec.replacement_id}"
                )
        elif callable(original):
            if not callable(replacement_value) or isinstance(replacement_value, type):
                raise HandlerError(
                    f"callable replacement must be non-class callable: "
                    f"{spec.replacement_id}"
                )
        else:
            raise HandlerError(
                f"replace target must be a callable or class: {spec.replacement_id}"
            )
        installed, condition = prepare_runtime_condition(
            spec,
            original,
            replacement_value,
            import_missing=import_missing,
        )
        return PreparedReplacement(
            spec,
            targets,
            installed,
            replacement_value,
            condition,
        )
