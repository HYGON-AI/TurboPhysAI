# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure conflict analysis shared by declarations, config generation and preparation."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from ..contracts import Conflict, ReplacementSpec
from ..definitions.registry import Registry


def _targets_overlap(left: str, right: str, left_kind: str, right_kind: str) -> bool:
    if left == right:
        return True
    if "import_replace" not in {left_kind, right_kind}:
        return False
    return left.startswith(right + ".") or right.startswith(left + ".")


def find_target_conflicts(
    groups: Iterable[Tuple[str, Sequence[ReplacementSpec]]],
) -> List[Conflict]:
    """Find duplicate or incompatible target replacements between Groups."""

    replacements = []
    for group_id, specs in groups:
        for spec in specs:
            for target in (spec.target,) + spec.aliases:
                replacements.append(
                    (
                        group_id,
                        spec.replacement_id,
                        spec.mechanism.value,
                        target,
                        spec.replacement,
                        spec.runtime_condition,
                    )
                )

    conflicts: List[Conflict] = []
    seen = set()
    for index, left in enumerate(replacements):
        for right in replacements[index + 1 :]:
            if left[0] == right[0] and left[1] == right[1]:
                continue
            if not _targets_overlap(
                left[3],
                right[3],
                left[2],
                right[2],
            ):
                continue

            same_group = left[0] == right[0]
            same_replacement = (
                left[4] == right[4]
                and left[5] == right[5]
            )
            groups_in_conflict = (
                (left[0],)
                if same_group
                else tuple(sorted((left[0], right[0])))
            )
            key = (
                groups_in_conflict,
                left[1],
                right[1],
                left[3],
                right[3],
            )
            if key in seen:
                continue
            seen.add(key)

            if same_group:
                code = (
                    "target.intra_group_duplicate"
                    if same_replacement
                    else "target.intra_group_conflict"
                )
            else:
                code = (
                    "target.group_duplicate"
                    if same_replacement
                    else "target.group_conflict"
                )
            relation = "duplicate" if same_replacement else "incompatible"
            conflicts.append(
                Conflict(
                    code,
                    groups_in_conflict,
                    target=left[3],
                    detail=(
                        f"{relation} target replacements from {left[1]} and {right[1]}; "
                        "runtime_conditions="
                        f"{left[5] or '<none>'},"
                        f"{right[5] or '<none>'}"
                    ),
                    replacement_ids=(left[1], right[1]),
                    replacements=(
                        left[4],
                        right[4],
                    ),
                )
            )
    return conflicts


def find_group_conflicts(
    registry: Registry, enabled_ids: Sequence[str]
) -> List[Conflict]:
    """Find all conflicts for the enabled Group composition."""

    enabled = set(enabled_ids)
    conflicts: List[Conflict] = []
    grouped_specs = []

    for group_id in enabled_ids:
        group = registry.get_group(group_id)
        if group is None:
            continue
        for dependency in group.depends_on:
            if dependency not in enabled:
                conflicts.append(
                    Conflict(
                        "dependency.missing",
                        (group_id,),
                        detail=(
                            "dependency OptimizationGroup is not selected: "
                            f"{dependency}"
                        ),
                    )
                )
        grouped_specs.append(
            (
                group_id,
                tuple(
                    spec
                    for replacement_id in group.members
                    if (spec := registry.get_spec(replacement_id)) is not None
                ),
            )
        )

    conflicts.extend(find_target_conflicts(grouped_specs))
    return conflicts


def format_conflict(conflict: Conflict) -> str:
    """Render deterministic evidence for generation errors."""

    parts = [conflict.code, f"groups={','.join(conflict.groups)}"]
    if conflict.replacement_ids:
        parts.append(f"members={','.join(conflict.replacement_ids)}")
    if conflict.target:
        parts.append(f"target={conflict.target}")
    if conflict.replacements:
        parts.append(f"replacements={','.join(conflict.replacements)}")
    if conflict.detail:
        parts.append(conflict.detail)
    return "; ".join(parts)


__all__ = [
    "find_target_conflicts",
    "find_group_conflicts",
    "format_conflict",
]
