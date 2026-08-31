#!/usr/bin/env python3
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Check local Markdown links and basic documentation hygiene."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_DOCS = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "RELEASE_NOTES.md",
)
DOC_TREES = (ROOT / "docs", ROOT / "model_examples")

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
ABSOLUTE_LOCAL_PATH_RE = re.compile(
    r"(?:/Users/[^/\s]+|/home/[^/\s]+|/root|/public/home)/[^\s`'\")]+"
)
INTERNAL_IPV4_RE = re.compile(
    r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}"
    r"|192\.168\.(?:\d{1,3}\.)\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.(?:\d{1,3}\.)\d{1,3})\b"
)


def markdown_files() -> list[Path]:
    files = [path for path in TOP_LEVEL_DOCS if path.exists()]
    for tree in DOC_TREES:
        if tree.exists():
            files.extend(tree.rglob("*.md"))
    return sorted(set(files))


def normalize_link(raw_link: str) -> str:
    link = raw_link.strip()
    if link.startswith("<") and link.endswith(">"):
        link = link[1:-1]
    # Markdown permits an optional title after a URL. The repository docs do
    # not use spaces in local paths, so splitting here remains deterministic.
    return unquote(link.split(maxsplit=1)[0]) if link else link


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    relative = path.relative_to(ROOT)

    if len(FENCE_RE.findall(text)) % 2:
        errors.append(f"{relative}: Markdown code fence is not closed")

    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.rstrip() != line:
            errors.append(f"{relative}:{line_number}: trailing whitespace")

    for match in LINK_RE.finditer(text):
        link = normalize_link(match.group(1))
        if not link or link.startswith(("#", "http://", "https://", "mailto:")):
            continue
        local_path = link.split("#", 1)[0].split("?", 1)[0]
        if not local_path:
            continue
        resolved = (path.parent / local_path).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            errors.append(f"{relative}: local link escapes repository: {link}")
            continue
        if not resolved.exists():
            errors.append(f"{relative}: broken local link: {link}")

    for match in ABSOLUTE_LOCAL_PATH_RE.finditer(text):
        errors.append(
            f"{relative}: personal or machine-local path must not be published: "
            f"{match.group(0)}"
        )

    for match in INTERNAL_IPV4_RE.finditer(text):
        errors.append(
            f"{relative}: private network address must not be published: "
            f"{match.group(0)}"
        )

    return errors


def main() -> int:
    files = markdown_files()
    errors: list[str] = []

    required = (
        ROOT / "README.md",
        ROOT / "docs/README.md",
        ROOT / "docs/zh/get_started/quick_start.md",
        ROOT / "docs/zh/models/support_list.md",
    )
    for path in required:
        if not path.exists():
            errors.append(f"missing required document: {path.relative_to(ROOT)}")

    for path in files:
        errors.extend(check_file(path))

    if errors:
        print("Documentation check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Documentation check passed: {len(files)} Markdown files checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
