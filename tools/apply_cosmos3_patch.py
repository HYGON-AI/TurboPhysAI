# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit source patch lifecycle; never stage, commit, or alter the source checkout HEAD."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


class PatchError(RuntimeError):
    """Patch application or verification error."""


PATCH_DIR = Path(__file__).resolve().parents[1] / "patches" / "cosmos3"


def _git(repo, *arguments):
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *arguments], capture_output=True, check=False
        )
    except OSError as exc:
        raise PatchError(f"Cannot execute git: {exc}") from exc
    if result.returncode:
        raise PatchError(result.stderr.decode(errors="replace").strip())
    return result.stdout.decode(errors="replace").strip()


def _bundle():
    try:
        manifest = json.loads((PATCH_DIR / "manifest.json").read_text(encoding="utf-8"))
        patch = PATCH_DIR / "cosmos3.patch"
        actual_hash = hashlib.sha256(patch.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise PatchError(f"Cannot load Cosmos3 patch bundle: {exc}") from exc
    if actual_hash != manifest["patch_sha256"]:
        raise PatchError("Cosmos3 patch checksum mismatch")
    return manifest, patch


def _state(repo, manifest):
    repo = Path(repo).expanduser().resolve()
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise PatchError("--repo must be the Cosmos repository root")
    head = _git(repo, "rev-parse", "HEAD")
    if head not in (manifest["base_commit"], manifest["source_commit"]):
        raise PatchError(
            f"Unsupported Cosmos HEAD {head}; expected {manifest['base_commit']} "
            f"or {manifest['source_commit']}. Regenerate/review the patch for a new baseline."
        )
    states = {"base", "target"}
    for name, expected in manifest["files"].items():
        path = repo / name
        if path.is_symlink():
            raise PatchError(f"Refusing symlink patch target: {name}")
        if path.exists():
            raw = path.read_bytes()
            hashes = {hashlib.sha256(raw).hexdigest()}
            if b"\0" not in raw:
                hashes.add(hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest())
        else:
            hashes = {None}
        states &= {
            label for label in ("base", "target")
            if expected[label] in hashes or (
                expected.get(label + "_lfs") is not None
                and expected[label + "_lfs"] in hashes
            )
        }
        if not states:
            raise PatchError(f"Modified or partially patched Cosmos tree at {name}; refusing to overwrite")
    return repo, ("target" if "target" in states else "base")


def manage_patch(repo, action="check"):
    manifest, patch = _bundle()
    repo, state = _state(repo, manifest)
    if action == "verify":
        if state != "target":
            raise PatchError("Cosmos3 patch is not applied; run apply_cosmos3_patch.py apply first")
    elif action == "check":
        if state == "base":
            _git(repo, "apply", "--check", str(patch))
    elif action in ("apply", "reverse"):
        wanted = "target" if action == "apply" else "base"
        if state != wanted:
            staged = set(_git(repo, "diff", "--cached", "--name-only", "-z").split("\0"))
            if staged.intersection(manifest["files"]):
                raise PatchError("Patch paths have staged changes; refusing to alter their worktree content")
            # Only reverse changes on the baseline checkout.
            if action == "reverse" and _git(repo, "rev-parse", "HEAD") != manifest["base_commit"]:
                raise PatchError("Cannot reverse a committed adapted version; use a baseline checkout")
            options = ["--reverse"] if action == "reverse" else []
            _git(repo, "apply", "--check", *options, str(patch))
            _git(repo, "apply", *options, str(patch))
            _, state = _state(repo, manifest)
            if state != wanted:
                raise PatchError(f"Patch {action} verification failed")
    else:
        raise PatchError(f"Unknown patch action: {action}")
    return {"repo": str(repo), "state": state, "source_commit": manifest["source_commit"],
            "paths": len(manifest["files"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "apply", "reverse", "verify"))
    parser.add_argument("--repo", default=".", type=Path, help="Cosmos Git repository root")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(manage_patch(args.repo, args.action), ensure_ascii=False))
    except (PatchError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
