"""Export an exact, reproducible Cosmos source delta without changing either repo."""
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--base", default="9726697a83315540c6885baefd2fe353d9c74920")
    parser.add_argument("--commit", default="fb7c6facdfed9bd5ae8442ccd501d23be573c3ac")
    args = parser.parse_args()
    source = args.source.resolve()

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={source.as_posix()}", "-C", str(source), *arguments]
        )

    base = git("rev-parse", f"{args.base}^{{commit}}").decode().strip()
    commit = git("rev-parse", f"{args.commit}^{{commit}}").decode().strip()
    patch = git("diff", "--binary", "--full-index", "--no-renames", "--no-ext-diff",
                "--no-textconv", base, commit, "--")
    paths = git("diff", "--name-only", "-z", "--no-renames", base, commit).decode().split("\0")
    files = {}
    for name in filter(None, paths):
        entry = {}
        for label, revision in (("base", base), ("target", commit)):
            if not git("ls-tree", revision, "--", name):
                entry[label] = None
            else:
                blob = git("show", f"{revision}:{name}")
                # Git text checkouts can have CRLF on Windows.
                entry[label] = hashlib.sha256(blob).hexdigest()
                if blob.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
                    for line in blob.decode("ascii").splitlines():
                        if line.startswith("oid sha256:"):
                            entry[label + "_lfs"] = line.removeprefix("oid sha256:")
        files[name] = entry
    output = Path(__file__).resolve().parents[1] / "patches" / "cosmos3"
    output.mkdir(parents=True, exist_ok=True)
    (output / "cosmos3.patch").write_bytes(patch)
    manifest = {
        "base_commit": base, "source_commit": commit,
        "base_tree": git("rev-parse", f"{base}^{{tree}}").decode().strip(),
        "source_tree": git("rev-parse", f"{commit}^{{tree}}").decode().strip(),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "files": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(files)} paths, {len(patch)} bytes: {base[:8]} -> {commit[:8]}")


if __name__ == "__main__":
    main()
