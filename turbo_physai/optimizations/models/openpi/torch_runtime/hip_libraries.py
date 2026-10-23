# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Build audited HIP helpers lazily into an external cache, not site-packages."""

from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


_FLAGS = ("-O3", "-ffp-contract=fast", "--offload-arch=gfx938", "-std=c++17", "-shared", "-fPIC")


def ensure_library(source_name: str, library_stem: str) -> Path:
    """Return the compiled helper, building it once per source/compiler/arch."""

    source = Path(__file__).with_name(source_name)
    if not source.is_file():
        raise FileNotFoundError(f"Turbo OpenPI HIP source is missing from the installed package: {source}")

    compiler_hint = os.environ.get("HIPCC") or shutil.which("hipcc") or "/opt/dtk/bin/hipcc"
    compiler = shutil.which(compiler_hint) or compiler_hint
    if not Path(compiler).is_file():
        raise RuntimeError(f"HIP compiler not found at {compiler!r}; check the HCU runtime image")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("OpenPI Turbo HIP helper requires an initialized HCU device")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if architecture != "gfx938":
        raise RuntimeError(f"OpenPI Turbo HIP helper supports gfx938, observed {architecture or 'unknown'}")

    version = subprocess.run([compiler, "--version"], text=True, capture_output=True, check=True)
    identity = hashlib.sha256(
        source.read_bytes() + version.stdout.encode() + "\0".join(_FLAGS).encode()
    ).hexdigest()[:20]
    cache_root = Path(
        os.environ.get(
            "OPENPI_TORCH_KERNEL_CACHE_DIR",
            str(Path(tempfile.gettempdir()) / "openpi_turbo_kernels"),
        )
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    library = cache_root / f"{library_stem}-{identity}.so"
    lock_path = cache_root / f"{library_stem}-{identity}.lock"

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if library.is_file():
            return library
        temporary = cache_root / f".{library_stem}-{identity}-{os.getpid()}.building.so"
        command = [compiler, *_FLAGS, str(source), "-o", str(temporary)]
        subprocess.run(command, cwd=source.parent, check=True)
        os.replace(temporary, library)
    return library
