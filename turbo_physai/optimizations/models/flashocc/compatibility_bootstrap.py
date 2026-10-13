"""Load FlashOCC runtime shims before Turbo resolves replacement targets."""

from __future__ import annotations

import os


def install_for_turbo_run() -> bool:
    """Install the external HCU compatibility layer in Turbo training ranks."""
    if os.environ.get("TURBO_PHYSAI_BOOTSTRAP") != "1":
        return False

    model_root = os.environ.get("FLASHOCC_ROOT")
    runtime_root = os.environ.get("FLASHOCC_RUNTIME_ROOT")
    if not model_root or not runtime_root:
        raise RuntimeError("FlashOCC compatibility roots are not configured")

    from .compatibility import install_compatibility

    install_compatibility(model_root, runtime_root)
    return True
