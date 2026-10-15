"""Always-loaded startup dispatcher for OpenPI compatibility and opt-in hooks."""

from turbo_physai.optimizations.models.openpi.startup_hooks import install_pre_torch_hooks

install_pre_torch_hooks()

from turbo_physai.optimizations.models.openpi.runtime_compatibility import apply_compatibility_shims

apply_compatibility_shims()
