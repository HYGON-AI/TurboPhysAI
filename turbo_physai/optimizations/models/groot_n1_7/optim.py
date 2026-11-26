"""Select Apex AdamW through the Transformers optimizer factory."""

from functools import wraps
from importlib import import_module

from turbo_physai import CompatibilityResult


def apex_available_check(context):
    try:
        fused = import_module("apex.optimizers").FusedAdam
        applier = import_module("apex.multi_tensor_apply").multi_tensor_applier
        kernels = import_module("amp_C")
        ok = callable(fused) and bool(applier.available) and callable(getattr(kernels, "multi_tensor_adam", None))
    except (ImportError, OSError, AttributeError) as exc:
        return CompatibilityResult(compatible=False, reason=f"Apex kernels unavailable: {exc}")
    return CompatibilityResult(compatible=ok, reason="Apex FusedAdam and multi-tensor kernels required")


def create_optimizer_apex_wrapper(original, options):
    @wraps(original)
    def wrapped(self, *args, **kwargs):
        args_obj = self.args
        original_optim = args_obj.optim
        plugin = getattr(getattr(getattr(self, "accelerator", None), "state", None), "deepspeed_plugin", None)
        config = getattr(plugin, "deepspeed_config", None) or {}
        zero = config.get("zero_optimization", {})
        offload = zero.get("offload_optimizer") or {}
        supported = (getattr(original_optim, "value", original_optim) in {"adamw_torch", "adamw_torch_fused"}
                     and "optimizer" not in config and not zero.get("cpu_offload")
                     and offload.get("device", "none") == "none")
        if not supported:
            return original(self, *args, **kwargs)
        try:
            args_obj.optim = "adamw_apex_fused"
            return original(self, *args, **kwargs)
        finally:
            args_obj.optim = original_optim
    return wrapped
