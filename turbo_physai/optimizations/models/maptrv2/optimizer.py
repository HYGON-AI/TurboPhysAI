# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in Apex AdamW with lossless, ordinary AdamW checkpoints.

Only Apex's non-capturable, group-step implementation and FP32 parameters are
supported. AMP with FP32 parameters is supported; FP16 master-weight optimizers
are not. Imports and registration are deferred until an AdamW is constructed.
"""

import ast
import copy
import functools
import inspect
import logging
import math
import textwrap


_NAME = "TurboPhysAIMapTRv2FusedAdamW"
_LOG = logging.getLogger(__name__)
_CLASSES = {}


def _integer_step(value, where):
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{where}: optimizer step must be a scalar")
        value = value.item()
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or int(value) != value):
        raise ValueError(f"{where}: invalid optimizer step {value!r}")
    return int(value)


def _validate_group(group, where):
    for flag in ("amsgrad", "maximize", "capturable", "differentiable", "master_weights"):
        if group.get(flag, False):
            raise ValueError(f"{where}: {flag}=True is unsupported by this Apex adapter")
    if not group.get("bias_correction", True):
        raise ValueError(f"{where}: bias_correction=False is not AdamW-compatible")
    if not group.get("adam_w_mode", True):
        raise ValueError(f"{where}: adam_w_mode must be True")


def _to_apex_state(state_dict):
    """Copy and validate *all* groups before modifying a live optimizer."""
    result = copy.deepcopy(state_dict)
    for index, group in enumerate(result["param_groups"]):
        where = f"optimizer group {index}"
        _validate_group(group, where)
        group_step = group.get("step")
        if group_step is not None:
            group_step = _integer_step(group_step, where)
        steps = []
        for parameter_id in group["params"]:
            state = result["state"].get(parameter_id, {})
            if not state:
                # Frozen/never-used parameters have no clock. They may remain
                # unused after resume; a late first gradient is checked in step.
                continue
            if set(state) - {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(f"{where}: unsupported optimizer state fields")
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                raise ValueError(f"{where}: missing AdamW moment tensors")
            if "step" not in state and group_step is None:
                raise ValueError(f"{where}: missing optimizer step")
            step = _integer_step(state.get("step", group_step), where)
            state["step"] = step
            steps.append(step)
        # Never silently collapse divergent parameter clocks to a group clock.
        if len(set(steps)) > 1:
            raise ValueError(f"{where}: unequal parameter steps cannot be converted to Apex")
        step = steps[0] if steps else 0
        if group_step is not None and steps and step != group_step:
            raise ValueError(f"{where}: parameter and group steps disagree")
        group["step"] = step
        group["bias_correction"] = True
        for key in ("amsgrad", "maximize", "foreach", "capturable", "differentiable",
                    "fused", "master_weights", "adam_w_mode"):
            group.pop(key, None)
    return result


def _to_adamw_state(state_dict):
    """Export portable PyTorch state, including per-parameter step tensors."""
    import torch

    # Copy dictionaries only: match Optimizer.state_dict's tensor aliasing and
    # avoid duplicating the full moment tensors on every checkpoint save.
    result = dict(state_dict)
    result["state"] = {key: dict(value) for key, value in state_dict["state"].items()}
    result["param_groups"] = [dict(group) for group in state_dict["param_groups"]]
    for group in result["param_groups"]:
        for parameter_id in group["params"]:
            state = result["state"].get(parameter_id)
            if state:
                step = _integer_step(state["step"], "checkpoint")
                if step > 2**53:
                    raise ValueError("optimizer step exceeds the exact portable floating-point range")
                state["step"] = torch.tensor(
                    step, dtype=torch.float32 if step <= 2**24 else torch.float64)
        group.pop("step", None)
        group.pop("bias_correction", None)
        group.update(amsgrad=False, maximize=False, foreach=None,
                     capturable=False, differentiable=False, fused=None)
    return result


def _validate_apex_contract(base):
    """Reject forks using a different clock representation instead of guessing."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(base.step)))
    except (OSError, TypeError, SyntaxError) as exc:
        raise RuntimeError("Cannot inspect Apex FusedAdam step contract") from exc
    clocks = {
        node.value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Constant) and node.slice.value == "step"
    }
    if "group" not in clocks or "state" in clocks:
        raise RuntimeError("Unsupported Apex FusedAdam: expected the group-step implementation")


def _load_apex():
    try:
        import torch
        from apex.optimizers import FusedAdam
        from apex.multi_tensor_apply import multi_tensor_applier
        import amp_C
    except (ImportError, OSError) as exc:
        raise RuntimeError("maptrv2.fused_adamw requires Apex with compiled amp_C kernels") from exc
    if not torch.cuda.is_available() or not multi_tensor_applier.available:
        raise RuntimeError("maptrv2.fused_adamw requires a device and Apex fused extensions")
    if not callable(getattr(amp_C, "multi_tensor_adam", None)):
        raise RuntimeError("Apex amp_C.multi_tensor_adam is unavailable")
    _validate_apex_contract(FusedAdam)
    return FusedAdam


def _make_fused_adamw(base):
    """Build the adapter lazily; also allows CPU contract tests with a test double."""
    import torch

    if base in _CLASSES:
        return _CLASSES[base]

    class PortableFusedAdamW(base):
        __turbo_physai_fused_adamw_optimizer__ = True

        def __init__(self, params, **kwargs):
            _validate_group(kwargs, "optimizer config")
            for key in ("foreach", "fused"):
                if kwargs.pop(key, None) not in (None, False):
                    raise ValueError(f"{key}=True is unsupported; select the Apex group instead")
            for key in ("maximize", "differentiable"):
                kwargs.pop(key, None)
            kwargs["adam_w_mode"] = True
            kwargs["bias_correction"] = True
            kwargs.setdefault("weight_decay", 0.01)  # PyTorch AdamW default
            inspect.signature(base.__init__).bind(None, params, **kwargs)
            super().__init__(params, **kwargs)
            for index, group in enumerate(self.param_groups):
                _validate_group(group, f"optimizer group {index}")
                if any(p.dtype != torch.float32 for p in group["params"]):
                    raise ValueError("Apex adapter requires FP32 parameters (AMP is supported)")
            if getattr(self, "capturable", False) or getattr(self, "master_weights", False):
                raise ValueError("Apex capturable/master_weights modes are unsupported")

        def zero_grad(self, set_to_none=None):
            # Support MMCV's no-argument call and PyTorch's explicit keyword.
            if set_to_none is None:
                set_to_none = getattr(self, "set_grad_none", True)
            return torch.optim.Optimizer.zero_grad(self, set_to_none=set_to_none)

        def load_state_dict(self, state_dict):
            converted = _to_apex_state(state_dict)
            if len(converted["param_groups"]) != len(self.param_groups):
                raise ValueError("optimizer checkpoint has a different number of parameter groups")
            for index, (saved, live) in enumerate(zip(converted["param_groups"], self.param_groups)):
                if len(saved["params"]) != len(live["params"]):
                    raise ValueError(f"optimizer group {index}: parameter count mismatch")
                for identifier, parameter in zip(saved["params"], live["params"]):
                    state = converted["state"].get(identifier, {})
                    for key in ("exp_avg", "exp_avg_sq"):
                        value = state.get(key)
                        if value is not None and (
                            not isinstance(value, torch.Tensor)
                            or value.shape != parameter.shape or value.dtype != torch.float32
                        ):
                            raise ValueError(f"optimizer group {index}: incompatible {key}")
            result = super().load_state_dict(converted)
            for state in self.state.values():
                if state:
                    state["step"] = _integer_step(state["step"], "loaded optimizer")
            return result

        def state_dict(self):
            return _to_adamw_state(super().state_dict())

        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            active_groups = []
            for index, group in enumerate(self.param_groups):
                _validate_group(group, f"optimizer group {index}")
                step = group.get("step", 0)
                active = [p for p in group["params"] if p.grad is not None]
                if active and any(
                    p.grad is None and self.state.get(p)
                    for p in group["params"]
                ):
                    raise ValueError(
                        f"optimizer group {index}: partial grad=None would diverge parameter steps; "
                        "use ordinary AdamW or separate parameter groups")
                for parameter in active:
                    if parameter.grad.is_sparse:
                        raise ValueError("Apex AdamW does not support sparse gradients")
                    if self.state.get(parameter, {}).get("step", 0) != step:
                        raise ValueError(
                            f"optimizer group {index}: unequal parameter steps after grad=None; "
                            "use ordinary AdamW or separate parameter groups")
                active_groups.append((group, step, active))
            super().step()
            for group, previous_step, active in active_groups:
                if active:
                    for parameter in active:
                        self.state[parameter]["step"] = previous_step + 1
                else:
                    # Apex increments even entirely inactive groups. AdamW
                    # does not: preserve the clock for a later first gradient.
                    group["step"] = previous_step
            return loss

    PortableFusedAdamW.__name__ = _NAME
    PortableFusedAdamW.__qualname__ = _NAME
    _CLASSES[base] = PortableFusedAdamW
    return PortableFusedAdamW


def fused_adamw_builder_wrapper(original, options):
    if options:
        raise ValueError("maptrv2.fused_adamw does not accept options")
    if getattr(original, "__turbo_physai_fused_adamw__", False):
        return original

    @functools.wraps(original)
    def build_optimizer(model, cfg):
        if cfg.get("type") != "AdamW":
            return original(model, cfg)
        from mmcv.runner.optimizer.builder import OPTIMIZERS

        adapter = _make_fused_adamw(_load_apex())
        existing = OPTIMIZERS.get(_NAME)
        if existing is None:
            OPTIMIZERS.register_module(name=_NAME, module=adapter)
        elif existing is not adapter:
            raise RuntimeError(f"Optimizer registry conflict: {_NAME}")
        config = copy.deepcopy(cfg)
        config["type"] = _NAME
        optimizer = original(model, config)
        if not isinstance(optimizer, adapter):
            raise RuntimeError("MMCV optimizer constructor did not return the requested Apex adapter")
        _LOG.info("MapTRv2 optimizer: %s (portable AdamW checkpoint format)", _NAME)
        return optimizer

    build_optimizer.__turbo_physai_fused_adamw__ = True
    return build_optimizer
