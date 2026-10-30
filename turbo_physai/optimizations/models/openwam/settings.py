# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Read options from installed Turbo wrappers, never optimization env vars.

Wrapper preparation is side-effect free. Only wrappers actually installed on
their declared target contribute options, so checks and failed transactions do
not activate a Group. Options are frozen once execution begins in each rank.
"""

from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Mapping
from importlib import import_module

GROUP_DEFAULTS = {
    "openwam.data": {
        "uint8_preprocess": True,
        "vae_latent_cache": False,
        "vae_latent_cache_dir": None,
    },
    "openwam.compute": {
        "pointwise_compile": False,
        "rope_real": True,
        "lightop_norm": False,
        "linear_bias2d": False,
    },
    "openwam.architecture": {"constant_cache": True},
    "openwam.mot": {
        "mot_split_attn": True,
        "compile_mot": False,
        "checkpoint_skip_layers": 0,
        "sac_attention": False,
        "sac_ffn_full": False,
        "sac_gemm": False,
        "sac_layers": None,
        "sac_extra_ops": "",
    },
    "openwam.text": {
        "text_cache": True,
        "text_cache_size": 32,
        "text_cache_dir": None,
        "t5_prefix": True,
        "text_trim": True,
        "padding_plan_cpu": True,
        "fa2_padding": True,
        "fa2_mask_fused": False,
    },
    "openwam.vae": {
        "vae_channels_last": True,
        "vae_pointwise_compile": False,
        "compile_vae": False,
    },
    "openwam.training": {
        "zero_overlap": True,
        "zero_reduce_scatter": False,
        "zero_scaled_pack": False,
        "zero_norm_overlap": False,
        "online_encode": False,
        "deferred_gather": False,
        "fused_adamw": False,
        "host_async": False,
        "hw_queues": 0,
    },
}

_values = None


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return list(map(_plain, value))
    return value


def _installed_groups():
    from turbo_physai.engine.definitions.registry import default_registry

    import_module(f"{__package__}.catalog")
    groups = {}
    for group in GROUP_DEFAULTS:
        for member in default_registry.get_group(group).members:
            parts = default_registry.get_spec(member).target.split(".")
            for length in range(len(parts) - 1, 0, -1):
                parent = sys.modules.get(".".join(parts[:length]))
                if parent is not None:
                    for part in parts[length:]:
                        parent = getattr(parent, part, None)
                    if getattr(parent, "__openwam_group__", None) == group:
                        options = _plain(parent.__openwam_options__)
                        if group in groups and groups[group] != options:
                            raise ValueError(
                                f"Inconsistent options on installed {group} wrappers"
                            )
                        groups[group] = options
                    break
    return groups


def initialize(*, force=False):
    """Freeze the installed Group options at the training/runtime boundary."""
    global _values
    if _values is not None and not force:
        return
    groups = _installed_groups()
    result = {}
    for group, defaults in GROUP_DEFAULTS.items():
        # Unselected Groups do not enable any optimization.
        result.update(
            {
                key: False if isinstance(val, bool) else val
                for key, val in defaults.items()
            }
        )
        if group not in groups:
            continue
        options = groups[group]
        unknown = options.keys() - defaults.keys()
        if unknown:
            raise ValueError(f"Unknown {group} options: {', '.join(sorted(unknown))}")
        candidate = {**defaults, **options}
        for key, default in defaults.items():
            item = candidate[key]
            if isinstance(default, bool) and not isinstance(item, bool):
                raise TypeError(f"{group}.{key} must be boolean")
            if isinstance(default, int) and not isinstance(default, bool):
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    raise ValueError(f"{group}.{key} must be a non-negative integer")
            if (
                key.endswith("_dir")
                and item is not None
                and (not isinstance(item, str) or not item.strip())
            ):
                raise ValueError(f"{group}.{key} must be a non-empty path or null")
        result.update(candidate)
    if (result["zero_scaled_pack"] or result["zero_norm_overlap"]) and not result[
        "zero_reduce_scatter"
    ]:
        raise ValueError(
            "ZeRO scaled packing and norm overlap require zero_reduce_scatter"
        )
    if result["text_cache_size"] < 1:
        raise ValueError("openwam.text.text_cache_size must be positive")
    if result["hw_queues"] == 1:
        raise ValueError("openwam.training.hw_queues must be zero or >= 2")
    if result["sac_layers"] is not None and (
        isinstance(result["sac_layers"], bool)
        or not isinstance(result["sac_layers"], int)
        or result["sac_layers"] < 0
    ):
        raise ValueError(
            "openwam.mot.sac_layers must be a non-negative integer or null"
        )
    if not isinstance(result["sac_extra_ops"], str):
        raise TypeError("openwam.mot.sac_extra_ops must be a comma-separated string")
    if result["vae_latent_cache"] and not result["vae_latent_cache_dir"]:
        raise ValueError("openwam.data.vae_latent_cache requires vae_latent_cache_dir")
    if result["fa2_mask_fused"] and not result["fa2_padding"]:
        raise ValueError("openwam.text.fa2_mask_fused requires fa2_padding")
    scopes = [name for name in ("mot", "vae") if result["compile_" + name]]
    result["global_compile"] = bool(scopes)
    result["compile_scope"] = (
        "all" if len(scopes) == 2 else (scopes[0] if scopes else "all")
    )
    _values = result


def value(name, default=None):
    # Import-time decorators must remain inert during Engine preparation.
    result = default if _values is None else _values.get(name, default)
    return default if result is None else result


def enabled(name, default=False):
    item = value(name, default)
    if name in {"checkpoint_skip_layers", "hw_queues"}:
        return item > 0
    if not isinstance(item, bool):
        raise TypeError(f"OpenWAM option {name} is not boolean")
    return item


class _CallableClassMethod(classmethod):
    """Keep the classmethod descriptor while satisfying Turbo's callable check."""

    def __call__(self, *args, **kwargs):
        return self.__func__(*args, **kwargs)


def port_wrapper(implementation, target, group):
    """Build a standard Turbo wrapper factory for a source-bound function."""

    def factory(original, options):
        @functools.wraps(original)
        def invoke(*args, **kwargs):
            initialize()
            return implementation(*args, **kwargs)

        invoke.__openwam_group__ = group
        invoke.__openwam_options__ = options
        if inspect.ismethod(original) and isinstance(original.__self__, type):
            function = implementation.__func__

            @functools.wraps(original)
            def class_invoke(cls, *args, **kwargs):
                initialize()
                return function(cls, *args, **kwargs)

            class_invoke.__openwam_group__ = group
            class_invoke.__openwam_options__ = options
            return _CallableClassMethod(class_invoke)
        return invoke

    return factory
