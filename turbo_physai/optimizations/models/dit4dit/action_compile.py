# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Action compilation and native RNG fallback are one indivisible operation."""

import functools


def build_framework_wrapper(original, options):
    from torch._inductor import config

    if not hasattr(config, "fallback_random"):
        raise RuntimeError("This PyTorch build does not support fallback_random")
    if options.get("backend", "inductor") != "inductor":
        raise ValueError("Action RNG fallback requires backend=inductor")
    if options.get("fallback_random", True) is not True:
        raise ValueError("Action compile requires fallback_random=True")
    if not hasattr(config, "alignment_asserts"):
        raise RuntimeError("This PyTorch build does not support alignment_asserts")
    if options.get("alignment_asserts", False) is not False:
        raise ValueError("Action compile requires alignment_asserts=False")
    if options.get("mode", "default") != "default":
        raise ValueError("Action RNG fallback requires mode=default")
    # The current PyTorch 2.10 DAS build emits an unused scalar from the HCU
    # Flash-Attention tuple with an 8-byte storage offset, then incorrectly
    # asserts 16-byte alignment for that scalar.  Disable only the generated
    # assertion for this Action graph; Flash-Attention remains enabled.
    kwargs = dict(
        backend="inductor", fullgraph=bool(options.get("fullgraph", False)),
        dynamic=bool(options.get("dynamic", False)),
        options={"fallback_random": True, "alignment_asserts": False},
    )

    @functools.wraps(original)
    def wrapped(cfg):
        model = original(cfg)
        action = getattr(getattr(model, "action_model", None), "model", None)
        if action is None:
            raise RuntimeError("Expected model.action_model.model")
        # In-place compilation preserves parameter identities and checkpoint
        # keys. Tracing starts on the first forward, after weights are loaded.
        action.compile(**kwargs)
        return model
    return wrapped
