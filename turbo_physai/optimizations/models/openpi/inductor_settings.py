# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Independent torch.compile and Inductor configuration replacement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from functools import wraps


def _set_if_present(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def wrap_torch_compile(original, group_options: Mapping[str, Any]):
    compile_group_options = dict(group_options)

    @wraps(original)
    def wrapped(
        model,
        *,
        fullgraph=False,
        dynamic=None,
        backend="inductor",
        mode=None,
        options=None,
        disable=False,
    ):
        import torch

        inductor_options = compile_group_options.get(
            "inductor", {"layout_optimization": True, "shape_padding": True}
        )
        if isinstance(inductor_options, Mapping):
            config = getattr(getattr(torch, "_inductor", None), "config", None)
            if config is not None:
                for name in ("shape_padding", "layout_optimization", "epilogue_fusion"):
                    if name in inductor_options:
                        _set_if_present(config, name, bool(inductor_options[name]))

        if mode is None and "mode" in compile_group_options:
            mode = compile_group_options["mode"]
        if "fullgraph" in compile_group_options:
            fullgraph = bool(compile_group_options["fullgraph"])
        if "dynamic" in compile_group_options:
            dynamic = compile_group_options["dynamic"]

        return original(
            model,
            fullgraph=fullgraph,
            dynamic=dynamic,
            backend=backend,
            mode=mode,
            options=options,
            disable=disable,
        )

    return wrapped
