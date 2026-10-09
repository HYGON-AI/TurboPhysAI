# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamo-safe replacement for timm's ``VisionTransformer._intermediate_layers``.

Why this exists
---------------
Prismatic replaces each ViT tower's ``forward`` with

    unpack_tuple(partial(featurizer.get_intermediate_layers, n={len(blocks) - 2}))

so ``n`` is a Python ``set`` *captured outside* the compiled region.  The FSDP1
compile flow (``compile_fsdp1.py``) whole-model-``torch.compile``s each
FSDP-wrapped tower; on the first real forward Dynamo traces timm's original
line (``timm/models/vision_transformer.py``)

    take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

i.e. ``set(n)`` applied to the already-existing (sourced) set.  Dynamo's
builtin ``set()`` handler clones that SetVariable with
``mutation_type=ValueMutationNew()``; the clone copies the variable's
``source`` along, and ``torch/_dynamo/variables/base.py``
(``VariableTracker.__init__``) forbids a "new" variable to carry a ``source``
=> ``assert source is None`` raises a bare ``AssertionError`` (torch 2.7.1;
Dynamo internal limitation, not a user-code bug).

Fix
---
The list-based reimplementation below never runs ``set(...)`` over a sourced
set, so Dynamo traces it cleanly.  It is exactly behaviour-equivalent to the
timm original for every ``n`` form the monkey-patch can pass (``int`` or any
iterable such as ``set`` / ``list`` / ``tuple`` / ``range``), so eager
numerics are unchanged.

Wiring
------
Declared in ``catalog.py`` as a member of the ``openvla.compile.fsdp1`` Group.
It is a ``wrap`` whose factory (``timm_intermediate_layers_wrapper``) swaps the
class attribute **only when the Group opts in via ``options.compile``** (the
vision towers are then actually ``torch.compile``d and hit the Dynamo crash).
When ``options.compile`` is false the tower stays eager, where the original
timm implementation is already fine, so the factory returns the original method
untouched (no global timm mutation).  Importing this module itself has no side
effects; the class attribute is swapped only when that Group is applied with
compilation enabled.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Union

import torch


def dynamo_safe_intermediate_layers(
    self,
    x: torch.Tensor,
    n: Union[int, Sequence[int]] = 1,
) -> List[torch.Tensor]:
    """timm ``_intermediate_layers`` with a Dynamo-safe ``take_indices``.

    Mirrors timm 0.9.16 exactly except the ``take_indices`` construction:
    indices are kept in a list instead of being routed through ``set(n)`` on a
    possibly-sourced set (which crashes ``torch._dynamo``, see module docstring).
    """
    outputs, num_blocks = [], len(self.blocks)
    if isinstance(n, int):
        take_indices = list(range(num_blocks - n, num_blocks))
    else:
        take_indices = list(n)  # accepts set / list / tuple / range ...

    # forward pass
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in take_indices:
            outputs.append(x)

    return outputs


def timm_intermediate_layers_wrapper(
    original: Callable, options
) -> Callable:
    """wrap factory for ``timm...VisionTransformer._intermediate_layers``.

    The Dynamo crash this fixes only occurs when the ViT tower is actually
    ``torch.compile``d.  When the compile Group's ``options.compile`` is False
    the tower stays eager, where the original timm implementation is already
    fine, so we return the original method untouched (no global timm mutation).
    """
    options = dict(options or {})
    if not options.get("compile", False):
        return original
    return dynamo_safe_intermediate_layers
