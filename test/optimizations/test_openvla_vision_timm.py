# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ``openvla.compile.fsdp1`` vision-tower rewrite.

Prismatic swaps each ViT tower's ``forward`` for
``partial(featurizer.get_intermediate_layers, n={len(blocks) - 2})``, and the
FSDP1 compile flow whole-model-``torch.compile``s that tower.  On the first real
forward Dynamo traces timm's

    take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

whose ``set(...)`` over an already-sourced set trips ``assert source is None`` in
``torch/_dynamo/variables/base.py``.  ``vision_timm`` reimplements the method
with a list instead of a set, which Dynamo traces cleanly.

``n`` is the only thing the rewrite changes, so the equivalence claim is exactly
"the same blocks, in the same order, with the same numbers, for every ``n`` form
the monkey-patch can pass".  Two properties are invisible to a bit-identity
check and are therefore asserted separately:

* **The swap must actually happen.**  The rewrite is behaviour-equivalent, so a
  run where nothing was installed compares equal to itself and passes.
* **With ``options.compile`` off, the factory must hand back the original
  method.**  The towers stay eager there, so timm must be left globally
  untouched.  An accidental install would still produce identical numbers, yet
  would mutate ``timm.models.vision_transformer.VisionTransformer`` for the
  whole process -- the one consequence a numerical comparison can never see.
"""

from __future__ import annotations

import inspect

import pytest


torch = pytest.importorskip("torch")

from turbo_physai.engine.contracts import Mechanism
from turbo_physai.optimizations.models.openvla.catalog import COMPILE_FSDP1
from turbo_physai.optimizations.models.openvla.vision_timm import (
    dynamo_safe_intermediate_layers,
    timm_intermediate_layers_wrapper,
)


# timm ships as part of the upstream model stack rather than the repository
# requirements, so only its absence skips.  A broken install must fail loudly.
timm_vision_transformer = pytest.importorskip("timm.models.vision_transformer")
VisionTransformer = timm_vision_transformer.VisionTransformer


pytestmark = pytest.mark.model_deps


DEPTH = 4
SEED = 20240607
TARGET = "timm.models.vision_transformer.VisionTransformer._intermediate_layers"
REPLACEMENT = (
    "turbo_physai.optimizations.models.openvla."
    "vision_timm.timm_intermediate_layers_wrapper"
)

# Every form the monkey-patch can pass: OpenVLA's set, plus the int and generic
# iterables `timm_intermediate_layers_wrapper` has to tolerate.  `int-over` and
# `int-zero` pin the two range boundaries the rewrite reproduces implicitly.
INDEX_FORMS = [
    (1, "int-1"),
    (2, "int-2"),
    (DEPTH, "int-all"),
    (DEPTH + 2, "int-beyond-depth"),
    (0, "int-zero"),
    ({DEPTH - 2}, "set-openvla"),
    ({1, 3}, "set-two"),
    ([1, 3], "list"),
    ((0, 2), "tuple"),
    (range(0, DEPTH, 2), "range"),
]


def _verify_timm_method():
    """Fail loudly if timm changed, or if an earlier test left timm patched."""

    source = inspect.getsource(VisionTransformer._intermediate_layers)
    assert "take_indices = set(" in source, (
        "timm's _intermediate_layers no longer matches the implementation this "
        "rewrite mirrors; re-derive vision_timm.py before trusting this comparison. "
        "A swapped-in method here also means an earlier test left timm patched."
    )


def _timm_method():
    """timm's own method, verified to still be the ``set(...)`` version we mirror."""

    _verify_timm_method()
    return VisionTransformer._intermediate_layers


def _vision_tower():
    """A tiny timm ViT tower, as ``eval`` so no dropout can make arms incomparable."""

    return VisionTransformer(
        img_size=32,
        patch_size=8,
        in_chans=3,
        embed_dim=32,
        depth=DEPTH,
        num_heads=2,
        num_classes=0,
    ).eval()


def _input():
    generator = torch.Generator().manual_seed(SEED)
    return torch.randn(2, 3, 32, 32, generator=generator)


def _flatten(value):
    """Flatten the nested tuples ``get_intermediate_layers`` can return."""

    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]


@pytest.mark.parametrize("n,case", INDEX_FORMS, ids=[case for _, case in INDEX_FORMS])
def test_rewrite_returns_the_same_blocks_for_every_index_form(n, case, monkeypatch):
    """``set(n)`` -> ``list(n)`` must not move, drop or reorder a single block."""

    original = _timm_method()
    model, x = _vision_tower(), _input()

    with torch.no_grad():
        expected = original(model, x, n)
    monkeypatch.setattr(
        VisionTransformer, "_intermediate_layers", dynamo_safe_intermediate_layers
    )
    with torch.no_grad():
        actual = dynamo_safe_intermediate_layers(model, x, n)

    assert len(actual) == len(expected), f"different block count for {case}"
    for index, (left, right) in enumerate(zip(expected, actual)):
        assert left.shape == right.shape, f"block {index} changed shape for {case}"
        assert torch.equal(left, right), f"block {index} differs for {case}"


@pytest.mark.parametrize(
    "keywords",
    [{}, {"norm": True}, {"return_prefix_tokens": True}, {"reshape": True}],
    ids=["plain", "norm", "prefix-tokens", "reshape"],
)
def test_prismatic_call_path_is_bit_identical(keywords, monkeypatch):
    """The path OpenVLA really calls post-processes the same blocks unchanged."""

    _verify_timm_method()
    model, x = _vision_tower(), _input()

    with torch.no_grad():
        expected = model.get_intermediate_layers(x, n={DEPTH - 2}, **keywords)
    monkeypatch.setattr(
        VisionTransformer, "_intermediate_layers", dynamo_safe_intermediate_layers
    )
    with torch.no_grad():
        actual = model.get_intermediate_layers(x, n={DEPTH - 2}, **keywords)

    left, right = _flatten(expected), _flatten(actual)
    assert len(left) == len(right)
    for index, (a, b) in enumerate(zip(left, right)):
        assert a.shape == b.shape, f"tensor {index} changed shape"
        assert torch.equal(a, b), f"tensor {index} differs"


def test_factory_installs_the_rewrite_only_when_compile_is_enabled():
    """Eager runs must leave timm globally untouched; compiled ones must swap."""

    original = _timm_method()

    assert timm_intermediate_layers_wrapper(original, None) is original
    assert timm_intermediate_layers_wrapper(original, {}) is original
    assert timm_intermediate_layers_wrapper(original, {"compile": False}) is original
    assert (
        timm_intermediate_layers_wrapper(original, {"compile": True})
        is dynamo_safe_intermediate_layers
    )
    # Building the replacement must not mutate timm itself: installing it is the
    # engine's job, and a factory that also swapped the class attribute would
    # patch every timm ViT in the process before the Group was even applied.
    assert VisionTransformer._intermediate_layers is original


def test_group_declares_the_timm_rewrite():
    """The declaration must keep pointing at the real timm method."""

    (spec,) = [
        item
        for item in COMPILE_FSDP1.specs
        if item.target.startswith("timm.models.vision_transformer.")
    ]
    assert spec.mechanism is Mechanism.WRAPPER
    assert spec.target == TARGET
    assert spec.replacement == REPLACEMENT
