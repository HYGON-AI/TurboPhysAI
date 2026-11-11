# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import os

import pytest

pytest.importorskip("einops")

from turbo_physai.optimizations.models.lingbot_va import group_backend as backend
from turbo_physai.optimizations.models.lingbot_va import attention, compatibility
from turbo_physai.optimizations.models.lingbot_va.catalog import FLEX_TEMPLATE
from turbo_physai.engine.contracts import CompatibilityResult
from turbo_physai.engine.execution.replacements.wrapper import WrapperHandler


@pytest.mark.parametrize("custom_cache", [False, True])
@pytest.mark.parametrize("supported_input", [False, True])
def test_backend_routes_without_requiring_custom_cache(
    monkeypatch, tmp_path, custom_cache, supported_input,
):
    keys = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
    for key in keys:
        if custom_cache:
            monkeypatch.setenv(key, str(tmp_path / key.lower()))
        else:
            monkeypatch.delenv(key, raising=False)
    before = {key: os.environ.get(key) for key in keys}
    conditions = []

    def supported(q, k, v, *, block_mask, kernel_options):
        conditions.append((q, k, v, block_mask, kernel_options))
        return supported_input

    monkeypatch.setattr(backend, "supported", supported)
    calls = []
    result = object()

    def original(q, k, v, *, block_mask, kernel_options):
        calls.append((q, k, v, block_mask, kernel_options))
        return result

    monkeypatch.setattr(attention, "flex_call", original)
    handler = WrapperHandler()
    prepared = handler.prepare(FLEX_TEMPLATE.specs[0], {"fast_bf16": True})
    q, k, v, mask = object(), object(), object(), object()
    options = {"BLOCK_M": 128}
    snapshot = handler.snapshot(prepared)
    try:
        handler.apply(prepared)
        assert attention.flex_call(q, k, v, block_mask=mask, kernel_options=options) is result
    finally:
        handler.restore(snapshot)
    expected = dict(options)
    if supported_input:
        expected.update(
            LINGBOT_VA_TEMPLATE_OPT=True, LINGBOT_VA_BACKEND_REV=backend.REVISION,
            LINGBOT_FAST_BF16=True,
        )
    assert calls == [(q, k, v, mask, expected)]
    assert options == {"BLOCK_M": 128}
    assert {key: os.environ.get(key) for key in keys} == before
    assert conditions == [(q, k, v, mask, options)]
    assert attention.flex_call is original


def test_compatibility_still_rejects_invalid_build_without_cache_overrides(monkeypatch):
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)

    def invalid_build():
        raise RuntimeError("Torch/HIP/Triton build mismatch")

    monkeypatch.setattr(backend, "validate_backend", invalid_build)
    monkeypatch.setattr(compatibility, "baseline", lambda context: CompatibilityResult(True))
    result = backend.compatible(None)
    assert not result.compatible
    assert result.reason == "Torch/HIP/Triton build mismatch"
