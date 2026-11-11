# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import os

import pytest

from turbo_physai.optimizations.models.lingbot_va import group_backend as backend


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
    validations = []
    monkeypatch.setattr(backend, "validate_backend", lambda: validations.append(True))
    monkeypatch.setattr(backend, "supported", lambda q, k, v: supported_input)
    calls = []
    result = object()

    def original(q, k, v, *, block_mask, kernel_options):
        calls.append((q, k, v, block_mask, kernel_options))
        return result

    wrapped = backend.backend_wrapper(original, {"fast_bf16": True})
    q, k, v, mask = object(), object(), object(), object()
    options = {"BLOCK_M": 128}
    assert wrapped(q, k, v, block_mask=mask, kernel_options=options) is result
    expected = dict(options)
    if supported_input:
        expected.update(
            LINGBOT_VA_TEMPLATE_OPT=True, LINGBOT_VA_BACKEND_REV=backend.REVISION,
            LINGBOT_FAST_BF16=True,
        )
    assert calls == [(q, k, v, mask, expected)]
    assert options == {"BLOCK_M": 128}
    assert {key: os.environ.get(key) for key in keys} == before
    assert validations == [True]


def test_backend_still_rejects_invalid_build_without_cache_overrides(monkeypatch):
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)

    def invalid_build():
        raise RuntimeError("Torch/HIP/Triton build mismatch")

    monkeypatch.setattr(backend, "validate_backend", invalid_build)
    with pytest.raises(RuntimeError, match="Torch/HIP/Triton build mismatch"):
        backend.backend_wrapper(lambda *args, **kwargs: None, {})
