# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")

from turbo_physai.engine.contracts import CompatibilityResult
from turbo_physai.optimizations.models.lingbot_va import attention, compatibility
from turbo_physai.optimizations.models.lingbot_va import group_backend as backend
from turbo_physai.optimizations.models.lingbot_va.catalog import FLEX_TEMPLATE


def test_supported_signature_matches_flex_call():
    assert inspect.signature(backend.supported) == inspect.signature(attention.flex_call)
    assert FLEX_TEMPLATE.specs[0].runtime_condition == (
        "turbo_physai.optimizations.models.lingbot_va.group_backend.supported"
    )


@pytest.mark.parametrize("case", [
    "supported", "length", "dtype", "architecture", "cpu", "batch", "heads", "stride",
])
def test_supported_input_allowlist_with_tensor_metadata(monkeypatch, case):
    monkeypatch.setattr(torch.version, "hip", "test")
    device = SimpleNamespace(type="cpu" if case == "cpu" else "cuda")
    tensor = SimpleNamespace(
        device=device, ndim=4,
        dtype=torch.float32 if case == "dtype" else torch.bfloat16,
        shape=(2 if case == "batch" else 1, 12 if case == "heads" else 24,
               7041 if case == "length" else 7040, 128),
        stride=lambda dim: 2 if case == "stride" else 1,
    )
    architecture = "gfx000" if case == "architecture" else "gfx936:test"
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: SimpleNamespace(gcnArchName=architecture))
    assert backend.supported(
        tensor, tensor, tensor, block_mask=None, kernel_options={},
    ) is (case == "supported")


def test_factories_defer_build_validation_to_group_compatibility(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "validate_backend", lambda: calls.append("validated"))
    monkeypatch.setattr(compatibility, "baseline", lambda context: CompatibilityResult(True))
    # The checker prepares these candidates before its compatibility check.
    original = lambda *args, **kwargs: None
    backend.backend_wrapper(original, {})
    backend.template_wrapper(original, {})
    backend.compile_options_wrapper(original, {})
    assert calls == []
    assert FLEX_TEMPLATE.definition.compatibility_check == (
        "turbo_physai.optimizations.models.lingbot_va.group_backend.compatible"
    )
    assert backend.compatible(None).compatible
    assert calls == ["validated"]


def test_incompatible_model_does_not_validate_assets(monkeypatch):
    baseline = CompatibilityResult(False, reason="model source mismatch")
    monkeypatch.setattr(compatibility, "baseline", lambda context: baseline)

    def unexpected_validation():
        pytest.fail("assets must not be validated after a model mismatch")

    monkeypatch.setattr(backend, "validate_backend", unexpected_validation)
    assert backend.compatible(None) is baseline
