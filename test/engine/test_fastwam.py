# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys
from unittest.mock import patch

from turbo_physai.optimizations.models.fastwam import catalog
from turbo_physai.optimizations.models.fastwam.compile import compile_hotpath


def test_catalog_is_lazy():
    subprocess.run(
        [sys.executable, "-c", "import sys; import turbo_physai.optimizations.models.fastwam.catalog; "
         "assert 'torch' not in sys.modules; assert 'fastwam' not in sys.modules"],
        check=True,
    )
    assert catalog.MODEL.definition.depends_on == ("fastwam.vae",)


def test_compile_is_lazy_cached_and_disableable():
    import torch

    calls = []

    def compiler(original, **options):
        calls.append(options)
        return lambda x: original(x) + 10

    with patch.object(torch, "compile", compiler), patch.dict(os.environ, {}, clear=False):
        os.environ.pop("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", None)
        os.environ.pop("TURBO_PHYSAI_FASTWAM_COMPILE_MODE", None)
        fn = compile_hotpath(lambda x: x + 1)
        assert calls == []
        assert fn(1) == 12
        assert fn(2) == 13
        assert calls == [{"mode": "max-autotune-no-cudagraphs"}]
        os.environ["TURBO_PHYSAI_DISABLE_TORCH_COMPILE"] = "1"
        assert fn(1) == 2


def test_compile_failure_is_propagated():
    import torch

    with patch.dict(os.environ, {"TURBO_PHYSAI_DISABLE_TORCH_COMPILE": "0"}), patch.object(
        torch, "compile", side_effect=RuntimeError("backend failure")
    ):
        import pytest

        with pytest.raises(RuntimeError, match="backend failure"):
            compile_hotpath(lambda: None)()
