# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from unittest.mock import patch

import pytest

from turbo_physai.optimizations.models.flashocc import compatibility


def test_bootstrap_needs_model_root_only():
    with patch.dict(
        "os.environ",
        {"TURBO_PHYSAI_BOOTSTRAP": "1", "FLASHOCC_ROOT": "/model"},
        clear=True,
    ), patch.object(compatibility, "install_compatibility") as install:
        assert compatibility.install_for_turbo_run()
        install.assert_called_once_with("/model")


def test_import_roots_use_model_and_not_runtime():
    paths = compatibility.order_import_roots(["/existing"], "/model", "/plugin")
    assert paths[:2] == ["/model", "/model/mmdetection3d"]
    assert "/plugin" not in paths


def test_encoding_finder_is_idempotent_and_rejects_different_model_root():
    with patch("sys.meta_path", []), patch("sys.path", ["/existing"]), patch.object(
        compatibility, "install_dvr_compatibility"
    ):
        compatibility.install_compatibility("/model")
        compatibility.install_compatibility("/model")
        finders = [
            finder for finder in __import__("sys").meta_path
            if isinstance(finder, compatibility.LegacyEncodingSourceFinder)
        ]
        assert len(finders) == 1
        with pytest.raises(RuntimeError, match="different roots"):
            compatibility.install_compatibility("/different-model")


def test_readme_uses_upstream_editable_build():
    readme = (
        Path(__file__).resolve().parents[2] / "model_examples/FlashOCC/README.md"
    ).read_text(encoding="utf-8")
    assert 'cd "/workspace/FlashOCC_TurboPhysAI/FlashOCC/projects"' in readme
    assert "python -m pip install -e . --no-build-isolation --no-deps" in readme
