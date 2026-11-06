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


def test_build_script_targets_official_projects():
    script = (
        Path(__file__).resolve().parents[2]
        / "turbo_physai/optimizations/models/flashocc/scripts/build_flashocc_ops_external.sh"
    ).read_text()
    assert '--build-lib "$FLASHOCC_ROOT/projects"' in script
    assert "FLASHOCC_RUNTIME_ROOT" not in script
    assert "source /root/.hygon_base_env" not in script
    assert "conda activate" not in script
