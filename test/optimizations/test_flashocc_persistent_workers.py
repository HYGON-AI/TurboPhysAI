# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from types import SimpleNamespace

import pytest

from turbo_physai.optimizations.models.flashocc.persistent_workers import (
    persistent_workers_wrapper,
)


@pytest.mark.parametrize(("workers", "expected"), [(0, False), (4, True)])
@pytest.mark.parametrize("keyword_cfg", [False, True])
def test_training_config_follows_worker_count(workers, expected, keyword_cfg):
    cfg = SimpleNamespace(data=SimpleNamespace(workers_per_gpu=workers))
    calls = []

    def original(model, dataset, cfg, distributed=False, validate=False):
        calls.append((model, dataset, cfg.data.persistent_workers, distributed, validate))
        return "result"

    wrapped = persistent_workers_wrapper(original, {})
    if keyword_cfg:
        result = wrapped("model", "train", cfg=cfg, distributed=True, validate=True)
    else:
        result = wrapped("model", "train", cfg, distributed=True, validate=True)
    assert result == "result"
    assert calls == [("model", "train", expected, True, True)]


def test_catalog_targets_mmdet3d_train_entry():
    catalog = (
        Path(__file__).resolve().parents[2]
        / "turbo_physai/optimizations/models/flashocc/catalog.py"
    ).read_text()
    section = catalog.split("PERSISTENT_WORKERS = group(", 1)[1].split(
        "VIEW_TRANSFORMER_PREPARATION = group(", 1
    )[0]
    assert 'target="mmdet3d.apis.train.train_detector"' in section
    assert "torch.utils.data.DataLoader.__init__" not in section


@pytest.mark.parametrize(("workers", "expected"), [(0, False), (2, True)])
def test_mmdet3d_training_and_validation_loader_settings(monkeypatch, workers, expected):
    from unittest.mock import MagicMock

    mmcv = pytest.importorskip("mmcv")
    train_api = pytest.importorskip("mmdet3d.apis.train")
    torch_data = pytest.importorskip("torch.utils.data")
    dataset = range(2)
    val_dataset = range(1)
    loaders = []

    def build_loader(data, samples_per_gpu, workers_per_gpu, **kwargs):
        loader = torch_data.DataLoader(
            data,
            batch_size=samples_per_gpu,
            num_workers=workers_per_gpu,
            persistent_workers=kwargs.get("persistent_workers", False),
        )
        loaders.append(loader)
        return loader

    cfg = mmcv.Config(dict(
        log_level="INFO",
        data=dict(samples_per_gpu=1, workers_per_gpu=workers, val={}),
        gpu_ids=[0], seed=0, runner=dict(type="EpochBasedRunner", max_epochs=1),
        optimizer={}, work_dir="/tmp", lr_config={}, optimizer_config={},
        checkpoint_config={}, log_config={}, resume_from=None, load_from=None,
        workflow=[("train", 1)],
    ))
    monkeypatch.setattr(train_api, "get_mmdet_root_logger", MagicMock())
    monkeypatch.setattr(train_api, "build_mmdet_dataloader", build_loader)
    monkeypatch.setattr(train_api, "build_dataset", lambda *args: val_dataset)
    monkeypatch.setattr(train_api, "MMDataParallel", MagicMock())
    monkeypatch.setattr(train_api, "build_optimizer", MagicMock())
    monkeypatch.setattr(train_api, "build_runner", MagicMock())
    monkeypatch.setattr(train_api, "MMDET_EvalHook", MagicMock())

    original_loader_init = torch_data.DataLoader.__init__
    persistent_workers_wrapper(train_api.train_detector, {})(
        MagicMock(), dataset, cfg, validate=True
    )

    assert [loader.persistent_workers for loader in loaders] == [expected, False]
    assert torch_data.DataLoader.__init__ is original_loader_init
