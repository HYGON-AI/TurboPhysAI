# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the real Group wrapper and Python module entry semantics on CPU."""

import importlib
import runpy
import sys
from types import SimpleNamespace

import pytest

from turbo_physai.engine.execution.replacements.wrapper import WrapperHandler
from turbo_physai.optimizations.models.lingbot_va.catalog import SYNC


@pytest.fixture
def training_module(tmp_path, monkeypatch):
    probe = SimpleNamespace(events=[], accumulation=2)
    monkeypatch.setitem(sys.modules, "lingbot_entry_probe", probe)
    upstream = tmp_path / "wan_va"
    upstream.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(upstream))
    monkeypatch.delitem(sys.modules, "train", raising=False)
    monkeypatch.delitem(
        sys.modules, "turbo_physai.optimizations.models.lingbot_va.train_sync",
        raising=False,
    )
    (upstream / "train.py").write_text('''
from lingbot_entry_probe import events, accumulation
events.append(("load", __name__))
logger = None
dist_mean = dist_max = lambda value: value

class Trainer:
    gradient_accumulation_steps = accumulation

    def train(self):
        events.append(("original", type(self)))

def init_logger():
    events.append(("logger", __name__))

def main():
    events.append(("main", __name__))
    Trainer().train()

if __name__ == "__main__":
    init_logger()
    main()
''')
    return probe


def install_sync(monkeypatch, probe):
    handler = WrapperHandler()
    prepared = handler.prepare(SYNC.specs[0], {})
    sync = importlib.import_module(
        "turbo_physai.optimizations.models.lingbot_va.train_sync"
    )
    # Isolate entry routing from the copied loop's GPU/distributed work.
    monkeypatch.setattr(
        sync, "train", lambda self: probe.events.append(("optimized", type(self)))
    )
    handler.apply(prepared)
    return importlib.import_module("train")


@pytest.mark.parametrize("accumulation", [1, 2])
def test_entrypoint_calls_patched_trainer_once(training_module, monkeypatch, accumulation):
    probe = training_module
    probe.accumulation = accumulation
    training = install_sync(monkeypatch, probe)
    runpy.run_module(
        "turbo_physai.optimizations.models.lingbot_va.entrypoint",
        run_name="__main__", alter_sys=True,
    )
    expected_path = "original" if accumulation == 1 else "optimized"
    assert probe.events == [
        ("load", "train"), ("logger", "train"), ("main", "train"),
        (expected_path, training.Trainer),
    ]
    assert hasattr(training.Trainer.train, "__wrapped__")


def test_direct_module_execution_bypasses_imported_trainer(training_module, monkeypatch):
    probe = training_module
    training = install_sync(monkeypatch, probe)
    executed = runpy.run_module("train", run_name="__main__", alter_sys=True)
    assert executed["Trainer"] is not training.Trainer
    assert probe.events[-1] == ("original", executed["Trainer"])
    assert not any(event[0] == "optimized" for event in probe.events)
