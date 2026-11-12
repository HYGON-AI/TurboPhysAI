# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Check the launch contract without starting training or allocating GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from turbo_physai.runtime import load_runtime_config, prepare_environment


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "model_examples/LingBot-VA/run_8card.sh"
RUNTIME = ROOT / "turbo_physai/optimizations/models/lingbot_va/configs/runtime.yaml"


@pytest.mark.parametrize("override", [False, True])
def test_launcher_uses_root_and_preserves_patched_entry(tmp_path, override):
    model = tmp_path / "upstream repo"
    (model / "wan_va/modules").mkdir(parents=True)
    (model / "wan_va/train.py").touch()
    (model / "wan_va/modules/model.py").touch()
    executable = tmp_path / "bin"
    executable.mkdir()
    probe = executable / "turbo-physai"
    probe.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'cwd': os.getcwd(), 'args': sys.argv[1:], "
        "'env': dict(os.environ)}))\n"
    )
    probe.chmod(0o755)
    env = dict(os.environ)
    for key in ["NGPU", "MASTER_PORT", "LOG_RANK", "CONFIG_NAME", "CACHE_ROOT",
                "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHFT_LIGHTHOUSE"]:
        env.pop(key, None)
    env.update(
        PATH=f"{executable}{os.pathsep}{env.get('PATH', '')}",
        LINGBOT_VA_ROOT=str(model), RUN_ROOT=str(tmp_path / "run outputs"),
        PYTHONPATH="/existing/modules", LD_LIBRARY_PATH="/existing/libs",
        OMP_NUM_THREADS="3", ROCBLAS_TENSILE_LIBPATH="/existing/tensile",
        TORCHINDUCTOR_MAX_AUTOTUNE="1",
        TORCHINDUCTOR_COORDINATE_DESCENT_TUNING="1",
    )
    if override:
        env.update(NGPU="4", MASTER_PORT="29603", LOG_RANK="0,1",
                   CONFIG_NAME="robotwin_train",
                   TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "custom inductor"),
                   TRITON_CACHE_DIR=str(tmp_path / "custom triton"),
                   TORCHFT_LIGHTHOUSE="http://custom:29510")
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--extra-argument", "value with spaces"],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    assert result["cwd"] == str(model)
    assert result["args"] == [
        "run", "--model", "lingbot_va", "--log-report",
        "python", "-m", "torch.distributed.run",
        f"--nproc_per_node={4 if override else 8}",
        f"--local-ranks-filter={'0,1' if override else '0'}",
        "--master_port", "29603" if override else "29501", "--tee", "3",
        "-m", "turbo_physai.optimizations.models.lingbot_va.entrypoint",
        "--config-name", "robotwin_train" if override else "libero_train",
        "--save-root", env["RUN_ROOT"], "--extra-argument", "value with spaces",
    ]
    actual = result["env"]
    assert actual["PYTHONPATH"] == f"{model / 'wan_va'}:/existing/modules"
    for key in ["LD_LIBRARY_PATH", "OMP_NUM_THREADS", "ROCBLAS_TENSILE_LIBPATH",
                "TORCHINDUCTOR_MAX_AUTOTUNE", "TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"]:
        assert actual[key] == env[key]
    assert actual["TORCHFT_LIGHTHOUSE"] == env.get("TORCHFT_LIGHTHOUSE", "http://localhost:29510")
    for key, subdirectory in [("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("TRITON_CACHE_DIR", "triton")]:
        assert actual[key] == env.get(key, str(Path(env["RUN_ROOT"]) / "cache" / subdirectory))


def test_runtime_sets_upstream_allocator_and_tokenizer_defaults(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    env = prepare_environment(load_runtime_config(RUNTIME))
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
