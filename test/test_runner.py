# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import json
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from turbo_physai.cli import (
    _resolve_run_configs,
    _run_training_command,
    main as cli_main,
)
from turbo_physai.runner import (
    _discover_rank_numa_node,
    _print_optimization_result,
    _set_rank_numa_binding,
    run,
)
from turbo_physai.runtime import load_runtime_config, prepare_environment
from turbo_physai.engine.config import loader
from turbo_physai.engine.errors import TurboPhysAIError
from turbo_physai.launchers import rewrite_command


class _Report:
    run_id = "test-run"
    summary = {"applied": 1, "blocked": 0, "failed": 0, "rolled_back": 0, "not_started": 0}


def test_runner_applies_optimization_before_executing_target(tmp_path, monkeypatch):
    target = tmp_path / "target.py"
    result = tmp_path / "result.txt"
    target.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        f"Path({str(result)!r}).write_text('|'.join(sys.argv))\n",
        encoding="utf-8",
    )
    calls = []

    def apply_optimization(**kwargs):
        calls.append(kwargs)
        return _Report()

    original_argv = sys.argv[:]
    try:
        run(
            str(target),
            ("--optimization-config", "config.py"),
            optimization_config_path="config.yaml",
            report_dir="reports",
            apply_optimization=apply_optimization,
        )
    finally:
        sys.argv = original_argv

    assert calls == [
        {
            "optimization_config_path": "config.yaml",
            "report_dir": "reports",
            "force_groups": [],
            "disable_groups": [],
        }
    ]
    assert result.read_text(encoding="utf-8") == f"{target}|--optimization-config|config.py"


def test_runner_reports_actual_optimization_summary(capsys, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    _print_optimization_result(_Report())
    assert capsys.readouterr().out == (
        "TURBO_PHYSAI_OPTIMIZATION_COMPLETED rank=3 applied=1 skipped=0 blocked=0 "
        "failed=0 rolled_back=0 not_started=0 run_id=test-run\n"
    )


def test_builtin_optimization_config_catalog_ignores_appledouble_metadata(tmp_path, monkeypatch):
    source = (
        Path(__file__).parents[1]
        / "turbo_physai/optimizations/common/configs/optimization.yaml"
    )
    config_dir = tmp_path / "common" / "configs"
    config_dir.mkdir(parents=True)
    (config_dir / "optimization.yaml").write_text(
        source.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (config_dir / "._optimization.yaml").write_bytes(b"\x00\xa3appledouble")
    monkeypatch.setattr(loader, "PACKAGED_OPTIMIZATION_ROOT", tmp_path)
    assert loader.OptimizationConfigCatalog.from_builtin_files().get("common.hcu.base") is not None


def test_python_api_resolves_packaged_model_optimization_config(tmp_path, monkeypatch):
    model_config = tmp_path / "bevformer" / "configs" / "optimization.yaml"
    model_config.parent.mkdir(parents=True)
    model_config.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(loader, "PACKAGED_MODEL_OPTIMIZATION_ROOT", tmp_path)

    assert loader.resolve_optimization_config_path(model="BEVFormer") == model_config.resolve()


def test_python_api_model_takes_priority_over_environment_config(tmp_path, monkeypatch):
    model_config = tmp_path / "models" / "bevformer" / "configs" / "optimization.yaml"
    model_config.parent.mkdir(parents=True)
    model_config.write_text("placeholder", encoding="utf-8")
    environment_config = tmp_path / "environment.yaml"
    environment_config.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        loader,
        "PACKAGED_MODEL_OPTIMIZATION_ROOT",
        tmp_path / "models",
    )
    monkeypatch.setenv(
        "TURBO_PHYSAI_OPTIMIZATION_CONFIG",
        str(environment_config),
    )

    assert loader.resolve_optimization_config_path(model="bevformer") == model_config.resolve()


def test_explicit_optimization_config_takes_priority_over_model(tmp_path):
    explicit = tmp_path / "custom.yaml"
    explicit.write_text("placeholder", encoding="utf-8")

    assert loader.resolve_optimization_config_path(
        explicit,
        model="unknown-model",
    ) == explicit.resolve()


def test_runtime_config_sets_environment_and_affinity(tmp_path, monkeypatch):
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(
        "schema_version: turbophysai/runtime-config/v1\nkind: RuntimeConfig\n"
        "environment:\n  set:\n    FEATURE: enabled\n  unset: [REMOVE_ME]\n"
        "process:\n  rank_affinity:\n    '0': 0-1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("REMOVE_ME", "yes")
    environment = prepare_environment(load_runtime_config(runtime_path))
    assert environment["FEATURE"] == "enabled"
    assert "REMOVE_ME" not in environment
    assert json.loads(environment["TURBO_PHYSAI_RANK_AFFINITY"]) == {"0": "0-1"}


def test_runtime_config_sets_rank_numa(tmp_path):
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(
        "schema_version: turbophysai/runtime-config/v1\nkind: RuntimeConfig\n"
        "process:\n  rank_numa:\n    '0': 2\n",
        encoding="utf-8",
    )
    environment = prepare_environment(load_runtime_config(runtime_path))
    assert json.loads(environment["TURBO_PHYSAI_RANK_NUMA"]) == {"0": 2}


def test_runtime_config_enables_automatic_numa(tmp_path):
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(
        "schema_version: turbophysai/runtime-config/v1\nkind: RuntimeConfig\n"
        "process:\n  numa: auto\n",
        encoding="utf-8",
    )
    environment = prepare_environment(load_runtime_config(runtime_path))
    assert environment["TURBO_PHYSAI_NUMA_AUTO"] == "1"


def test_runner_reexecs_rank_with_numactl(monkeypatch):
    monkeypatch.setenv("TURBO_PHYSAI_RANK_NUMA", '{"1": 3}')
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr("turbo_physai.runner.shutil.which", lambda _: "/usr/bin/numactl")
    with patch("turbo_physai.runner.os.execvpe") as execvpe:
        _set_rank_numa_binding()
    executable, command, environment = execvpe.call_args.args
    assert executable == "/usr/bin/numactl"
    assert command[:3] == ["/usr/bin/numactl", "--cpunodebind=3", "--membind=3"]
    assert command[3:6] == [sys.executable, "-m", "turbo_physai.runner"]
    assert environment["TURBO_PHYSAI_NUMA_BOUND"] == "1"


def test_runner_discovers_rank_numa_from_hy_smi(monkeypatch):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")
    monkeypatch.setattr("turbo_physai.runner.shutil.which", lambda _: "/usr/bin/hy-smi")
    completed = type("Completed", (), {
        "returncode": 0,
        "stderr": "",
        "stdout": "HCU[4] : (Topology) Numa Node 1\n",
    })()
    with patch("turbo_physai.runner.subprocess.run", return_value=completed) as command:
        assert _discover_rank_numa_node("0") == 1
    assert command.call_args.args[0] == ["/usr/bin/hy-smi", "--showtopo"]


def test_runner_starts_training_when_an_independent_group_is_blocked(tmp_path):
    target = tmp_path / "target.py"
    result = tmp_path / "result.txt"
    target.write_text(
        f"from pathlib import Path\nPath({str(result)!r}).write_text('ran')\n",
        encoding="utf-8",
    )

    class BlockedReport:
        run_id = "blocked"
        summary = {"blocked": 1}

    run(str(target), (), apply_optimization=lambda **_: BlockedReport())
    assert result.read_text(encoding="utf-8") == "ran"


def test_run_cli_rewrites_torchrun_and_prepares_runtime(tmp_path):
    runtime_path = tmp_path / "runtime.yaml"
    optimization_config_path = tmp_path / "config.yaml"
    runtime_path.write_text(
        "schema_version: turbophysai/runtime-config/v1\nkind: RuntimeConfig\n"
        "environment:\n  set: {FEATURE: enabled}\n"
        "process:\n  rank_affinity: {'0': 0-1}\n  rank_numa: {'0': 0}\n",
        encoding="utf-8",
    )
    optimization_config_path.write_text("placeholder", encoding="utf-8")
    with patch("turbo_physai.cli._run_training_command", return_value=17) as launch:
        result = cli_main([
            "run", "--optimization-config", str(optimization_config_path), "--runtime-config", str(runtime_path),
            "--force-group", "customer.encoder", "customer.decoder",
            "--disable-group", "customer.compile", "customer.training",
            "--set-rank-affinity", "1=2-3", "--set-rank-numa", "1=1", "--enable-numa", "--", "torchrun",
            "--nproc-per-node=2", "tools/train.py", "config.py",
        ])
    command, environment = launch.call_args.args
    assert result == 17
    assert command[:2] == ["torchrun", "--nproc-per-node=2"]
    assert command[2:5] == ["-m", "turbo_physai.runner", "--optimization-config"]
    assert command[command.index("--force-group") + 1] == "customer.encoder"
    assert command[command.index("--disable-group") + 1 : command.index("--")] == [
        "customer.compile", "customer.training"
    ]
    assert command[-3:] == ["--", "tools/train.py", "config.py"]
    assert environment["FEATURE"] == "enabled"
    assert json.loads(environment["TURBO_PHYSAI_RANK_AFFINITY"]) == {
        "0": "0-1", "1": "2-3"
    }
    assert json.loads(environment["TURBO_PHYSAI_RANK_NUMA"]) == {"0": 0, "1": 1}
    assert environment["TURBO_PHYSAI_NUMA_AUTO"] == "1"
    assert environment["TURBO_PHYSAI_RUNTIME_CONFIG_PATH"] == str(
        runtime_path.resolve()
    )


def test_python_launcher_preserves_isolated_mode(tmp_path):
    optimization = tmp_path / "optimization.yaml"
    rewritten = rewrite_command(
        ["python", "-I", "-u", "tools/train.py", "config.py"],
        str(optimization),
        "reports",
    )

    assert rewritten[:5] == [
        "python",
        "-I",
        "-u",
        "-m",
        "turbo_physai.runner",
    ]
    assert rewritten[-3:] == ["--", "tools/train.py", "config.py"]


def test_torchpack_launcher_preserves_launcher_and_python_options(tmp_path):
    optimization = tmp_path / "optimization.yaml"
    rewritten = rewrite_command(
        [
            "torchpack",
            "dist-run",
            "-np",
            "8",
            "--hostfile",
            "hosts",
            "python",
            "-I",
            "tools/train.py",
            "config.py",
        ],
        str(optimization),
        "reports",
        ("bevfusion.compile",),
    )

    assert rewritten[:8] == [
        "torchpack",
        "dist-run",
        "-np",
        "8",
        "--hostfile",
        "hosts",
        "python",
        "-I",
    ]
    assert rewritten[8:10] == ["-m", "turbo_physai.runner"]
    assert rewritten[rewritten.index("--force-group") + 1] == "bevfusion.compile"
    assert rewritten[-3:] == ["--", "tools/train.py", "config.py"]


def test_torchpack_launcher_supports_python_module(tmp_path):
    rewritten = rewrite_command(
        [
            "torchpack",
            "dist-run",
            "-np",
            "8",
            "python3.10",
            "-I",
            "-m",
            "customer.train",
            "--epochs",
            "1",
        ],
        str(tmp_path / "optimization.yaml"),
        "reports",
    )

    assert rewritten[:7] == [
        "torchpack",
        "dist-run",
        "-np",
        "8",
        "python3.10",
        "-I",
        "-m",
    ]
    assert rewritten[7] == "turbo_physai.runner"
    assert rewritten[-5:] == [
        "--module",
        "customer.train",
        "--",
        "--epochs",
        "1",
    ]


def test_torchpack_launcher_rejects_non_python_entry(tmp_path):
    try:
        rewrite_command(
            ["torchpack", "dist-run", "-np", "8", "bash", "train.sh"],
            str(tmp_path / "optimization.yaml"),
            "reports",
        )
    except TurboPhysAIError as error:
        assert "requires a supported Python command" in str(error)
    else:
        raise AssertionError("non-Python TorchPack command was accepted")


def test_run_config_defaults_to_common_optimization():
    optimization_config_path, runtime_path = _resolve_run_configs(None, None, None)

    assert optimization_config_path.name == "optimization.yaml"
    assert optimization_config_path.parent.parent.name == "common"
    assert runtime_path is None


def test_run_config_selects_builtin_model_configs():
    for model in ("BEVFormer", "BEVFusion"):
        optimization_config_path, runtime_path = _resolve_run_configs(
            model, None, None
        )

        assert optimization_config_path.name == "optimization.yaml"
        assert optimization_config_path.parent.parent.name == model.lower()
        assert runtime_path is not None
        assert runtime_path.name == "runtime.yaml"
        assert runtime_path.parent == optimization_config_path.parent


def test_explicit_run_configs_override_model_configs(tmp_path):
    config = tmp_path / "custom-config.yaml"
    runtime = tmp_path / "custom-runtime.yaml"
    config.write_text("config", encoding="utf-8")
    runtime.write_text("runtime", encoding="utf-8")

    optimization_config_path, runtime_path = _resolve_run_configs(
        "bevformer", str(config), str(runtime)
    )

    assert optimization_config_path == config.resolve()
    assert runtime_path == runtime.resolve()


def test_run_config_rejects_unknown_builtin_model():
    try:
        _resolve_run_configs("unknown-model", None, None)
    except TurboPhysAIError as error:
        assert "unknown built-in model" in str(error)
        assert "bevformer" in str(error)
    else:
        raise AssertionError("unknown model was accepted")


def test_run_cli_uses_builtin_model_configs():
    with patch("turbo_physai.cli._run_training_command", return_value=0) as launch:
        result = cli_main([
            "run", "--model", "bevformer", "--", "python", "tools/train.py"
        ])

    command, environment = launch.call_args.args
    assert result == 0
    assert "optimizations/models/bevformer/configs/optimization.yaml" in "/".join(
        command
    )
    assert environment["NCCL_ALGO"] == "Ring"


def test_run_training_command_creates_process_group():
    process = type("Process", (), {"wait": lambda self: 0})()
    with patch("turbo_physai.cli.subprocess.Popen", return_value=process) as popen:
        assert _run_training_command(["torchrun"], {"FEATURE": "1"}) == 0
    assert popen.call_args.kwargs["start_new_session"] is True


def test_run_training_command_forwards_interrupt_to_process_group():
    class Process:
        pid = 321

        def __init__(self):
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                from turbo_physai.cli import _LaunchSignal

                raise _LaunchSignal(signal.SIGINT)
            return -signal.SIGINT

        def poll(self):
            return None

    process = Process()
    with (
        patch("turbo_physai.cli.subprocess.Popen", return_value=process),
        patch("turbo_physai.cli.os.killpg") as killpg,
    ):
        assert _run_training_command(["torchrun"], {}) == 130
    killpg.assert_called_once_with(321, signal.SIGINT)


def test_run_training_command_escalates_when_process_does_not_stop():
    class Process:
        pid = 654

        def __init__(self):
            self.waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                from turbo_physai.cli import _LaunchSignal

                raise _LaunchSignal(signal.SIGINT)
            if self.waits in {2, 3}:
                raise subprocess.TimeoutExpired("torchrun", timeout)
            return -signal.SIGKILL

        def poll(self):
            return None

    process = Process()
    with (
        patch("turbo_physai.cli.subprocess.Popen", return_value=process),
        patch("turbo_physai.cli.os.killpg") as killpg,
    ):
        assert _run_training_command(["torchrun"], {}) == 130
    assert [call.args for call in killpg.call_args_list] == [
        (654, signal.SIGINT),
        (654, signal.SIGTERM),
        (654, signal.SIGKILL),
    ]
