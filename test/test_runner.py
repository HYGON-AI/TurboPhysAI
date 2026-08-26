# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from turbo_physai.bootstrap import (
    ACTIVATION_FAILURE_EXIT_CODE,
    RUN_ID,
    RUNTIME_CONFIG_PATH,
    SITE_DIR,
    bootstrap_environment,
    isolation_flags,
    should_activate,
)
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PACKAGED_CONFIG = (
    _REPO_ROOT / "turbo_physai/optimizations/common/configs/optimization.yaml"
)


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


def test_runtime_config_defaults_to_automatic_numa():
    environment = prepare_environment(load_runtime_config(None))
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


def test_runner_replays_the_original_command_when_reexecing(monkeypatch):
    """The bootstrap channel must not rewrite the user's command to bind NUMA."""

    monkeypatch.setenv("TURBO_PHYSAI_RANK_NUMA", '{"0": 2}')
    monkeypatch.setattr("turbo_physai.runner.shutil.which", lambda _: "/usr/bin/numactl")
    original = ["python", "-u", "tools/train.py", "config.py"]
    with patch("turbo_physai.runner.os.execvpe") as execvpe:
        _set_rank_numa_binding(reexec_command=original)
    _, command, _ = execvpe.call_args.args
    assert command == [
        "/usr/bin/numactl", "--cpunodebind=2", "--membind=2", *original
    ]


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


def test_run_cli_passes_torchrun_command_through_unchanged(tmp_path):
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(
        "schema_version: turbophysai/runtime-config/v1\nkind: RuntimeConfig\n"
        "environment:\n  set: {FEATURE: enabled}\n"
        "process:\n  rank_affinity: {'0': 0-1}\n  rank_numa: {'0': 0}\n",
        encoding="utf-8",
    )
    with patch("turbo_physai.cli._run_training_command", return_value=17) as launch:
        result = cli_main([
            "run", "--optimization-config", str(_PACKAGED_CONFIG),
            "--runtime-config", str(runtime_path),
            "--force-group", "mmcv.msda",
            "--disable-group", "mmdet3d.gaussian,mmdet3d.bev_pool",
            "torchrun", "--nproc-per-node=2", "tools/train.py", "config.py",
        ])
    command, environment = launch.call_args.args
    assert result == 17
    assert command == [
        "torchrun", "--nproc-per-node=2", "tools/train.py", "config.py"
    ]
    assert environment["FEATURE"] == "enabled"
    assert environment["TURBO_PHYSAI_BOOTSTRAP"] == "1"
    assert environment["PYTHONPATH"].split(os.pathsep)[0] == str(SITE_DIR)
    assert environment["TURBO_PHYSAI_OPTIMIZATION_CONFIG"] == str(_PACKAGED_CONFIG)
    assert environment["TURBO_PHYSAI_FORCE_GROUPS"] == "mmcv.msda"
    assert environment["TURBO_PHYSAI_DISABLE_GROUPS"] == os.pathsep.join(
        ("mmdet3d.gaussian", "mmdet3d.bev_pool")
    )
    assert json.loads(environment["TURBO_PHYSAI_RANK_AFFINITY"]) == {"0": "0-1"}
    assert json.loads(environment["TURBO_PHYSAI_RANK_NUMA"]) == {"0": 0}
    assert environment["TURBO_PHYSAI_NUMA_AUTO"] == "1"


def test_run_cli_rejects_invalid_group_lists_before_launch(capsys):
    for value, message in (
        ("mmdet3d.gaussian,,mmdet3d.bev_pool", "non-empty Group IDs"),
        ("mmdet3d.gaussian,mmdet3d.gaussian", "duplicate Group ID"),
        ("mmdet3d.gaussian,missing.group", "unknown Optimization Groups"),
    ):
        with patch("turbo_physai.cli._run_training_command") as launch:
            assert cli_main([
                "run", "--optimization-config", str(_PACKAGED_CONFIG),
                "--disable-group", value,
                "python", "tools/train.py",
            ]) == 2
        assert message in capsys.readouterr().err
        launch.assert_not_called()


def test_run_cli_reports_a_missing_group_separator(capsys):
    with patch("turbo_physai.cli._run_training_command") as launch:
        assert cli_main([
            "run", "--optimization-config", str(_PACKAGED_CONFIG),
            "--disable-group", "mmcv.msda",
            "mmdet3d.gaussian", "torchrun", "tools/train.py",
        ]) == 2
    assert "separate multiple Group IDs with commas" in capsys.readouterr().err
    launch.assert_not_called()


def test_run_cli_group_validation_does_not_import_optimization_modules():
    with (
        patch("turbo_physai.cli._run_training_command", return_value=0),
        patch(
            "turbo_physai.engine.config.loader.importlib.import_module"
        ) as import_module,
    ):
        assert cli_main([
            "run", "--optimization-config", str(_PACKAGED_CONFIG),
            "--disable-group", "mmdet3d.gaussian",
            "python", "tools/train.py",
        ]) == 0
    import_module.assert_not_called()


def test_run_cli_disables_all_numa_binding(tmp_path):
    runtime_path = tmp_path / "runtime.yaml"
    runtime_path.write_text(
        "schema_version: turbophysai/runtime-config/v1\nkind: RuntimeConfig\n"
        "process:\n  numa: auto\n  rank_numa: {'0': 0}\n",
        encoding="utf-8",
    )
    with patch("turbo_physai.cli._run_training_command", return_value=0) as launch:
        result = cli_main([
            "run", "--optimization-config", str(_PACKAGED_CONFIG),
            "--runtime-config", str(runtime_path), "--disable-numa",
            "--", "python", "tools/train.py",
        ])
    _, environment = launch.call_args.args
    assert result == 0
    assert "TURBO_PHYSAI_NUMA_AUTO" not in environment
    assert "TURBO_PHYSAI_RANK_NUMA" not in environment
    assert environment["TURBO_PHYSAI_RUNTIME_CONFIG_PATH"] == str(
        runtime_path.resolve()
    )


def test_run_cli_accepts_launchers_the_rewrite_channel_rejected(tmp_path):
    """Commands the launcher grammar could not parse now pass through verbatim."""

    for command in (
        ["deepspeed", "--num_gpus=8", "tools/train.py"],
        ["accelerate", "launch", "tools/train.py"],
        ["bash", "scripts/train.sh"],
        ["srun", "--ntasks=8", "python", "tools/train.py"],
        ["torchrun", "--no-python", "./train_wrapper"],
    ):
        with patch("turbo_physai.cli._run_training_command", return_value=0) as launch:
            assert cli_main([
                "run", "--optimization-config", str(_PACKAGED_CONFIG), "--", *command
            ]) == 0
        assert launch.call_args.args[0] == command


def test_run_cli_rejects_commands_that_would_silently_skip_the_bootstrap(capsys):
    for flag in ("-E", "-I", "-S"):
        assert cli_main([
            "run", "--optimization-config", str(_PACKAGED_CONFIG),
            "--", "python", flag, "tools/train.py",
        ]) == 2
        assert "silently run training unoptimized" in capsys.readouterr().err


def test_isolation_flags_ignores_training_arguments():
    assert isolation_flags(["python", "-I", "-u", "train.py"]) == ("-I",)
    assert isolation_flags(["python", "-ES", "train.py"]) == ("-E", "-S")
    # --seed and -Iou are not interpreter flag clusters.
    assert isolation_flags(["python", "train.py", "--seed", "1", "-Iou"]) == ()


def test_bootstrap_activates_only_in_training_ranks():
    active = {"TURBO_PHYSAI_BOOTSTRAP": "1"}
    assert should_activate(["python", "tools/train.py", "cfg.py"], active) is True
    assert should_activate(["python", "-u", "tools/train.py"], active) is True
    assert should_activate(["python", "-m", "customer.train"], active) is True
    # Launchers are Python too, but they must not apply.
    assert should_activate(["python", "/usr/bin/torchrun", "-n", "2"], active) is False
    assert should_activate(["python", "-m", "torch.distributed.run"], active) is False
    assert should_activate(["python", "-m", "deepspeed.launcher.runner"], active) is False
    # multiprocessing spawn helpers and interactive shells.
    assert should_activate(["python", "-c", "pass"], active) is False
    assert should_activate(["python"], active) is False
    # Without the flag nothing activates, even for a training entry point.
    assert should_activate(["python", "tools/train.py"], {}) is False


def test_bootstrap_environment_preserves_existing_pythonpath():
    environment = bootstrap_environment(
        {"PYTHONPATH": "/opt/user"},
        optimization_config="/cfg.yaml",
        report_dir="reports",
    )
    assert environment["PYTHONPATH"] == f"{SITE_DIR}{os.pathsep}/opt/user"
    assert len(environment[RUN_ID]) == 32
    assert set(environment[RUN_ID]) <= set("0123456789abcdef")
    assert "TURBO_PHYSAI_FORCE_GROUPS" not in environment
    assert "TURBO_PHYSAI_DISABLE_GROUPS" not in environment


def _bootstrap_env(**extra):
    environment = bootstrap_environment(
        os.environ,
        optimization_config=str(_PACKAGED_CONFIG),
        report_dir=str(Path(extra.pop("report_dir", "turbophysai_reports"))),
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        [environment["PYTHONPATH"], str(_REPO_ROOT)]
    )
    environment.update(extra)
    return environment


def test_bootstrap_applies_before_the_training_script_runs(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(
        "import sys\n"
        "print('TRAIN_STARTED patched=%s' % ('turbo_physai' in sys.modules))\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        env=_bootstrap_env(report_dir=str(tmp_path / "reports")),
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    applied = completed.stdout.index("TURBO_PHYSAI_OPTIMIZATION_COMPLETED")
    started = completed.stdout.index("TRAIN_STARTED")
    assert applied < started, completed.stdout


def test_bootstrap_makes_the_training_working_directory_importable(tmp_path):
    package = tmp_path / "model_project"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'available'\n", encoding="utf-8")
    scripts = tmp_path / "tools"
    scripts.mkdir()
    script = scripts / "train.py"
    script.write_text(
        "import model_project\n"
        "print('MODEL_PROJECT=%s' % model_project.VALUE)\n",
        encoding="utf-8",
    )
    environment = _bootstrap_env(report_dir=str(tmp_path / "reports"))
    completed = subprocess.run(
        [sys.executable, str(script)],
        env=environment,
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "MODEL_PROJECT=available" in completed.stdout


def test_bootstrap_descendants_update_one_shared_report(tmp_path):
    scripts = []
    for name in ("prepare.py", "train.py"):
        script = tmp_path / name
        script.write_text(f"print({name!r})\n", encoding="utf-8")
        scripts.append(script)
    report_dir = tmp_path / "reports"
    environment = _bootstrap_env(report_dir=str(report_dir))

    for script in scripts:
        phase_environment = dict(environment)
        phase_environment[RUNTIME_CONFIG_PATH] = f"/{script.stem}/runtime.yaml"
        completed = subprocess.run(
            [sys.executable, str(script)],
            env=phase_environment,
            cwd=tmp_path,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 0, completed.stderr

    json_reports = list(report_dir.glob("optimization_report-*.json"))
    markdown_reports = list(report_dir.glob("optimization_report-*.md"))
    assert len(json_reports) == 1
    assert len(markdown_reports) == 1
    assert environment[RUN_ID] in json_reports[0].name
    report = json.loads(json_reports[0].read_text(encoding="utf-8"))
    assert report["runtime_config_path"] == "/train/runtime.yaml"
    assert not list(report_dir.glob("*.tmp"))


def test_bootstrap_aborts_instead_of_training_unoptimized(tmp_path):
    """site.execsitecustomize swallows exceptions; activation must not rely on them."""

    script = tmp_path / "train.py"
    script.write_text("print('TRAIN_STARTED')\n", encoding="utf-8")
    environment = _bootstrap_env(report_dir=str(tmp_path / "reports"))
    environment["TURBO_PHYSAI_OPTIMIZATION_CONFIG"] = str(tmp_path / "missing.yaml")
    completed = subprocess.run(
        [sys.executable, str(script)],
        env=environment, cwd=tmp_path, text=True, capture_output=True,
    )
    assert completed.returncode == ACTIVATION_FAILURE_EXIT_CODE
    assert "TRAIN_STARTED" not in completed.stdout
    assert "aborting instead of training unoptimized" in completed.stderr


def test_bootstrap_does_not_activate_in_launcher_processes(tmp_path):
    launcher = tmp_path / "torchrun"
    launcher.write_text("print('LAUNCHER_RAN')\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(launcher)],
        env=_bootstrap_env(report_dir=str(tmp_path / "reports")),
        cwd=tmp_path, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "LAUNCHER_RAN" in completed.stdout
    assert "TURBO_PHYSAI_OPTIMIZATION_COMPLETED" not in completed.stdout


def test_bootstrap_chains_to_an_existing_user_sitecustomize(tmp_path):
    user_site = tmp_path / "usersite"
    user_site.mkdir()
    (user_site / "sitecustomize.py").write_text(
        "print('USER_SITECUSTOMIZE_RAN')\n", encoding="utf-8"
    )
    script = tmp_path / "train.py"
    script.write_text("print('TRAIN_STARTED')\n", encoding="utf-8")
    environment = _bootstrap_env(report_dir=str(tmp_path / "reports"))
    environment["PYTHONPATH"] = os.pathsep.join(
        [environment["PYTHONPATH"], str(user_site)]
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        env=environment, cwd=tmp_path, text=True, capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "USER_SITECUSTOMIZE_RAN" in completed.stdout
    assert "TURBO_PHYSAI_OPTIMIZATION_COMPLETED" in completed.stdout


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
    assert command == ["python", "tools/train.py"]
    assert "optimizations/models/bevformer/configs/optimization.yaml" in environment[
        "TURBO_PHYSAI_OPTIMIZATION_CONFIG"
    ]
    assert environment["NCCL_ALGO"] == "Ring"


def test_run_training_command_replaces_this_process():
    """execvpe leaves no TurboPhysAI process in the tree to forward signals."""

    with patch("turbo_physai.cli.os.execvpe") as execvpe:
        try:
            _run_training_command(["torchrun", "-n", "2"], {"FEATURE": "1"})
        except AssertionError:
            pass  # execvpe is mocked, so it returns instead of replacing us
    executable, command, environment = execvpe.call_args.args
    assert executable == "torchrun"
    assert command == ["torchrun", "-n", "2"]
    assert environment == {"FEATURE": "1"}


def test_run_training_command_reports_a_missing_executable():
    with patch("turbo_physai.cli.os.execvpe", side_effect=FileNotFoundError("nope")):
        try:
            _run_training_command(["definitely-not-a-command"], {})
        except TurboPhysAIError as error:
            assert "failed to execute training command" in str(error)
        else:
            raise AssertionError("a missing executable was not reported")
