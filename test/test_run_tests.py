# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Behavior checks for imports, process failures and coverage reports."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_tests.py"
spec = importlib.util.spec_from_file_location("test_runner_tool", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_local_report_cleanup_keeps_unrelated_files_and_symlink_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    run = runner.prepare_local_report()
    (run / ".coverage.previous").write_text("old coverage")
    (run / "common.events.jsonl").write_text("old events")
    (run / "notes.txt").write_text("keep")
    external = tmp_path / "external"
    external.mkdir()
    (external / "index.html").write_text("keep target")
    (run / "htmlcov").symlink_to(external, target_is_directory=True)
    assert runner.prepare_local_report() == run
    assert not (run / ".coverage.previous").exists()
    assert not (run / "common.events.jsonl").exists()
    assert not (run / "htmlcov").exists()
    assert (external / "index.html").read_text() == "keep target"
    assert (run / "notes.txt").read_text() == "keep"


def test_output_directory_option_is_removed_and_ci_directory_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as exc:
        runner.main(["--output-dir", str(tmp_path / "other")])
    assert exc.value.code == 2
    assert not (tmp_path / "test-reports").exists()
    ci = tmp_path / "ci-run/coverage"
    ci.mkdir(parents=True)
    marker = ci / "summary.md"
    marker.write_text("previous CI run")
    with pytest.raises(FileExistsError):
        runner.main(["--_run-dir", str(ci)])
    assert marker.read_text() == "previous CI run"


def test_local_repeated_runs_replace_reports_without_merging_previous_coverage(tmp_path):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    script = root / "scripts/run_tests.py"
    script.write_bytes(SCRIPT.read_bytes())
    package = root / "turbo_physai"
    package.mkdir()
    (package / "__init__.py").write_text(
        "def compute(x):\n    if x > 0:\n        return x * 2\n    return 0\n")
    test = root / "test_repeat.py"
    test.write_text("from turbo_physai import compute\n"
                    "def test_positive(): assert compute(3) == 6\n"
                    "def test_negative(): assert compute(-1) == 0\n")
    run = root / "test-reports"
    for name in ("positive", "negative"):
        completed = subprocess.run([sys.executable, str(script), "--tests", str(test) + "::test_" + name,
                                    "--", "-q"], cwd=tmp_path, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads((run / "manifest.json").read_text())
        assert result["counts"]["tests"] == 1
        assert len((run / "common.events.jsonl").read_text().splitlines()) == 3
        if name == "positive":
            (run / ".coverage.stale").write_text("invalid old fragment")
            (run / "htmlcov/obsolete.html").write_text("old page")
    record = json.loads((run / "coverage.json").read_text())["files"]["turbo_physai/__init__.py"]
    assert 3 not in record["executed_lines"]
    assert 4 in record["executed_lines"]
    assert not (run / ".coverage.stale").exists()
    assert not (run / "htmlcov/obsolete.html").exists()


@pytest.mark.parametrize("selection", ["directory", "relative_node", "absolute_node"])
def test_test_selection_is_resolved_from_checkout_outside_working_directory(tmp_path, selection):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_tests.py").write_bytes(SCRIPT.read_bytes())
    (root / "turbo_physai").mkdir()
    (root / "turbo_physai/__init__.py").write_text("VALUE = 1\n")
    (root / "test").mkdir()
    test = root / "test/test_paths.py"
    test.write_text("def test_selected(): assert True\ndef test_unselected(): assert False\n")
    selectors = dict(directory="test", relative_node="test/test_paths.py::test_selected",
                     absolute_node=str(test) + "::test_selected")
    completed = subprocess.run([sys.executable, str(root / "scripts/run_tests.py"),
                                "--tests", selectors[selection],
                                "--", "-q"], cwd=tmp_path, capture_output=True, text=True)
    directory = selection == "directory"
    assert completed.returncode == int(directory), completed.stdout + completed.stderr
    run = root / "test-reports"
    result = json.loads((run / "manifest.json").read_text())
    assert "report_error" not in result, result
    assert result["counts"] == dict(tests=2 if directory else 1, failures=int(directory), errors=0, skipped=0)
    assert result["coverage_saved"]


def test_failure_continues_other_tests_and_keeps_report(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_common.py").write_text(
        'def test_failure(): assert False\ndef test_after_failure(): assert True\n')
    completed = subprocess.run([sys.executable, str(SCRIPT), "--tests", str(tests),
                                "--_run-dir", str(tmp_path / "out"), "--", "-q"],
                               capture_output=True, text=True)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    run = tmp_path / "out"
    result = json.loads((run / "manifest.json").read_text())
    assert "report_error" not in result, result
    assert result["counts"]["tests"] == 2
    assert result["counts"]["failures"] == 1
    assert result["status"] == "failed"
    events = [json.loads(line) for line in (run / "common.events.jsonl").read_text().splitlines()]
    assert any(event["test"].endswith("test_after_failure") and event["phase"] == "call"
               and event["outcome"] == "passed" for event in events)
    assert result["groups"]["算子 API（Python）"]["covered_lines"] == 0
    assert result["groups"]["算子 API（Python）"]["statements"] > 0
    assert (run / "htmlcov/index.html").is_file()
    assert "整体覆盖率" in (run / "summary.md").read_text()


def test_abrupt_process_exit_preserves_events_and_marks_missing_results(tmp_path):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_tests.py").write_bytes(SCRIPT.read_bytes())
    (root / "turbo_physai").mkdir()
    (root / "turbo_physai/__init__.py").write_text("VALUE = 1\n")
    test = root / "test_crash.py"
    test.write_text('import os\ndef test_first(): assert True\ndef test_crash(): os._exit(17)\n')
    completed = subprocess.run([sys.executable, str(root / "scripts/run_tests.py"), "--tests", str(test),
                                "--", "-q"],
                               capture_output=True, text=True)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    run = root / "test-reports"
    result = json.loads((run / "manifest.json").read_text())
    assert result["exit_code"] == 17
    assert result["status"] == "execution_error"
    assert result["counts"] is None
    assert not result["coverage_saved"]
    assert "report_error" not in result, result
    assert '"phase": "call", "outcome": "passed"' in (run / "common.events.jsonl").read_text()
    assert "覆盖数据不完整" in (run / "summary.md").read_text()


def test_report_missing_data_is_explicit_and_excludes_models(tmp_path):
    result = dict(status="execution_error", counts=None, coverage_saved=False, diff_base="origin/main")
    runner.report(tmp_path, result)
    assert result["groups"]["算子 API（Python）"]["covered_lines"] == 0
    records = json.loads((tmp_path / "coverage.json").read_text())
    assert not any("/optimizations/models/" in name for name in records["files"])
    assert "覆盖数据不完整" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("different_wheel", [False, True])
def test_installed_worker_merges_parent_and_child_without_loading_checkout(tmp_path, monkeypatch, different_wheel):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_tests.py").write_bytes(SCRIPT.read_bytes())
    package = root / "turbo_physai"
    package.mkdir()
    code = "def compute(x):\n    if x > 0:\n        return x * 2\n    return 0\n"
    (package / "__init__.py").write_text(code)
    installed = tmp_path / "site-packages/turbo_physai"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text(code + ("# different release\n" if different_wheel else ""))
    for base in (package, installed):
        (base / "unused.py").write_text("UNUSED = 1\n")
        models = base / "optimizations/models"
        models.mkdir(parents=True)
        (models / "implementation.py").write_text("def model_only():\n    return 7\n")
    test = tmp_path / "test_wheel.py"
    child_cwd = tmp_path / "child"
    child_cwd.mkdir()
    grandchild = "from turbo_physai import compute; assert compute(-1) == 0"
    child = ("import subprocess, sys; "
             f"subprocess.run([sys.executable, '-c', {grandchild!r}], check=True)")
    test.write_text(
        'import subprocess, sys\nfrom turbo_physai import compute\n'
        'from turbo_physai.optimizations.models.implementation import model_only\n'
        'def test_positive():\n'
        '    assert compute(3) == 6\n'
        '    assert model_only() == 7\n'
        f"    subprocess.run([sys.executable, '-c', {child!r}], cwd={str(child_cwd)!r}, check=True)\n")
    run = tmp_path / "report"
    run.mkdir()
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps(dict(installed=True, tests=[str(test)], pytest_args=["-q"])))
    completed = subprocess.run([sys.executable, str(root / "scripts/run_tests.py"), "--_worker", str(manifest)],
                               cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(installed.parent)),
                               capture_output=True, text=True)
    if different_wheel:
        assert completed.returncode != 0
        assert "wheel Python source differs" in completed.stderr
    else:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads(manifest.read_text())
        assert result["component_package"] == str(installed)
        result.update(status="passed", counts=dict(tests=1, failures=0, errors=0, skipped=0), diff_base="origin/main")
        monkeypatch.setattr(runner, "ROOT", root)
        runner.report(run, result)
        records = json.loads((run / "coverage.json").read_text())["files"]
        assert set(records) == {"turbo_physai/__init__.py", "turbo_physai/unused.py"}
        assert 3 in records["turbo_physai/__init__.py"]["executed_lines"]
        assert 4 in records["turbo_physai/__init__.py"]["executed_lines"]
        assert records["turbo_physai/__init__.py"]["summary"]["missing_branches"] == 0
        assert records["turbo_physai/unused.py"]["summary"]["covered_lines"] == 0
        assert 'filename="turbo_physai/__init__.py"' in (run / "coverage.xml").read_text()


def test_source_worker_uses_checkout_without_installed_package(tmp_path):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_tests.py").write_bytes(SCRIPT.read_bytes())
    (root / "turbo_physai").mkdir()
    (root / "turbo_physai/__init__.py").write_text("VALUE = 'source'\n")
    other = tmp_path / "other/turbo_physai"
    other.mkdir(parents=True)
    (other / "__init__.py").write_text("VALUE = 'installed'\n")
    test = tmp_path / "test_source.py"
    test.write_text("from turbo_physai import VALUE\ndef test_origin(): assert VALUE == 'source'\n")
    run = tmp_path / "report"
    run.mkdir()
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps(dict(installed=False, tests=[str(test)], pytest_args=["-q"])))
    completed = subprocess.run([sys.executable, str(root / "scripts/run_tests.py"), "--_worker", str(manifest)],
                               cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(other.parent)),
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(manifest.read_text())
    assert result["component_package"] == str(root / "turbo_physai")
    assert result["component_mode"] == "source"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["backward", "grad"])
def test_autograd_adapter_preserves_gradients_and_restores_functions(tmp_path, device, method):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a visible accelerator")
    import coverage
    source = tmp_path / "autograd_reference.py"
    source.write_text(
        "import torch\n"
        "class Square(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx, x):\n"
        "        ctx.save_for_backward(x)\n"
        "        return x * x\n"
        "    @staticmethod\n"
        "    def backward(ctx, gradient):\n"
        "        (x,) = ctx.saved_tensors\n"
        "        return gradient * 2 * x\n")
    spec = importlib.util.spec_from_file_location("autograd_reference", source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    cov = coverage.Coverage(config_file=False, source=[str(tmp_path)])
    cov.set_option("run:core", "ctrace")
    cov.start()
    backward, grad = torch.autograd.backward, torch.autograd.grad
    try:
        with runner.trace_autograd_threads():
            x = torch.tensor(3., device=device, requires_grad=True)
            if method == "backward":
                torch.autograd.backward(tensors=reference.Square.apply(x))
                assert x.grad.item() == 6.
            else:
                assert torch.autograd.grad(outputs=reference.Square.apply(x), inputs=x)[0].item() == 6.
            assert {9, 10} <= set(cov.get_data().lines(str(source)) or [])
            with pytest.raises(RuntimeError):
                torch.autograd.grad(outputs=x * x, inputs=torch.tensor(1., requires_grad=True))
        assert torch.autograd.backward is backward
        assert torch.autograd.grad is grad
    finally:
        cov.stop()


def diff_fixture(tmp_path, monkeypatch):
    import coverage
    root = tmp_path / "checkout"
    package = root / "turbo_physai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    module = package / "example.py"
    module.write_text("VALUE = 1\n")
    models = package / "optimizations/models"
    models.mkdir(parents=True)
    (models / "example.py").write_text("MODEL = 1\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.setattr(runner, "ROOT", root)
    runner.git("add", ".")
    runner.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
    base = runner.git("rev-parse", "HEAD")
    run = tmp_path / "report"
    run.mkdir()
    data = coverage.CoverageData(basename=str(run / ".coverage.fixture"))
    result = dict(status="passed", counts=dict(tests=1, failures=0, errors=0, skipped=0),
                  coverage_saved=True, diff_base=base)
    return module, run, data, result


@pytest.mark.parametrize("change", ["unstaged", "staged", "untracked"])
def test_diff_report_counts_changed_executable_lines_and_unimported_files(tmp_path, monkeypatch, change):
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    module.write_text("VALUE = 2\nOTHER = 3\n# not executable\n")
    (module.parent / "optimizations/models/example.py").write_text("MODEL = 2\n")
    if change == "untracked":
        (module.parent / "new.py").write_text("NEW = 4\n")
    elif change == "staged":
        runner.git("add", str(module))
    data.add_arcs({str(module): [(-1, 1), (1, -1)]})
    data.write()
    runner.report(run, result)
    diff = result["diff"]
    assert diff["status"] == "measured"
    assert diff["changed_lines"] == (3 if change == "untracked" else 2)
    assert diff["covered_lines"] == 1
    assert diff["percent"] == pytest.approx(100 / diff["changed_lines"])
    assert "optimizations/models/example.py" not in (run / "diff-cover.html").read_text()
    assert "本次变更覆盖率" in (run / "summary.md").read_text()
    assert not (run / "diff-cover.json").exists()


@pytest.mark.parametrize("comment_only", [False, True])
def test_diff_report_without_executable_changes_is_not_applicable(tmp_path, monkeypatch, comment_only):
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    if comment_only:
        module.write_text("# comment only\nVALUE = 1\n")
    line = 2 if comment_only else 1
    data.add_arcs({str(module): [(-1, line), (line, -1)]})
    data.write()
    runner.report(run, result)
    assert result["diff"]["status"] == "no_changes"
    assert result["diff"]["percent"] is None
    assert "不适用" in (run / "diff-cover.html").read_text()


def test_diff_report_missing_baseline_keeps_full_report(tmp_path, monkeypatch):
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    data.add_arcs({str(module): [(-1, 1), (1, -1)]})
    data.write()
    result["diff_base"] = "missing-baseline"
    runner.report(run, result)
    assert result["diff"]["status"] == "unavailable"
    assert "percent" not in result["diff"]
    assert (run / "htmlcov/index.html").is_file()
    assert "未生成" in (run / "diff-cover.html").read_text()
