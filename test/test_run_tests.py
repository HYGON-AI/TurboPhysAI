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


def make_checkout(tmp_path, code):
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/run_tests.py").write_bytes(SCRIPT.read_bytes())
    (root / "turbo_physai").mkdir()
    (root / "turbo_physai/__init__.py").write_text(code)
    return root


def commit(message, *, allow_empty=False):
    options = ["--allow-empty"] if allow_empty else []
    runner.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
               "commit", *options, "-qm", message)


def test_local_report_cleanup_keeps_unrelated_files_and_symlink_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    run = runner.prepare_local_report()
    (run / ".coverage.previous").write_text("old coverage")
    (run / "diff-cover.md").write_text("old standalone diff summary")
    (run / "common.events.jsonl").write_text("old events")
    (run / "notes.txt").write_text("keep")
    external = tmp_path / "external"
    external.mkdir()
    (external / "index.html").write_text("keep target")
    (run / "htmlcov").symlink_to(external, target_is_directory=True)
    assert runner.prepare_local_report() == run
    assert not (run / ".coverage.previous").exists()
    assert not (run / "diff-cover.md").exists()
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
    root = make_checkout(tmp_path,
        "def compute(x):\n    if x > 0:\n        return x * 2\n    return 0\n")
    script = root / "scripts/run_tests.py"
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
    root = make_checkout(tmp_path, "VALUE = 1\n")
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
    assert result["groups"]["Operator API (Python)"]["covered_lines"] == 0
    assert result["groups"]["Operator API (Python)"]["statements"] > 0
    assert (run / "htmlcov/index.html").is_file()
    assert "Overall coverage" in (run / "summary.md").read_text()


def test_abrupt_process_exit_preserves_events_and_marks_missing_results(tmp_path):
    root = make_checkout(tmp_path, "VALUE = 1\n")
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
    assert "Incomplete coverage data" in (run / "summary.md").read_text()


def test_report_missing_data_is_explicit_and_excludes_models(tmp_path):
    result = dict(status="execution_error", counts=None, coverage_saved=False, diff_base="origin/main")
    runner.report(tmp_path, result)
    assert result["groups"]["Operator API (Python)"]["covered_lines"] == 0
    records = json.loads((tmp_path / "coverage.json").read_text())
    assert not any("/optimizations/models/" in name for name in records["files"])
    assert "Incomplete coverage data" in (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("different_wheel", [False, True])
def test_installed_worker_merges_parent_and_child_without_loading_checkout(tmp_path, monkeypatch, different_wheel):
    code = "def compute(x):\n    if x > 0:\n        return x * 2\n    return 0\n"
    root = make_checkout(tmp_path, code)
    package = root / "turbo_physai"
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
    root = make_checkout(tmp_path, "VALUE = 'source'\n")
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
    assert not result["installed"]


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
    commit("baseline")
    base = runner.git("rev-parse", "HEAD")
    run = tmp_path / "report"
    run.mkdir()
    data = coverage.CoverageData(basename=str(run / ".coverage.fixture"))
    result = dict(status="passed", counts=dict(tests=1, failures=0, errors=0, skipped=0),
                  coverage_saved=True, diff_base=base)
    return module, run, data, result


@pytest.mark.parametrize("explicit,environment,expected", [
    ("chosen", "ci-base", "chosen"),
    (None, "ci-base", "ci-base"),
    (None, None, "upstream/main"),
])
def test_diff_baseline_precedence(tmp_path, monkeypatch, explicit, environment, expected):
    diff_fixture(tmp_path, monkeypatch)
    if environment:
        monkeypatch.setenv("TURBOPHYSAI_DIFF_BASE", environment)
    else:
        monkeypatch.delenv("TURBOPHYSAI_DIFF_BASE", raising=False)
    runner.git("update-ref", "refs/remotes/upstream/main", "HEAD")
    assert runner.resolve_diff_base(explicit) == expected


def test_diff_baseline_falls_back_without_fetching(tmp_path, monkeypatch):
    diff_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("TURBOPHYSAI_DIFF_BASE", raising=False)
    runner.git("update-ref", "refs/remotes/origin/main", "HEAD")
    assert runner.resolve_diff_base() == "origin/main"
    assert runner.git("for-each-ref", "--format=%(refname)", "refs/remotes") == "refs/remotes/origin/main"


def test_fork_main_at_head_measures_committed_changes_against_upstream(tmp_path, monkeypatch):
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("TURBOPHYSAI_DIFF_BASE", raising=False)
    runner.git("update-ref", "refs/remotes/upstream/main", "HEAD")
    module.write_text("VALUE = 2\n")
    runner.git("add", str(module))
    commit("update public component")
    runner.git("update-ref", "refs/remotes/origin/main", "HEAD")
    data.add_arcs({str(module): [(-1, 1), (1, -1)]})
    data.write()
    result["diff_base"] = runner.resolve_diff_base()
    result["component"] = runner.source_identity()
    runner.report(run, result)
    diff = result["diff"]
    assert diff["status"] == "measured"
    assert diff["changed_lines"] == diff["covered_lines"] == 1
    assert diff["commit_count"] == 1
    assert not result["component"]["dirty"]
    assert diff["commits"][0]["subject"] == "update public component"
    assert "Working tree has uncommitted changes: No" in (run / "summary.md").read_text()


def test_stale_upstream_lists_extra_comparison_commits(tmp_path, monkeypatch):
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv("TURBOPHYSAI_DIFF_BASE", raising=False)
    runner.git("update-ref", "refs/remotes/upstream/main", "HEAD")
    for index in range(12):
        commit(f"change {index} | title", allow_empty=True)
    runner.git("update-ref", "refs/remotes/origin/main", "HEAD")
    module.write_text("VALUE = 2\n")
    data.add_arcs({str(module): [(-1, 1), (1, -1)]})
    data.write()
    result["diff_base"] = runner.resolve_diff_base()
    result["component"] = runner.source_identity()
    runner.report(run, result)
    diff = result["diff"]
    assert diff["base_ref"] == "upstream/main"
    assert diff["commit_count"] == 12
    assert len(diff["commits"]) == 10
    assert diff["commits"][0]["subject"] == "change 11 | title"
    assert result["component"]["dirty"]
    summary = (run / "summary.md").read_text()
    assert "Commits after comparison start: 12" in summary
    assert "Working tree has uncommitted changes: Yes" in summary
    assert "change 11 &#124; title" in summary
    assert "Only the 10 most recent commits" in summary
    assert "change 1 &#124; title" not in summary
    assert "hint" not in diff


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
    result["component"] = runner.source_identity()
    runner.report(run, result)
    diff = result["diff"]
    assert diff["status"] == "measured"
    assert diff["changed_lines"] == (3 if change == "untracked" else 2)
    assert diff["covered_lines"] == 1
    summary = (run / "summary.md").read_text()
    assert (f"| Executable lines | {diff['changed_lines']} | 1 | {diff['changed_lines'] - 1} | "
            f"{1 / diff['changed_lines']:.2%} |") in summary
    assert diff["commit_count"] == 0
    assert result["component"]["dirty"]
    assert "Comparison starts at the current commit" in diff["hint"]
    assert diff["hint"] in (run / "summary.md").read_text()
    assert "N/A: " not in (run / "summary.md").read_text()
    assert "optimizations/models/example.py" not in (run / "diff-cover.html").read_text()
    assert "Changed-line coverage" in (run / "summary.md").read_text()
    assert not (run / "diff-cover.json").exists()
    assert not (run / "diff-cover.md").exists()


def test_diff_report_counts_nested_component_files(tmp_path, monkeypatch):
    # Component code lives in subpackages such as operators/; a top-level-only
    # pattern would silently report no changed lines.
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    nested = module.parent / "operators/nested.py"
    nested.parent.mkdir()
    (nested.parent / "__init__.py").write_text("")
    nested.write_text("FIRST = 1\nSECOND = 2\n")
    model = module.parent / "optimizations/models/example.py"
    model.write_text("MODEL = 2\n")
    data.add_arcs({str(nested): [(-1, 1), (1, -1)], str(model): [(-1, 1), (1, -1)]})
    data.write()
    runner.report(run, result)
    diff = result["diff"]
    assert (diff["changed_lines"], diff["covered_lines"]) == (2, 1)
    assert "turbo_physai/operators/nested.py" in (run / "diff-cover.html").read_text()
    assert "optimizations/models" not in (run / "diff-cover.html").read_text()


def test_diff_report_counts_branches_starting_on_changed_lines(tmp_path, monkeypatch):
    import coverage
    import runpy
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    module.write_text("def choose(x):\n    if x:\n        x += 1\n    return x\n")
    cov = coverage.Coverage(data_file=str(run / ".coverage.fixture"), config_file=False, branch=True)
    cov.start()
    runpy.run_path(str(module))["choose"](1)
    cov.stop()
    cov.save()
    runner.report(run, result)
    diff = result["diff"]
    # Every changed line runs, but the false branch of the new condition does not.
    assert diff["changed_lines"] == diff["covered_lines"] == 4
    assert (diff["changed_branches"], diff["covered_branches"]) == (2, 1)
    assert diff["missing_branches"] == [dict(path="turbo_physai/example.py", line=2, target=4)]
    summary = (run / "summary.md").read_text()
    assert "| Branches | 2 | 1 | 1 | 50.00% |" in summary
    assert "| turbo_physai/example.py | 2 | 4 |" in summary


@pytest.mark.parametrize("change", ["clean", "comment", "note"])
def test_diff_report_without_executable_changes_is_not_applicable(tmp_path, monkeypatch, change):
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    if change == "comment":
        module.write_text("# comment only\nVALUE = 1\n")
    elif change == "note":
        (module.parent.parent / "notes.txt").write_text("untracked notes")
    line = 2 if change == "comment" else 1
    data.add_arcs({str(module): [(-1, line), (line, -1)]})
    data.write()
    result["component"] = runner.source_identity()
    runner.report(run, result)
    assert result["diff"]["status"] == "no_changes"
    assert "Not applicable" in (run / "diff-cover.html").read_text()
    assert result["component"]["dirty"] == (change != "clean")
    assert "Comparison starts at the current commit" in (run / "summary.md").read_text()
    assert "fetch upstream main" in (run / "diff-cover.html").read_text()
    summary = (run / "summary.md").read_text()
    assert "N/A" not in summary
    # A completed measurement with nothing to count reads "0 of 0", not 0%.
    assert "| Executable lines | 0 | 0 | 0 | 0/0 |" in summary
    assert "| Branches | 0 | 0 | 0 | 0/0 |" in summary
    assert "Not generated" not in summary
    assert not (run / "diff-cover.md").exists()


def test_synced_fork_with_uncommitted_changes_explains_comparison_scope(tmp_path, monkeypatch):
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    module.write_text("VALUE = 2\n")
    runner.git("add", str(module))
    commit("already in fork main")
    runner.git("update-ref", "refs/remotes/origin/main", "HEAD")
    result["diff_base"] = "origin/main"
    module.write_text("VALUE = 2\nOTHER = 3\n")
    data.add_arcs({str(module): [(-1, 1), (1, -1)]})
    data.write()
    result["component"] = runner.source_identity()
    runner.report(run, result)
    diff = result["diff"]
    assert diff["status"] == "measured"
    assert diff["changed_lines"] == 1
    assert diff["commit_count"] == 0
    assert result["component"]["dirty"]
    assert diff["hint"] in (run / "summary.md").read_text()


def test_clean_comparison_with_commits_does_not_warn_baseline_contains_head(tmp_path, monkeypatch):
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    commit("metadata change", allow_empty=True)
    data.add_arcs({str(module): [(-1, 1), (1, -1)]})
    data.write()
    result["component"] = runner.source_identity()
    runner.report(run, result)
    assert result["diff"]["status"] == "no_changes"
    assert result["diff"]["commit_count"] == 1
    assert not result["component"]["dirty"]
    assert "hint" not in result["diff"]


def test_diff_report_missing_baseline_keeps_full_report(tmp_path, monkeypatch):
    module, run, data, result = diff_fixture(tmp_path, monkeypatch)
    data.add_arcs({str(module): [(-1, 1), (1, -1)]})
    data.write()
    result["diff_base"] = "missing-baseline"
    runner.report(run, result)
    assert result["diff"]["status"] == "unavailable"
    assert "changed_lines" not in result["diff"]
    assert (run / "htmlcov/index.html").is_file()
    assert "Not generated" in (run / "diff-cover.html").read_text()
    # An unavailable baseline must not look like a completed zero-item measurement.
    summary = (run / "summary.md").read_text()
    assert "| Executable lines | - | - | - | Not generated |" in summary
    assert "| Branches | - | - | - | Not generated |" in summary
    assert "0/0" not in summary.split("## Changed-line coverage", 1)[1]
    with pytest.raises(subprocess.CalledProcessError) as exc:
        runner.git("merge-base", "missing-baseline", "HEAD")
    assert exc.value.output.strip() in result["diff"]["reason"]
    assert exc.value.output.strip() in (run / "summary.md").read_text()
    assert exc.value.output.strip() in (run / "diff-cover.html").read_text()
    assert not (run / "diff-cover.md").exists()
