#!/usr/bin/env python3
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Run pytest and report public Python coverage.

Examples:
  python scripts/run_tests.py -- -q
  python scripts/run_tests.py --tests test/operators -- -q
  python scripts/run_tests.py --installed --diff-base origin/main -- -q
"""
from __future__ import annotations

import argparse
import configparser
from contextlib import contextmanager
import functools
import html
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import types
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MODEL_OMIT = "*/optimizations/models/*"


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT}", *args],
                                   cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


@contextmanager
def trace_autograd_threads():
    """Attach coverage to native-created autograd threads without changing gradients."""
    if importlib.util.find_spec("torch") is None:
        yield
        return
    import torch
    originals = {name: getattr(torch.autograd, name) for name in ("backward", "grad")}

    def trace_thread(gradient):
        if sys.gettrace() is None:
            # Native-created threads do not inherit threading's trace hook.
            sys.settrace(threading.gettrace())
        return None

    def instrument(original, keyword):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            roots = args[0] if args else kwargs.get(keyword)
            roots = (roots,) if isinstance(roots, torch.Tensor) else roots
            handles = []
            try:
                if isinstance(roots, (tuple, list)):
                    for tensor in roots:
                        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                            handles.append(tensor.register_hook(trace_thread))
                return original(*args, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()
        return wrapped

    try:
        for name, original in originals.items():
            setattr(torch.autograd, name, instrument(original, "tensors" if name == "backward" else "outputs"))
        yield
    finally:
        for name, original in originals.items():
            setattr(torch.autograd, name, original)


def coverage_config(run, package_root=None):
    """Use coverage's own path mapping for checkout and installed-wheel records."""
    roots = list(dict.fromkeys([str(ROOT / "turbo_physai"), str(package_root or ROOT / "turbo_physai")]))
    aliases = roots + [str(Path(p).relative_to(Path.cwd())) for p in roots if Path(p).is_relative_to(Path.cwd())]
    settings = configparser.ConfigParser(interpolation=None)
    settings["run"] = dict(branch="True", parallel="True", core="ctrace", patch="subprocess",
                           relative_files="True", source="\n".join(roots), omit=MODEL_OMIT,
                           data_file=str(run / ".coverage"))
    settings["paths"] = dict(component="\n".join(dict.fromkeys(["turbo_physai", *aliases])))
    settings["report"] = dict(include="turbo_physai/*", omit=MODEL_OMIT)
    for section in settings.values():
        for key in section:
            section[key] = section[key].replace("$", "$$")
    path = run / ".coveragerc"
    with path.open("w", encoding="utf-8") as stream:
        settings.write(stream)
    return path


def worker(manifest_path):
    import coverage
    import pytest
    path = Path(manifest_path)
    run, result = path.parent, json.loads(path.read_text())
    package_root = ROOT / "turbo_physai"
    if result["installed"]:
        sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != ROOT]
        spec = importlib.util.find_spec("turbo_physai")
        if spec is None or not spec.origin:
            raise ValueError("the tested wheel is not installed")
        package_root = Path(spec.origin).resolve().parent
        if package_root.is_relative_to(ROOT):
            raise ValueError("installed-wheel test imported the source checkout")
        for source in package_root.rglob("*.py"):
            reference = ROOT / "turbo_physai" / source.relative_to(package_root)
            if not reference.is_file() or source.read_bytes() != reference.read_bytes():
                raise ValueError(f"wheel Python source differs from tested checkout: {source}")
        support = types.ModuleType("test")
        support.__path__ = [str(ROOT / "test")]
        sys.modules["test"] = support
    else:
        sys.path.insert(0, str(ROOT))
    result.update(component_mode="installed" if result["installed"] else "source",
                  component_package=str(package_root))
    path.write_text(json.dumps(result), encoding="utf-8")

    class Events:
        def pytest_runtest_logreport(self, report):
            event = dict(test=report.nodeid, phase=report.when,
                         outcome=report.outcome, duration=report.duration)
            if report.failed:
                event["error"] = str(report.longrepr)
            with (run / "common.events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    cov = coverage.Coverage(config_file=str(coverage_config(run, package_root)))
    cov.start()
    try:
        with trace_autograd_threads():
            return int(pytest.main([*result["tests"], "-o", "addopts=", "-p", "no:cacheprovider",
                                   "--import-mode=importlib", "--junitxml=" + str(run / "common.xml"),
                                   *result["pytest_args"]], plugins=[Events()]))
    finally:
        cov.stop()
        cov.save()
        result["coverage_saved"] = True
        path.write_text(json.dumps(result), encoding="utf-8")


def coverage_diff(run, base_ref):
    result = dict(base_ref=base_ref, status="unavailable")
    raw_path = run / "diff-cover.json"
    try:
        result["compare_commit"] = git("merge-base", base_ref, "HEAD")
        command = [sys.executable, "-c", "from diff_cover.diff_cover_tool import main; raise SystemExit(main())",
                   str(run / "coverage.xml"), "--compare-branch", result["compare_commit"],
                   "--include-untracked", "--fail-under=0", "--total-percent-float", "--show-covered",
                   "--include", "turbo_physai/*", "--exclude", "turbo_physai/optimizations/models/*",
                   "--format", f"json:{raw_path},html:{run / 'diff-cover.html'},markdown:{run / 'diff-cover.md'}"]
        env = dict(os.environ)
        index = int(env.get("GIT_CONFIG_COUNT", "0"))
        env.update(GIT_CONFIG_COUNT=str(index + 1))
        env[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
        env[f"GIT_CONFIG_VALUE_{index}"] = str(ROOT)
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        (run / "diff-cover.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise ValueError("diff-cover failed; see diff-cover.log")
        raw = json.loads(raw_path.read_text())
        total, missing = raw["total_num_lines"], raw["total_num_violations"]
        result.update(status="measured" if total else "no_changes", changed_lines=total,
                      covered_lines=total - missing, percent=100 * (total - missing) / total if total else None)
        raw_path.unlink()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        result["reason"] = str(exc)
    if result["status"] != "measured":
        message = ("不适用：没有新增或修改的可统计 Python 可执行行。" if result["status"] == "no_changes"
                   else "未生成：" + result.get("reason", "基准不可用。"))
        (run / "diff-cover.md").write_text("# 本次变更覆盖率\n\n" + message + "\n", encoding="utf-8")
        (run / "diff-cover.html").write_text('<!doctype html><meta charset="utf-8"><h1>本次变更覆盖率</h1><p>'
                                            + html.escape(message) + '</p><a href="summary.md">查看汇总</a>', encoding="utf-8")
    return result


def category(path):
    parts = Path(path).parts[1:]
    if parts[0] == "engine":
        return "Engine"
    if parts[0] == "operators":
        return "算子 API（Python）"
    if parts[:2] == ("optimizations", "common"):
        return "公共优化"
    return "其他组件代码"


def summary_reports(run, result):
    """Render the Markdown summary."""
    md = []

    def text(value, level=0):
        md.extend(["#" * level + " " + value if level else value, ""])

    def table(headers, rows):
        md.extend(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
                   *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""])

    def links(items):
        md.extend([" · ".join(f"[{label}]({url})" for label, url in items), ""])

    def ratio(hit, total):
        return f"{hit}/{total} ({hit / total:.2%})" if total else "不适用"

    text("测试与覆盖率报告", 1)
    text("测试结果", 2)
    counts = result.get("counts") or {}
    table(["状态", "结果项（含子测试）", "失败", "错误", "跳过"],
          [[result["status"], *(counts.get(k, "—") for k in ("tests", "failures", "errors", "skipped"))]])
    if not result.get("coverage_saved"):
        text("覆盖数据不完整：测试进程未正常保存完整覆盖数据；以下比例仅依据已保存数据。")
    text("整体覆盖率", 2)
    metrics = ["covered_lines", "statements", "covered_branches", "branches"]
    groups = dict(result["groups"])
    groups["整体"] = {k: sum(g[k] for g in groups.values()) for k in metrics}
    table(["代码范围", "行覆盖", "分支覆盖"],
          [[name, ratio(g["covered_lines"], g["statements"]), ratio(g["covered_branches"], g["branches"])]
           for name, g in groups.items()])
    links([("逐文件覆盖率及未覆盖代码", "htmlcov/index.html")])
    text("本次变更覆盖率", 2)
    diff = result["diff"]
    table(["比较信息", "值"], [["基准", diff["base_ref"]],
          ["比较起点", diff.get("compare_commit", "—")],
          ["待测提交", result.get("component", {}).get("commit", "—")],
          ["比较范围", "提交及工作区修改"]])
    if diff["status"] in ("measured", "no_changes"):
        total, covered = diff["changed_lines"], diff["covered_lines"]
        row = [total, covered, total - covered, f"{covered / total:.2%}" if total else "不适用"]
    else:
        row = ["—", "—", "—", "未生成"]
    table(["变更可执行行", "已覆盖", "未覆盖", "覆盖率"], [row])
    if diff["status"] == "unavailable":
        text("未生成原因：" + diff.get("reason", "基准不可用。"))
    links([("变更行明细", "diff-cover.html")])
    (run / "summary.md").write_text("\n".join(md), encoding="utf-8")


def report(run, result):
    import coverage
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        config = run / ".coveragerc"
        if not config.exists():
            config = coverage_config(run)
        cov = coverage.Coverage(config_file=str(config))
        # Report against the checkout root so XML retains turbo_physai/ prefixes.
        cov.set_option("run:source", ["."])
        cov.load()
        cov.combine(strict=False, keep=True)
        if not cov.get_data().measured_files():
            cov.get_data().add_arcs({"turbo_physai/__init__.py": []})
        cov.save()
        files = [str(p.relative_to(ROOT)) for p in (ROOT / "turbo_physai").rglob("*.py")
                 if p.relative_to(ROOT / "turbo_physai").parts[:2] != ("optimizations", "models")]
        cov.json_report(morfs=files, outfile=str(run / "coverage.json"))
        cov.xml_report(morfs=files, outfile=str(run / "coverage.xml"))
        cov.html_report(morfs=files, directory=str(run / "htmlcov"))
        raw = json.loads((run / "coverage.json").read_text())
        groups = {}
        for path, record in raw["files"].items():
            group = groups.setdefault(category(path), dict(statements=0, covered_lines=0, branches=0, covered_branches=0))
            for dst, src in (("statements", "num_statements"), ("covered_lines", "covered_lines"),
                             ("branches", "num_branches"), ("covered_branches", "covered_branches")):
                group[dst] += record["summary"].get(src, 0)
        result["groups"] = dict(sorted(groups.items()))
        result["diff"] = coverage_diff(run, result["diff_base"])
        summary_reports(run, result)
    finally:
        os.chdir(previous)


def prepare_local_report():
    run = ROOT / "test-reports"
    if run.is_symlink():
        raise ValueError("test-reports must not be a symbolic link")
    run.mkdir(parents=True, exist_ok=True)
    generated = [run / name for name in (
        "summary.md", "index.html", "manifest.json", "coverage.xml", "coverage.json",
        "diff-cover.html", "diff-cover.md", "diff-cover.json", "diff-cover.log",
        "common.log", "common.xml", "common.events.jsonl", "htmlcov")]
    for path in [*generated, *run.glob(".coverage*")]:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tests", nargs="+", default=[str(ROOT / "test")], help="pytest directories, files or node IDs")
    parser.add_argument("--installed", action="store_true", help="test an already installed wheel")
    parser.add_argument("--diff-base", default=os.environ.get("TURBOPHYSAI_DIFF_BASE") or "origin/main")
    parser.add_argument("--_run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args._worker:
        return worker(args._worker)
    pytest_args = args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    if any(v in ("-x", "--exitfirst", "--collect-only", "--co") or v.startswith("--maxfail") for v in pytest_args):
        raise ValueError("fail-fast/collection-only options are not supported; use pytest --collect-only separately")
    tests = [str((ROOT / p.split("::")[0]).resolve()) + ("::" + p.split("::", 1)[1] if "::" in p else "") for p in args.tests]
    if args._run_dir:
        run = args._run_dir.resolve()
        run.mkdir(parents=True)
    else:
        run = prepare_local_report()
    try:
        component = dict(commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")))
    except (OSError, subprocess.CalledProcessError):
        component = dict(commit=os.environ.get("GITHUB_SHA"), dirty=None)
    result = dict(component=component, python=sys.version, tests=tests, installed=args.installed,
                  pytest_args=pytest_args, diff_base=args.diff_base, coverage_saved=False)
    if os.environ.get("TURBOPHYSAI_IMAGE_METADATA"):
        image = json.loads(Path(os.environ["TURBOPHYSAI_IMAGE_METADATA"]).read_text())[0]
        result["image"] = dict(address=os.environ.get("TURBOPHYSAI_CI_IMAGE"), id=image["Id"], digests=image.get("RepoDigests", []))
    manifest = run / "manifest.json"
    manifest.write_text(json.dumps(result), encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="" if args.installed else str(ROOT))
    env.pop("PYTEST_ADDOPTS", None)
    try:
        with (run / "common.log").open("w") as log, tempfile.TemporaryDirectory(prefix="turbophysai-pytest-") as cwd:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", str(manifest)],
                                       cwd=cwd if args.installed else ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads(manifest.read_text())
        result.update(exit_code=completed.returncode, status="passed" if completed.returncode == 0 else "failed", counts=None)
        xml = run / "common.xml"
        if xml.exists():
            suites = list(ET.parse(xml).getroot().iter("testsuite"))
            result["counts"] = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
        else:
            result.update(status="execution_error", error="no complete JUnit result; see common.log and common.events.jsonl")
    except (OSError, ValueError, ET.ParseError) as exc:
        result.update(status="execution_error", error=str(exc))
    try:
        report(run, result)
    except Exception as exc:
        result["report_error"] = str(exc)
        print(f"Report failed: {exc}", file=sys.stderr)
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tests: {result['status']}; counts: {result.get('counts')}; reports: {run}")
    return int(result["status"] != "passed" or "report_error" in result)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, SyntaxError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2)
