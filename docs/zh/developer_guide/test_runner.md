# 单测执行与覆盖率报告

`scripts/run_tests.py` 运行 `test/` 下的测例，生成整体覆盖率和变更行覆盖率报告。覆盖率只提供评审证据，不按比例自动阻断。

覆盖率统计 `turbo_physai/` 下的 Python 代码，按路径自动分类，新增文件无需配置：

| 路径 | 报告分类 |
|---|---|
| `turbo_physai/engine/` | Engine |
| `turbo_physai/operators/` | 算子 API（Python） |
| `turbo_physai/optimizations/common/` | 公共优化 |
| `turbo_physai/optimizations/models/` | 不计入 |
| 其他 | 其他组件代码 |

未被任何测例执行的文件也计入分母，显示零覆盖。测试代码、配置文件和 C++/HIP 内核不计入。不计入覆盖率的代码，其测例照常执行。

## 准备环境

在 Python 3.10+ 环境中安装测试依赖：

```bash
pip install -r requirements-dev.txt
```

调用原生算子的测例还需要已编译的组件扩展和可用 HCU。

## 运行

```bash
# 测试当前仓库源码
python scripts/run_tests.py -- -q

# 只运行 test/operators 目录中的测例
python scripts/run_tests.py --tests test/operators -- -q

# 测试已安装的 wheel
python scripts/run_tests.py --installed -- -q
```

- `--` 之后的参数原样传给 pytest。不接受 `-x`、`--maxfail` 和 `--collect-only`；查看测例清单使用 `python -m pytest test --collect-only -q`。
- `--tests` 接受目录、文件和 `文件::测试函数`。相对路径相对仓库根目录解析，与执行命令时的当前目录无关。
- 默认直接测试仓库源码，不要求先安装 wheel。
- `--installed` 测试已安装的包，不负责构建或安装。开始前逐一比对已安装包与仓库中的 Python 文件，不一致则不执行测例；不校验原生二进制内容。

运行结束后查看 `test-reports/summary.md`，逐行明细见 `test-reports/htmlcov/index.html`。

普通 `pytest` 和 `scripts/test_unit.sh` 仍可执行原有测例，但不生成这套报告。

## 报告内容

本地报告固定保存到仓库的 `test-reports/`。每次运行前清理该目录中上一次的报告，不要同时运行多个本工具进程。

| 文件 | 内容 |
|---|---|
| `summary.md` | 测试结果、分类覆盖率和变更行覆盖率汇总 |
| `htmlcov/index.html` | 逐文件、逐行及分支明细；分享时保留整个目录 |
| `diff-cover.html`、`diff-cover.md` | 变更行覆盖明细 |
| `diff-cover.log` | diff-cover 的输出；变更报告未生成时在此查看原因 |
| `manifest.json` | 代码版本、工作区修改状态、测试结果、分类覆盖率和实际导入的组件位置 |
| `common.log`、`common.xml` | 测试日志和 JUnit 结果 |
| `common.events.jsonl` | 随执行逐条保存的测例阶段结果 |
| `coverage.xml`、`coverage.json`、`.coverage` | 可继续处理的覆盖数据 |

`summary.md` 中的结果数量包含 JUnit 子测试，不等同于测试函数数量。失败测例执行过的代码也计入覆盖率，高覆盖率不代表结果正确。

提交评审时可粘贴 `summary.md`，并将 `htmlcov/` 作为附件或链接。

## 失败处理

| 情况 | 表现 | 退出码 |
|---|---|---|
| 全部通过且报告生成成功 | 状态为 `passed` | 0 |
| 测例断言失败 | 其他测例继续执行，照常生成报告，状态为 `failed` | 1 |
| 测试收集失败 | 本次测试中止，不自动重跑，状态为 `failed` | 1 |
| 测试进程异常退出（如原生代码崩溃） | 状态为 `execution_error`；已完成的测例见 `common.events.jsonl`；覆盖数据可能不完整，`summary.md` 会注明 | 1 |
| 已安装 wheel 的导入位置或 Python 文件检查未通过 | 不执行测例，原因见 `common.log` | 1 |
| 报告生成失败 | 原因记录在 `manifest.json` 的 `report_error` | 1 |
| 命令行参数不受支持 | 不执行测例，不生成报告 | 2 |

## 变更行覆盖率

将 Git 新增、修改行与本次测试的覆盖数据对照，只统计上表范围内的 Python 可执行行。删除行、注释和空行不参与统计。只测试当前代码，基准仅用于确定变更行，不运行旧版本测试。

比较基准依次取 `--diff-base`、环境变量 `TURBOPHYSAI_DIFF_BASE`、`origin/main`。工具从基准与 `HEAD` 的共同祖先开始比较，包含已暂存、未暂存和未跟踪的文件，因此测试期间不要修改待测源码。

```bash
git fetch origin main:refs/remotes/origin/main
python scripts/run_tests.py --diff-base origin/main -- -q
```

| 运行场景 | 基准来源 |
|---|---|
| 本地 | 默认 `origin/main`，可用 `--diff-base` 指定 |
| Fork 提交 PR | HCU workflow 跳过 Fork PR，需本地测试。Fork 仓库的 `origin` 通常指向个人仓库，应获取主仓库 main 后指定对应引用，例如 `--diff-base upstream/main` |
| 主仓库内提交 PR | CI 使用 PR 目标分支的基准提交 |
| 合入 main 后触发 push | CI 使用本次推送前的提交 |
| 手动触发或定时 CI | CI 使用 `origin/main` |

`summary.md` 记录基准引用、实际比较起点和待测提交。没有可统计的变更时覆盖率显示“不适用”；Git 或基准不可用时显示“未生成”及原因，整体覆盖率报告不受影响。

## CI

`.github/workflows/hcu-ci.yml` 在测试容器内执行 `.github/scripts/run_hcu_ci.sh`：构建并安装 wheel，然后在仓库外调用 `scripts/run_tests.py --installed`。组件源码只读挂载，构建在临时副本中进行。

报告保存在宿主机产物目录 `<运行编号>-<尝试次数>/coverage/` 下，结构与本地的 `test-reports/` 相同，`summary.md` 同时写入 workflow 的 Step Summary。CI 的 `manifest.json` 另记录测试镜像地址、ID 和 digest。产物默认保留 14 天。

