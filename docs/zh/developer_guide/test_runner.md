# 单测执行与覆盖率报告

`scripts/run_tests.py` 运行 `test/` 下的测例，生成整体覆盖率和变更行覆盖率报告。覆盖率只提供评审证据，不按比例自动阻断。

覆盖率统计 `turbo_physai/` 下的 Python 代码，按路径自动分类，新增文件无需配置：

| 路径 | 报告分类 |
|---|---|
| `turbo_physai/engine/` | Engine |
| `turbo_physai/operators/` | Operator API (Python) |
| `turbo_physai/optimizations/common/` | Common optimizations |
| `turbo_physai/optimizations/models/` | 不计入 |
| 其他 | Other public components |

未被任何测例执行的文件也计入分母，显示零覆盖。测试代码、配置文件和 C++/HIP 内核不计入。不计入覆盖率的代码，其测例照常执行。

## 准备环境

在 Python 3.10+ 环境中安装测试依赖：

```bash
pip install -r requirements-test.txt
```

调用原生算子的测例还需要已编译的组件扩展和可用 HCU。

Fork 开发者在仓库根目录配置主仓库远程（只需一次，已有 `upstream` 时不重复添加）：

```bash
git remote add upstream https://github.com/HYGON-AI/TurboPhysAI.git
```

每次运行前更新主仓库引用，包括在 GitHub 上同步 Fork 后：

```bash
git fetch upstream main
```

工具不会自动 fetch，运行前需自行更新。直接克隆主仓库时，运行 `git fetch origin main`。

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
| `diff-cover.html` | 变更行覆盖明细；汇总统计见 `summary.md` |
| `diff-cover.log` | diff-cover 的输出；变更报告未生成时在此查看原因 |
| `manifest.json` | 代码版本、工作区修改状态、测试结果、分类覆盖率和实际导入的组件位置 |
| `common.log`、`common.xml` | 测试日志和 JUnit 结果 |
| `common.events.jsonl` | 随执行逐条保存的测例阶段结果 |
| `coverage.xml`、`coverage.json`、`.coverage` | 可继续处理的覆盖数据 |

`summary.md` 中的结果数量包含 JUnit 子测试，不等同于测试函数数量。失败测例执行过的代码也计入覆盖率，高覆盖率不代表结果正确。

### 提交评审

Fork 提交的 PR 不触发 HCU CI，评审依据贡献者提供的报告。用于评审的报告按以下步骤生成：

1. 提交全部改动，确认 `git status` 没有未提交的修改；
2. 运行 `git fetch upstream main` 更新基准；
3. 运行工具，将 `summary.md` 粘贴到 PR，`htmlcov/` 作为附件或链接。

评审时核对 `summary.md`：待测提交等于 PR 的最新提交，工作区未提交修改为 `No`，提交列表与 PR 的提交一致。开发过程中可以随时在有未提交修改时运行，但这类报告不作为评审依据。

## 失败处理

| 情况 | 表现 | 退出码 |
|---|---|---|
| 全部通过且报告生成成功 | 状态为 `passed` | 0 |
| 测例断言失败 | 其他测例继续执行，照常生成报告，状态为 `failed` | 1 |
| 测试收集失败 | 本次测试中止，不自动重跑，状态为 `failed` | 1 |
| 测试进程异常退出（如原生代码崩溃） | 状态为 `execution_error`；已完成的测例见 `common.events.jsonl`；覆盖数据可能不完整，`summary.md` 会注明 | 1 |
| 已安装 wheel 的导入位置或 Python 文件检查未通过 | 不执行测例，原因见 `common.log` | 1 |
| 报告生成失败 | 原因记录在 `manifest.json` 的 `report_error` | 1 |
| 变更覆盖率未生成（基准不可用或 diff-cover 失败） | 整体覆盖率报告照常生成；变更部分显示 `Not generated` 及原因 | 不受影响，由测试结果决定 |
| 命令行参数不受支持 | 不执行测例，不生成报告 | 2 |

## 变更行覆盖率

将 Git 新增、修改行与本次测试的覆盖数据对照，只统计上表范围内的 Python 可执行行。删除行、注释和空行不参与统计。只测试当前代码，基准仅用于确定变更行，不运行旧版本测试。

变更覆盖率分别统计行和分支。分支指从变更行出发的跳转，例如新增或修改的 `if` 行有“条件成立”和“条件不成立”两个分支；只走了其中一个时，该行显示已覆盖，分支只覆盖一半。

- `summary.md` 列出从变更行出发、未执行的分支，最多 20 条；`exit` 表示跳出函数。
- 只修改了分支内部的语句、没有修改条件行时，该条件的分支不计入变更统计。`diff-cover.html` 只有按行的明细。

比较基准依次取 `--diff-base`、环境变量 `TURBOPHYSAI_DIFF_BASE`、本地已有的 `upstream/main`，没有 `upstream/main` 时退回 `origin/main`。工具从基准与 `HEAD` 的共同祖先开始比较，包含已暂存、未暂存和未跟踪的文件，因此测试期间不要修改待测源码。

| 运行场景 | 基准来源 |
|---|---|
| Fork 提交 PR | HCU workflow 跳过 Fork PR，需本地测试；先配置并 fetch `upstream`，默认与主仓库 main 的共同祖先比较 |
| 主仓库内提交 PR | CI 使用 PR 目标分支的基准提交 |
| 合入 main 后触发 push | CI 使用本次推送前的提交 |
| 手动触发或定时 CI | CI 使用 `origin/main` |

`summary.md` 记录基准引用、实际比较起点、待测提交、比较起点之后的提交数量，以及工作区是否有修改（整个仓库是否有未提交的改动，与 `manifest.json` 中的工作区修改状态一致）。提交标题最多列最近 10 条，超出时注明截断；评审时可与 PR 的提交列表核对。基准引用过期时可能包含额外提交，提交列表不能代替运行前的 fetch，也不能独立证明报告可信。

主仓库内的 PR CI 默认检出 GitHub 生成的合并提交，报告中的提交列表会包含该合并提交，可能比 PR 页面多一条。

提交数量为 0 时，报告说明比较起点就是当前提交、只比较工作区修改，并提醒 Fork 开发者检查、fetch `upstream`。

| 状态 | 覆盖率显示 |
|---|---|
| 已统计 | 实际数量和比例 |
| 已统计，但没有可统计的变更 | `0/0`，表示共 0 项、覆盖 0 项 |
| 未生成（Git 或基准不可用、diff-cover 失败） | `Not generated` 及具体原因；Git 命令失败时附原始报错 |

## CI

`.github/workflows/hcu-ci.yml` 在测试容器内执行 `.github/scripts/run_hcu_ci.sh`：检查镜像未预装组件，构建并安装 wheel，检查组件及原生扩展的导入来源、框架版本和 HCU 可用性，再调用 `scripts/run_tests.py --installed`。构建在临时副本中进行，导入检查和测试在仓库外执行。

报告保存在宿主机产物目录 `<运行编号>-<尝试次数>/coverage/` 下，结构与本地的 `test-reports/` 相同，`summary.md` 同时写入 workflow 的 Step Summary。镜像信息保存在产物根目录的 `image-metadata.txt` 中。产物默认保留 14 天。
