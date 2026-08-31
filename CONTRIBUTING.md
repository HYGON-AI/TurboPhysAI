# 贡献指南

感谢参与 TurboPhysAI。欢迎通过 Issue 反馈问题和需求，或通过 Pull Request（PR）提交代码、优化实现、测试和文档改进。

## 提交前沟通

- 提交前请先搜索已有 Issue 和 PR，避免重复工作。
- Bug 修复应说明复现条件、实际结果和预期结果。
- 新功能、公共接口调整或较大范围重构，建议先创建 Issue，确认目标和影响范围后再开发。
- 安全问题请按[安全问题反馈](SECURITY.md)处理，不要在公开 Issue 中披露完整细节。

## 开发流程

1. Fork 并克隆仓库，从最新默认分支创建独立开发分支。
2. 按[安装指南](docs/zh/get_started/installation.md)准备开发环境。
3. 实现变更并补充与变更范围对应的测试和文档。
4. 完成本地检查后，将分支推送到个人仓库并创建 PR。

分支名称应简洁说明变更类型和内容，例如：

```text
feat/runtime-check
fix/config-loader
docs/quick-start
```

一个 PR 应聚焦一项完整变更，避免混入无关格式调整或重构。

## 变更要求

### 框架与接口

- Bug 修复应包含最小复现和回归测试。
- 新增或修改公共接口时，应同步更新类型、异常行为、测试和对应文档。
- 不应绕过目标检查、冲突分析、执行顺序或回滚机制来处理单一模型问题。

### 模型与算子优化

- 新优化可先在仓库外独立开发和验证，再迁入公共优化或模型专用优化目录。
- 提交应记录优化接入基线、适用条件、配置、验证命令和结果。
- 涉及计算逻辑的优化应完成数值验证；拟交付的性能优化应提供可复现的模型精度和端到端性能结果。
- 新增支持模型时，应增加对应的 `model_examples/<Model>/README.md` 和模型支持清单条目。

完整流程见[优化开发与接入流程](docs/zh/developer_guide/optimization_workflow.md)，验收要求见[优化验证与交付](docs/zh/developer_guide/validation.md)。

### 文档

- 文档内容应与当前代码、CLI 和配置格式一致。
- 修改 CLI、OptimizationConfig、RuntimeConfig 或执行报告时，应同步更新对应指南和参考文档。
- 文档不得包含个人绝对路径、内部地址、凭据、令牌或未脱敏日志。

## 本地检查

所有变更至少运行与修改范围相关的测试。框架和文档的基础检查为：

```bash
bash scripts/test_engine.sh
python scripts/check_docs.py
git diff --check
```

原生算子和模型优化还应在支持的 HCU 环境中运行对应单元测试和模型验证。若某项检查受硬件、数据集或权重限制，请在 PR 中说明未执行项、原因和已有验证结果。

## Pull Request 要求

PR 描述至少应包括：

- 变更目的和范围；
- 关联 Issue（如有）；
- 关键实现和兼容性影响；
- 已执行的测试命令及结果；
- 未完成的验证或已知限制；
- 涉及模型性能时的测试环境、配置和对比结果。

请及时处理评审意见，并在新增提交后更新验证结果。所有必需检查和评审完成后，由维护人员合并 PR。

## 许可证与第三方代码

提交至本仓库的自研代码应能够按仓库 [LICENSE](LICENSE) 发布。引入或修改第三方代码时，应保留原始版权和许可证声明，并提供上游仓库、固定 commit、许可证及修改说明；必要时同步更新 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)、[NOTICE](NOTICE) 和 `third_party/licenses/`。
