# 贡献指南

感谢参与 TurboPhysAI。模型优化与框架修改采用不同的开发路径。

## 提交模型或算子优化

1. 使用 `turbo-physai optimization init <name>` 在 TurboPhysAI 仓库外创建开发工程。
2. 记录官方模型仓库、基线 commit、优化目标和真实启动命令。
3. 完成优化实现、Group 声明和最终 OptimizationConfig 生成。
4. 提供正向结果、必要梯度、真实训练、性能和报告证据。
5. 通过评审后再迁入 TurboPhysAI 长期维护目录。

详细流程见[优化开发与接入流程](docs/zh/developer_guide/optimization_workflow.md)。

## 修改框架

只有真实模型适配能够证明是通用框架缺口时才修改框架。提交应先增加最小复现测试，再修复通用能力，不在模型 Replacement 中绕过检查、冲突或恢复规则。

## 提交前检查

```bash
python -m unittest discover -s test/engine
python scripts/check_docs.py
git diff --check
```

涉及 HCU 算子或模型优化时，还需要附上对应设备环境中的数值、梯度、训练和性能验证结果。

## 文档要求

- 新增模型必须增加 `model_examples/<Model>/README.md`。
- 修改 CLI、OptimizationConfig 或报告字段时必须同步更新对应参考文档。
- 文档中不得包含个人绝对路径、内部地址、令牌或未脱敏日志。
