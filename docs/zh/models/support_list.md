# 模型支持清单

本页列出 TurboPhysAI 已支持的模型、基线和使用说明。

优化接入基线是模型优化开发、正确性验证和性能验证所使用的模型 commit。推荐使用表中基线，
以保持模型代码与随包优化配置的验证条件一致。

TurboPhysAI 不强制模型仓库停留在优化接入基线（[可以在优化接入基线之外的模型版本上使用吗？](../faq.md#可以在优化接入基线之外的模型版本上使用吗)）。

## 已支持模型

| 模型 | Method | Backbone | 优化接入基线 | 使用说明 |
| :---: | :---: | :---: | :---: | :---: |
| BEVFormer | BEVFormer-base | R101-DCN | `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` | [BEVFormer](../../../model_examples/BEVFormer/README.md) |
| BEVFusion | BEVFusion | — | `326653dc06e0938edf1aae7d01efcd158ba83de5` | [BEVFusion](../../../model_examples/BEVFusion/README.md) |
