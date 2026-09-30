# 算子兼容层 API 清单

本目录记录 `turbo_physai/operators/` 中基于自定义 kernel 的 Python 兼容接口，用于验证参数整理、输出转换、dtype 分派和自动求导。构建与自定义算子接入说明见 [`../developer_guide/custom_operator.md`](../developer_guide/custom_operator.md)。

## 调用方式

以下示例面向已编译 GPU 扩展的 HCU/ROCm 环境；PyTorch ROCm 环境通常仍使用 `device="cuda"`。兼容接口按算子模块导入：

```python
from turbo_physai.operators import grid_sample

out = grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
out.sum().backward()
```

兼容层内部最终调用 `turbo_physai._C`，但会按接口需要完成缓冲区分配、参数转换、输出布局调整或 Autograd 注册。

## API 清单

| 分类 | 兼容接口 | 实现位置 | 额外行为 |
|---|---|---|---|
| 特征融合 | [bev_pool](./context/bev_pool.md) | `operators/_bev_pool.py` | 排序、区间生成、Autograd、输出布局转换 |
| 采样 | [deformable_aggregation_function](./context/deformable_aggregation.md) | `operators/deformable_aggregation.py` | dtype/连续性整理与 Autograd |
| 采样 | [grid_sample](./context/grid_sample.md) | `operators/_grid_sample.py` | 字符串参数转换与 Autograd |
| 采样 | [interpolate](./context/upsample_bilinear_2d.md) | `operators/upsample_bilinear_2d.py` | 尺寸处理与 Autograd |
| 体素化 | [hard_voxelize](./context/hard_voxelize.md) | `operators/voxelization.py` | 输出缓冲区分配和有效区间切片 |
| 体素化 | [dynamic_voxelize](./context/dynamic_voxelize.md) | `operators/voxelization.py` | 输出坐标分配 |
| 体素化 | [dynamic_scatter](./context/dynamic_point_to_voxel.md) | `operators/voxelization.py` | 点到体素归约与 Autograd |
| 稀疏卷积 | [get_indice_pairs](./context/get_indice_pairs.md) | `operators/sparse_conv.py` | 维度广播和输出形状推导 |
| 稀疏卷积 | [indice_conv](./context/indice_conv.md) | `operators/sparse_conv.py` | dtype 分派、bias 和 Autograd |
| 稀疏卷积 | [indice_maxpool](./context/indice_maxpool.md) | `operators/sparse_conv.py` | dtype 分派与 Autograd |

## 测试原则

- 调用兼容接口，而不是直接调用 `turbo_physai._C`。
- 对 Autograd 接口执行 `backward()` 并检查输入梯度。
- 对负责分配输出的 helper 检查返回 shape、dtype 和有效数据范围。
- 能力范围以对应 wrapper 的参数检查和底层 kernel 支持范围为准。
