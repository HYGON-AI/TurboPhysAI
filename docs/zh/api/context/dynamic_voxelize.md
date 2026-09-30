# dynamic_voxelize

## 接口原型

```python
turbo_physai.operators.dynamic_voxelize(
    points, voxel_size, coors_range, ndim=3
) -> Tensor
```

接口位置：`turbo_physai/operators/voxelization.py`。与原地写入的 native 接口不同，该 helper 自动分配并返回坐标 Tensor。

## 功能描述

为每个点计算所属体素坐标，不执行体素内聚合。封装根据点数和 `ndim` 自动分配 int32 输出缓冲区，再调用 native dynamic voxelization。

## 参数说明

- `points`：shape `[N, F]`。
- `voxel_size`：长度为 `ndim` 的体素尺寸。
- `coors_range`：长度为 `2 * ndim` 的坐标范围。

## 返回值

返回 int32 坐标 Tensor，shape `[N, ndim]`。越界点至少将坐标首列置为 `-1`。

## 约束说明

- `points` 必须是二维 Tensor，特征列数至少覆盖 `ndim` 个坐标分量。
- `voxel_size` 和 `coors_range` 的长度必须与 `ndim` 匹配。
- GPU 路径要求 CUDA/HIP Tensor；具体浮点 dtype 支持取决于编译后的 native 类型分派。

## 调用示例

```python
import torch
from turbo_physai.operators import dynamic_voxelize

points = torch.rand(20000, 3, device="cuda", dtype=torch.float32) * 70.0
coords = dynamic_voxelize(
    points,
    [0.05, 0.05, 0.1],
    [0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
    ndim=3,
)
torch.cuda.synchronize()
valid = coords[:, 0] >= 0
print(
    f"points shape: {points.shape}, points dtype: {points.dtype}, "
    f"coords shape: {coords.shape}, coords dtype: {coords.dtype}, "
    f"valid points: {valid.sum().item()}, "
    f"invalid points: {(~valid).sum().item()}"
)
```

一次运行的输出示例（随机点云会使有效点数量变化）：

```text
points shape: torch.Size([20000, 3]), points dtype: torch.float32, coords shape: torch.Size([20000, 3]), coords dtype: torch.int32, valid points: 157, invalid points: 19843

```

## 参考

- 兼容层：`turbo_physai/operators/voxelization.py`
- Kernel：`kernel/voxelization/`
- 返回[兼容层 API 清单](../README.md)
