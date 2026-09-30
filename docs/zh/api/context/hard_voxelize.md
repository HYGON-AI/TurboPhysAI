# hard_voxelize

## 接口原型

```python
turbo_physai.operators.hard_voxelize(
    points, voxel_size, coors_range, max_points=35,
    max_voxels=20000, ndim=3, deterministic=True
) -> Tuple[Tensor, Tensor, Tensor]
```

接口位置：`turbo_physai/operators/voxelization.py`。封装负责分配 native 输出缓冲区，并按有效体素数切片。

## 功能描述

将点云划分到固定网格，并限制每个体素的点数和总输出体素数。封装自动分配 `voxels`、`coords` 和 `counts` 缓冲区，调用 native kernel 后仅返回有效部分。

## 参数说明

- `points`：`[N, F]` 点特征。
- `voxel_size` / `coors_range`：体素尺寸和坐标范围。
- `max_points`：每个体素最多保留的点数。
- `max_voxels`：最多输出的体素数。
- `ndim`：坐标维数，目标模型场景使用 3。
- `deterministic`：是否使用确定性实现。

## 返回值

返回 `(voxels, coords, counts)`：shape 分别为 `[num_voxels, max_points, F]`、`[num_voxels, ndim]` 和 `[num_voxels]`，第一维均已切到有效体素数。坐标和计数为 int32。

## 约束说明

- `max_points` 和 `max_voxels` 必须为可用于分配输出缓冲区的正整数。
- 超出坐标范围的点、超过体素容量的点以及超过 `max_voxels` 的体素会被忽略。
- `deterministic=False` 时，同一体素内的点顺序不保证稳定。
- CPU/HCU 的具体 dtype 支持以编译后的 native 类型分派为准。

## 调用示例

```python
import torch
from turbo_physai.operators import hard_voxelize

points = torch.rand(20000, 3, device="cuda", dtype=torch.float32) * 70.0
voxels, coords, counts = hard_voxelize(
    points,
    [0.05, 0.05, 0.1],
    [0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
    max_points=10,
    max_voxels=40000,
    ndim=3,
    deterministic=True,
)
torch.cuda.synchronize()
print(
    f"points shape: {points.shape}, points dtype: {points.dtype}, "
    f"voxels shape: {voxels.shape}, voxels dtype: {voxels.dtype}, "
    f"coords shape: {coords.shape}, coords dtype: {coords.dtype}, "
    f"counts shape: {counts.shape}, counts dtype: {counts.dtype}, "
    f"num_voxels: {coords.shape[0]}"
)
```

一次运行的输出示例（随机点云会使体素数量变化）：

```text
points shape: torch.Size([20000, 3]), points dtype: torch.float32, voxels shape: torch.Size([150, 10, 3]), voxels dtype: torch.float32, coords shape: torch.Size([150, 3]), coords dtype: torch.int32, counts shape: torch.Size([150]), counts dtype: torch.int32, num_voxels: 150
```

## 参考

- 兼容层：`turbo_physai/operators/voxelization.py`
- Kernel：`kernel/voxelization/`
- 返回[兼容层 API 清单](../README.md)
