# dynamic_point_to_voxel

## 接口原型

```python
turbo_physai.operators.dynamic_scatter(
    feats, coors, reduce_type="max"
) -> Tuple[Tensor, Tensor]
```

接口位置：`turbo_physai/operators/voxelization.py`。`dynamic_scatter` 使用内部 `_DynamicScatterFunction` 保存 native 前向产生的映射和计数，并自动调用反向 kernel。

## 功能描述

按体素坐标将点特征执行 max、sum 或 mean 归约，返回去重后的体素特征和坐标。封装保存点到体素的映射与计数，并在反向时自动把体素梯度传播回点特征。

## 参数说明

- `feats`：float32 点特征 `[N, C]`。
- `coors`：int32 坐标 `[N, 3]`；含负值的行作为无效点。
- `reduce_type`：`"max"`、`"sum"` 或 `"mean"`。

## 返回值

返回 `(reduced_feats, out_coors)`：前者 shape 为 `[num_voxels, C]`，后者为 int32 `[num_voxels, 3]`。`out_coors` 不参与求导，梯度仅回传至 `feats`。

## 约束说明

- `reduce_type` 只能为 `"max"`、`"sum"` 或 `"mean"`，否则抛出 `ValueError`。
- 当前 point-to-voxel kernel 仅支持 HCU（ROCm/HIP）设备和 float32 特征。
- 封装会将 `feats` 连续化，并将 `coors` 转为连续 int32。
- max 归约出现并列最大值时，梯度回传至输入索引最小的对应点。

## 调用示例

```python
import torch
from turbo_physai.operators import dynamic_scatter

feats = torch.randn(20000, 32, device="cuda", dtype=torch.float32, requires_grad=True)
coors = torch.randint(0, 100, (20000, 3), device="cuda", dtype=torch.int32)
reduced, out_coors = dynamic_scatter(feats, coors, reduce_type="max")
reduced.sum().backward()
torch.cuda.synchronize()
print(
    f"feats shape: {feats.shape}, feats dtype: {feats.dtype}, "
    f"coors shape: {coors.shape}, coors dtype: {coors.dtype}, "
    f"reduced shape: {reduced.shape}, reduced dtype: {reduced.dtype}, "
    f"out_coors shape: {out_coors.shape}, out_coors dtype: {out_coors.dtype}, "
    f"feats grad shape: {feats.grad.shape}, feats grad dtype: {feats.grad.dtype}"
)
```

一次运行的输出示例（随机坐标会使体素数量变化）：

```text
feats shape: torch.Size([20000, 32]), feats dtype: torch.float32, coors shape: torch.Size([20000, 3]), coors dtype: torch.int32, reduced shape: torch.Size([19823, 32]), reduced dtype: torch.float32, out_coors shape: torch.Size([19823, 3]), out_coors dtype: torch.int32, feats grad shape: torch.Size([20000, 32]), feats grad dtype: torch.float32
```



## 参考

- 兼容层：`turbo_physai/operators/voxelization.py`
- Kernel：`kernel/voxelization/src/scatter_points_cuda.cu`
- 返回[兼容层 API 清单](../README.md)
