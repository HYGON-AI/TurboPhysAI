# deformable_aggregation

## 接口原型

```python
turbo_physai.operators.deformable_aggregation_function(
    feature_maps, spatial_shape, scale_start_index, sampling_location, weights
) -> Tensor
```

接口位置：`turbo_physai/operators/deformable_aggregation.py`。封装会将特征、采样位置和权重整理为连续 float32，将形状索引整理为连续 int32，并注册反向。

## 功能描述

在多相机、多尺度特征上按 `sampling_location` 做双线性采样，再使用 `weights` 加权聚合。封装负责输入类型和连续性整理，并通过 `DeformableAggregationFunction` 调用 native forward/backward。

## 参数说明

- `feature_maps`：`[B, num_feat, C]`。
- `spatial_shape`：`[cam, scale, 2]`。
- `scale_start_index`：`[cam, scale]`。
- `sampling_location`：`[B, anchor, pts, cam, 2]`，使用约 `[0, 1]` 的坐标。
- `weights`：`[B, anchor, pts, cam, scale, num_groups]`。

## 返回值

返回 float32 Tensor，shape `[B, anchor, C]`。梯度回传至 `feature_maps`、`sampling_location` 和 `weights`；形状及起始索引不参与求导。

## 约束说明

- 封装会把浮点输入转换为连续 float32，把 `spatial_shape` 和 `scale_start_index` 转换为连续 int32。
- `C` 必须能被 `num_groups` 整除，各 shape 必须满足底层 kernel 的布局约定。
- 当前实现面向 CUDA/HIP 设备，且 backward 被 `once_differentiable` 标记，不支持二阶梯度。

## 调用示例

```python
import torch
from turbo_physai.operators import deformable_aggregation_function

B, C, groups = 1, 32, 8
anchor, pts, cam, scale, H, W = 10, 10, 1, 1, 32, 88
features = torch.rand(B, cam * scale * H * W, C, device="cuda", requires_grad=True)
shapes = torch.tensor([[[H, W]]], dtype=torch.int32, device="cuda")
starts = torch.zeros((cam, scale), dtype=torch.int32, device="cuda")
locations = torch.rand(B, anchor, pts, cam, 2, device="cuda", requires_grad=True)
weights = torch.rand(B, anchor, pts, cam, scale, groups, device="cuda", requires_grad=True)
out = deformable_aggregation_function(features, shapes, starts, locations, weights)
out.sum().backward()
torch.cuda.synchronize()
print(
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"features grad shape: {features.grad.shape}, features grad dtype: {features.grad.dtype}, "
    f"locations grad shape: {locations.grad.shape}, locations grad dtype: {locations.grad.dtype}, "
    f"weights grad shape: {weights.grad.shape}, weights grad dtype: {weights.grad.dtype}"
)
```

一次运行的输出示例：

```text
output shape: torch.Size([1, 10, 32]), output dtype: torch.float32, features grad shape: torch.Size([1, 2816, 32]), features grad dtype: torch.float32, locations grad shape: torch.Size([1, 10, 10, 1, 2]), locations grad dtype: torch.float32, weights grad shape: torch.Size([1, 10, 10, 1, 1, 8]), weights grad dtype: torch.float32
```

## 参考

- 兼容层：`turbo_physai/operators/deformable_aggregation.py`
- Kernel：`kernel/deformable_aggregation/DeformableAggregation.cu`
- 返回[兼容层 API 清单](../README.md)
