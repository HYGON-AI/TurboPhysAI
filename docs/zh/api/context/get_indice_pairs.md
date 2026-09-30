# get_indice_pairs

## 接口原型

```python
turbo_physai.operators.get_indice_pairs(
    indices, batch_size, spatial_shape, ksize=3, stride=1,
    padding=0, dilation=1, out_padding=0, subm=False,
    transpose=False, grid=None
) -> Tuple[Tensor, Tensor, Tensor]
```

接口位置：`turbo_physai/operators/sparse_conv.py`。封装根据 `indices` 推导维度，将标量卷积参数广播到各空间维，并自动计算输出形状。

## 功能描述

生成稀疏卷积或稀疏池化使用的输出坐标、索引对和每个 kernel offset 的有效连接数。封装根据普通卷积、转置卷积或子流形卷积规则推导输出空间形状，并选择对应维度的 native binding。

## 参数说明

- `indices`：int32，shape `[num_points, ndim + 1]`，第一列是 batch 索引。
- `batch_size`：输入 batch 大小。
- `spatial_shape`：输入空间形状。
- `ksize`、`stride`、`padding`、`dilation`、`out_padding`：标量或长度为 `ndim` 的序列。
- `subm` / `transpose`：子流形卷积和转置卷积开关。
- `grid`：可选预分配网格；仅 2D/3D 支持 grid 路径。

## 返回值

返回 `(out_indices, indice_pairs, indice_num)`：shape 分别为 `[num_act_out, ndim + 1]`、`[kernel_volume, 2, num_act_in]` 和 `[kernel_volume]`，dtype 均为 int32。`subm=True` 时 `out_indices` 与输入 `indices` 相同。

## 约束说明

- `ndim` 由 `indices.shape[1] - 1` 推导；支持 2D、3D 和 4D，grid 路径支持 2D 和 3D。
- 标量参数会广播到所有空间维；序列参数长度必须与 `ndim` 一致。
- `stride` 与 `dilation` 不能在同一维同时大于 1，否则抛出 `ValueError`。
- `kernel_volume` 必须不大于 4096；`indices` 和可选 `grid` 必须满足 native binding 的 int32、设备和布局要求。

## 调用示例

```python
import torch
from turbo_physai.operators import get_indice_pairs

indices = torch.tensor(
    [[0, 1, 2, 3], [0, 2, 2, 3], [0, 1, 3, 3], [0, 2, 2, 3]],
    dtype=torch.int32,
    device="cuda",
)
out_indices, indice_pairs, indice_num = get_indice_pairs(
    indices,
    batch_size=1,
    spatial_shape=[5, 5, 5],
    ksize=3,
    stride=1,
    padding=1,
    dilation=1,
    out_padding=0,
    subm=True,
)
torch.cuda.synchronize()
print(
    f"indices shape: {indices.shape}, indices dtype: {indices.dtype}, "
    f"out_indices shape: {out_indices.shape}, out_indices dtype: {out_indices.dtype}, "
    f"indice_pairs shape: {indice_pairs.shape}, indice_pairs dtype: {indice_pairs.dtype}, "
    f"indice_num shape: {indice_num.shape}, indice_num dtype: {indice_num.dtype}"
)
```

一次运行的输出示例：

```text
indices shape: torch.Size([4, 4]), indices dtype: torch.int32, out_indices shape: torch.Size([4, 4]), out_indices dtype: torch.int32, indice_pairs shape: torch.Size([27, 2, 4]), indice_pairs dtype: torch.int32, indice_num shape: torch.Size([27]), indice_num dtype: torch.int32
```

## 参考

- 兼容层：`turbo_physai/operators/sparse_conv.py`
- Kernel：`kernel/sparse_conv/`
- 返回[兼容层 API 清单](../README.md)
