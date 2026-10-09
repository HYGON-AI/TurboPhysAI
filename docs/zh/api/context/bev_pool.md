# bev_pool

## 接口原型

```python
turbo_physai.operators.bev_pool(
    feats, coords, batch, depth, height, width, ranks=None
) -> Tensor

turbo_physai.operators.bev_pool_prepare(
    geom_feats, bx, dx, nx, batch, depth, height, width
) -> Tuple[Tensor, Tensor, Tensor]
```

接口位置：`turbo_physai/operators/_bev_pool.py`。`bev_pool` 负责计算或接收 ranks、排序输入、生成区间、注册反向，并将 native 输出转换为 `[B, C, D, H, W]`。

## 功能描述

将具有相同 BEV 网格坐标的点特征累加到输出体素。相比 native 接口，该封装自动计算排序秩、重排输入、生成连续区间，并通过内部 Autograd Function 接入 PyTorch Autograd。

## 参数说明

- `feats`：float32，shape `[N, C]`。
- `coords`：shape `[N, 4]`，每行为 `(height, width, depth, batch)`。
- `batch, depth, height, width`：输出网格大小。
- `ranks`：可选排序秩；省略时由 `coords` 计算。

## 返回值

返回 float32 Tensor，shape `[B, C, D, H, W]`。反向传播时仅计算 `feats` 的梯度，shape 与 `feats` 相同。

`bev_pool_prepare` 和 `bev_pool_prepare_geometry` 保留 native 返回布局，并负责输入连续化和标量转换。

## 约束说明

- `feats` 与 `coords` 的第一维必须相等，否则抛出 `ValueError`。
- 当前 native BEV pool kernel 使用 float32 特征和 CUDA/HIP 张量。
- `coords` 会在封装内部转换为连续 int32；坐标必须落在给定输出网格范围内。
- 封装输出为 `[B, C, D, H, W]`，不同于 native forward 的 `[B, D, H, W, C]`。

## 调用示例

```python
import torch
from turbo_physai.operators import bev_pool

B, D, H, W, C = 1, 4, 128, 128, 64
N = 1024
feats = torch.randn(N, C, device="cuda", dtype=torch.float32, requires_grad=True)
coords = torch.empty(N, 4, device="cuda", dtype=torch.int32)
coords[:, 0] = torch.randint(0, H, (N,), device="cuda")
coords[:, 1] = torch.randint(0, W, (N,), device="cuda")
coords[:, 2] = torch.randint(0, D, (N,), device="cuda")
coords[:, 3] = torch.randint(0, B, (N,), device="cuda")
out = bev_pool(feats, coords, B, D, H, W)
out.sum().backward()
torch.cuda.synchronize()
print(
    f"feats shape: {feats.shape}, feats dtype: {feats.dtype}, "
    f"coords shape: {coords.shape}, coords dtype: {coords.dtype}, "
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"feats grad shape: {feats.grad.shape}, feats grad dtype: {feats.grad.dtype}"
)
```

一次运行的输出示例：

```text
feats shape: torch.Size([1024, 64]), feats dtype: torch.float32, coords shape: torch.Size([1024, 4]), coords dtype: torch.int32, output shape: torch.Size([1, 64, 4, 128, 128]), output dtype: torch.float32, feats grad shape: torch.Size([1024, 64]), feats grad dtype: torch.float32
```

## 参考

- 兼容接口实现：`turbo_physai/operators/_bev_pool.py`
- Kernel：`kernel/bev_pool/`
- 返回[兼容层 API 清单](../README.md)
