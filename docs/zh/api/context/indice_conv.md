# indice_conv

## 接口原型

```python
turbo_physai.operators.indice_conv(
    features, filters, indice_pairs, indice_num, num_act_out,
    inverse=False, subm=False, bias=None
) -> Tensor
```

接口位置：`turbo_physai/operators/sparse_conv.py`。内部 `_IndiceConvFunction` 根据 `features.dtype` 在 fp32 和 half native kernel 间分派，并自动调用 backward；可选 `bias` 由 PyTorch 加法处理。

## 功能描述

根据 `indice_pairs` 描述的稀疏连接执行 gather、矩阵乘和 scatter-add。封装自动选择 float32 或 float16 native kernel，通过内部 `_IndiceConvFunction` 计算输入和卷积核梯度，并可在输出上添加 bias。

## 参数说明

- `features`：输入激活 `[num_act_in, num_in_channels]`。
- `filters`：`[..., num_in_channels, num_out_channels]`，dtype 必须与 `features` 相同。
- `indice_pairs` / `indice_num`：`get_indice_pairs` 的输出。
- `num_act_out`：输出激活行数。
- `inverse` / `subm`：需与索引对生成方式一致。
- `bias`：可选，通常为 `[num_out_channels]`。

## 返回值

返回 `[num_act_out, num_out_channels]`。反向传播产生与 `features`、`filters` 相同 shape 的梯度；传入 bias 时，PyTorch 加法会计算其梯度。

## 约束说明

- `features` 仅支持 float32 或 float16；`filters.dtype` 必须与其相同，否则抛出 `TypeError`。
- `features` 必须为二维，filters 的输入通道维必须与 `features.shape[1]` 一致。
- `indice_pairs`、`indice_num`、`num_act_out`、`inverse` 和 `subm` 必须与索引生成阶段一致。
- bias 通过普通 Tensor 加法应用，并不调用 `fused_indice_conv_*` native binding。

## 调用示例

```python
import torch
from turbo_physai.operators import get_indice_pairs, indice_conv

indices = torch.tensor(
    [[0, 1, 2, 3], [0, 2, 2, 3], [0, 1, 3, 3], [0, 2, 2, 3]],
    dtype=torch.int32,
    device="cuda",
)
out_indices, pairs, counts = get_indice_pairs(
    indices, 1, [5, 5, 5], ksize=3, padding=1, subm=True
)
features = torch.randn(4, 16, device="cuda", dtype=torch.float32, requires_grad=True)
filters = torch.randn(3, 3, 3, 16, 32, device="cuda", dtype=torch.float32, requires_grad=True)
bias = torch.randn(32, device="cuda", dtype=torch.float32, requires_grad=True)
out = indice_conv(
    features, filters, pairs, counts, out_indices.size(0),
    inverse=False, subm=True, bias=bias,
)
out.sum().backward()
torch.cuda.synchronize()
print(
    f"features shape: {features.shape}, features dtype: {features.dtype}, "
    f"filters shape: {filters.shape}, filters dtype: {filters.dtype}, "
    f"bias shape: {bias.shape}, bias dtype: {bias.dtype}, "
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"features grad shape: {features.grad.shape}, features grad dtype: {features.grad.dtype}, "
    f"filters grad shape: {filters.grad.shape}, filters grad dtype: {filters.grad.dtype}, "
    f"bias grad shape: {bias.grad.shape}, bias grad dtype: {bias.grad.dtype}"
)
```

一次运行的输出示例：

```text
features shape: torch.Size([4, 16]), features dtype: torch.float32, filters shape: torch.Size([3, 3, 3, 16, 32]), filters dtype: torch.float32, bias shape: torch.Size([32]), bias dtype: torch.float32, output shape: torch.Size([4, 32]), output dtype: torch.float32, features grad shape: torch.Size([4, 16]), features grad dtype: torch.float32, filters grad shape: torch.Size([3, 3, 3, 16, 32]), filters grad dtype: torch.float32, bias grad shape: torch.Size([32]), bias grad dtype: torch.float32
```

## 参考

- 兼容层：`turbo_physai/operators/sparse_conv.py`
- Kernel：`kernel/sparse_conv/`
- 返回[兼容层 API 清单](../README.md)
