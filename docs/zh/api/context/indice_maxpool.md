# indice_maxpool

## 接口原型

```python
turbo_physai.operators.indice_maxpool(
    features, indice_pairs, indice_num, num_act
) -> Tensor
```

接口位置：`turbo_physai/operators/sparse_conv.py`。内部 `_IndiceMaxPoolFunction` 根据输入 dtype 分派 fp32 或 half kernel，并保存前向输出以自动计算输入梯度。

## 功能描述

根据稀疏索引对在输入激活之间执行逐通道最大池化。封装自动选择 float32 或 float16 native kernel，并通过内部 `_IndiceMaxPoolFunction` 保存前向结果以计算输入梯度。

## 参数说明

- `features`：输入激活 `[num_act_in, channels]`，支持 float32 或 float16。
- `indice_pairs` / `indice_num`：`get_indice_pairs` 的输出。
- `num_act`：输出激活行数。

## 返回值

返回 `[num_act, channels]`。反向传播产生与 `features` shape 和 dtype 相同的梯度。

## 约束说明

- `features` 仅支持 float32 或 float16，且必须为二维。
- `indice_pairs`、`indice_num` 和 `num_act` 必须来自匹配的索引生成过程。
- 底层输出以 0 初始化；如果某个输出位置的候选值全部为负，其结果仍为 0。
- 多个输入值同时等于池化输出时，native backward 会向所有相等位置累加梯度。

## 调用示例

```python
import torch
from turbo_physai.operators import get_indice_pairs, indice_maxpool

indices = torch.tensor(
    [[0, 1, 2, 3], [0, 2, 2, 3], [0, 1, 3, 3], [0, 2, 2, 3]],
    dtype=torch.int32,
    device="cuda",
)
out_indices, pairs, counts = get_indice_pairs(
    indices, 1, [5, 5, 5], ksize=3, padding=1, subm=True
)
features = torch.rand(4, 16, device="cuda", dtype=torch.float32, requires_grad=True)
out = indice_maxpool(features, pairs, counts, out_indices.size(0))
out.sum().backward()
torch.cuda.synchronize()
print(
    f"features shape: {features.shape}, features dtype: {features.dtype}, "
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"features grad shape: {features.grad.shape}, features grad dtype: {features.grad.dtype}"
)
```

一次运行的输出示例：

```text
features shape: torch.Size([4, 16]), features dtype: torch.float32, output shape: torch.Size([4, 16]), output dtype: torch.float32, features grad shape: torch.Size([4, 16]), features grad dtype: torch.float32
```

## 参考

- 兼容层：`turbo_physai/operators/sparse_conv.py`
- Kernel：`kernel/sparse_conv/`
- 返回[兼容层 API 清单](../README.md)
