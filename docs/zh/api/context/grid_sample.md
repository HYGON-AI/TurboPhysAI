# grid_sample

## 接口原型

```python
turbo_physai.operators.grid_sample(
    input, grid, mode="bilinear", padding_mode="zeros", align_corners=None
) -> Tensor
```

接口位置：`turbo_physai/operators/_grid_sample.py`。该封装将字符串模式转换为 native 枚举，并通过 `GridSampleFunction` 注册反向。

## 功能描述

按照归一化二维坐标对输入特征执行双线性采样，接口形式与 `torch.nn.functional.grid_sample` 的目标模型用法兼容。封装将字符串参数转换为 native 枚举，并保存输入用于自动反向。

## 参数说明

- `input`：`[N, C, H_in, W_in]`。
- `grid`：`[N, H_out, W_out, 2]`，坐标范围通常为 `[-1, 1]`。
- `mode`：当前仅支持 `"bilinear"`。
- `padding_mode`：`"zeros"`、`"border"` 或 `"reflection"`。
- `align_corners`：`None` 时按 `False` 处理。

## 返回值

返回采样结果 `[N, C, H_out, W_out]`。反向传播产生与 `input` 和 `grid` shape 相同的梯度。

## 约束说明

- `input` 和 `grid` 都必须为 4D，且 batch 大小一致。
- `mode` 只能为 `"bilinear"`；`padding_mode` 只能为 `"zeros"`、`"border"` 或 `"reflection"`。
- 底层 kernel 支持 float32、float16 和 bfloat16；输入、grid 和梯度应具有兼容 dtype 并位于同一 HCU 设备。
- backward 被 `once_differentiable` 标记，不支持二阶梯度。

## 调用示例

```python
import torch
from turbo_physai.operators import grid_sample

input = torch.randn(2, 32, 64, 64, device="cuda", dtype=torch.float32, requires_grad=True)
grid = (torch.rand(2, 128, 128, 2, device="cuda") * 2 - 1).requires_grad_()
out = grid_sample(input, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
out.sum().backward()
torch.cuda.synchronize()
print(
    f"input shape: {input.shape}, input dtype: {input.dtype}, "
    f"grid shape: {grid.shape}, grid dtype: {grid.dtype}, "
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"input grad shape: {input.grad.shape}, input grad dtype: {input.grad.dtype}, "
    f"grid grad shape: {grid.grad.shape}, grid grad dtype: {grid.grad.dtype}"
)
```

一次运行的输出示例：

```text
input shape: torch.Size([2, 32, 64, 64]), input dtype: torch.float32, grid shape: torch.Size([2, 128, 128, 2]), grid dtype: torch.float32, output shape: torch.Size([2, 32, 128, 128]), output dtype: torch.float32, input grad shape: torch.Size([2, 32, 64, 64]), input grad dtype: torch.float32, grid grad shape: torch.Size([2, 128, 128, 2]), grid grad dtype: torch.float32
```

## 参考

- 兼容接口实现：`turbo_physai/operators/_grid_sample.py`
- Kernel：`kernel/grid_sample/GridSampler.cu`
- 返回[兼容层 API 清单](../README.md)
