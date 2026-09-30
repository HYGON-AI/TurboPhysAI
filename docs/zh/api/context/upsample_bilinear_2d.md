# upsample_bilinear_2d

## 接口原型

```python
turbo_physai.operators.interpolate(
    input, size=None, scale_factor=None, mode="nearest",
    align_corners=None, recompute_scale_factor=None, antialias=False
) -> Tensor
```

接口位置：`turbo_physai/operators/upsample_bilinear_2d.py`。封装负责输出尺寸处理，并通过 `UpSampleBilinear2dFunction` 注册反向。

## 功能描述

对 4D 特征图执行二维双线性上采样，接口形式覆盖目标模型使用的 `torch.nn.functional.interpolate` 子集。封装负责处理 `size` 或 `scale_factor`，并注册 native backward。

## 参数说明

- `input`：4D Tensor `[N, C, H_in, W_in]`。
- `size` / `scale_factor`：必须且只能指定一个，可使用标量或两个元素的序列。
- `mode`：必须显式传 `"bilinear"`。
- `align_corners`：支持 `None` 或 `False`，`True` 会报错。
- `recompute_scale_factor`：与 `scale_factor` 配合使用；显式指定 `size` 时不能设为 `True`。
- `antialias`：当前必须为 `False`。

## 返回值

返回 `[N, C, H_out, W_out]`。反向传播产生与 `input` shape 和 dtype 相同的梯度。

## 约束说明

- 仅支持 4D 输入和 `mode="bilinear"`。
- `size` 与 `scale_factor` 必须且只能指定一个，序列长度必须为 2。
- `align_corners=True` 和 `antialias=True` 当前不支持，会抛出 `ValueError`。
- backward 被 `once_differentiable` 标记，不支持二阶梯度。

## 调用示例

```python
import torch
from turbo_physai.operators import interpolate

input = torch.randn(2, 64, 32, 32, device="cuda", dtype=torch.float32, requires_grad=True)
out = interpolate(input, size=(64, 64), mode="bilinear", align_corners=False)
out.sum().backward()
torch.cuda.synchronize()
print(
    f"input shape: {input.shape}, input dtype: {input.dtype}, "
    f"output shape: {out.shape}, output dtype: {out.dtype}, "
    f"input grad shape: {input.grad.shape}, input grad dtype: {input.grad.dtype}"
)
```

一次运行的输出示例：

```text
input shape: torch.Size([2, 64, 32, 32]), input dtype: torch.float32, output shape: torch.Size([2, 64, 64, 64]), output dtype: torch.float32, input grad shape: torch.Size([2, 64, 32, 32]), input grad dtype: torch.float32
```

## 参考

- 兼容层：`turbo_physai/operators/upsample_bilinear_2d.py`
- Kernel：`kernel/upsample_bilinear_2d/UpSampleBilinear2d.cu`
- 返回[兼容层 API 清单](../README.md)
