# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Dynamo-safe 版 timm ``VisionTransformer._intermediate_layers``。

timm 原实现在已 sourced 的 set 上再做一次 ``set(n)``；
FSDP1 把 ViT 整塔 compile 后，Dynamo 的 set() 处理会 clone 出带 source 的 SetVariable，
撞上assert source is None，会报错
这里改成 list 展开，语义与 timm 原实现逐位一致。
"""

import os


def dynamo_safe_intermediate_layers(self, x, n=1):
    outputs, num_blocks = [], len(self.blocks)
    take_indices = (
        list(range(num_blocks - n, num_blocks)) if isinstance(n, int) else list(n)
    )

    # forward pass
    x = self.patch_embed(x)
    x = self._pos_embed(x)
    x = self.patch_drop(x)
    x = self.norm_pre(x)
    for i, blk in enumerate(self.blocks):
        x = blk(x)
        if i in take_indices:
            outputs.append(x)

    return outputs


def timm_intermediate_layers_wrapper(original, options=None):
    """wrap 工厂：``TURBO_PHYSAI_DISABLE_TORCH_COMPILE=1`` 时返回原方法。"""

    del options
    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original
    return dynamo_safe_intermediate_layers
