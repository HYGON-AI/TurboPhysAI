# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""编译蒸馏 teacher 的 forward，并去掉 depth target 的两个冗余 op。"""

from __future__ import annotations

import functools

import torch


def compiled_forward(original, _options=None):
    """把函数包成首次调用时才编译的版本。"""
    compiled = None

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        nonlocal compiled
        if compiled is None:  # 装配期 teacher 还没加载，只能等首次调用
            compiled = torch.compile(original)
        return compiled(*args, **kwargs)

    return wrapped


def fast_get_depth_target(model_type, depth_model, pil_images):
    """get_depth_target 的等价实现，少一次 clone 和一次乘 1。"""
    if model_type != "MoRGBD":
        raise ValueError(f"Only Lingbot-Depth depth targets are supported, got {model_type!r}.")

    import einops
    from lingbotvla.models.vla.vision_models.module_utils import _rgb_to_unit_float

    pil_images = pil_images[:, :1, :, :, :]  # 只取第一帧，和下面 n=1 对齐
    images = einops.rearrange(pil_images, "b n c h w -> (b n) c h w", n=1).contiguous().float()

    input_images = _rgb_to_unit_float(images)
    moge_model, morgbd_model = depth_model
    depth_only_kwargs = {}
    if getattr(moge_model.infer, "_turbophysai_depth_only_supported", False):
        depth_only_kwargs["_turbophysai_depth_only"] = True
    output_moge = moge_model.infer(
        input_images, resolution_level=3, num_tokens=256, apply_mask=False,
        **depth_only_kwargs,
    )

    depth_pred = torch.nan_to_num(
        output_moge["depth"].squeeze().detach(), nan=0.0, posinf=0.0, neginf=0.0
    )

    depth_target, cls_token = morgbd_model.infer_feat(
        input_images,
        depth_pred,
        depth_down_scale=1,
        resolution_level=3,
        num_tokens=256,
        enable_depth_mask=False,
    )

    depth_target = depth_target.permute(0, 2, 3, 1)
    depth_target = depth_target.view(depth_target.shape[0], -1, depth_target.shape[-1])

    return depth_target.to(dtype=torch.bfloat16), cls_token
