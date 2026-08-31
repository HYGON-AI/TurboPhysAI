# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python catalog declarations for BEVFormer HCU optimizations."""

from __future__ import annotations

from ....engine.definitions import group, replace, wrap


_MODULE_PREFIX = "projects.mmdet3d_plugin.bevformer.modules"
FP16_TARGET = f"{_MODULE_PREFIX}.multi_scale_deformable_attn_function.MultiScaleDeformableAttnFunction_fp16"
FP16_ALIASES = (
    f"{_MODULE_PREFIX}.spatial_cross_attention.MultiScaleDeformableAttnFunction_fp16",
    f"{_MODULE_PREFIX}.decoder.MultiScaleDeformableAttnFunction_fp16",
)
FP32_TARGET = f"{_MODULE_PREFIX}.multi_scale_deformable_attn_function.MultiScaleDeformableAttnFunction_fp32"
FP32_ALIASES = (
    f"{_MODULE_PREFIX}.spatial_cross_attention.MultiScaleDeformableAttnFunction_fp32",
    f"{_MODULE_PREFIX}.temporal_self_attention.MultiScaleDeformableAttnFunction_fp32",
    f"{_MODULE_PREFIX}.decoder.MultiScaleDeformableAttnFunction_fp32",
)


MDC = group(
    "bevformer.mdc",
    replace(
        target="mmcv.ops.modulated_deform_conv.modulated_deform_conv2d",
        aliases=("mmcv.ops.modulated_deform_conv2d",),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.mdc."
            "modulated_deform_conv2d"
        ),
    ),
    replace(
        target="mmcv.ops.modulated_deform_conv.ModulatedDeformConv2d",
        aliases=("mmcv.ops.ModulatedDeformConv2d",),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.mdc."
            "ModulatedDeformConv2d"
        ),
    ),
    replace(
        target="mmcv.ops.modulated_deform_conv.ModulatedDeformConv2dPack",
        aliases=(
            "mmcv.ops.ModulatedDeformConv2dPack",
            "mmcv.cnn.bricks.conv.CONV_LAYERS.module_dict.DCNv2",
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.mdc."
            "ModulatedDeformConv2dPack"
        ),
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.bevformer.detectors.bevformer."
            "BEVFormer.extract_img_feat"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.backbone."
            "compiled_extract_img_feat"
        ),
    ),
)

MSDA = group(
    "bevformer.msda",
    replace(
        target=FP16_TARGET,
        aliases=FP16_ALIASES,
        replacement=(
            "turbo_physai.optimizations.models.bevformer.msda."
            "MultiScaleDeformableAttnFunction_fp16"
        ),
    ),
    replace(
        target=FP32_TARGET,
        aliases=FP32_ALIASES,
        replacement=(
            "turbo_physai.optimizations.models.bevformer.msda."
            "MultiScaleDeformableAttnFunction_fp32"
        ),
    ),
)

GEOMETRY = group(
    "bevformer.geometry_sca",
    replace(
        target="projects.mmdet3d_plugin.datasets.builder.build_dataloader",
        aliases=(
            "projects.mmdet3d_plugin.bevformer.apis.mmdet_train.build_dataloader",
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.data.build_dataloader"
        ),
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.bevformer.modules.transformer."
            "PerceptionTransformer.get_bev_features"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.geometry_sca."
            "get_bev_features"
        ),
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.bevformer.modules.encoder."
            "BEVFormerEncoder.point_sampling"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.geometry_sca.point_sampling"
        ),
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.bevformer.modules.encoder."
            "BEVFormerEncoder.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.geometry_sca.encoder_forward"
        ),
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention."
            "SpatialCrossAttention.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.geometry_sca."
            "spatial_cross_attention_forward"
        ),
    ),
)

TSA = group(
    "bevformer.tsa",
    replace(
        target=(
            "projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention."
            "TemporalSelfAttention.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.tsa."
            "temporal_self_attention_forward"
        ),
    ),
)

GRID_MASK = group(
    "bevformer.grid_mask",
    replace(
        target="projects.mmdet3d_plugin.models.utils.grid_mask.GridMask.forward",
        replacement=(
            "turbo_physai.optimizations.models.bevformer.grid_mask.grid_mask_forward"
        ),
    ),
)

COMPILE_ENCODER = group(
    "bevformer.compile.encoder",
    wrap(
        target=(
            "projects.mmdet3d_plugin.bevformer.modules.encoder."
            "BEVFormerLayer.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.compile.compile_wrapper"
        ),
    ),
)

COMPILE_DECODER = group(
    "bevformer.compile.decoder",
    wrap(
        target=(
            "projects.mmdet3d_plugin.bevformer.modules.decoder."
            "DetectionTransformerDecoder.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.compile.compile_wrapper"
        ),
    ),
)

TRAINING = group(
    "bevformer.training",
    wrap(
        target=(
            "projects.mmdet3d_plugin.bevformer.apis.mmdet_train."
            "custom_train_detector"
        ),
        aliases=(
            "projects.mmdet3d_plugin.bevformer.apis.custom_train_detector",
            "projects.mmdet3d_plugin.bevformer.apis.train.custom_train_detector",
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevformer.training."
            "training_runtime_wrapper"
        ),
    ),
)
