# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Python catalog declarations for BEVFusion HCU optimizations."""

from __future__ import annotations

from ....compatibility import import_alias, optional_import, registry_override
from ....engine.definitions import (
    group,
    replace,
    replace_import,
    wrap,
)


IMPORT_COMPATIBILITY = group(
    "bevfusion.import_compatibility",
    import_alias(
        module="flash_attn.modules.mha",
        source="MHA",
        alias="FlashMHA",
    ),
    replace_import(
        target="flash_attn.flash_attention",
        replacement="flash_attn.modules.mha",
    ),
    optional_import(
        module="mmdet3d.ops.feature_decorator.feature_decorator_ext",
    ),
    registry_override(
        module="mmdet3d.ops.spconv.conv",
        registry="mmcv.cnn.CONV_LAYERS",
        names=(
            "SparseConv2d",
            "SparseConv3d",
            "SparseConv4d",
            "SparseConvTranspose2d",
            "SparseConvTranspose3d",
            "SparseInverseConv2d",
            "SparseInverseConv3d",
            "SubMConv2d",
            "SubMConv3d",
            "SubMConv4d",
        ),
    ),
)


BACKBONE = group(
    "bevfusion.backbone",
    replace(
        target=(
            "mmdet3d.models.fusion_models.bevfusion."
            "BEVFusion.extract_camera_features"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.backbone."
            "extract_camera_features"
        ),
    ),
    replace(
        target=(
            "mmdet3d.models.fusion_models.bevfusion."
            "BEVFusion.extract_features"
        ),
        replacement="turbo_physai.optimizations.models.bevfusion.backbone.extract_features",
    ),
)

LOSS_REDUCTION = group(
    "bevfusion.loss_reduction",
    replace(
        target=(
            "mmdet3d.models.fusion_models.base."
            "Base3DFusionModel._parse_losses"
        ),
        replacement="turbo_physai.optimizations.models.bevfusion.training.parse_losses",
    ),
)

TRAINING = group(
    "bevfusion.training",
    wrap(
        target="mmdet3d.apis.train.train_model",
        aliases=("mmdet3d.apis.train_model",),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.training.training_wrapper"
        ),
    ),
)

GAUSSIAN = group(
    "bevfusion.gaussian",
    wrap(
        target="mmdet3d.core.utils.gaussian.draw_heatmap_gaussian",
        aliases=(
            "mmdet3d.core.utils.draw_heatmap_gaussian",
            "mmdet3d.core.draw_heatmap_gaussian",
            "mmdet3d.models.heads.bbox.transfusion.draw_heatmap_gaussian",
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.gaussian."
            "compiled_draw_heatmap_gaussian_wrapper"
        ),
    ),
    wrap(
        target="mmdet3d.core.utils.gaussian.gaussian_radius",
        aliases=(
            "mmdet3d.core.utils.gaussian_radius",
            "mmdet3d.core.gaussian_radius",
            "mmdet3d.models.heads.bbox.transfusion.gaussian_radius",
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.gaussian."
            "compiled_gaussian_radius_wrapper"
        ),
    ),
    depends_on=("mmdet3d.gaussian",),
)

DEPTH_FACTORIZATION = group(
    "bevfusion.depth_factorization",
    wrap(
        target="mmdet3d.models.vtransforms.base.BaseTransform.__init__",
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.depth."
            "base_transform_init_wrapper"
        ),
    ),
    wrap(
        target=(
            "mmdet3d.models.vtransforms.base."
            "BaseDepthTransform.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.depth."
            "base_depth_transform_forward_wrapper"
        ),
    ),
    wrap(
        target=(
            "mmdet3d.models.vtransforms.depth_lss."
            "DepthLSSTransform.get_cam_feats"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.depth."
            "depth_lss_get_cam_feats_wrapper"
        ),
    ),
    wrap(
        target=(
            "mmdet3d.models.vtransforms.base.BaseTransform.bev_pool"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.depth."
            "base_transform_bev_pool_wrapper"
        ),
    ),
    depends_on=("mmdet3d.bev_pool",),
)

BEV_GEOMETRY = group(
    "bevfusion.bev_geometry",
    wrap(
        target=(
            "mmdet3d.models.vtransforms.base."
            "BaseTransform.get_geometry"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.depth."
            "base_transform_get_geometry_wrapper"
        ),
    ),
    depends_on=("bevfusion.depth_factorization",),
)

HUNGARIAN_TRANSFER = group(
    "bevfusion.hungarian_transfer",
    wrap(
        target=(
            "mmdet3d.models.heads.bbox.transfusion."
            "TransFusionHead.__init__"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.transfusion."
            "transfusion_init_wrapper"
        ),
    ),
    replace(
        target=(
            "mmdet3d.models.heads.bbox.transfusion."
            "TransFusionHead.get_targets"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.transfusion."
            "transfusion_get_targets"
        ),
    ),
    replace(
        target=(
            "mmdet3d.models.heads.bbox.transfusion."
            "TransFusionHead.get_targets_single"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.transfusion."
            "transfusion_get_targets_single"
        ),
    ),
    wrap(
        target=(
            "mmdet3d.models.heads.bbox.transfusion."
            "TransFusionHead.loss"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.transfusion."
            "transfusion_loss_wrapper"
        ),
    ),
    replace(
        target=(
            "mmdet3d.core.bbox.assigners.hungarian_assigner."
            "HungarianAssigner3D.assign"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.transfusion."
            "hungarian_assign"
        ),
    ),
)

COMPILE = group(
    "bevfusion.compile",
    wrap(
        target="mmdet3d.models.backbones.sparse_encoder.SparseEncoder",
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.compile.compile_class_wrapper"
        ),
    ),
    wrap(
        target="mmdet3d.models.fusers.conv.ConvFuser",
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.compile.compile_class_wrapper"
        ),
    ),
    wrap(
        target="mmdet3d.models.necks.generalized_lss.GeneralizedLSSFPN",
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.compile.compile_class_wrapper"
        ),
    ),
    wrap(
        target=(
            "mmdet3d.models.heads.bbox.transfusion."
            "TransFusionHead.forward_single"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.compile."
            "compile_transfusion_forward_single_wrapper"
        ),
    ),
    wrap(
        target="mmdet3d.models.vtransforms.depth_lss.DepthLSSTransform",
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.compile.compile_class_wrapper"
        ),
    ),
    wrap(
        target=(
            "mmdet3d.models.utils.transformer."
            "PositionEmbeddingLearned.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.compile."
            "dynamo_disable_wrapper"
        ),
    ),
    wrap(
        target="mmdet3d.models.utils.transformer.FFN.forward",
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.compile."
            "dynamo_disable_wrapper"
        ),
    ),
)

TRANSFUSION_BBOX_CODER = group(
    "bevfusion.transfusion_bbox_coder",
    wrap(
        target=(
            "mmdet3d.core.bbox.coders.transfusion_bbox_coder."
            "TransFusionBBoxCoder"
        ),
        aliases=("mmdet3d.core.bbox.coders.TransFusionBBoxCoder",),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.transfusion_bbox_coder."
            "transfusion_bbox_coder_class_wrapper"
        ),
    ),
    wrap(
        target="mmdet.core.bbox.builder.build_bbox_coder",
        aliases=(
            "mmdet.core.build_bbox_coder",
            "mmdet3d.models.heads.bbox.transfusion.build_bbox_coder",
        ),
        replacement=(
            "turbo_physai.optimizations.models.bevfusion.transfusion_bbox_coder."
            "build_bbox_coder_wrapper"
        ),
    ),
)
