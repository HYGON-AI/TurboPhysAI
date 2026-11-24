# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""MapTRv2 optimization declarations.

Every Group below maps one item of the validated MapTRv2 reference optimization
onto a symbol that exists in the official ``maptrv2`` branch, so the recipe can
be applied to a clean baseline checkout:

===========================================  ==================================
Group                                        Reference change
===========================================  ==================================
``maptrv2.training``                         channels-last backbone, cuDNN
                                             benchmark/非确定性, ``fork`` start
                                             method
``maptrv2.data``                             ``pin_memory=True``
``maptrv2.grid_mask``                        Dynamo-safe ``GridMask.forward``
``maptrv2.compile``                          ``torch.compile`` on every baseline
                                             symbol carrying a compile hook in
                                             the reference
``maptrv2.match_cost``                       ``torch.cdist(p=1)`` replaced by a
                                             broadcast subtraction
``maptrv2.pv_mask``                          ``line_ego_to_pvmask`` drops the
                                             shapely interpolation loop and
                                             ``line_ego_to_mask`` merges the
                                             two affine transform steps;
                                             ``gen_vectorized_samples`` drops one
                                             vertex copy
``maptrv2.assigner``                         padded GT contract and Dynamo
                                             isolation for Hungarian matching
``maptrv2.efficientnet``                     ``EfficientNet`` allowed to
                                             override the registry entry
``maptrv2.spconv_registry``                  SparseConv classes allowed to
                                             override registry entries
``maptrv2.reference_boundaries``             eight extracted helper boundaries
                                             plus the reference BEV layout
``maptrv2.dataset_vectorization``            arc-length resampling for
                                             ``shift_fixed_num_sampled_points_v2``
``maptrv2.eval_compat``                      chamfer evaluation tolerates the
                                             index array returned by Shapely 2
                                             ``STRtree.query``
===========================================  ==================================

``maptrv2.training`` also pins the fp32 matmul precision to the reference's
``"high"`` and defaults ``_use_replicated_tensor_module`` on
``MMDistributedDataParallel`` so validation can run on newer Torch.

``optimization_modules`` loads this module, so importing it registers every
Group with TurboPhysAI.
"""

from __future__ import annotations

from turbo_physai import group, replace, wrap
from turbo_physai.compatibility import registry_override


_TRAIN_API = (
    "projects.mmdet3d_plugin.bevformer.apis.mmdet_train.custom_train_detector"
)
_BUILD_DATALOADER_API = "projects.mmdet3d_plugin.datasets.builder.build_dataloader"
_GRID_MASK_API = "projects.mmdet3d_plugin.models.utils.grid_mask.GridMask.forward"
_ASSIGN_API = (
    "projects.mmdet3d_plugin.maptr.assigners.maptr_assigner.MapTRAssigner.assign"
)
# ``mmcv.parallel``'s subclass does not define ``__init__``; installing the
# wrapper shadows the inherited Torch DDP constructor on the subclass only.
# MMCV 1.6 reads ``_use_replicated_tensor_module`` here, which newer Torch DDP
# no longer defines; reached by ``model(return_loss=False, ...)`` at validation.
_DDP_FORWARD_API = (
    "mmcv.parallel.distributed.MMDistributedDataParallel._run_ddp_forward"
)
_CHAMFER_SCORE_API = (
    "projects.mmdet3d_plugin.datasets.map_utils.tpfp_chamfer.custom_polyline_score"
)

# ``torch.compile``/``torch._dynamo`` need a new enough Torch and the user's
# consent, so the Groups that depend on them dispatch per call instead of
# failing while ``apply`` installs them.  See ``turbo_physai.optimizations.models.maptrv2.compat``.
_COMPILE_CONDITION = "turbo_physai.optimizations.models.maptrv2.compat.torch_compile_available"
_STATIC_ASSIGNER_CONDITION = (
    "turbo_physai.optimizations.models.maptrv2.compat."
    "static_assigner_available"
)
_PV_MASK_CONDITION = "turbo_physai.optimizations.models.maptrv2.compat.pv_mask_sampling_enabled"
_EVAL_COMPAT_CONDITION = (
    "turbo_physai.optimizations.models.maptrv2.compat."
    "shapely_query_returns_indices"
)


TRAINING = group(
    "maptrv2.training",
    wrap(
        target=_TRAIN_API,
        aliases=("projects.mmdet3d_plugin.bevformer.apis.custom_train_detector",),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2.training.training_runtime_wrapper"
        ),
    ),
    wrap(
        target=_DDP_FORWARD_API,
        replacement=(
            "turbo_physai.optimizations.models.maptrv2.training."
            "ddp_forward_compat_wrapper"
        ),
    ),
    wrap(
        target="mmcv.runner.dist_utils.init_dist",
        aliases=("mmcv.runner.init_dist",),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2.training."
            "init_dist_fork_wrapper"
        ),
    ),
)

DATA = group(
    "maptrv2.data",
    replace(
        target=_BUILD_DATALOADER_API,
        aliases=(
            "projects.mmdet3d_plugin.bevformer.apis.mmdet_train.build_dataloader",
        ),
        replacement="turbo_physai.optimizations.models.maptrv2.data.build_dataloader",
    ),
)

GRID_MASK = group(
    "maptrv2.grid_mask",
    replace(
        target=_GRID_MASK_API,
        replacement="turbo_physai.optimizations.models.maptrv2.grid_mask.grid_mask_forward",
    ),
)

DATACONTAINER_PIN = group(
    "maptrv2.datacontainer_pin",
    wrap(
        target="torch.utils.data._utils.pin_memory.pin_memory",
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "datacontainer_pin.datacontainer_pin_wrapper"
        ),
    ),
)

FUSED_ADAMW = group(
    "maptrv2.fused_adamw",
    wrap(
        target="projects.mmdet3d_plugin.bevformer.apis.mmdet_train.build_optimizer",
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "optimizer.fused_adamw_builder_wrapper"
        ),
    ),
)

# The reference carries ten compile hooks on baseline symbols listed below.
# Its eight additional helper hooks are implemented by the
# ``maptrv2.reference_boundaries`` Group, which replaces the existing callers.
_COMPILE_TARGETS = (
    "projects.mmdet3d_plugin.maptr.modules.transformer."
    "MapTRPerceptionTransformer.format_feats",
    "projects.mmdet3d_plugin.maptr.modules.decoder.MapTRDecoder.forward",
    "projects.mmdet3d_plugin.maptr.detectors.maptrv2.MapTRv2.extract_img_feat",
    "projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.get_cam_feats",
    "projects.mmdet3d_plugin.maptr.modules.encoder.LSSTransform.get_mlp_input",
    "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.normalize_3d_pts",
    "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.normalize_2d_bbox",
    "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head.normalize_2d_pts",
    "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
    "denormalize_2d_bbox",
    "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
    "denormalize_2d_pts",
)

COMPILE = group(
    "maptrv2.compile",
    *(
        wrap(
            target=target,
            replacement="turbo_physai.optimizations.models.maptrv2.compile.compile_wrapper",
            runtime_condition=_COMPILE_CONDITION,
        )
        for target in _COMPILE_TARGETS
    ),
)

MATCH_COST = group(
    "maptrv2.match_cost",
    replace(
        target=(
            "projects.mmdet3d_plugin.maptr.losses.map_loss."
            "OrderedPtsL1Cost.__call__"
        ),
        replacement="turbo_physai.optimizations.models.maptrv2.match_cost.ordered_pts_l1_cost_call",
    ),
)

SYNC_ELIMINATION = group(
    "maptrv2.sync_elimination",
    replace(
        target=("projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
                "MapTRv2Head.loss_single"),
        replacement=("turbo_physai.optimizations.models.maptrv2."
                     "sync_elimination.maptrv2_head_loss_single"),
    ),
    replace(
        target=("projects.mmdet3d_plugin.maptr.dense_heads.maptr_head."
                "MapTRHead.loss_single"),
        replacement=("turbo_physai.optimizations.models.maptrv2."
                     "sync_elimination.maptr_head_loss_single"),
    ),
    replace(
        target=("projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head."
                "BEVFormerHead.loss_single"),
        replacement=("turbo_physai.optimizations.models.maptrv2."
                     "sync_elimination.bevformer_head_loss_single"),
    ),
    replace(
        target=("projects.mmdet3d_plugin.bevformer.modules.encoder."
                "BEVFormerEncoder.point_sampling"),
        replacement=("turbo_physai.optimizations.models.maptrv2."
                     "sync_elimination.bevformer_point_sampling"),
    ),
)

PV_MASK = group(
    "maptrv2.pv_mask",
    replace(
        target=(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.line_ego_to_pvmask"
        ),
        replacement="turbo_physai.optimizations.models.maptrv2.pv_mask.line_ego_to_pvmask",
        runtime_condition=_PV_MASK_CONDITION,
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.line_ego_to_mask"
        ),
        replacement="turbo_physai.optimizations.models.maptrv2.pv_mask.line_ego_to_mask",
        runtime_condition=_PV_MASK_CONDITION,
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.gen_vectorized_samples"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "pv_mask.gen_vectorized_samples"
        ),
        runtime_condition=_PV_MASK_CONDITION,
    ),
)

DATASET_VECTORIZATION = group(
    "maptrv2.dataset_vectorization",
    replace(
        target=(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "LiDARInstanceLines.shift_fixed_num_sampled_points_v2"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "dataset_vectorization.shift_fixed_num_sampled_points_v2"
        ),
    ),
)

ASSIGNER = group(
    "maptrv2.assigner",
    replace(
        target=_ASSIGN_API,
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "assigner.assign"
        ),
        runtime_condition=_STATIC_ASSIGNER_CONDITION,
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "MapTRv2Head.loss"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "assigner.static_loss"
        ),
        runtime_condition=_STATIC_ASSIGNER_CONDITION,
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "MapTRv2Head._get_target_single"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "assigner.get_target_single"
        ),
        runtime_condition=_STATIC_ASSIGNER_CONDITION,
    ),
)

EFFICIENTNET = group(
    "maptrv2.efficientnet",
    registry_override(
        module="projects.mmdet3d_plugin.models.backbones.efficientnet",
        registry="mmdet.models.builder.BACKBONES",
        names=("EfficientNet",),
    ),
)

BEV_POOL_FIX = group(
    "maptrv2.bev_pool_fix",
    replace(
        target=(
            "projects.mmdet3d_plugin.maptr.modules.encoder.BaseTransform.bev_pool"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2"
            ".bev_pool_fix.base_transform_bev_pool"
        ),
    ),
)

REFERENCE_BOUNDARIES = group(
    "maptrv2.reference_boundaries",
    wrap(
        target=(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "BaseTransform.get_geometry_v1"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "reference.base_transform_get_geometry_wrapper"
        ),
        runtime_condition=_COMPILE_CONDITION,
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "BaseTransform.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "reference.base_transform_forward_wrapper"
        ),
        runtime_condition=_COMPILE_CONDITION,
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "LSSTransform.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "reference.ls_transform_forward_wrapper"
        ),
        runtime_condition=_COMPILE_CONDITION,
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.maptr.modules.transformer."
            "MapTRPerceptionTransformer.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "reference.transformer_forward_wrapper"
        ),
        runtime_condition=_COMPILE_CONDITION,
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "MapTRv2Head.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "reference.head_forward_wrapper"
        ),
        runtime_condition=_COMPILE_CONDITION,
    ),
)

SPARSE_CONV_REGISTRY = group(
    "maptrv2.spconv_registry",
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

# ``tpfp.py`` binds the function with ``from .tpfp_chamfer import ...``, so the
# importing module's name is an alias the replacement must cover as well.
EVAL_COMPAT = group(
    "maptrv2.eval_compat",
    replace(
        target=_CHAMFER_SCORE_API,
        aliases=(
            "projects.mmdet3d_plugin.datasets.map_utils.tpfp.custom_polyline_score",
        ),
        replacement=(
            "turbo_physai.optimizations.models.maptrv2."
            "eval_compat.custom_polyline_score"
        ),
        runtime_condition=_EVAL_COMPAT_CONDITION,
    ),
)

__all__ = [
    "ASSIGNER",
    "BEV_POOL_FIX",
    "COMPILE",
    "DATA",
    "DATACONTAINER_PIN",
    "DATASET_VECTORIZATION",
    "EFFICIENTNET",
    "EVAL_COMPAT",
    "FUSED_ADAMW",
    "GRID_MASK",
    "MATCH_COST",
    "PV_MASK",
    "REFERENCE_BOUNDARIES",
    "SPARSE_CONV_REGISTRY",
    "SYNC_ELIMINATION",
    "TRAINING",
]
