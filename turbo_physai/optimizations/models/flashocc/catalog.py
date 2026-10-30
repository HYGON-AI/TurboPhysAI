# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""External TurboPhysAI optimization groups for official FlashOCC."""

from turbo_physai import group, wrap


DISABLE_ACTIVATION_CHECKPOINT = group(
    "flashocc.activation_checkpoint.disable",
    wrap(
        target=(
            "projects.mmdet3d_plugin.models.detectors.bevdet."
            "BEVDet.__init__"
        ),
        replacement=(
            f"{__package__}.disable_activation_checkpoint."
            "disable_activation_checkpoint_wrapper"
        ),
    ),
)


COMPILE_IMAGE_ENCODER = group(
    "flashocc.compile.image_encoder",
    wrap(
        target=(
            "projects.mmdet3d_plugin.models.detectors.bevdet."
            "BEVDet.image_encoder"
        ),
        replacement=(
            f"{__package__}.compile_image_encoder."
            "compile_image_encoder_wrapper"
        ),
    ),
)

COMPILE_BEV_ENCODER = group(
    "flashocc.compile.bev_encoder",
    wrap(
        target=(
            "projects.mmdet3d_plugin.models.detectors.bevdet."
            "BEVDet.bev_encoder"
        ),
        replacement=(
            f"{__package__}.compile_bev_encoder."
            "compile_bev_encoder_wrapper"
        ),
    ),
)

COMPILE_PREPARE_INPUTS = group(
    "flashocc.compile.prepare_inputs",
    wrap(
        target=(
            "projects.mmdet3d_plugin.models.detectors.bevdet."
            "BEVDet.prepare_inputs"
        ),
        replacement=(
            f"{__package__}.compile_prepare_inputs."
            "compile_prepare_inputs_wrapper"
        ),
    ),
)

COMPILE_OCC_TRAIN = group(
    "flashocc.compile.occ_train",
    wrap(
        target=(
            "projects.mmdet3d_plugin.models.detectors.bevdet_occ."
            "BEVDetOCC.forward_occ_train"
        ),
        replacement=(
            f"{__package__}.compile_occ_train."
            "compile_occ_train_wrapper"
        ),
    ),
)

PIN_MEMORY = group(
    "flashocc.loader.pin_memory",
    wrap(
        target="mmdet.datasets.builder.build_dataloader",
        aliases=(
            "mmdet.datasets.build_dataloader",
            "mmdet.apis.train.build_dataloader",
            "mmdet3d.apis.train.build_mmdet_dataloader",
        ),
        replacement=f"{__package__}.pin_memory.pin_memory_wrapper",
    ),
)

PERSISTENT_WORKERS = group(
    "flashocc.loader.persistent_workers",
    wrap(
        target="torch.utils.data.DataLoader.__init__",
        replacement=(
            f"{__package__}.persistent_workers."
            "persistent_workers_wrapper"
        ),
    ),
)

VIEW_TRANSFORMER_PREPARATION = group(
    "flashocc.view_transformer.accelerated_preparation",
    wrap(
        target=(
            "projects.mmdet3d_plugin.models.necks.view_transformer."
            "LSSViewTransformer.__init__"
        ),
        replacement=(
            f"{__package__}.device_buffers."
            "register_device_buffers_wrapper"
        ),
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.models.necks.view_transformer."
            "LSSViewTransformer.get_ego_coor"
        ),
        replacement=(
            f"{__package__}.geometry_compile."
            "get_ego_coor_wrapper"
        ),
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.models.necks.view_transformer."
            "LSSViewTransformer.voxel_pooling_prepare_v2"
        ),
        replacement=(
            f"{__package__}.voxel_pooling_compile."
            "voxel_pooling_prepare_wrapper"
        ),
    ),
)
