"""sparse4d optimization declarations.

Add model-specific Groups here. Importing this module registers the declarations
with TurboPhysAI; the generated OptimizationConfig loads it through optimization_modules.
"""

from turbo_physai import group, replace, wrap  # noqa: F401
from turbo_physai.compatibility import optional_import, registry_override  # noqa: F401


# Let Sparse4D import ops before the deformable aggregation callable is replaced.
DAF_IMPORT_COMPATIBILITY = group(
    "sparse4d.daf_import_compatibility",
    optional_import("projects.mmdet3d_plugin.ops.deformable_aggregation_ext"),
)


# Use FusedAdam optimizer for Sparse4D.
FUSED_ADAM = group(
    "sparse4d.fused_adam",
    registry_override(
        module="turbo_physai.optimizations.models.sparse4d.fused_adam",
        registry="mmcv.runner.OPTIMIZERS",
        names=("FusedAdam",),
    ),
)

# Channels-last image features and cached spatial metadata for Sparse4D.
CHANNELS_LAST_AND_CACHED_SPATIAL_SHAPE = group(
    "sparse4d.channels_last_and_cached_spatial_shape",
    wrap(
        target="projects.mmdet3d_plugin.models.sparse4d.Sparse4D.__init__",
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.channels_last."
            "channels_last_init_wrapper"
        ),
    ),
    replace(
        target="projects.mmdet3d_plugin.models.sparse4d.Sparse4D.extract_feat",
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.channels_last."
            "extract_feat"
        ),
    ),
    wrap(
        target="projects.mmdet3d_plugin.ops.feature_maps_format",
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.channels_last."
            "feature_maps_format_wrapper"
        ),
    ),
)

# Configure only the DataLoader constructor used by the Sparse4D builder.
DATALOADER = group(
    "sparse4d.dataloader",
    wrap(
        target="projects.mmdet3d_plugin.datasets.builder.DataLoader",
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.dataloader.dataloader_wrapper"
        ),
    ),
)

# Reduce Detection3D H2D syncs and compile the box encoder and refinement.
DETECTION3D_BLOCKS = group(
    "sparse4d.detection3d_blocks",
    wrap(
        target=(
            "projects.mmdet3d_plugin.models.detection3d.detection3d_blocks."
            "SparseBox3DEncoder.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.detection3d_blocks."
            "box_encoder_forward_wrapper"
        ),
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.models.detection3d.detection3d_blocks."
            "SparseBox3DKeyPointsGenerator.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.detection3d_blocks."
            "point_generator_forward"
        ),
    ),
    replace(
        target=(
            "projects.mmdet3d_plugin.models.detection3d.detection3d_blocks."
            "SparseBox3DKeyPointsGenerator.anchor_projection"
        ),
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.detection3d_blocks."
            "anchor_projection"
        ),
    ),
    wrap(
        target=(
            "projects.mmdet3d_plugin.models.detection3d.detection3d_blocks."
            "SparseBox3DRefinementModule.forward"
        ),
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.detection3d_blocks."
            "box_refinement_forward_wrapper"
        ),
    ),
)

# Use TurboPhysAI's native deformable aggregation kernel.
DEFORMABLE_AGGREGATION = group(
    "sparse4d.deformable_aggregation",
    replace(
        target="projects.mmdet3d_plugin.ops.deformable_aggregation_function",
        aliases=("projects.mmdet3d_plugin.models.blocks.DAF",),
        replacement=(
            "turbo_physai.operators.deformable_aggregation_function"
        ),
    ),
)

# Rewrite SparseBox3DTarget: flatten the per-sample Python loops in
# encode_reg_target/_cls_cost/_box_cost into batched tensor ops, swap the
# CPU-bound scipy Hungarian assignment in get_dn_anchors for batched CUDA
# linear assignment.
DETECTION3D_TARGET = group(
    "sparse4d.detection3d_target",
    replace(
        target=(
            "projects.mmdet3d_plugin.models.detection3d.target."
            "BBOX_SAMPLERS.module_dict.SparseBox3DTarget"
        ),
        replacement="turbo_physai.optimizations.models.sparse4d.target.SparseBox3DTarget",
    ),
)

# Resolve optimized methods during preparation, before the registry entry is replaced.
DETECTION3D_TARGET_COMPILE = group(
    "sparse4d.detection3d_target_compile",
    *(
        wrap(
            target=(
                "turbo_physai.optimizations.models.sparse4d.target."
                f"SparseBox3DTarget.{method}"
            ),
            replacement=(
                "turbo_physai.optimizations.models.sparse4d.compile."
                "compile_target_wrapper"
            ),
        )
        for method in (
            "_encode_flat",
            "encode_reg_target_fix",
            "_compute_cls_diff",
            "_box_cost_flat",
            "_assign_and_scatter",
            "_box_cost_fix",
            "cache_dn",
        )
    ),
    depends_on=("sparse4d.detection3d_target",),
)

# Optimize Sparse4d head forward H2D
SPARSE4D_HEAD = group(
    "sparse4d.sparse4d_head",
    replace(
        target="projects.mmdet3d_plugin.models.sparse4d_head.Sparse4DHead.forward",
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.sparse4d_head.forward"
        ),
    ),
    replace(
        target="projects.mmdet3d_plugin.models.blocks.DeformableFeatureAggregation._get_weights",
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.sparse4d_head._get_weights"
        ),
    ),
    depends_on=("sparse4d.detection3d_target",),
)

GRID_MASK_H2D = group(
    "sparse4d.grid_mask_h2d",
    replace(
        target="projects.mmdet3d_plugin.models.grid_mask.GridMask.forward",
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.grid_mask.forward"
        ),
    ),
)

INSTANCE_BANK_H2D = group(
    "sparse4d.instance_bank_h2d",
    replace(
        target="projects.mmdet3d_plugin.models.instance_bank.InstanceBank.get",
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.instance_bank.get"
        ),
    ),
)

# Compile the image backbone (ResNet) and neck (FPN) as whole modules.
COMPILE_BACKBONE = group(
    "sparse4d.compile",
    wrap(
        target="mmdet.models.backbones.resnet.ResNet",
        aliases=("mmdet.models.backbones.ResNet",),
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.compile.compile_class_wrapper"
        ),
    ),
    wrap(
        target="mmdet.models.necks.fpn.FPN",
        aliases=("mmdet.models.necks.FPN", "mmdet.models.FPN"),
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.compile.compile_class_wrapper"
        ),
    ),
    wrap(
        target="projects.mmdet3d_plugin.models.blocks.AsymmetricFFN.forward",
        replacement=(
            "turbo_physai.optimizations.models.sparse4d.compile.compile_wrapper"
        ),
    ),
)
