"""Engine-managed PointPillars targets for the official MMDetection3D model."""
from turbo_physai import group, wrap

P = "turbo_physai.optimizations.models.PointPillars"

ASSIGNER = group(
    "pointpillars.assigner.tensorized_compiled",
    wrap(target="mmdet3d.models.task_modules.assigners.max_3d_iou_assigner.Max3DIoUAssigner.assign_wrt_overlaps",
         replacement=f"{P}.tensorized_assigner.tensorized_assign_wrt_overlaps_wrapper"),
)
SAMPLER = group(
    "pointpillars.sampler.unique_free",
    wrap(target="mmdet3d.models.task_modules.samplers.pseudosample.PseudoSampler.sample",
         replacement=f"{P}.unique_free_sampler.unique_free_sampler_wrapper"),
)
IOU = group(
    "pointpillars.iou.nearest_bev_compiled",
    wrap(target="mmdet3d.structures.ops.iou3d_calculator.BboxOverlapsNearest3D.__call__",
         replacement=f"{P}.compiled_nearest_bev_iou.compiled_nearest_bev_iou_wrapper"),
)
VOXEL = group(
    "pointpillars.voxelization.legacy_sorted",
    wrap(target="mmdet3d.models.data_preprocessors.voxelize.VoxelizationByGridShape.forward",
         replacement=f"{P}.voxelize_legacy_sorted.voxelization_layer_forward_wrapper",
         runtime_condition=f"{P}.voxelize_legacy_sorted.supports_legacy_sorted"),
)
SCATTER = group(
    "pointpillars.scatter.flattened_batch",
    wrap(target="mmdet3d.models.middle_encoders.pillar_scatter.PointPillarsScatter.forward_batch",
         replacement=f"{P}.pillar_scatter.flat_pillar_scatter_wrapper"),
)
ANCHOR = group(
    "pointpillars.anchor.grid_compile",
    wrap(target="mmdet3d.models.task_modules.anchor.anchor_3d_generator.Anchor3DRangeGenerator.grid_anchors",
         replacement=f"{P}.compiled_anchor_grid.compile_anchor_grid_wrapper"),
)
PERIOD = group(
    "pointpillars.geometry.limit_period",
    wrap(target="mmdet3d.structures.bbox_3d.utils.limit_period",
         aliases=("mmdet3d.structures.bbox_3d.limit_period",
                  "mmdet3d.structures.limit_period",
                  "mmdet3d.structures.bbox_3d.base_box3d.limit_period",
                  "mmdet3d.structures.bbox_3d.box_3d_mode.limit_period",
                  "mmdet3d.models.dense_heads.base_3d_dense_head.limit_period",
                  "mmdet3d.models.dense_heads.train_mixins.limit_period"),
         replacement=f"{P}.native_limit_period.native_limit_period_wrapper"),
)
COMPILE_MODULES = group(
    "pointpillars.framework.compile_point_modules",
    wrap(target="mmengine.runner.runner.Runner._maybe_compile",
         replacement=f"{P}.compile_model_modules.compile_pointpillars_modules_wrapper"),
)
BACKEND = group(
    "pointpillars.framework.cudnn_benchmark",
    wrap(target="mmengine.runner.runner.Runner.train",
         replacement=f"{P}.cudnn_benchmark.cudnn_benchmark_wrapper"),
)
DDP = group(
    "pointpillars.framework.disable_dynamo_ddp",
    wrap(target="mmengine.runner.loops.EpochBasedTrainLoop.run",
         replacement=f"{P}.optimize_ddp.disable_dynamo_ddp_wrapper"),
)
