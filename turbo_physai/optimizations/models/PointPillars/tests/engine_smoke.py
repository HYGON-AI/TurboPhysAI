"""Run as an entry script under turbo-physai run to verify target patching."""
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
from mmdet3d.models.task_modules.assigners.max_3d_iou_assigner import Max3DIoUAssigner
from mmdet3d.models.task_modules.samplers.pseudosample import PseudoSampler
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter
from mmdet3d.models.task_modules.anchor.anchor_3d_generator import Anchor3DRangeGenerator
from mmdet3d.structures.ops.iou3d_calculator import BboxOverlapsNearest3D
from mmdet3d.structures.bbox_3d import utils
from mmengine.runner.runner import Runner
from mmengine.runner.loops import EpochBasedTrainLoop

targets = [VoxelizationByGridShape.forward, Max3DIoUAssigner.assign_wrt_overlaps,
           PseudoSampler.sample, PointPillarsScatter.forward_batch,
           Anchor3DRangeGenerator.grid_anchors, BboxOverlapsNearest3D.__call__,
           utils.limit_period, Runner._maybe_compile, Runner.train,
           EpochBasedTrainLoop.run]
for target in targets:
    assert hasattr(target, "__wrapped__"), f"not wrapped: {target}"
print(f"verified {len(targets)} wrapped targets")
