"""Compare replacements against the pinned official implementation."""
from types import SimpleNamespace

import torch
from mmengine.structures import InstanceData
from mmdet.models.task_modules import AssignResult
from mmdet.models.task_modules.assigners.max_iou_assigner import MaxIoUAssigner
from mmdet3d.models.task_modules.samplers.pseudosample import PseudoSampler
from mmdet3d.structures.ops.iou3d_calculator import BboxOverlapsNearest3D
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter

from test_contract import load


def test_assigner_ties_and_thresholds():
    overlaps = torch.tensor([[.9, .15, .2, .31, .4],
                             [.9, .29, .21, .31, .25],
                             [.5, .1, .19, .29, .2]])
    labels = torch.tensor([2, 5, 7])
    assigner = SimpleNamespace(neg_iou_thr=.3, pos_iou_thr=.6,
                               min_pos_iou=.3, match_low_quality=True,
                               gt_max_assign_all=True)
    original = MaxIoUAssigner.assign_wrt_overlaps
    wrapper = load("tensorized_assigner").tensorized_assign_wrt_overlaps_wrapper
    expected = original(assigner, overlaps, labels)
    actual = wrapper(original, {})(assigner, overlaps, labels)
    for field in ("gt_inds", "max_overlaps", "labels"):
        assert torch.equal(getattr(actual, field), getattr(expected, field))


def test_sampler_parity():
    sampler = PseudoSampler()
    assignment = AssignResult(num_gts=2, gt_inds=torch.tensor([0, 1, 1, -1, 0]),
                              max_overlaps=torch.tensor([.1, .8, .7, 0., .2]),
                              labels=torch.tensor([-1, 2, 2, -1, -1]))
    pred = InstanceData(priors=torch.zeros(5, 7))
    gt = InstanceData(bboxes_3d=torch.ones(2, 7))
    wrapper = load("unique_free_sampler").unique_free_sampler_wrapper
    expected = sampler.sample(assignment, pred, gt)
    actual = wrapper(PseudoSampler.sample, {})(sampler, assignment, pred, gt)
    for field in ("pos_inds", "neg_inds", "pos_is_gt", "priors", "pos_gt_bboxes"):
        assert torch.equal(getattr(actual, field), getattr(expected, field))


def test_nearest_bev_iou_parity():
    boxes1 = torch.tensor([[0., 0., 0., 2., 4., 1., 0.],
                           [1., 1., 0., 3., 2., 1., .9]])
    boxes2 = torch.tensor([[.5, 0., 0., 2., 2., 1., .1],
                           [4., 4., 0., 1., 1., 1., -.8]])
    calculator = BboxOverlapsNearest3D(coordinate="lidar")
    wrapper = load("compiled_nearest_bev_iou").compiled_nearest_bev_iou_wrapper
    actual = wrapper(BboxOverlapsNearest3D.__call__, {})(calculator, boxes1, boxes2)
    assert torch.allclose(actual, calculator(boxes1, boxes2), rtol=1e-5, atol=1e-6)


def test_pillar_scatter_outputs_and_gradients():
    scatter = PointPillarsScatter(in_channels=2, output_shape=(3, 4))
    coors = torch.tensor([[0, 0, 0, 1], [1, 0, 2, 3], [0, 0, 1, 2]])
    a = torch.randn(3, 2, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    wrapped = load("pillar_scatter").flat_pillar_scatter_wrapper(
        PointPillarsScatter.forward_batch, {})
    expected = scatter.forward_batch(a, coors, 2)
    actual = wrapped(scatter, b, coors, 2)
    assert torch.equal(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert torch.equal(a.grad, b.grad)


def test_pillar_scatter_duplicate_falls_back():
    called = []
    scatter = PointPillarsScatter(in_channels=2, output_shape=(3, 4))
    coors = torch.tensor([[0, 0, 0, 1], [0, 0, 0, 1]])
    def original(*args):
        called.append(True)
        return "original"
    wrapped = load("pillar_scatter").flat_pillar_scatter_wrapper(original, {})
    assert wrapped(scatter, torch.randn(2, 2), coors, 1) == "original"
    assert called == [True]
