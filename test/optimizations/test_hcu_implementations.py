# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Real-HCU correctness tests without BEVFormer or BEVFusion repositories."""

from __future__ import annotations

import sys
import types

import pytest


torch = pytest.importorskip("torch")

from test.reference_operators import multi_scale_deformable_attn_reference
from test.reference_operators import modulated_deform_conv2d_reference


pytestmark = pytest.mark.hcu


def _require_hcu():
    if not torch.cuda.is_available():
        pytest.skip("a real HCU device is required")
    # This import must fail, rather than skip, when the formal HCU environment
    # has not built the bundled extension.
    import turbo_physai.ops  # noqa: F401


def _msda_inputs(device):
    torch.manual_seed(17)
    shapes = torch.tensor([[3, 4], [2, 2]], dtype=torch.long, device=device)
    starts = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
    value = (torch.rand(1, 16, 2, 4, device=device) * 0.01).requires_grad_()
    locations = torch.rand(1, 5, 2, 2, 2, 2, device=device, requires_grad=True)
    raw_weights = torch.rand(1, 5, 2, 2, 2, device=device, requires_grad=True)
    weights = raw_weights.softmax(dim=-1)
    return shapes, starts, value, locations, raw_weights, weights


def test_common_lightop_msda_forward_and_backward_match_reference():
    _require_hcu()
    from turbo_physai.optimizations.common.mmcv import msda

    cpu = _msda_inputs("cpu")
    shapes, starts, value, locations, raw_weights, weights = cpu
    reference = multi_scale_deformable_attn_reference(
        value, shapes, locations, weights
    )
    grad_output = torch.linspace(0.1, 1.0, reference.numel()).reshape_as(reference)
    expected_grads = torch.autograd.grad(
        reference, (value, locations, raw_weights), grad_output
    )

    hcu_shapes = shapes.cuda()
    hcu_starts = starts.cuda()
    hcu_value = value.detach().cuda().requires_grad_()
    hcu_locations = locations.detach().cuda().requires_grad_()
    hcu_raw_weights = raw_weights.detach().cuda().requires_grad_()
    hcu_weights = hcu_raw_weights.softmax(dim=-1)
    actual = msda.ms_deform_attn_forward(
        hcu_value,
        hcu_shapes,
        hcu_starts,
        hcu_locations,
        hcu_weights,
        64,
    )
    grad_value = torch.zeros_like(hcu_value)
    grad_locations = torch.zeros_like(hcu_locations)
    grad_weights = torch.zeros_like(hcu_weights)
    msda.ms_deform_attn_backward(
        hcu_value,
        hcu_shapes,
        hcu_starts,
        hcu_locations,
        hcu_weights,
        grad_output.cuda(),
        grad_value,
        grad_locations,
        grad_weights,
        64,
    )
    actual_raw_weight_grad = torch.autograd.grad(
        hcu_weights,
        hcu_raw_weights,
        grad_weights,
    )[0]

    torch.testing.assert_close(actual.cpu(), reference.detach(), rtol=1e-3, atol=1e-4)
    for actual_grad, expected_grad in zip(
        (grad_value, grad_locations, actual_raw_weight_grad), expected_grads
    ):
        torch.testing.assert_close(
            actual_grad.cpu(), expected_grad, rtol=2e-3, atol=2e-4
        )


def test_bevformer_lightop_msda_autograd_matches_reference():
    _require_hcu()
    from turbo_physai.optimizations.models.bevformer import msda

    cpu = _msda_inputs("cpu")
    shapes, starts, value, locations, raw_weights, weights = cpu
    reference = multi_scale_deformable_attn_reference(
        value, shapes, locations, weights
    )
    grad_output = torch.linspace(0.1, 1.0, reference.numel()).reshape_as(reference)
    expected_grads = torch.autograd.grad(
        reference, (value, locations, raw_weights), grad_output
    )

    hcu_shapes = shapes.cuda()
    hcu_starts = starts.cuda()
    hcu_value = value.detach().cuda().requires_grad_()
    hcu_locations = locations.detach().cuda().requires_grad_()
    hcu_raw_weights = raw_weights.detach().cuda().requires_grad_()
    hcu_weights = hcu_raw_weights.softmax(dim=-1)
    actual = msda.MultiScaleDeformableAttnFunction_fp32.apply(
        hcu_value,
        hcu_shapes,
        hcu_starts,
        hcu_locations,
        hcu_weights,
        64,
    )
    actual_grads = torch.autograd.grad(
        actual,
        (hcu_value, hcu_locations, hcu_raw_weights),
        grad_output.cuda(),
    )
    torch.testing.assert_close(actual.cpu(), reference.detach(), rtol=1e-3, atol=1e-4)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad.cpu(), expected_grad, rtol=2e-3, atol=2e-4
        )


def test_bev_pool_forward_backward_and_prepare_match_independent_golden():
    _require_hcu()
    from turbo_physai.optimizations.common.mmdet3d import bev_pool

    features = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        device="cuda",
        requires_grad=True,
    )
    coords = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    actual = bev_pool.bev_pool(features, coords, 1, 1, 2, 2)
    expected = torch.zeros(1, 2, 1, 2, 2)
    for row, coord in zip(features.detach().cpu(), coords.cpu()):
        x, y, z, batch = coord.tolist()
        expected[batch, :, z, x, y] += row
    torch.testing.assert_close(actual.cpu(), expected)
    grad_output = torch.arange(1.0, actual.numel() + 1).reshape_as(actual)
    actual.backward(grad_output.cuda())
    expected_grad = torch.stack(
        [grad_output[batch, :, z, x, y] for x, y, z, batch in coords.cpu().tolist()]
    )
    torch.testing.assert_close(features.grad.cpu(), expected_grad)

    geometry = torch.tensor(
        [[[[[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [4.0, 0.0, 0.0]]]]]],
        device="cuda",
    )
    bx = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    dx = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    nx = torch.tensor([2, 2, 1], dtype=torch.long, device="cuda")
    prepared_coords, ranks, kept = bev_pool.bev_pool_prepare(
        geometry, bx, dx, nx, 1, 1, 2, 2
    )
    assert prepared_coords.cpu().tolist() == [
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [4, 0, 0, 0],
    ]
    assert ranks.cpu().tolist() == [0, 3, -1]
    assert kept.cpu().tolist() == [True, True, False]


def test_quick_cumsum_forward_and_backward_match_segment_sums():
    _require_hcu()
    from turbo_physai.optimizations.common.mmdet3d import bev_pool

    class QuickCumsum(torch.autograd.Function):
        forward = staticmethod(bev_pool.quick_cumsum_forward)
        backward = staticmethod(bev_pool.quick_cumsum_backward)

    features = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        device="cuda",
        requires_grad=True,
    )
    geometry = torch.tensor(
        [[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [2, 1, 0, 0]],
        dtype=torch.int32,
        device="cuda",
    )
    ranks = torch.tensor([0, 0, 3, 3, 3], dtype=torch.int64, device="cuda")

    actual, actual_geometry = QuickCumsum.apply(features, geometry, ranks)
    expected = torch.tensor([[4.0, 6.0], [21.0, 24.0]])
    torch.testing.assert_close(actual.cpu(), expected)
    assert actual_geometry.cpu().tolist() == [[1, 0, 0, 0], [2, 1, 0, 0]]

    grad_output = torch.tensor([[0.5, 1.0], [2.0, 3.0]], device="cuda")
    actual.backward(grad_output)
    expected_grad = torch.tensor(
        [[0.5, 1.0], [0.5, 1.0], [2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]
    )
    torch.testing.assert_close(features.grad.cpu(), expected_grad)


def test_bev_pool_prepare_geometry_matches_identity_transform_golden():
    _require_hcu()
    from turbo_physai.optimizations.common.mmdet3d import bev_pool

    frustum = torch.tensor(
        [[[[1.0, 1.0, 0.25], [1.0, 0.25, 1.0], [8.8, 1.0, 0.25]]]],
        device="cuda",
    )
    matrix = torch.eye(3, device="cuda").reshape(1, 1, 3, 3)
    translation = torch.zeros(1, 1, 3, device="cuda")
    extra_matrix = torch.eye(3, device="cuda").reshape(1, 3, 3)
    extra_translation = torch.zeros(1, 3, device="cuda")
    bx = torch.tensor([0.5, 0.5, 0.5], device="cuda")
    dx = torch.ones(3, device="cuda")
    nx = torch.tensor([2, 2, 2], dtype=torch.long, device="cuda")

    coords, ranks, kept, boundary = bev_pool.bev_pool_prepare_geometry(
        frustum,
        matrix,
        translation,
        matrix,
        translation,
        extra_matrix,
        extra_translation,
        bx,
        dx,
        nx,
        1,
        2,
        2,
        2,
        boundary_eps=1.0e-2,
    )
    assert coords.cpu().tolist() == [
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [2, 0, 0, 0],
    ]
    assert ranks.cpu().tolist() == [0, 5, -1]
    assert kept.cpu().tolist() == [True, True, False]
    assert boundary.cpu().tolist() == [False, True, False]


def test_gaussian_2d_runs_on_hcu_and_matches_reference_formula():
    _require_hcu()
    from turbo_physai.optimizations.common.mmdet3d.gaussian import gaussian_2d

    actual = gaussian_2d((4, 6), sigma=1.25, device="cuda", dtype=torch.float32)
    y = torch.arange(-1.5, 2.0, dtype=torch.float32).unsqueeze(1)
    x = torch.arange(-2.5, 3.0, dtype=torch.float32).unsqueeze(0)
    expected = torch.exp(-(x.square() + y.square()) / (2 * 1.25**2))
    expected = torch.where(
        expected < torch.finfo(expected.dtype).eps * expected.max(), 0, expected
    )
    assert actual.is_cuda
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)


def test_dynamic_and_hard_voxelization_match_coordinate_golden():
    _require_hcu()
    from turbo_physai.optimizations.common.mmdet3d import voxelization

    points = torch.tensor(
        [
            [0.1, 0.1, 0.1, 1.0],
            [0.2, 0.3, 0.4, 2.0],
            [1.1, 0.1, 0.1, 3.0],
            [-0.1, 0.0, 0.0, 4.0],
            [2.0, 0.0, 0.0, 5.0],
        ],
        device="cuda",
    )
    voxel_size = [1.0, 1.0, 1.0]
    coordinate_range = [0.0, 0.0, 0.0, 2.0, 2.0, 2.0]
    dynamic = voxelization.voxelization_forward(
        None, points, voxel_size, coordinate_range, -1, -1, True
    )
    dynamic_cpu = dynamic.cpu()
    assert dynamic_cpu[:3].tolist() == [
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
    ]
    assert dynamic_cpu[3:, 0].tolist() == [-1, -1]

    voxels, coords, counts = voxelization.voxelization_forward(
        None, points, voxel_size, coordinate_range, 2, 4, True
    )
    by_coord = {
        tuple(coord): (voxel[: int(count)].cpu(), int(count))
        for voxel, coord, count in zip(voxels, coords.cpu().tolist(), counts.cpu())
    }
    assert set(by_coord) == {(0, 0, 0), (1, 0, 0)}
    torch.testing.assert_close(
        by_coord[(0, 0, 0)][0], points[:2].cpu()
    )
    torch.testing.assert_close(
        by_coord[(1, 0, 0)][0], points[2:3].cpu()
    )
    assert by_coord[(0, 0, 0)][1] == 2
    assert by_coord[(1, 0, 0)][1] == 1


def test_hard_voxelization_enforces_limits_and_supports_nondeterministic_mode():
    _require_hcu()
    from turbo_physai.optimizations.common.mmdet3d import voxelization

    points = torch.tensor(
        [
            [0.1, 0.1, 0.1, 1.0],
            [0.2, 0.2, 0.2, 2.0],
            [0.3, 0.3, 0.3, 3.0],
            [1.1, 0.1, 0.1, 4.0],
            [2.1, 0.1, 0.1, 5.0],
        ],
        device="cuda",
    )
    voxel_size = [1.0, 1.0, 1.0]
    coordinate_range = [0.0, 0.0, 0.0, 3.0, 2.0, 2.0]

    voxels, coords, counts = voxelization.voxelization_forward(
        None, points, voxel_size, coordinate_range, 2, 2, True
    )
    assert len(coords) == 2
    by_coord = {
        tuple(coord): (voxel[: int(count)].cpu(), int(count))
        for voxel, coord, count in zip(voxels, coords.cpu().tolist(), counts.cpu())
    }
    assert set(by_coord) == {(0, 0, 0), (1, 0, 0)}
    assert by_coord[(0, 0, 0)][1] == 2
    torch.testing.assert_close(by_coord[(0, 0, 0)][0], points[:2].cpu())
    torch.testing.assert_close(by_coord[(1, 0, 0)][0], points[3:4].cpu())

    unique_points = points[[0, 3, 4]]
    voxels, coords, counts = voxelization.voxelization_forward(
        None, unique_points, voxel_size, coordinate_range, 2, 3, False
    )
    by_coord = {
        tuple(coord): voxel[0].cpu()
        for voxel, coord, count in zip(voxels, coords.cpu().tolist(), counts.cpu())
        if int(count) == 1
    }
    assert set(by_coord) == {(0, 0, 0), (1, 0, 0), (2, 0, 0)}
    for point in unique_points.cpu():
        coord = (int(point[0]), int(point[1]), int(point[2]))
        torch.testing.assert_close(by_coord[coord], point)


def _assert_identity_indice_mapping(result, input_indices):
    output_indices, pairs, counts = result
    point_count = len(input_indices)
    assert counts.cpu().tolist() == [point_count]
    input_ids = pairs[0, 0, :point_count].cpu().tolist()
    output_ids = pairs[0, 1, :point_count].cpu().tolist()
    assert sorted(input_ids) == list(range(point_count))
    assert sorted(output_ids) == list(range(point_count))
    for input_id, output_id in zip(input_ids, output_ids):
        torch.testing.assert_close(
            output_indices[output_id].cpu(), input_indices[input_id].cpu()
        )


def test_submanifold_indice_pairs_preserve_center_mapping():
    _require_hcu()
    from turbo_physai.optimizations.common.mmdet3d import sparse_conv

    indices = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 1, 1], [0, 2, 2, 2]],
        dtype=torch.int32,
        device="cuda",
    )
    output_indices, pairs, counts = sparse_conv.get_indice_pairs(
        indices,
        1,
        [4, 4, 4],
        ksize=3,
        stride=1,
        padding=1,
        dilation=1,
        subm=True,
    )
    torch.testing.assert_close(output_indices.cpu(), indices.cpu())
    center = 13
    assert int(counts[center]) == len(indices)
    expected = torch.arange(len(indices), dtype=torch.int32)
    torch.testing.assert_close(pairs[center, 0, : len(indices)].cpu(), expected)
    torch.testing.assert_close(pairs[center, 1, : len(indices)].cpu(), expected)


@pytest.mark.parametrize(
    ("spatial_shape", "indices", "transpose"),
    [
        ([4, 4], [[0, 0, 0], [0, 1, 2], [0, 3, 3]], False),
        (
            [3, 3, 3],
            [[0, 0, 0, 0], [0, 1, 1, 2], [0, 2, 2, 2]],
            True,
        ),
        (
            [3, 3, 3, 3],
            [[0, 0, 0, 0, 0], [0, 1, 1, 1, 2], [0, 2, 2, 2, 2]],
            False,
        ),
    ],
)
def test_sparse_convolution_indice_pairs_cover_supported_ranks_and_transpose(
    spatial_shape, indices, transpose
):
    _require_hcu()
    from turbo_physai.optimizations.common.mmdet3d import sparse_conv

    input_indices = torch.tensor(indices, dtype=torch.int32, device="cuda")
    result = sparse_conv.get_indice_pairs(
        input_indices,
        1,
        spatial_shape,
        ksize=1,
        stride=1,
        padding=0,
        dilation=1,
        transpose=transpose,
    )
    _assert_identity_indice_mapping(result, input_indices)


@pytest.mark.parametrize(
    ("spatial_shape", "indices"),
    [
        ([4, 4], [[0, 0, 0], [0, 1, 2], [0, 3, 3]]),
        ([3, 3, 3], [[0, 0, 0, 0], [0, 1, 1, 2], [0, 2, 2, 2]]),
    ],
)
def test_sparse_convolution_grid_path_matches_identity_mapping(
    spatial_shape, indices
):
    _require_hcu()
    from turbo_physai.optimizations.common.mmdet3d import sparse_conv

    input_indices = torch.tensor(indices, dtype=torch.int32, device="cuda")
    spatial_volume = 1
    for dimension in spatial_shape:
        spatial_volume *= dimension
    grid = torch.full(
        (spatial_volume,), -1, dtype=torch.int32, device="cuda"
    )
    result = sparse_conv.get_indice_pairs(
        input_indices,
        1,
        spatial_shape,
        ksize=1,
        stride=1,
        padding=0,
        dilation=1,
        grid=grid,
    )
    _assert_identity_indice_mapping(result, input_indices)


def _install_mmcv_stubs(monkeypatch):
    mmcv = types.ModuleType("mmcv")
    utils = types.ModuleType("mmcv.utils")
    cnn = types.ModuleType("mmcv.cnn")

    class Registry:
        def register_module(self, *args, **kwargs):
            del args, kwargs
            return lambda cls: cls

    def deprecated_api_warning(*args, **kwargs):
        del args, kwargs
        return lambda function: function

    utils.deprecated_api_warning = deprecated_api_warning
    utils.print_log = lambda *args, **kwargs: None
    utils.ext_loader = types.SimpleNamespace(load_ext=lambda *args, **kwargs: object())
    cnn.CONV_LAYERS = Registry()
    mmcv.utils = utils
    mmcv.cnn = cnn
    monkeypatch.setitem(sys.modules, "mmcv", mmcv)
    monkeypatch.setitem(sys.modules, "mmcv.utils", utils)
    monkeypatch.setitem(sys.modules, "mmcv.cnn", cnn)


def test_bevformer_mdc_matches_reference_for_supported_gradients(monkeypatch):
    _require_hcu()
    _install_mmcv_stubs(monkeypatch)
    from turbo_physai.optimizations.models.bevformer import mdc

    torch.manual_seed(29)
    input = torch.randn(1, 4, 5, 6, requires_grad=True)
    weight = (torch.randn(4, 4, 3, 3) * 0.01).requires_grad_()
    offset = (torch.rand(1, 18, 5, 6) - 0.5).requires_grad_()
    mask = torch.sigmoid(torch.randn(1, 9, 5, 6)).requires_grad_()
    expected = modulated_deform_conv2d_reference(
        input, offset, mask, weight, padding=1
    )
    grad_output = torch.linspace(0.1, 1.0, expected.numel()).reshape_as(expected)
    expected_input_grad, expected_weight_grad = torch.autograd.grad(
        expected, (input, weight), grad_output
    )

    hcu_input = input.detach().to(
        memory_format=torch.channels_last
    ).cuda().requires_grad_()
    hcu_weight = weight.detach().to(
        memory_format=torch.channels_last
    ).cuda().requires_grad_()
    hcu_offset = offset.detach().to(
        memory_format=torch.channels_last
    ).cuda().requires_grad_()
    hcu_mask = mask.detach().to(
        memory_format=torch.channels_last
    ).cuda().requires_grad_()
    actual = mdc.modulated_deform_conv2d(
        hcu_input, hcu_offset, hcu_mask, hcu_weight, None, 1, 1, 1, 1, 1
    )
    actual.backward(grad_output.cuda())
    torch.testing.assert_close(actual.cpu(), expected.detach(), rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        hcu_input.grad.cpu(), expected_input_grad, rtol=1e-3, atol=1e-3
    )
    torch.testing.assert_close(
        hcu_weight.grad.cpu(), expected_weight_grad, rtol=1e-1, atol=1e-1
    )
    assert hcu_offset.grad is None
    assert hcu_mask.grad is None
