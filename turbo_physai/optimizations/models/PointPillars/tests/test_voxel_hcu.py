"""HCU parity check; run only on an explicitly selected idle device."""
import pytest
import torch
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape

from test_contract import load


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.hip,
                    reason="requires an HCU device")
@pytest.mark.parametrize("max_voxels", [2, 20])
def test_hard_voxel_matches_official(max_voxels):
    points = torch.tensor([[2.1, 1.1, .1, 1], [.1, .1, .1, 2],
                           [2.2, 1.1, .1, 3], [1.1, 1.1, .1, 4],
                           [2.3, 1.1, .1, 5], [9., 0., 0., 6]], device="cuda")
    layer = VoxelizationByGridShape(point_cloud_range=[0., 0., 0., 4., 4., 2.],
                                    max_num_points=2, voxel_size=[1., 1., 1.],
                                    max_voxels=max_voxels, deterministic=True)
    module = load("voxelize_legacy_sorted")
    assert module.supports_legacy_sorted(layer, points)
    actual = module.voxelization_layer_forward_wrapper(
        VoxelizationByGridShape.forward, {})(layer, input=points)
    expected = layer(points)
    for x, y in zip(actual, expected):
        assert torch.equal(x, y), (x, y)
