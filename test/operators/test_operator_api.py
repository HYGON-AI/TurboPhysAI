# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Public imports and native-extension wiring without HCU dependencies."""

import subprocess
import sys
import textwrap
import unittest


class OperatorApiTest(unittest.TestCase):
    def run_script(self, script):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_namespace_and_cli_do_not_load_backends(self):
        self.run_script(
            """
            import importlib.abc
            import sys

            class BlockBackends(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in {'torch', 'lightop', 'hipdnn', 'mmcv', 'mmdet3d'} or fullname in {'turbo_physai.ops', 'turbo_physai._C'}:
                        raise AssertionError(fullname)

            sys.meta_path.insert(0, BlockBackends())
            import turbo_physai
            from turbo_physai import operators
            import turbo_physai.cli
            from turbo_physai.optimizations.common.mmdet3d import bev_pool, voxelization, sparse_conv

            assert callable(turbo_physai.apply)
            assert operators is turbo_physai.operators
            for name in operators.__all__:
                assert name in dir(operators)
                assert not hasattr(turbo_physai, name), name
            assert not hasattr(operators, 'unknown_operator')
            assert 'torch' not in sys.modules
            """
        )

    def test_public_exports_resolve_with_only_backend_stubs(self):
        self.run_script(
            """
            import sys
            import types
            import torch

            extension = types.ModuleType('turbo_physai._C')
            def unused(*args, **kwargs):
                raise AssertionError('unexpected operator execution')
            for name in ('grid_sample_forward', 'grid_sample_backward',
                         'upsample_bilinear_2d_forward', 'upsample_bilinear_2d_backward',
                         'deformable_aggregation_forward', 'deformable_aggregation_backward'):
                setattr(extension, name, unused)
            sys.modules[extension.__name__] = extension
            sys.modules['hipdnn'] = types.ModuleType('hipdnn')
            lightop = types.ModuleType('lightop')
            lightop.op = types.SimpleNamespace(ms_deform_attn_forward=unused, ms_deform_attn_backward=unused)
            sys.modules['lightop'] = lightop

            from turbo_physai import operators
            for name in operators.__all__:
                exported = getattr(operators, name)
                assert callable(exported), name
                assert getattr(operators, name) is exported
            assert 'turbo_physai.ops' not in sys.modules
            """
        )

    def test_grid_sample_import_order_and_native_forward_backward(self):
        self.run_script(
            """
            import importlib
            import sys
            import types
            import warnings
            import torch

            calls = []
            extension = types.ModuleType('turbo_physai._C')
            def forward(x, grid, mode, padding, align_corners):
                calls.append(('forward', mode, padding, align_corners))
                return torch.full_like(x, 2.)
            def backward(grad, x, grid, mode, padding, align_corners, mask):
                calls.append(('backward', mode, padding, align_corners, mask))
                return torch.full_like(x, 3.), torch.full_like(grid, 4.)
            extension.grid_sample_forward = forward
            extension.grid_sample_backward = backward
            sys.modules[extension.__name__] = extension

            implementation = importlib.import_module('turbo_physai.operators._grid_sample')
            from turbo_physai.operators import grid_sample
            assert grid_sample is implementation.grid_sample
            x = torch.ones(1, 1, 2, 2, requires_grad=True)
            grid = torch.zeros(1, 2, 2, 2, requires_grad=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                output = grid_sample(x, grid)
            assert caught and 'align_corners=False' in str(caught[0].message)
            output.sum().backward()
            torch.testing.assert_close(output, torch.full_like(x, 2.))
            torch.testing.assert_close(x.grad, torch.full_like(x, 3.))
            torch.testing.assert_close(grid.grad, torch.full_like(grid, 4.))
            assert calls == [('forward', 0, 0, False), ('backward', 0, 0, False, [True, True])]
            """
        )

    def test_framework_adapters_delegate_to_public_operators(self):
        self.run_script(
            """
            import inspect
            from unittest.mock import Mock, patch
            from turbo_physai import operators
            from turbo_physai.optimizations.common.mmdet3d import bev_pool, voxelization, sparse_conv

            cases = [
                (bev_pool.bev_pool, 'bev_pool', {'B': 2, 'D': 3, 'H': 4, 'W': 5, 'ranks': object()}),
                (bev_pool.bev_pool_prepare, 'bev_pool_prepare', {'B': 2, 'D': 3, 'H': 4, 'W': 5}),
                (bev_pool.bev_pool_prepare_geometry, 'bev_pool_prepare_geometry', {'boundary_eps': .25}),
                (voxelization.voxelization_forward, 'voxelize', {'deterministic': False}),
                (sparse_conv.get_indice_pairs, 'get_indice_pairs', {'transpose': True, 'grid': object()}),
            ]
            for adapter, name, overrides in cases:
                kwargs = {}
                for key, parameter in inspect.signature(adapter).parameters.items():
                    kwargs[key] = overrides.get(key, object() if parameter.default is inspect.Parameter.empty else parameter.default)
                expected = tuple(value for key, value in kwargs.items() if key != 'ctx')
                mock = Mock(return_value=object())
                with patch.dict(operators.__dict__, {name: mock}):
                    assert adapter(**kwargs) is mock.return_value
                mock.assert_called_once_with(*expected)
            """
        )

    def test_bev_pool_layout_sorting_and_gradients_through_adapter(self):
        self.run_script(
            """
            import importlib
            import sys
            import types
            import torch

            extension = types.ModuleType('turbo_physai._C')
            def forward(x, coords, lengths, starts, B, D, H, W):
                assert coords.dtype == torch.int32
                assert lengths.tolist() == [2, 1]
                assert starts.tolist() == [0, 2]
                out = x.new_zeros((B, D, H, W, x.shape[1]))
                for feature, (h, w, d, b) in zip(x, coords.tolist()):
                    out[b, d, h, w] += feature
                return out
            def backward(grad, coords, lengths, starts, B, D, H, W):
                assert grad.is_contiguous()
                return torch.stack([grad[b, d, h, w] for h, w, d, b in coords.tolist()])
            extension.bev_pool_forward = forward
            extension.bev_pool_backward = backward
            sys.modules[extension.__name__] = extension

            implementation = importlib.import_module('turbo_physai.operators._bev_pool')
            from turbo_physai import operators
            from turbo_physai.optimizations.common.mmdet3d import bev_pool as adapter
            assert operators.bev_pool is implementation.bev_pool
            coords = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])
            weights = torch.tensor([[[[[1., 2.], [3., 4.]]], [[[5., 6.], [7., 8.]]]]])
            for function in (operators.bev_pool, adapter.bev_pool):
                for ranks in (None, torch.tensor([2, 1, 1])):
                    features = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
                    output = function(features, coords, B=1, D=1, H=2, W=2, ranks=ranks)
                    expected = torch.tensor([[[[[0., 8.], [1., 0.]]], [[[0., 10.], [2., 0.]]]]])
                    torch.testing.assert_close(output, expected)
                    assert output.is_contiguous()
                    (output * weights).sum().backward()
                    torch.testing.assert_close(features.grad, torch.tensor([[3., 7.], [2., 6.], [2., 6.]]))
            """
        )

    def test_voxelize_dynamic_and_hard_native_contracts(self):
        self.run_script(
            """
            import sys
            import types
            import torch

            extension = types.ModuleType('turbo_physai._C')
            points = torch.ones(4, 5)
            voxel_size, coors_range = [.5] * 3, [0.] * 3 + [2.] * 3
            def dynamic(p, coords, size, bounds, ndim):
                assert p is points and size is voxel_size and bounds is coors_range and ndim == 3
                assert coords.shape == (4, 3) and coords.dtype == torch.int32
                coords.fill_(2)
            def hard(p, voxels, coords, counts, size, bounds, max_points, max_voxels, ndim, deterministic):
                assert p is points and size is voxel_size and bounds is coors_range
                assert (max_points, max_voxels, ndim, deterministic) == (2, 3, 3, False)
                assert voxels.shape == (3, 2, 5)
                assert coords.dtype == counts.dtype == torch.int32
                voxels[0].fill_(3)
                coords[0].fill_(1)
                counts[0] = 2
                return 1
            extension.dynamic_voxelize, extension.hard_voxelize = dynamic, hard
            sys.modules[extension.__name__] = extension
            from turbo_physai import operators
            from turbo_physai.optimizations.common.mmdet3d.voxelization import voxelization_forward
            for function in (operators.voxelize, lambda *args, **kwargs: voxelization_forward(None, *args, **kwargs)):
                for limits in ({'max_points': -1}, {'max_voxels': -1}):
                    coords = function(points, voxel_size, coors_range, **limits)
                    torch.testing.assert_close(coords, torch.full((4, 3), 2, dtype=torch.int32))
                voxels, coords, counts = function(points, voxel_size, coors_range, 2, 3, False)
                torch.testing.assert_close(voxels, torch.full((1, 2, 5), 3.))
                torch.testing.assert_close(coords, torch.ones((1, 3), dtype=torch.int32))
                torch.testing.assert_close(counts, torch.tensor([2], dtype=torch.int32))
            """
        )

    def test_sparse_indice_dispatch_and_shape_contracts(self):
        self.run_script(
            """
            import sys
            import types
            import torch
            from unittest.mock import Mock

            extension = types.ModuleType('turbo_physai._C')
            sys.modules[extension.__name__] = extension
            from turbo_physai import operators
            from turbo_physai.optimizations.common.mmdet3d.sparse_conv import get_indice_pairs
            for function in (operators.get_indice_pairs, get_indice_pairs):
                for ndim in (2, 3, 4):
                    indices = torch.zeros((1, ndim + 1), dtype=torch.int32)
                    for grid in (None, torch.zeros(1, dtype=torch.int32)):
                        if grid is not None and ndim == 4:
                            continue
                        name = f'get_indice_pairs_{ndim}d' if grid is None else f'get_indice_pairs_grid_{ndim}d'
                        mock = Mock(return_value=object())
                        setattr(extension, name, mock)
                        for subm, transpose, expected_shape in ((False, False, [4]*ndim), (True, False, [8]*ndim), (False, True, [15]*ndim)):
                            result = function(indices, 2, [8]*ndim, ksize=3, stride=2, padding=1, subm=subm, transpose=transpose, grid=grid)
                            assert result is mock.return_value
                            prefix = (indices,) if grid is None else (indices, grid)
                            mock.assert_called_with(*(prefix + (2, expected_shape, [8]*ndim, [3]*ndim, [2]*ndim, [1]*ndim, [1]*ndim, [0]*ndim, int(subm), int(transpose))))
            """
        )
