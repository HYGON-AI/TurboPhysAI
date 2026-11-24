# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical equivalence tests for the MapTRv2 replacement implementations."""

import os
import sys
import types
import unittest
import unittest.mock

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for catalog checks
    torch = None

try:
    import mmcv  # noqa: F401
except Exception:  # pragma: no cover - the model environment ships mmcv
    mmcv = None

try:
    import cv2
except Exception:  # pragma: no cover - opencv ships with the model stack
    cv2 = None

try:
    from shapely.geometry import LineString
except Exception:  # pragma: no cover - shapely ships with the model stack
    LineString = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - Pillow ships with the model stack
    Image = None

_GridMaskBase = torch.nn.Module if torch is not None else object
class _GridMaskStub(_GridMaskBase):
    """Minimal stand-in for ``GridMask`` so the math can be tested alone."""

    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5,
                 mode=0, prob=1.0):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.l = None
        self.training = True


def _official_mask(mask, use_h, use_w, mode, distance, start_h, start_w,
                   keep, expanded_h, expanded_w, rotate_draw):
    """Official numpy/PIL mask construction."""

    if use_h:
        for index in range(expanded_h // distance):
            start = distance * index + start_h
            mask[start:min(start + keep, expanded_h), :] *= 0
    if use_w:
        for index in range(expanded_w // distance):
            start = distance * index + start_w
            mask[:, start:min(start + keep, expanded_w)] *= 0
    mask = np.asarray(
        Image.fromarray(np.uint8(mask)).rotate(rotate_draw)
    )
    if mode == 1:
        mask = 1 - mask
    return mask


def _official_pv_mask(line_ego, proj_mat, shape, color=1, thickness=1,
                      z=-1.6):
    """Baseline ``VectorizedLocalMap.line_ego_to_pvmask`` body, inlined."""

    distances = np.linspace(0, line_ego.length, 200)
    coords = np.array(
        [list(line_ego.interpolate(distance).coords) for distance in distances]
    ).reshape(-1, 2)
    pts_num = coords.shape[0]
    zeros = np.zeros((pts_num, 1))
    zeros[:] = z
    ones = np.ones((pts_num, 1))
    lidar_coords = np.concatenate([coords, zeros, ones], axis=1).transpose(1, 0)
    pix_coords = proj_mat @ lidar_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.polylines(mask, np.int32([pix_coords]), False, color=color,
                  thickness=thickness)
    return mask


def _official_line_ego_to_mask(canvas_size, scale_x, scale_y, line_ego,
                               color=1, thickness=3):
    """Baseline ``VectorizedLocalMap.line_ego_to_mask`` body, inlined."""

    from shapely import affinity

    trans_x = canvas_size[1] / 2
    trans_y = canvas_size[0] / 2
    line_ego = affinity.scale(line_ego, scale_x, scale_y, origin=(0, 0))
    line_ego = affinity.affine_transform(
        line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
    coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
    coords = coords.reshape((-1, 2))
    assert len(coords) >= 2
    mask = np.zeros(canvas_size, dtype=np.uint8)
    cv2.polylines(mask, np.int32([coords]), False, color=color,
                  thickness=thickness)
    return mask


@unittest.skipIf(torch is None or mmcv is None or Image is None,
                 "torch, mmcv and Pillow are required for GridMask tests")
class GridMaskTest(unittest.TestCase):
    def _run(self, module, stub, x, distance, start_h, start_w, prob_draw,
             rotate_draw):
        original_rand = np.random.rand
        original_randint = np.random.randint
        randint_calls = iter((
            ((2, x.size(2)), distance),
            ((distance,), start_h),
            ((distance,), start_w),
            ((stub.rotate,), rotate_draw),
        ))

        def fake_rand(*args):
            if args:
                return np.full(args, 0.25, dtype=np.float32)
            return prob_draw

        def fake_randint(*args):
            expected_args, value = next(randint_calls)
            if args != expected_args:
                raise AssertionError(
                    f"unexpected randint args: {args}; "
                    f"expected {expected_args}"
                )
            return value

        np.random.rand = fake_rand
        np.random.randint = fake_randint
        runner = types.ModuleType("mmcv.runner")
        runner.auto_fp16 = lambda *args, **kwargs: (
            lambda function: function
        )
        try:
            with unittest.mock.patch.dict(
                sys.modules, {"mmcv.runner": runner}
            ):
                with unittest.mock.patch.object(
                    mmcv, "runner", runner, create=True
                ):
                    return module.grid_mask_forward(stub, x)
        finally:
            np.random.rand = original_rand
            np.random.randint = original_randint

    def _expected(self, stub, x, distance, start_h, start_w, rotate_draw):
        _, _, height, width = x.shape
        expanded_h, expanded_w = int(1.5 * height), int(1.5 * width)
        keep = min(max(int(distance * stub.ratio + 0.5), 1), distance - 1)
        mask = np.ones((expanded_h, expanded_w), np.float32)
        mask = _official_mask(mask, stub.use_h, stub.use_w, stub.mode, distance,
                              start_h, start_w, keep, expanded_h, expanded_w,
                              rotate_draw)
        mask = mask[
            (expanded_h - height) // 2:
            (expanded_h - height) // 2 + height,
            (expanded_w - width) // 2:
            (expanded_w - width) // 2 + width,
        ]
        return torch.from_numpy(np.asarray(mask).copy())

    def test_mask_matches_official_pil_construction(self):
        from turbo_physai.optimizations.models.maptrv2 import grid_mask

        for rotate, rotate_draw in ((1, 0), (2, 0), (4, 1)):
            with self.subTest(rotate=rotate, rotate_draw=rotate_draw):
                stub = _GridMaskStub(
                    True, True, rotate=rotate, offset=False, ratio=0.5,
                    mode=1, prob=0.7)
                x = torch.arange(
                    2 * 3 * 32 * 40, dtype=torch.float32).reshape(
                        2, 3, 32, 40).contiguous()
                expected_mask = self._expected(
                    stub, x, distance=6, start_h=2, start_w=4,
                    rotate_draw=rotate_draw)
                actual = self._run(
                    grid_mask, stub, x, 6, 2, 4, prob_draw=0.0,
                    rotate_draw=rotate_draw)
                expected = x.view(-1, 32, 40) * expected_mask
                torch.testing.assert_close(
                    actual.view(-1, 32, 40), expected, rtol=0, atol=0)

    def test_offset_path_matches_official_numpy_construction(self):
        from turbo_physai.optimizations.models.maptrv2 import grid_mask

        stub = _GridMaskStub(True, True, rotate=4, offset=True, ratio=0.5,
                             mode=0, prob=0.7)
        x = torch.arange(1 * 2 * 16 * 24, dtype=torch.float32).reshape(
            1, 2, 16, 24).contiguous()
        expected_mask = self._expected(
            stub, x, distance=4, start_h=1, start_w=3, rotate_draw=1)
        actual = self._run(
            grid_mask, stub, x, 4, 1, 3, prob_draw=0.0, rotate_draw=1)
        offset = torch.full((16, 24), 2 * (0.25 - 0.5), dtype=torch.float32)
        expected = (x.view(-1, 16, 24) * expected_mask
                    + offset * (1 - expected_mask))
        torch.testing.assert_close(
            actual.view(-1, 16, 24), expected, rtol=0, atol=0)

    def test_probability_shortcut_returns_input(self):
        from turbo_physai.optimizations.models.maptrv2 import grid_mask

        stub = _GridMaskStub(True, True, prob=0.7)
        x = torch.ones(1, 1, 8, 8)
        self.assertIs(
            self._run(
                grid_mask, stub, x, 3, 0, 0, prob_draw=0.9,
                rotate_draw=0),
            x,
        )

    def test_eval_mode_returns_input(self):
        from turbo_physai.optimizations.models.maptrv2 import grid_mask

        stub = _GridMaskStub(True, True, prob=0.7)
        stub.training = False
        x = torch.ones(1, 1, 8, 8)
        self.assertIs(
            self._run(
                grid_mask, stub, x, 3, 0, 0, prob_draw=0.0,
                rotate_draw=0),
            x,
        )


@unittest.skipIf(torch is None, "torch is required for implementation tests")
class MatchCostTest(unittest.TestCase):
    def test_broadcast_cost_equals_cdist(self):
        from turbo_physai.optimizations.models.maptrv2.match_cost import ordered_pts_l1_cost_call

        class _Cost:
            weight = 0.5

        generator = torch.Generator().manual_seed(0)
        # float64 first: the two spellings are algebraically the same sum of
        # absolute differences, so only the reduction order may differ.
        bbox64 = torch.rand(100, 20, 2, generator=generator,
                            dtype=torch.float64)
        gt64 = torch.rand(7, 2, 20, 2, generator=generator,
                          dtype=torch.float64)
        expected = torch.cdist(
            bbox64.view(bbox64.size(0), -1),
            gt64.flatten(2).view(7 * 2, -1), p=1) * _Cost.weight
        actual = ordered_pts_l1_cost_call(_Cost(), bbox64, gt64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)

        # float32 (the training dtype) may only differ by accumulated rounding.
        bbox32 = torch.rand(100, 20, 2, generator=generator)
        gt32 = torch.rand(7, 2, 20, 2, generator=generator)
        expected32 = torch.cdist(
            bbox32.view(bbox32.size(0), -1),
            gt32.flatten(2).view(7 * 2, -1), p=1) * _Cost.weight
        actual32 = ordered_pts_l1_cost_call(_Cost(), bbox32, gt32)
        torch.testing.assert_close(actual32, expected32, rtol=1e-5, atol=1e-5)


class TrainingFlagsTest(unittest.TestCase):
    def test_option_flag_prefers_explicit_options(self):
        from turbo_physai.optimizations.models.maptrv2 import training

        self.assertFalse(training._option_flag(
            {"channels_last": False}, "channels_last", "NO_SUCH_ENV", True))
        self.assertTrue(training._option_flag(
            {"channels_last": True}, "channels_last", "NO_SUCH_ENV", False))

    def test_option_flag_falls_back_to_default(self):
        from turbo_physai.optimizations.models.maptrv2 import training

        self.assertTrue(training._option_flag(
            {}, "channels_last", "NO_SUCH_ENV", True))

    def test_channels_last_scope_defaults_to_model(self):
        from turbo_physai.optimizations.models.maptrv2 import training

        with unittest.mock.patch.dict(os.environ):
            os.environ.pop("TURBO_PHYSAI_CHANNELS_LAST_SCOPE", None)
            self.assertEqual(training._channels_last_scope({}), "model")
            self.assertEqual(
                training._channels_last_scope(
                    {"channels_last_scope": "BACKBONE"}),
                "backbone")
            with self.assertRaises(ValueError):
                training._channels_last_scope(
                    {"channels_last_scope": "layer"})

    def test_matmul_precision_defaults_to_reference_value(self):
        from turbo_physai.optimizations.models.maptrv2 import training

        with unittest.mock.patch.dict(os.environ):
            os.environ.pop("TURBO_PHYSAI_MATMUL_PRECISION", None)
            self.assertEqual(training._matmul_precision({}), "high")
            self.assertEqual(
                training._matmul_precision({"matmul_precision": "OFF"}),
                "off")


class CompatProbeTest(unittest.TestCase):
    """The probes back the ``runtime_condition`` of the optional Groups."""

    @staticmethod
    def _stub_module(name, **attributes):
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        return module

    def test_compile_probe_follows_torch_capability(self):
        from turbo_physai.optimizations.models.maptrv2 import compat

        compiled = self._stub_module("torch", compile=lambda target, **_: target)
        with unittest.mock.patch.dict(sys.modules, {"torch": compiled}):
            self.assertTrue(compat.torch_compile_available())
        with unittest.mock.patch.dict(
                sys.modules, {"torch": self._stub_module("torch")}):
            self.assertFalse(compat.torch_compile_available())

    def test_compile_probe_honours_the_kill_switch(self):
        from turbo_physai.optimizations.models.maptrv2 import compat

        compiled = self._stub_module("torch", compile=lambda target, **_: target)
        with unittest.mock.patch.dict(sys.modules, {"torch": compiled}):
            with unittest.mock.patch.dict(
                    os.environ, {"TURBO_PHYSAI_DISABLE_TORCH_COMPILE": "1"}):
                self.assertFalse(compat.torch_compile_available())

    def test_dynamo_probe_checks_dynamo_disable(self):
        from turbo_physai.optimizations.models.maptrv2 import compat

        dynamo = self._stub_module("torch._dynamo", disable=lambda item: item)
        with unittest.mock.patch.dict(
                sys.modules, {"torch": self._stub_module("torch", _dynamo=dynamo)}):
            self.assertTrue(compat.dynamo_available())
        with unittest.mock.patch.dict(
                sys.modules, {"torch": self._stub_module("torch")}):
            self.assertFalse(compat.dynamo_available())

    def test_probes_tolerate_a_missing_torch(self):
        from turbo_physai.optimizations.models.maptrv2 import compat

        with unittest.mock.patch.dict(sys.modules, {"torch": None}):
            self.assertFalse(compat.torch_compile_available())
            self.assertFalse(compat.dynamo_available())

    def test_probes_accept_the_wrapped_call_arguments(self):
        from turbo_physai.optimizations.models.maptrv2 import compat

        # The engine calls a condition as ``condition(*args, **kwargs)``.
        with unittest.mock.patch.dict(sys.modules, {"torch": None}):
            self.assertIsInstance(
                compat.torch_compile_available(object(), key=1), bool)
            self.assertIsInstance(
                compat.dynamo_available(object(), key=1), bool)


_PROJECTION = np.array([
    [200.0, 0.0, 0.0, 100.0],
    [0.0, 10.0, 0.0, 40.0],
    [0.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 0.0, 1.0],
])


@unittest.skipIf(LineString is None, "shapely is required for the sampling test")
class PvMaskSamplingTest(unittest.TestCase):
    def test_resampling_matches_shapely_interpolation(self):
        from turbo_physai.optimizations.models.maptrv2.pv_mask import sample_line_arc_length

        line = LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 8.0)])
        distances = np.linspace(0.0, line.length, 200)
        expected = np.array(
            [list(line.interpolate(distance).coords) for distance in distances]
        ).reshape(-1, 2)
        np.testing.assert_allclose(
            sample_line_arc_length(line), expected, rtol=0.0, atol=1e-9)

    def test_resampling_uses_equal_arc_length_steps(self):
        from turbo_physai.optimizations.models.maptrv2.pv_mask import sample_line_arc_length

        line = LineString([(0.0, 0.0), (3.0, 4.0)])
        np.testing.assert_allclose(
            sample_line_arc_length(line, n_points=5),
            [[0.0, 0.0], [0.75, 1.0], [1.5, 2.0], [2.25, 3.0], [3.0, 4.0]],
            rtol=0.0,
            atol=1e-12,
        )


class PvMaskProjectionTest(unittest.TestCase):
    def test_projection_matches_the_perspective_helper(self):
        from turbo_physai.optimizations.models.maptrv2.pv_mask import project_points

        coords = np.array([[0.0, 0.0], [1.0, -2.0], [4.0, 3.0]])
        zeros = np.full((coords.shape[0], 1), -1.6)
        ones = np.ones((coords.shape[0], 1))
        lidar_coords = np.concatenate(
            [coords, zeros, ones], axis=1).transpose(1, 0)
        pix_coords = _PROJECTION @ lidar_coords
        valid = pix_coords[2, :] > 0
        expected = (pix_coords[:2, valid]
                    / (pix_coords[2, valid] + 1e-7)).transpose(1, 0)
        np.testing.assert_allclose(
            project_points(coords, _PROJECTION, z=-1.6),
            expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_points_behind_the_camera_are_dropped(self):
        from turbo_physai.optimizations.models.maptrv2.pv_mask import project_points

        coords = np.array([[1.0, 1.0], [-1.0, 1.0]])
        visible = project_points(coords, _PROJECTION, z=0.0)
        behind = project_points(coords, _PROJECTION, z=-3.0)
        self.assertEqual(visible.shape, (2, 2))
        # ``perspective`` divides by ``z + 1e-7`` and both points project to
        # ``z = 3``, so the exact expectation carries the same epsilon factor.
        expected = np.array(
            [[100.0, 50.0 / 3.0], [-100.0 / 3.0, 50.0 / 3.0]]
        ) * (3.0 / (3.0 + 1e-7))
        np.testing.assert_allclose(
            visible,
            expected,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(behind.shape, (0, 2))


@unittest.skipIf(LineString is None or cv2 is None,
                 "shapely and opencv are required for the drawing test")
class PvMaskDrawingTest(unittest.TestCase):
    def test_vectorized_drawing_reproduces_the_official_loop(self):
        from turbo_physai.optimizations.models.maptrv2 import pv_mask

        line = LineString([(0.0, 0.0), (10.0, 0.0), (10.0, 8.0)])
        shape = (180, 320)
        expected = _official_pv_mask(line, _PROJECTION, shape, thickness=3)
        actual = np.zeros(shape, dtype=np.uint8)
        pv_mask.line_ego_to_pvmask(None, line, actual, _PROJECTION, thickness=3)
        np.testing.assert_array_equal(actual, expected)

    def test_semantic_mask_reproduces_the_official_two_step_transform(self):
        from turbo_physai.optimizations.models.maptrv2 import (
            pv_mask,
        )

        canvas_size = (200, 100)
        scale_x = canvas_size[1] / 100.0
        scale_y = canvas_size[0] / 200.0
        line = LineString([(-12.0, 3.5), (4.0, -9.0), (18.0, 21.0)])
        container = types.SimpleNamespace(
            canvas_size=canvas_size, scale_x=scale_x, scale_y=scale_y)
        for thickness in (1, 3, 5):
            with self.subTest(thickness=thickness):
                expected = _official_line_ego_to_mask(
                    canvas_size, scale_x, scale_y, line, thickness=thickness)
                actual = np.zeros(canvas_size, dtype=np.uint8)
                pv_mask.line_ego_to_mask(
                    container, line, actual, thickness=thickness)
                np.testing.assert_array_equal(actual, expected)

    def test_scale_translate_geom_matches_two_step_affinity(self):
        from shapely import affinity

        from turbo_physai.optimizations.models.maptrv2.pv_mask import (
            scale_translate_geom,
        )

        line = LineString([(0.0, 0.0), (10.0, -3.0), (12.5, 7.25)])
        for origin in ((0.0, 0.0), (3.5, -2.25), (-11.0, 4.0)):
            with self.subTest(origin=origin):
                two_step = affinity.affine_transform(
                    affinity.scale(line, 2.5, 0.75, origin=origin),
                    [1.0, 0.0, 0.0, 1.0, 7.0, -9.0])
                one_step = scale_translate_geom(
                    line, 2.5, 0.75, 7.0, -9.0, origin=origin)
                np.testing.assert_array_equal(
                    np.asarray(one_step.coords), np.asarray(two_step.coords))


def _import_bev_pool_fix():
    mmdet3d = types.ModuleType("mmdet3d")
    ops = types.ModuleType("mmdet3d.ops")
    ops.bev_pool = lambda *args: None
    mmdet3d.ops = ops
    with unittest.mock.patch.dict(
        sys.modules, {"mmdet3d": mmdet3d, "mmdet3d.ops": ops}
    ):
        from turbo_physai.optimizations.models.maptrv2 import (
            bev_pool_fix,
        )
    return bev_pool_fix


@unittest.skipIf(torch is None, "torch is required for bev_pool_fix tests")
class BevPoolFixTest(unittest.TestCase):
    """Verify that ``base_transform_bev_pool`` produces the correct output shape
    for both the legacy Python-wrapper layout ``[B, C, Z, H, W]`` and the
    raw/optimized wrapper layout ``[B, Z, H, W, C]`` returned by ``_bev_pool``.

    The real ``mmdet3d.ops.bev_pool`` is replaced with a stub that returns a
    pre-built tensor of the desired layout so the test runs without HCU hardware
    or the model stack.
    """

    # Grid dimensions matching a typical MapTRv2 config (nx=[200,400,1], C=256).
    B, N, D, H_in, W_in, C = 2, 6, 1, 8, 8, 4
    NX0, NX1, NX2 = 10, 20, 1  # nx[0], nx[1], nx[2] (x, y, z bins)

    def _make_self(self):
        """Minimal stub standing in for a ``BaseTransform`` instance."""
        import torch as _torch

        stub = types.SimpleNamespace(
            C=self.C,
            nx=_torch.tensor([self.NX0, self.NX1, self.NX2]),
            bx=_torch.tensor([0.0, 0.0, 0.0]),
            dx=_torch.tensor([1.0, 1.0, 1.0]),
        )
        return stub

    def _make_inputs(self):
        """Return ``(geom_feats, x)`` with shapes the real method expects."""
        import torch as _torch

        Nprime = self.B * self.N * self.D * self.H_in * self.W_in
        x = _torch.zeros(self.B, self.N, self.D, self.H_in, self.W_in, self.C)
        # geom_feats must survive the bounds filter; put everything at (1,1,0).
        geom_feats = _torch.ones(self.B, self.N, self.D, self.H_in, self.W_in, 3)
        return geom_feats, x

    def _expected_output_shape(self):
        # collapse Z: C * nx[2] channels, spatial H=nx[0], W=nx[1]
        return (self.B, self.C * self.NX2, self.NX0, self.NX1)

    def _run_with_stub_layout(self, layout):
        """Patch ``_bev_pool`` inside ``bev_pool_fix`` to return *layout* and
        call ``base_transform_bev_pool``; return the output tensor."""
        import torch as _torch
        bev_pool_fix = _import_bev_pool_fix()

        self_stub = self._make_self()
        geom_feats, x = self._make_inputs()

        if layout == "NCHW":
            # [B, C, Z, H, W] - legacy Python wrapper layout.
            stub_out = _torch.zeros(
                self.B, self.C, self.NX2, self.NX0, self.NX1)
        elif layout == "NHWC":
            # [B, Z, H, W, C] - raw or optimized wrapper layout.
            stub_out = _torch.zeros(
                self.B, self.NX2, self.NX0, self.NX1, self.C)
        else:
            raise ValueError(layout)

        with unittest.mock.patch.object(
            bev_pool_fix, "_bev_pool", return_value=stub_out
        ):
            return bev_pool_fix.base_transform_bev_pool(self_stub, geom_feats, x)

    def test_legacy_nchw_layout_produces_correct_shape(self):
        out = self._run_with_stub_layout("NCHW")
        self.assertEqual(tuple(out.shape), self._expected_output_shape())

    def test_raw_nhwc_layout_produces_correct_shape(self):
        out = self._run_with_stub_layout("NHWC")
        self.assertEqual(tuple(out.shape), self._expected_output_shape())

    def test_both_layouts_produce_identical_output(self):
        """A zero-filled BEV grid collapses to zeros regardless of layout;
        verify the two paths agree element-wise for a non-trivial tensor."""
        import torch as _torch
        bev_pool_fix = _import_bev_pool_fix()

        self_stub = self._make_self()
        geom_feats, x = self._make_inputs()

        base = _torch.arange(
            self.B * self.C * self.NX2 * self.NX1 * self.NX0,
            dtype=_torch.float32,
        ).reshape(self.B, self.C, self.NX2, self.NX0, self.NX1)

        nchw_out = None
        nhwc_out = None

        with unittest.mock.patch.object(
            bev_pool_fix, "_bev_pool", return_value=base.clone()
        ):
            nchw_out = bev_pool_fix.base_transform_bev_pool(
                self_stub, geom_feats, x)

        nhwc_base = base.permute(0, 2, 3, 4, 1).contiguous()
        with unittest.mock.patch.object(
            bev_pool_fix, "_bev_pool", return_value=nhwc_base
        ):
            nhwc_out = bev_pool_fix.base_transform_bev_pool(
                self_stub, geom_feats, x)

        torch.testing.assert_close(nchw_out, nhwc_out, rtol=0, atol=0)

    def test_5d_return_preserves_reference_layout(self):
        import torch as _torch

        bev_pool_fix = _import_bev_pool_fix()
        self.assertTrue(
            bev_pool_fix.base_transform_bev_pool._turbo_physai_returns_5d
        )
        self_stub = self._make_self()
        geom_feats, x = self._make_inputs()
        nchw = _torch.arange(
            self.B * self.C * self.NX2 * self.NX0 * self.NX1,
            dtype=_torch.float32,
        ).reshape(self.B, self.C, self.NX2, self.NX0, self.NX1)
        expected = nchw.permute(0, 2, 3, 4, 1).contiguous()

        for pooled in (nchw, expected):
            with self.subTest(layout=tuple(pooled.shape)):
                with unittest.mock.patch.object(
                    bev_pool_fix, "_bev_pool", return_value=pooled
                ):
                    result = bev_pool_fix.base_transform_bev_pool(
                        self_stub, geom_feats, x, return_5d=True
                    )
                torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_equal_width_and_channels_preserves_values_and_gradients(self):
        bev_pool_fix = _import_bev_pool_fix()
        batch, channels, depth, height, width = 2, 4, 2, 3, 4
        owner = types.SimpleNamespace(
            C=channels, nx=torch.tensor([height, width, depth]),
            bx=torch.zeros(3), dx=torch.ones(3),
        )
        features = torch.zeros(batch, 1, 1, 1, 1, channels)
        geometry = torch.zeros(batch, 1, 1, 1, 1, 3)
        values = torch.arange(
            batch * channels * depth * height * width, dtype=torch.float32
        ).reshape(batch, channels, depth, height, width)

        for channels_last in (False, True):
            for return_5d in (False, True):
                with self.subTest(channels_last=channels_last, return_5d=return_5d):
                    source = values.clone().requires_grad_()
                    pooled = (source.permute(0, 2, 3, 4, 1).contiguous()
                              if channels_last else source)
                    with unittest.mock.patch.object(
                        bev_pool_fix, "_bev_pool", return_value=pooled
                    ):
                        result = bev_pool_fix.base_transform_bev_pool(
                            owner, geometry, features, return_5d=return_5d
                        )
                    if return_5d:
                        expected = values.permute(0, 2, 3, 4, 1)
                    else:
                        expected = values.permute(0, 2, 1, 3, 4).reshape(
                            batch, depth * channels, height, width
                        )
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    weights = torch.arange(
                        1, result.numel() + 1, dtype=result.dtype
                    ).reshape(result.shape)
                    (result * weights).sum().backward()
                    expected_grad = (
                        weights.permute(0, 4, 1, 2, 3) if return_5d else
                        weights.reshape(batch, depth, channels, height, width)
                        .permute(0, 2, 1, 3, 4)
                    )
                    torch.testing.assert_close(source.grad, expected_grad, rtol=0, atol=0)

    def test_ambiguous_and_invalid_pool_layouts_are_rejected(self):
        bev_pool_fix = _import_bev_pool_fix()
        owner = types.SimpleNamespace(
            C=2, nx=torch.tensor([2, 2, 2]),
            bx=torch.zeros(3), dx=torch.ones(3),
        )
        features = torch.zeros(1, 1, 1, 1, 1, 2)
        geometry = torch.zeros(1, 1, 1, 1, 1, 3)
        cases = (
            (torch.arange(16.).reshape(1, 2, 2, 2, 2), "ambiguous BEV pooling layout"),
            (torch.zeros(1, 2, 2, 2), "invalid BEV pooling shape"),
            (torch.zeros(1, 2, 2, 2, 3), "invalid BEV pooling shape"),
        )
        for pooled, message in cases:
            for return_5d in (False, True):
                with self.subTest(shape=tuple(pooled.shape), return_5d=return_5d):
                    with unittest.mock.patch.object(
                        bev_pool_fix, "_bev_pool", return_value=pooled
                    ):
                        with self.assertRaisesRegex(RuntimeError, message):
                            bev_pool_fix.base_transform_bev_pool(
                                owner, geometry, features, return_5d=return_5d
                            )

    def test_permute_guard_triggers_only_for_nhwc(self):
        """The permute guard must fire for NHWC (last dim == C) and stay silent
        for NCHW (last dim == nx[1], not C when nx[1] != C)."""
        import torch as _torch
        bev_pool_fix = _import_bev_pool_fix()

        self_stub = self._make_self()
        geom_feats, x = self._make_inputs()

        nchw_stub = _torch.zeros(
            self.B, self.C, self.NX2, self.NX0, self.NX1)
        nhwc_stub = _torch.zeros(
            self.B, self.NX2, self.NX0, self.NX1, self.C)

        # NCHW: last dim is NX1, not C - guard must not fire.
        self.assertNotEqual(self.NX1, self.C,
                            "test requires NX1 != C to distinguish layouts")
        self.assertEqual(nchw_stub.shape[-1], self.NX1)

        # NHWC: last dim is C — guard fires and permutes to NCHW
        self.assertEqual(nhwc_stub.shape[-1], self.C)

        with unittest.mock.patch.object(
            bev_pool_fix, "_bev_pool", return_value=nchw_stub
        ):
            nchw_out = bev_pool_fix.base_transform_bev_pool(
                self_stub, geom_feats, x)

        with unittest.mock.patch.object(
            bev_pool_fix, "_bev_pool", return_value=nhwc_stub
        ):
            nhwc_out = bev_pool_fix.base_transform_bev_pool(
                self_stub, geom_feats, x)

        self.assertEqual(tuple(nchw_out.shape), self._expected_output_shape())
        self.assertEqual(tuple(nhwc_out.shape), self._expected_output_shape())


if __name__ == "__main__":
    unittest.main()
