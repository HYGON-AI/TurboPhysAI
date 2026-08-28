# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

import os
import subprocess
import sys
import textwrap
import types
import unittest
from unittest import mock

from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine.definitions.registry import default_registry
from turbo_physai.engine.execution.replacements.base import resolve_replacement
from turbo_physai.optimizations.common.mmdet3d import (
    bev_pool,
    catalog as mmdet3d_catalog,
    sparse_conv as indice,
    voxelization as voxel,
)
from turbo_physai.optimizations.models.bevfusion import (
    backbone,
    catalog,
    compile as compile_patches,
    gaussian,
    training,
    transfusion,
    transfusion_bbox_coder,
)


BEVFUSION_GROUPS = (
    catalog.BACKBONE,
    catalog.LOSS_REDUCTION,
    catalog.TRAINING,
    catalog.GAUSSIAN,
    mmdet3d_catalog.GAUSSIAN,
    mmdet3d_catalog.BEV_POOL,
    mmdet3d_catalog.QUICK_CUMSUM,
    catalog.DEPTH_FACTORIZATION,
    catalog.BEV_GEOMETRY,
    catalog.HUNGARIAN_TRANSFER,
    catalog.TRANSFUSION_BBOX_CODER,
    mmdet3d_catalog.VOXELIZATION,
    mmdet3d_catalog.CANONICAL_INDICE_PAIRS,
    mmdet3d_catalog.SPARSE_TENSOR,
    catalog.COMPILE,
)


class BevFusionPatchFrameworkTest(unittest.TestCase):
    def test_catalog_registers_all_model_groups(self):
        for optimization in BEVFUSION_GROUPS:
            registered = default_registry.get_group(optimization.group_id)
            self.assertEqual(registered, optimization.definition)
        self.assertEqual(catalog.TRAINING.specs[0].mechanism, Mechanism.WRAPPER)

        depth_targets = {spec.target for spec in catalog.DEPTH_FACTORIZATION.specs}
        self.assertIn(
            "mmdet3d.models.vtransforms.base.BaseDepthTransform.forward",
            depth_targets,
        )
        self.assertIn(
            "mmdet3d.models.vtransforms.base.BaseTransform.__init__",
            depth_targets,
        )
        self.assertEqual(
            catalog.BEV_GEOMETRY.specs[0].target,
            "mmdet3d.models.vtransforms.base.BaseTransform.get_geometry",
        )

    def test_vtransform_base_custom_helpers_and_added_methods_are_complete(self):
        from turbo_physai.optimizations.models.bevfusion import depth

        with mock.patch.dict(
            os.environ,
            {
                "MMDET3D_BEV_POOL_PREPARE_OPT_MODE": "off",
                "MMDET3D_BEV_POOL_GEOMETRY_OPT_MODE": "disabled",
                "MMDET3D_BEV_POOL_GEOMETRY_BOUNDARY_EPS": "0.002",
                "MMDET3D_BEV_POOL_GEOMETRY_CORRECTION_CHUNK": "4096",
            },
            clear=False,
        ):
            self.assertFalse(depth.use_bev_pool_prepare_opt())
            self.assertFalse(depth.use_bev_pool_geometry_opt())
            self.assertEqual(depth.bev_pool_geometry_boundary_eps(), 0.002)
            self.assertEqual(depth.bev_pool_geometry_correction_chunk(), 4096)

        class BaseTransform:
            pass

        class DerivedTransform(BaseTransform):
            pass

        def original(model):
            model.nx = (4, 5, 6)

        wrapped = depth.base_transform_init_wrapper(original, {})
        model = DerivedTransform()
        wrapped(model)
        self.assertEqual(model._bev_output_shape, (4, 5, 6))
        self.assertIs(
            BaseTransform.bev_pool_prepared,
            depth.base_transform_bev_pool_prepared,
        )
        self.assertIs(
            BaseTransform.bev_pool_prepared_factorized,
            depth.base_transform_bev_pool_prepared_factorized,
        )
        self.assertIs(
            BaseTransform.correct_bev_pool_geometry_boundaries,
            depth.base_transform_correct_bev_pool_geometry_boundaries,
        )

    def test_base_transform_geometry_cpu_fallback_uses_inv_ex(self):
        script = textwrap.dedent(
            """
            import types
            import torch
            from turbo_physai.optimizations.models.bevfusion.depth import (
                PreparedGeometry,
                base_transform_get_geometry,
            )

            model = types.SimpleNamespace(
                frustum=torch.tensor([[[[1.0, 2.0, 2.0]]]]),
                bx=torch.ones(3),
                dx=torch.ones(3),
                nx=torch.ones(3, dtype=torch.long),
                xbound=(0.0, 1.0, 1.0),
                ybound=(0.0, 1.0, 1.0),
                zbound=(0.0, 1.0, 1.0),
            )
            rotations = torch.eye(3).reshape(1, 1, 3, 3)
            translations = torch.zeros(1, 1, 3)
            geometry = base_transform_get_geometry(
                model,
                rotations,
                translations,
                rotations,
                rotations,
                translations,
            )
            assert not isinstance(geometry, PreparedGeometry)
            assert geometry.shape == (1, 1, 1, 1, 1, 3)
            torch.testing.assert_close(
                geometry.reshape(3), torch.tensor([2.0, 4.0, 2.0])
            )
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_compile_catalog_covers_all_reference_decorators(self):
        compile_targets = {spec.target for spec in catalog.COMPILE.specs}
        self.assertEqual(
            compile_targets,
            {
                "mmdet3d.models.backbones.sparse_encoder.SparseEncoder",
                "mmdet3d.models.fusers.conv.ConvFuser",
                "mmdet3d.models.necks.generalized_lss.GeneralizedLSSFPN",
                (
                    "mmdet3d.models.heads.bbox.transfusion."
                    "TransFusionHead.forward_single"
                ),
                (
                    "mmdet3d.models.vtransforms.depth_lss."
                    "DepthLSSTransform"
                ),
                (
                    "mmdet3d.models.utils.transformer."
                    "PositionEmbeddingLearned.forward"
                ),
                "mmdet3d.models.utils.transformer.FFN.forward",
            },
        )
        self.assertEqual(
            tuple(spec.mechanism for spec in catalog.GAUSSIAN.specs),
            (Mechanism.WRAPPER, Mechanism.WRAPPER),
        )
        self.assertEqual(
            tuple(spec.mechanism for spec in mmdet3d_catalog.GAUSSIAN.specs),
            (Mechanism.REPLACE,),
        )
        transfusion_compile = next(
            spec
            for spec in catalog.COMPILE.specs
            if spec.target.endswith("TransFusionHead.forward_single")
        )
        self.assertTrue(
            transfusion_compile.replacement.endswith(
                "compile_transfusion_forward_single_wrapper"
            )
        )

    def test_transfusion_group_covers_all_optimized_class_boundaries(self):
        targets = {spec.target for spec in catalog.HUNGARIAN_TRANSFER.specs}
        compile_targets = {spec.target for spec in catalog.COMPILE.specs}
        self.assertEqual(
            targets,
            {
                "mmdet3d.models.heads.bbox.transfusion.TransFusionHead.__init__",
                "mmdet3d.models.heads.bbox.transfusion.TransFusionHead.get_targets",
                (
                    "mmdet3d.models.heads.bbox.transfusion."
                    "TransFusionHead.get_targets_single"
                ),
                "mmdet3d.models.heads.bbox.transfusion.TransFusionHead.loss",
                (
                    "mmdet3d.core.bbox.assigners.hungarian_assigner."
                    "HungarianAssigner3D.assign"
                ),
            },
        )
        self.assertIn(
            (
                "mmdet3d.models.heads.bbox.transfusion."
                "TransFusionHead.forward_single"
            ),
            compile_targets,
        )

    def test_transfusion_init_registers_non_persistent_bev_position_buffer(self):
        def original(model):
            model._buffers = {}
            model.bev_pos = object()

        wrapped = transfusion.transfusion_init_wrapper(original, {})

        class Head:
            def register_buffer(self, name, value, persistent=True):
                self._buffers[name] = value
                self.persistent = persistent

        model = Head()
        wrapped(model)
        self.assertIn("bev_pos", model._buffers)
        self.assertFalse(model.persistent)

    def test_transfusion_compile_wraps_migrated_forward(self):
        calls = []
        fake_torch = types.ModuleType("torch")
        fake_torch.compile = lambda function, **kwargs: (
            calls.append((function, kwargs)) or (lambda *args, **kw: (args, kw))
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            wrapped = compile_patches.compile_transfusion_forward_single_wrapper(
                lambda: "original",
                {
                    "mode": "max-autotune-no-cudagraphs",
                    "fullgraph": False,
                    "dynamic": False,
                },
            )
        self.assertIs(wrapped.__wrapped__, transfusion.transfusion_forward_single)
        self.assertIs(calls[0][0], transfusion.transfusion_forward_single)

    def test_transfusion_cpu_gaussian_helpers_cache_and_draw(self):
        script = textwrap.dedent(
            """
            import torch
            from turbo_physai.optimizations.models.bevfusion import transfusion

            heatmap = torch.zeros(7, 7)
            cache = {}
            transfusion.draw_heatmap_gaussian_cpu(
                heatmap, (3, 3), 2, gaussian_cache=cache
            )
            first = heatmap.clone()
            assert len(cache) == 1
            transfusion.draw_heatmap_gaussian_cpu(
                heatmap, (3, 3), 2, gaussian_cache=cache
            )
            torch.testing.assert_close(heatmap, first)
            assert float(heatmap[3, 3]) == 1.0
            radii = transfusion.gaussian_radius_cpu(
                (torch.tensor([2.0]), torch.tensor([4.0])), min_overlap=0.1
            )
            assert radii.shape == (1,)
            assert float(radii[0]) > 0
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_transfusion_bbox_coder_group_replaces_class_and_builder(self):
        targets = {
            spec.target for spec in catalog.TRANSFUSION_BBOX_CODER.specs
        }
        self.assertEqual(
            targets,
            {
                (
                    "mmdet3d.core.bbox.coders.transfusion_bbox_coder."
                    "TransFusionBBoxCoder"
                ),
                "mmdet.core.bbox.builder.build_bbox_coder",
            },
        )

        replacement_class = type("ReplacementCoder", (), {})
        calls = []

        def original(cfg, *args, **kwargs):
            calls.append((cfg, args, kwargs))
            return cfg

        with mock.patch.object(
            transfusion_bbox_coder,
            "get_transfusion_bbox_coder_class",
            return_value=replacement_class,
        ):
            wrapped = transfusion_bbox_coder.build_bbox_coder_wrapper(
                original, {}
            )
            source = {"type": "TransFusionBBoxCoder", "code_size": 10}
            result = wrapped(source, "default", strict=True)
        self.assertIs(result["type"], replacement_class)
        self.assertEqual(source["type"], "TransFusionBBoxCoder")
        self.assertEqual(calls[0][1], ("default",))
        self.assertEqual(calls[0][2], {"strict": True})

    def test_transfusion_bbox_coder_runtime_matches_optimized_class(self):
        script = textwrap.dedent(
            """
            import sys
            import types
            import torch

            class BaseBBoxCoder:
                pass

            mmdet = types.ModuleType("mmdet")
            core = types.ModuleType("mmdet.core")
            bbox = types.ModuleType("mmdet.core.bbox")
            bbox.BaseBBoxCoder = BaseBBoxCoder
            mmdet.core = core
            core.bbox = bbox
            sys.modules["mmdet"] = mmdet
            sys.modules["mmdet.core"] = core
            sys.modules["mmdet.core.bbox"] = bbox

            from turbo_physai.optimizations.models.bevfusion.transfusion_bbox_coder_runtime import (
                TransFusionBBoxCoder,
            )

            coder = TransFusionBBoxCoder(
                pc_range=[-10.0, -20.0],
                out_size_factor=2,
                voxel_size=[0.5, 0.25],
                post_center_range=[-100, -100, -10, 100, 100, 10],
                score_threshold=0.0,
                code_size=10,
            )
            assert isinstance(coder, torch.nn.Module)
            assert set(coder.state_dict()) == {"pc_range", "voxel_size"}

            boxes = torch.tensor(
                [[-9.0, -19.5, 1.0, 2.0, 3.0, 4.0, 0.0, 0.2, 0.3]]
            )
            encoded = coder.encode(boxes)
            torch.testing.assert_close(encoded[0, :2], torch.tensor([1.0, 1.0]))
            torch.testing.assert_close(encoded[0, 2], torch.tensor(3.0))
            torch.testing.assert_close(encoded[0, 8:], torch.tensor([0.2, 0.3]))

            heatmap = torch.tensor([[[0.8, 0.1], [0.2, 0.9]]])
            rotation = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
            dimensions = torch.zeros(1, 3, 2)
            centers = torch.ones(1, 2, 2)
            heights = torch.full((1, 1, 2), 2.0)
            velocity = torch.zeros(1, 2, 2)
            centers_before = centers.clone()
            dimensions_before = dimensions.clone()
            decoded = coder.decode(
                heatmap, rotation, dimensions, centers, heights, velocity
            )[0]
            torch.testing.assert_close(centers, centers_before)
            torch.testing.assert_close(dimensions, dimensions_before)
            torch.testing.assert_close(
                decoded["bboxes"][0, :2], torch.tensor([-9.0, -19.5])
            )
            # The 10-value encoded representation stores sin/cos rotation;
            # decoding collapses those two values back to one yaw angle.
            assert decoded["bboxes"].shape == (2, 9)

            coder.to(dtype=torch.float64)
            assert coder.pc_range.dtype == torch.float64
            assert coder.voxel_size.dtype == torch.float64
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_all_replacement_paths_resolve_without_model_dependencies(self):
        for optimization in BEVFUSION_GROUPS:
            for spec in optimization.specs:
                self.assertIsNotNone(resolve_replacement(spec.replacement))

    def test_catalog_import_does_not_import_model_stack(self):
        script = textwrap.dedent(
            """
            import sys

            from turbo_physai.optimizations.common.mmdet3d import catalog
            from turbo_physai.optimizations.models.bevfusion import (
                catalog as model_catalog,
            )

            assert catalog is not None
            assert model_catalog is not None
            assert "mmcv" not in sys.modules
            assert "mmdet3d" not in sys.modules
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_extract_features_uses_input_batch_size(self):
        class Backbone:
            def __call__(self, feats, coords, batch_size, sizes=None):
                return feats, coords, batch_size, sizes

        class Model:
            encoders = {"lidar": {"backbone": Backbone()}}

            def voxelize(self, values, sensor):
                del values, sensor
                return "features", "coords", "sizes"

        output = backbone.extract_features(Model(), [object(), object()], "lidar")
        self.assertEqual(output[2], 2)

    def test_training_wrapper_construction_has_no_side_effect(self):
        calls = []

        def original(model, dataset, cfg):
            calls.append((model, dataset, cfg))
            return "ok"

        original.__globals__["build_dataloader"] = object()
        wrapped = training.training_wrapper(
            original,
            {
                "channels_last": False,
                "cuda_prefetch": False,
                "compile_target": "off",
                "start_method": "",
            },
        )
        self.assertEqual(calls, [])
        marker = object()
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            result = wrapped(marker, "dataset", types.SimpleNamespace())
        self.assertEqual(result, "ok")
        self.assertEqual(calls[0][0], marker)

    def test_cuda_prefetch_loader_builds_bev_meta_cfg_inside_class(self):
        class ConfigNode(dict):
            __getattr__ = dict.__getitem__

        encoder = ConfigNode(
            pc_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0],
            num_points_in_pillar=4,
        )
        head = ConfigNode(
            bev_h=180,
            bev_w=180,
            transformer=ConfigNode(encoder=encoder),
        )
        cfg = ConfigNode(model=ConfigNode(pts_bbox_head=head))
        actual = training.CudaPrefetchLoader.build_bev_meta_prefetch_cfg(cfg)
        self.assertEqual(actual["bev_h"], 180)
        self.assertEqual(actual["bev_w"], 180)
        self.assertEqual(actual["num_points_in_pillar"], 4)
        self.assertEqual(actual["buckets"], (9728, 10240, 11264, 12288))

        with mock.patch.dict(os.environ, {"ENABLE_BEV_META_PREFETCH": "0"}):
            self.assertIsNone(
                training.CudaPrefetchLoader.build_bev_meta_prefetch_cfg(cfg)
            )

    def test_cuda_prefetch_loader_prepares_reference_bev_metadata(self):
        script = textwrap.dedent(
            """
            import numpy as np
            from turbo_physai.optimizations.models.bevfusion.training import CudaPrefetchLoader

            loader = object.__new__(CudaPrefetchLoader)
            loader.bev_meta_cfg = {
                "bev_h": 2,
                "bev_w": 2,
                "pc_range": [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
                "num_points_in_pillar": 1,
                "buckets": (4,),
            }
            loader._bev_ref_points = None
            loader._array_to_cuda = lambda array, dtype: np.asarray(array)
            meta = {
                "lidar2img": [np.eye(4, dtype=np.float32)],
                "img_shape": [(10, 10, 3)],
            }
            loader._prepare_bev_meta_group([meta])
            assert meta["_bev_reference_points_cam"].shape == (1, 4, 1, 2)
            assert meta["_bev_bev_mask"].shape == (1, 4, 1)
            assert meta["_bev_index_lengths"].shape == (1,)
            assert meta["_bev_indexes"].shape == (1, 4)
            assert loader._get_bev_reference_points() is loader._bev_ref_points
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_training_wrapper_passes_cfg_to_bev_metadata_prefetch(self):
        namespace = {"build_dataloader": lambda: "base-loader"}
        exec(
            "def original(model, dataset, cfg):\n"
            "    return build_dataloader()\n",
            namespace,
        )
        original = namespace["original"]
        cfg = types.SimpleNamespace()
        captured = {}

        class FakePrefetchLoader:
            @staticmethod
            def build_bev_meta_prefetch_cfg(value):
                self.assertIs(value, cfg)
                return {"bev_h": 180}

            def __init__(self, loader, device=None, bev_meta_cfg=None):
                captured.update(
                    loader=loader, device=device, bev_meta_cfg=bev_meta_cfg
                )

        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 3,
        )
        fake_torch.multiprocessing = types.SimpleNamespace(
            get_start_method=lambda allow_none=True: None,
            set_start_method=lambda *args, **kwargs: None,
        )
        wrapped = training.training_wrapper(
            original,
            {
                "channels_last": False,
                "cuda_prefetch": True,
                "compile_target": "off",
                "start_method": "",
            },
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch.object(
            training, "CudaPrefetchLoader", FakePrefetchLoader
        ):
            result = wrapped(object(), None, cfg)

        self.assertIsInstance(result, FakePrefetchLoader)
        self.assertEqual(
            captured,
            {
                "loader": "base-loader",
                "device": 3,
                "bev_meta_cfg": {"bev_h": 180},
            },
        )
        self.assertEqual(namespace["build_dataloader"](), "base-loader")

    def test_gaussian_helpers_match_reference_values(self):
        script = textwrap.dedent(
            """
            import torch
            from turbo_physai.optimizations.models.bevfusion import gaussian

            heatmap = torch.zeros(7, 7)
            center = torch.tensor([3, 3], dtype=torch.long)
            result = gaussian.draw_heatmap_gaussian(heatmap, center, 1)
            assert float(result[3, 3]) == 1.0
            assert torch.equal(result, result.flip(0))
            assert torch.equal(result, result.flip(1))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_gaussian_compile_wrappers_use_reference_mode(self):
        calls = []
        fake_torch = types.ModuleType("torch")

        def fake_compile(function, **kwargs):
            calls.append((function, kwargs))
            return function

        fake_torch.compile = fake_compile
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            wrapped = gaussian.compiled_gaussian_radius_wrapper(
                object(), {}
            )
        self.assertIsNotNone(wrapped)
        self.assertEqual(
            calls[0][1],
            {
                "mode": "max-autotune-no-cudagraphs",
                "fullgraph": False,
                "dynamic": False,
            },
        )

    def test_dynamo_disable_and_depth_softmax_boundaries(self):
        import torch

        if not hasattr(torch, "compiler"):
            self.skipTest("requires torch.compiler")
        marker = object()
        calls = []
        fake_torch = types.ModuleType("torch")
        fake_torch._dynamo = types.SimpleNamespace(
            disable=lambda function: calls.append(function) or marker
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            wrapped = compile_patches.dynamo_disable_wrapper(
                lambda value: value, {}
            )
        self.assertIs(wrapped, marker)
        self.assertEqual(len(calls), 1)

        script = textwrap.dedent(
            """
            import importlib
            import torch
            from unittest import mock

            with mock.patch(
                "torch.compiler.disable", side_effect=lambda function: function
            ) as disable:
                depth = importlib.import_module(
                    "turbo_physai.optimizations.models.bevfusion.depth"
                )
                result = depth.materialized_softmax(torch.tensor([[1.0, 2.0]]))
                torch.testing.assert_close(result.sum(1), torch.ones(1))
                depth.materialized_softmax(torch.tensor([[2.0, 3.0]]))
                assert disable.call_count == 1
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_factorized_depth_features_match_dense_outer_product_and_gradients(self):
        script = textwrap.dedent(
            """
            import torch
            from turbo_physai.optimizations.models.bevfusion.depth import (
                DepthFeatureFactorization,
                gather_factorized_depth_features,
            )

            torch.manual_seed(7)
            depth = torch.randn(2, 3, 2, 2, dtype=torch.double, requires_grad=True)
            features = torch.randn(2, 2, 2, 4, dtype=torch.double, requires_grad=True)
            kept = torch.tensor([0, 2, 5, 7, 12, 18, 23])
            actual = gather_factorized_depth_features(
                DepthFeatureFactorization(depth, features), kept
            )
            dense = depth.unsqueeze(-1) * features.unsqueeze(1)
            expected = dense.reshape(-1, 4).index_select(0, kept)
            torch.testing.assert_close(actual, expected)

            actual.square().sum().backward(retain_graph=True)
            actual_depth_grad = depth.grad.clone()
            actual_feature_grad = features.grad.clone()
            depth.grad.zero_()
            features.grad.zero_()
            expected.square().sum().backward()
            torch.testing.assert_close(depth.grad, actual_depth_grad)
            torch.testing.assert_close(features.grad, actual_feature_grad)
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_base_depth_transform_forward_vectorizes_camera_depth_writes(self):
        script = textwrap.dedent(
            """
            import types
            import torch
            from turbo_physai.optimizations.models.bevfusion.depth import (
                base_depth_transform_forward,
            )

            captured = {}
            output = torch.tensor([17.0])

            def get_geometry(*args, **kwargs):
                captured["geometry_kwargs"] = kwargs
                return "prepared-geometry"

            def get_cam_feats(image, depth, matrices):
                captured["depth"] = depth.clone()
                captured["matrices"] = matrices
                return torch.zeros(1, 2, 1, 1, 1, 1)

            def bev_pool(geometry, features):
                assert geometry == "prepared-geometry"
                assert features.shape == (1, 2, 1, 1, 1, 1)
                return output

            model = types.SimpleNamespace(
                use_points="lidar",
                height_expand=False,
                depth_input="scalar",
                add_depth_features=True,
                D=4,
                image_size=(4, 5),
                get_geometry=get_geometry,
                get_cam_feats=get_cam_feats,
                bev_pool=bev_pool,
            )
            image = torch.zeros(1, 2, 3, 4, 5)
            points = [torch.tensor([[1.0, 2.0, 2.0, 9.0]])]
            identity = torch.eye(4)
            cameras = identity.reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
            lidar_aug = identity.reshape(1, 4, 4)

            actual = base_depth_transform_forward(
                model,
                image,
                points,
                None,
                cameras,
                lidar_aug,
                cameras,
                cameras,
                cameras,
                cameras,
                cameras,
                lidar_aug,
                [{}],
            )
            assert actual is output
            depth = captured["depth"]
            assert depth.shape == (1, 2, 5, 4, 5)
            torch.testing.assert_close(depth[0, :, 0, 1, 0], torch.tensor([2.0, 2.0]))
            expected_features = torch.tensor([1.0, 2.0, 2.0, 9.0])
            torch.testing.assert_close(depth[0, 0, 1:, 1, 0], expected_features)
            torch.testing.assert_close(depth[0, 1, 1:, 1, 0], expected_features)
            assert set(captured["matrices"]) == {
                "intrin_mats", "ida_mats", "bda_mat", "sensor2ego_mats"
            }
            assert set(captured["geometry_kwargs"]) == {"extra_rots", "extra_trans"}
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_quick_cumsum_matches_reference_and_backward(self):
        script = textwrap.dedent(
            """
            import torch
            from turbo_physai.optimizations.common.mmdet3d.bev_pool import (
                quick_cumsum_backward,
                quick_cumsum_forward,
            )

            class OptimizedQuickCumsum(torch.autograd.Function):
                forward = staticmethod(quick_cumsum_forward)
                backward = staticmethod(quick_cumsum_backward)

            x = torch.randn(8, 3, dtype=torch.double, requires_grad=True)
            geom = torch.arange(32).reshape(8, 4)
            ranks = torch.tensor([0, 0, 1, 1, 1, 3, 4, 4])
            values, selected_geom = OptimizedQuickCumsum.apply(x, geom, ranks)
            expected = torch.stack((x[:2].sum(0), x[2:5].sum(0), x[5], x[6:].sum(0)))
            torch.testing.assert_close(values, expected)
            assert torch.equal(selected_geom, geom[torch.tensor([1, 4, 5, 7])])
            values.sum().backward()
            torch.testing.assert_close(x.grad, torch.ones_like(x))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_factorized_bev_pool_uses_dense_fallback_on_cpu(self):
        script = textwrap.dedent(
            """
            import sys
            import types
            import torch
            from turbo_physai.optimizations.models.bevfusion.depth import (
                DepthFeatureFactorization,
                base_transform_bev_pool,
            )

            calls = []
            def fake_bev_pool(values, coords, batch, depth, height, width, ranks=None):
                calls.append((values.clone(), coords.clone(), ranks))
                return values.new_zeros((batch, values.shape[1], depth, height, width))

            mmdet3d = types.ModuleType("mmdet3d")
            ops = types.ModuleType("mmdet3d.ops")
            ops.bev_pool = fake_bev_pool
            mmdet3d.ops = ops
            sys.modules["mmdet3d"] = mmdet3d
            sys.modules["mmdet3d.ops"] = ops

            model = types.SimpleNamespace(
                bx=torch.tensor([0.5, 0.5, 0.5]),
                dx=torch.ones(3),
                nx=torch.tensor([2, 2, 1]),
                xbound=(0.0, 2.0, 1.0),
                ybound=(0.0, 2.0, 1.0),
                zbound=(0.0, 1.0, 1.0),
            )
            depth = torch.tensor([[[[0.2, 0.3]], [[0.8, 0.7]]]])
            features = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
            factors = DepthFeatureFactorization(depth, features)
            geom = torch.tensor([[[[[[0.5, 0.5, 0.5], [3.5, 0.5, 0.5]]],
                                     [[[1.5, 0.5, 0.5], [0.5, 1.5, 0.5]]]]]])
            result = base_transform_bev_pool(model, geom, factors)
            assert result.shape == (1, 2, 2, 2)
            values, coords, ranks = calls[0]
            dense = (depth.unsqueeze(-1) * features.unsqueeze(1)).reshape(-1, 2)
            torch.testing.assert_close(
                values, dense.index_select(0, torch.tensor([0, 2, 3]))
            )
            assert coords.tolist() == [
                [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]
            ]
            assert ranks is None
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_hungarian_transfer_preserves_assignment(self):
        import torch

        try:
            torch.tensor([0]).numpy()
        except (RuntimeError, ImportError):
            self.skipTest("requires Torch/NumPy interoperability")
        script = textwrap.dedent(
            """
            import sys
            import types
            import torch
            from turbo_physai.optimizations.models.bevfusion.transfusion import hungarian_assign

            class AssignResult:
                def __init__(self, num_gts, gt_inds, overlaps, labels=None):
                    self.num_gts = num_gts
                    self.gt_inds = gt_inds
                    self.max_overlaps = overlaps
                    self.labels = labels
                    self.extra = {}
                def set_extra_property(self, name, value):
                    self.extra[name] = value

            module = types.ModuleType("mmdet.core.bbox.assigners")
            module.AssignResult = AssignResult
            sys.modules["mmdet"] = types.ModuleType("mmdet")
            sys.modules["mmdet.core"] = types.ModuleType("mmdet.core")
            sys.modules["mmdet.core.bbox"] = types.ModuleType("mmdet.core.bbox")
            sys.modules["mmdet.core.bbox.assigners"] = module

            class Assigner:
                cls_cost = staticmethod(lambda pred, labels: torch.tensor([[0., 9.], [9., 0.], [5., 6.]]))
                reg_cost = staticmethod(lambda boxes, gt, cfg: torch.zeros(3, 2))
                iou_cost = staticmethod(lambda iou: torch.zeros_like(iou))
                iou_calculator = staticmethod(lambda boxes, gt: torch.tensor([[.8, .1], [.2, .9], [.3, .4]]))

            boxes = torch.zeros(3, 4)
            gt = torch.zeros(2, 4)
            labels = torch.tensor([4, 7])
            cls_pred = torch.zeros(1, 2, 3)
            result = hungarian_assign(Assigner(), boxes, gt, labels, cls_pred, {})
            assert torch.equal(result.gt_inds, torch.tensor([1, 2, 0]))
            assert torch.equal(result.labels, torch.tensor([4, 7, -1]))
            assert torch.equal(result.extra["matched_row_inds"], torch.tensor([0, 1]))
            assert torch.equal(result.extra["matched_col_inds"], torch.tensor([0, 1]))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_native_bev_pool_frontend_forward_and_backward(self):
        script = textwrap.dedent(
            """
            import torch
            from turbo_physai.optimizations.common.mmdet3d import bev_pool

            class Extension:
                @staticmethod
                def bev_pool_forward(x, coords, lengths, starts, B, D, H, W):
                    output = x.new_zeros((B, D, H, W, x.shape[1]))
                    for row, coord in zip(x, coords):
                        px, py, pz, batch = coord.tolist()
                        output[batch, pz, py, px] += row
                    return output
                @staticmethod
                def bev_pool_backward(grad, coords, lengths, starts, B, D, H, W):
                    rows = []
                    for coord in coords:
                        px, py, pz, batch = coord.tolist()
                        rows.append(grad[batch, pz, py, px])
                    return torch.stack(rows)

            bev_pool._ops = lambda: Extension()
            bev_pool._NATIVE_BEV_POOL_AUTOGRAD = None
            features = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
            coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]])
            output = bev_pool.bev_pool(features, coords, 1, 1, 1, 1)
            assert output.shape == (1, 2, 1, 1, 1)
            torch.testing.assert_close(output.flatten(), torch.tensor([4., 6.]))
            output.sum().backward()
            torch.testing.assert_close(features.grad, torch.ones_like(features))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_voxel_and_indice_frontends_use_bundled_extensions(self):
        script = textwrap.dedent(
            """
            import torch
            from turbo_physai.optimizations.common.mmdet3d import (
                sparse_conv as indice,
                voxelization as voxel,
            )

            calls = []
            class VoxelExtension:
                @staticmethod
                def hard_voxelize(points, voxels, coords, counts, *args):
                    calls.append("hard")
                    voxels[0, 0] = points[0]
                    coords[0] = torch.tensor([1, 2, 3])
                    counts[0] = 1
                    return 1
            voxel._ops = lambda: VoxelExtension()
            points = torch.tensor([[1., 2., 3., 4.]])
            result = voxel.voxelization_forward(
                None, points, [1., 1., 1.], [0., 0., 0., 4., 4., 4.], 2, 8, True
            )
            assert calls == ["hard"]
            assert result[0].shape == (1, 2, 4)
            assert result[1].tolist() == [[1, 2, 3]]

            marker = object()
            class IndiceExtension:
                @staticmethod
                def get_indice_pairs_3d(*args):
                    calls.append(args)
                    return marker
            indice._ops = lambda: IndiceExtension()
            indices = torch.zeros((2, 4), dtype=torch.int32)
            output = indice.get_indice_pairs(
                indices, 1, [4, 5, 6], 3, 1, 1, 1, 0, False, False
            )
            assert output is marker
            assert calls[-1][2] == [4, 5, 6]
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_bev_geometry_native_entry_and_boundary_correction(self):
        script = textwrap.dedent(
            """
            import types
            import torch
            from turbo_physai.optimizations.common.mmdet3d import bev_pool
            from turbo_physai.optimizations.models.bevfusion import depth

            marker = (object(), object(), object(), object())
            calls = []
            class Extension:
                @staticmethod
                def bev_pool_prepare_geometry(*args):
                    calls.append(args)
                    return marker
            bev_pool._ops = lambda: Extension()
            tensor = torch.zeros(1, 1, 1, 1, 1, 3)
            matrix = torch.eye(3).reshape(1, 1, 3, 3)
            vector = torch.zeros(1, 1, 3)
            result = bev_pool.bev_pool_prepare_geometry(
                tensor, matrix, vector, matrix, vector,
                matrix[:, 0], vector[:, 0],
                torch.ones(3), torch.ones(3), torch.ones(3, dtype=torch.long),
                1, 1, 1, 1, boundary_eps=0.002,
            )
            assert result is marker
            assert calls[0][-1] == 0.002

            model = types.SimpleNamespace(
                frustum=torch.tensor([[[[0.0, 0.0, 1.0]]]]),
                bx=torch.tensor([0.5, 0.5, 0.5]),
                dx=torch.ones(3),
                nx=torch.tensor([2, 2, 2]),
                xbound=(0.0, 2.0, 1.0),
                ybound=(0.0, 2.0, 1.0),
                zbound=(0.0, 2.0, 1.0),
            )
            coords = torch.full((1, 4), -7, dtype=torch.int32)
            ranks = torch.full((1,), -7, dtype=torch.int32)
            kept = torch.zeros(1, dtype=torch.bool)
            boundary = torch.ones(1, dtype=torch.bool)
            corrected = depth._correct_geometry_boundaries(
                model, coords, ranks, kept, boundary,
                matrix, vector, matrix, vector,
                matrix[:, 0], vector[:, 0], 1,
            )
            assert corrected[0].tolist() == [[0, 0, 1, 0]]
            assert corrected[1].tolist() == [1]
            assert corrected[2].tolist() == [True]
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_prepared_geometry_consumes_only_kept_factorized_rows(self):
        script = textwrap.dedent(
            """
            import sys
            import types
            import torch
            from turbo_physai.optimizations.models.bevfusion.depth import (
                DepthFeatureFactorization,
                PreparedGeometry,
                base_transform_bev_pool,
            )

            calls = []
            def fake_bev_pool(values, coords, batch, depth, height, width, ranks=None):
                calls.append((values.clone(), coords.clone(), ranks.clone()))
                return values.new_zeros((batch, values.shape[1], depth, height, width))
            mmdet3d = types.ModuleType("mmdet3d")
            ops = types.ModuleType("mmdet3d.ops")
            ops.bev_pool = fake_bev_pool
            mmdet3d.ops = ops
            sys.modules["mmdet3d"] = mmdet3d
            sys.modules["mmdet3d.ops"] = ops

            model = types.SimpleNamespace(
                xbound=(0.0, 2.0, 1.0),
                ybound=(0.0, 2.0, 1.0),
                zbound=(0.0, 1.0, 1.0),
            )
            depth = torch.tensor([[[[0.2, 0.3]], [[0.8, 0.7]]]])
            features = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
            factors = DepthFeatureFactorization(depth, features)
            coords = torch.tensor(
                [[0, 0, 0, 0], [9, 9, 9, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
                dtype=torch.int32,
            )
            ranks = torch.tensor([0, -1, 2, 1], dtype=torch.int32)
            kept = torch.tensor([True, False, True, True])
            prepared = PreparedGeometry(coords, ranks, kept, 1)
            base_transform_bev_pool(model, prepared, factors)

            values, selected_coords, selected_ranks = calls[0]
            dense = (depth.unsqueeze(-1) * features.unsqueeze(1)).reshape(-1, 2)
            expected = dense.index_select(0, torch.tensor([0, 2, 3]))
            torch.testing.assert_close(values, expected)
            assert torch.equal(selected_coords, coords[kept])
            assert torch.equal(selected_ranks, ranks[kept])
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_compile_wrapper_reports_unknown_target_at_execution(self):
        def original(model, dataset, cfg):
            del model, dataset, cfg

        wrapped = training.training_wrapper(
            original,
            {
                "channels_last": False,
                "cuda_prefetch": False,
                "compile_target": "invalid",
                "start_method": "",
            },
        )
        fake_torch = types.ModuleType("torch")
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(ValueError, "unknown TurboPhysAI"):
                wrapped(mock.Mock(), None, types.SimpleNamespace())


if __name__ == "__main__":
    unittest.main()
