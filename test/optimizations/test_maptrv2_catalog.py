# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Structural tests for the MapTRv2 optimization catalog."""

import ast
import importlib
import inspect
import os
import pathlib
import subprocess
import sys
import unittest

from turbo_physai.engine.contracts import Mechanism
from turbo_physai.engine.definitions.registry import default_registry

import turbo_physai.optimizations.models.maptrv2.catalog as catalog


EXPECTED_GROUPS = {
    "maptrv2.training": Mechanism.WRAPPER,
    "maptrv2.data": Mechanism.REPLACE,
    "maptrv2.datacontainer_pin": Mechanism.WRAPPER,
    "maptrv2.fused_adamw": Mechanism.WRAPPER,
    "maptrv2.grid_mask": Mechanism.REPLACE,
    "maptrv2.compile": Mechanism.WRAPPER,
    "maptrv2.match_cost": Mechanism.REPLACE,
    "maptrv2.pv_mask": Mechanism.REPLACE,
    "maptrv2.assigner": Mechanism.REPLACE,
    "maptrv2.sync_elimination": Mechanism.REPLACE,
    "maptrv2.efficientnet": Mechanism.REGISTRY_OVERRIDE,
    "maptrv2.spconv_registry": Mechanism.REGISTRY_OVERRIDE,
    "maptrv2.bev_pool_fix": Mechanism.REPLACE,
    "maptrv2.reference_boundaries": Mechanism.WRAPPER,
    "maptrv2.dataset_vectorization": Mechanism.REPLACE,
    "maptrv2.eval_compat": Mechanism.REPLACE,
}

# Groups whose members must re-check an optional dependency on every call.
# The condition is resolved by the engine and must stay a plain callable.
EXPECTED_CONDITIONS = {
    "maptrv2.compile": (
        "turbo_physai.optimizations.models.maptrv2.compat.torch_compile_available", 10),
    "maptrv2.assigner": (
        "turbo_physai.optimizations.models.maptrv2.compat.static_assigner_available", 3),
    "maptrv2.pv_mask": (
        "turbo_physai.optimizations.models.maptrv2.compat.pv_mask_sampling_enabled", 3),
    "maptrv2.reference_boundaries": (
        "turbo_physai.optimizations.models.maptrv2.compat.torch_compile_available", 5),
    "maptrv2.eval_compat": (
        "turbo_physai.optimizations.models.maptrv2.compat.shapely_query_returns_indices", 1),
}

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _find_clean_map_checkout():
    model_root = REPO_ROOT.parent / "model"
    if not model_root.is_dir():
        return None
    return next(
        (
            path
            for path in sorted(model_root.iterdir())
            if path.is_dir() and path.name.lower() == "maptrv2"
        ),
        None,
    )


MODEL_ROOT = _find_clean_map_checkout()


def _target_exists(repo_root, target):
    parts = target.split(".")
    search_root = (
        repo_root / "mmdetection3d"
        if parts[0] == "mmdet3d"
        else repo_root
    )
    module_parts = None
    for split in range(len(parts), 0, -1):
        candidate = search_root.joinpath(*parts[:split]).with_suffix(".py")
        if candidate.is_file():
            module_parts = parts[:split]
            break
    if module_parts is None:
        # Targets that do not live in the checkout (site-packages, e.g.
        # ``mmcv.parallel.distributed.MMDistributedDataParallel.__init__``)
        # are declared against what the interpreter imports instead.
        from turbo_physai.engine.execution.replacements.base import (
            HandlerError,
            resolve_attribute,
        )

        try:
            resolve_attribute(target)
        except HandlerError:
            return False
        return True

    module_path = search_root.joinpath(*module_parts).with_suffix(".py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    body = tree.body
    for name in parts[len(module_parts):]:
        node = next(
            (
                item
                for item in body
                if isinstance(
                    item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and item.name == name
            ),
            None,
        )
        if node is None:
            # The optimizer hook targets a model-local imported alias, not a
            # function defined in this file. Resolve its original provider.
            for item in body:
                if isinstance(item, ast.ImportFrom) and item.module:
                    for alias in item.names:
                        if (alias.asname or alias.name) == name:
                            return _target_exists(repo_root, item.module + "." + alias.name)
            return False
        body = node.body
    return True


class CatalogTest(unittest.TestCase):
    def test_catalog_can_be_imported(self):
        importlib.import_module("turbo_physai.optimizations.models.maptrv2.catalog")

    def test_every_group_is_registered(self):
        self.assertEqual(
            sorted(catalog.__all__),
            [
                "ASSIGNER", "BEV_POOL_FIX", "COMPILE", "DATA",
                "DATACONTAINER_PIN", "DATASET_VECTORIZATION", "EFFICIENTNET",
                "EVAL_COMPAT", "FUSED_ADAMW", "GRID_MASK", "MATCH_COST", "PV_MASK",
                "REFERENCE_BOUNDARIES", "SPARSE_CONV_REGISTRY", "SYNC_ELIMINATION", "TRAINING",
            ],
        )
        for group_id in EXPECTED_GROUPS:
            with self.subTest(group_id=group_id):
                self.assertIsNotNone(default_registry.get_group(group_id))

    def test_exported_declarations_cover_every_group(self):
        declared = {
            getattr(catalog, name).group_id for name in catalog.__all__
        }
        self.assertEqual(declared, set(EXPECTED_GROUPS))

    def test_targets_point_at_the_official_baseline(self):
        targets = {
            spec.target
            for group_id in EXPECTED_GROUPS
            for replacement_id in default_registry.get_group(group_id).members
            for spec in (default_registry.get_spec(replacement_id),)
            if spec is not None
        }
        self.assertIn(
            "projects.mmdet3d_plugin.bevformer.apis.mmdet_train."
            "custom_train_detector",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.datasets.builder.build_dataloader", targets
        )
        self.assertIn(
            "projects.mmdet3d_plugin.models.utils.grid_mask.GridMask.forward",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.losses.map_loss."
            "OrderedPtsL1Cost.__call__",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.line_ego_to_pvmask",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.line_ego_to_mask",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.datasets.nuscenes_offlinemap_dataset."
            "VectorizedLocalMap.gen_vectorized_samples",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "LSSTransform.get_cam_feats",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "normalize_2d_pts",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.assigners.maptr_assigner."
            "MapTRAssigner.assign",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.modules.encoder.BaseTransform.bev_pool",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "MapTRv2Head.loss",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "MapTRv2Head._get_target_single",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "BaseTransform.get_geometry_v1",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "BaseTransform.forward",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.modules.encoder."
            "LSSTransform.forward",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.modules.transformer."
            "MapTRPerceptionTransformer.forward",
            targets,
        )
        self.assertIn(
            "projects.mmdet3d_plugin.maptr.dense_heads.maptrv2_head."
            "MapTRv2Head.forward",
            targets,
        )

    @unittest.skipIf(
        MODEL_ROOT is None, "sibling clean MapTRv2 checkout is unavailable"
    )
    def test_targets_resolve_in_clean_map_checkout(self):
        targets = {
            spec.target
            for group_id in EXPECTED_GROUPS
            for replacement_id in default_registry.get_group(group_id).members
            for spec in (default_registry.get_spec(replacement_id),)
            if spec is not None
        }
        for target in sorted(targets):
            with self.subTest(target=target):
                self.assertTrue(
                    _target_exists(MODEL_ROOT, target),
                    f"catalog target does not resolve: {target}",
                )

    def test_optional_groups_dispatch_through_compat_probes(self):
        for group_id, (condition, member_count) in EXPECTED_CONDITIONS.items():
            group = default_registry.get_group(group_id)
            with self.subTest(group_id=group_id):
                self.assertEqual(len(group.members), member_count)
            for replacement_id in group.members:
                spec = default_registry.get_spec(replacement_id)
                with self.subTest(group_id=group_id, target=spec.target):
                    self.assertEqual(spec.runtime_condition, condition)

    def test_declared_conditions_are_plain_callables(self):
        from turbo_physai.engine.execution.replacements.base import (
            resolve_replacement,
        )

        conditions = {item[0] for item in EXPECTED_CONDITIONS.values()}
        for condition in sorted(conditions):
            probe = resolve_replacement(condition)
            with self.subTest(condition=condition):
                self.assertTrue(callable(probe))
                self.assertFalse(inspect.isclass(probe))
                self.assertIsInstance(probe(), bool)

    def test_catalog_import_does_not_load_torch(self):
        """Declaring groups must not pull Torch into a fresh interpreter."""

        script = (
            "import sys;"
            "import turbo_physai.optimizations.models.maptrv2.catalog;"
            "sys.exit(1 if 'torch' in sys.modules or 'apex' in sys.modules else 0)"
        )
        environment = dict(os.environ)
        # Ship the parent's import path so ``turbo_physai`` resolves in the child
        # no matter whether it was installed or picked up from the CWD.
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT)] + [entry for entry in sys.path if entry])
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        self.assertEqual(
            completed.returncode,
            0,
            completed.stderr.decode("utf-8", "replace"),
        )

    def test_bev_helpers_import_without_model_dependencies(self):
        script = (
            "import sys;"
            "sys.modules['torch'] = None;"
            "sys.modules['mmdet3d'] = None;"
            "from turbo_physai.optimizations.models.maptrv2 "
            "import bev_pool_fix, reference;"
            "assert callable(bev_pool_fix.normalize_bev_pool_layout);"
            "assert reference.normalize_bev_pool_layout "
            "is bev_pool_fix.normalize_bev_pool_layout;"
            "assert sys.modules['torch'] is None;"
            "assert sys.modules['mmdet3d'] is None"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT)] + [entry for entry in sys.path if entry])
        completed = subprocess.run(
            [sys.executable, "-c", script], cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
        )
        self.assertEqual(completed.returncode, 0,
                         completed.stderr.decode("utf-8", "replace"))

    def test_mechanism_per_group(self):
        for group_id, mechanism in EXPECTED_GROUPS.items():
            group = default_registry.get_group(group_id)
            for replacement_id in group.members:
                spec = default_registry.get_spec(replacement_id)
                with self.subTest(group_id=group_id):
                    self.assertEqual(spec.mechanism, mechanism)

    def test_sparse_conv_registry_covers_all_reference_classes(self):
        group = default_registry.get_group("maptrv2.spconv_registry")
        spec = default_registry.get_spec(group.members[0])
        self.assertEqual(
            spec.mechanism_options["names"],
            (
                "SparseConv2d",
                "SparseConv3d",
                "SparseConv4d",
                "SparseConvTranspose2d",
                "SparseConvTranspose3d",
                "SparseInverseConv2d",
                "SparseInverseConv3d",
                "SubMConv2d",
                "SubMConv3d",
                "SubMConv4d",
            ),
        )


if __name__ == "__main__":
    unittest.main()
