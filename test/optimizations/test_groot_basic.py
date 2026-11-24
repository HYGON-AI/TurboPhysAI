"""Catalog, generated configuration and official-target injection."""

import importlib
import importlib.util
import unittest
from pathlib import Path

import pytest
import yaml

from turbo_physai.engine.config.generation_record import verify_generated
from turbo_physai.engine.execution.replacements import default_handlers
from turbo_physai.optimizations.models.groot import catalog


# Catalog

class CatalogTest(unittest.TestCase):
    def test_catalog_can_be_imported(self):
        importlib.import_module("turbo_physai.optimizations.models.groot.catalog")


# Configuration

CONFIGS = Path(catalog.__file__).resolve().parent / "configs"


def test_config_receipt_is_valid():
    assert verify_generated(CONFIGS / "optimization.yaml")["model_commit"] == "51d4c89f72fda44cbf77285c6a8114b52676b8a1"


def test_generated_config_matches_recipe_group_enablement():
    recipe = yaml.safe_load((CONFIGS / "recipe.yaml").read_text())
    config = yaml.safe_load((CONFIGS / "optimization.yaml").read_text())
    recipe_groups = {g["id"]: g["enabled"] for g in recipe["optimization_groups"]}
    config_groups = {g["id"]: g["enabled"] for g in config["optimization_groups"]}
    assert len(config_groups) == 18
    assert config_groups == recipe_groups
    assert sum(config_groups.values()) == 17
    assert {k for k, v in config_groups.items() if not v} == {"groot.compat.opencv_video"}
    assert config_groups["groot.compat.training_config"]
    assert not config_groups["groot.compat.opencv_video"]
    allgather = next(g for g in config["optimization_groups"] if g["id"] == "groot.zero2.allgather_pipeline")
    assert allgather["options"]["allow_checkpointing"] is True


@pytest.mark.model_deps
def test_all_groups_prepare_without_conflicts(tmp_path):
    """Enable every Group in a temporary copy of the recipe; no second config is shipped."""
    import importlib.util
    for name in ("gr00t", "transformers", "deepspeed", "apex", "amp_C"):
        if importlib.util.find_spec(name) is None:
            pytest.skip(f"{name} is unavailable")
    from turbo_physai.optimizations.models.groot import catalog  # noqa: F401
    from turbo_physai.engine.checking.context import detect_context
    from turbo_physai.engine.checking.ordering import Preparation
    from turbo_physai.engine.config.loader import load_optimization_config
    from turbo_physai.engine.contracts import Decision
    from turbo_physai.engine.definitions.registry import default_registry
    from turbo_physai.engine.execution.replacements import default_handlers

    recipe = yaml.safe_load((CONFIGS / "recipe.yaml").read_text())
    for group in recipe["optimization_groups"]:
        group["enabled"] = True
    path = tmp_path / "all_enabled.yaml"
    path.write_text(yaml.safe_dump(recipe))

    prepared = Preparation(default_registry, default_handlers()).prepare(
        run_id="groot-integration-check",
        config=load_optimization_config(path),
        environment=detect_context(),
    )
    assert not prepared.conflicts
    assert len(prepared.groups) == 18
    assert all(group.decision is Decision.APPLY for group in prepared.groups)


# Integration

@pytest.mark.model_deps
@pytest.mark.parametrize("declaration", [
    catalog.TRAINING_CONFIG, catalog.OPENCV_VIDEO,
    catalog.DIT_HIP_MATH_SDPA, catalog.QWEN_FAST_POS, catalog.QWEN_ROT_POS,
    catalog.QWEN_ROPE, catalog.FLASH_DISABLE, catalog.COMPILE_VISION,
    catalog.COMPILE_DIT, catalog.COMPILE_VL, catalog.PIXEL_VALUES_BF16, catalog.FIXED_PADDING,
    catalog.APEX_FUSED_ADAM, catalog.ZERO2_OVERLAP, catalog.DISABLE_KV_CACHE,
    catalog.COMPILE_TEXT, catalog.ZERO2_OWNER_REDUCE, catalog.ZERO2_ALLGATHER,
], ids=lambda item: item.group_id)
def test_handler_applies_and_restores_all_targets(declaration):
    for module in ("gr00t", "transformers", "deepspeed"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not installed")
    handlers = default_handlers()
    prepared = []
    try:
        for spec in declaration.specs:
            handler = handlers[spec.mechanism]
            item = handler.prepare(spec, {})
            snapshot = handler.snapshot(item)
            prepared.append((handler, snapshot))
            handler.apply(item)
            for target in item.targets:
                value = target.parent[target.attribute] if target.is_mapping else getattr(target.parent, target.attribute)
                assert value is item.replacement
    finally:
        for handler, snapshot in reversed(prepared):
            handler.restore(snapshot)
            for target in snapshot:
                value = target.parent[target.attribute] if target.is_mapping else getattr(target.parent, target.attribute)
                assert value is target.original


@pytest.mark.model_deps
def test_gro_and_owner_apply_together_and_restore_in_reverse():
    pytest.importorskip("deepspeed")
    handlers = default_handlers()
    prepared = []
    try:
        for declaration in (catalog.ZERO2_OVERLAP, catalog.ZERO2_OWNER_REDUCE):
            for spec in declaration.specs:
                handler = handlers[spec.mechanism]
                item = handler.prepare(spec, {})
                snapshot = handler.snapshot(item)
                prepared.append((handler, snapshot, item))
                handler.apply(item)
        for _, _, item in prepared:
            for target in item.targets:
                assert getattr(target.parent, target.attribute) is item.replacement
    finally:
        for handler, snapshot, _ in reversed(prepared):
            handler.restore(snapshot)
            for target in snapshot:
                assert getattr(target.parent, target.attribute) is target.original
