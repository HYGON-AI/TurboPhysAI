# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Small integration regressions; no Torch/model/GPU imports or execution."""

import ast
import inspect
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from turbo_physai.engine.contracts import RestoreStatus
from turbo_physai.engine.execution.replacements.base import (
    MechanismHandler, resolve_attribute, set_attribute,
)
from turbo_physai.optimizations.models.openwam.settings import port_wrapper

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "turbo_physai/optimizations/models/openwam"


def load_definition(relative, name, **namespace):
    # Execute only the named definition with stand-in dependencies. In particular
    # this must never import Torch, OpenWAM, Accelerate or DeepSpeed.
    path = MODEL / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    namespace.setdefault("logger", logging.getLogger(__name__))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name], namespace


class DescriptorRollbackTest(unittest.TestCase):
    def setUp(self):
        self.module = types.ModuleType("openwam_review_fixture")
        sys.modules[self.module.__name__] = self.module
        self.addCleanup(sys.modules.pop, self.module.__name__, None)

    def test_classmethod_install_and_rollback_keep_subclass_binding(self):
        class Dataset:
            @classmethod
            def from_config(cls, config):
                return cls, config

        class Child(Dataset):
            pass

        def implementation(cls, config):
            return cls, config + 1

        self.module.Dataset = Dataset
        target = resolve_attribute("openwam_review_fixture.Dataset.from_config")
        descriptor = inspect.getattr_static(Dataset, "from_config")
        factory = port_wrapper(types.MethodType(implementation, Dataset), target.path, "openwam.data")
        replacement = factory(target.original, {})
        with patch("turbo_physai.optimizations.models.openwam.settings.initialize"):
            set_attribute(target, replacement)
            self.assertEqual(Child.from_config(3), (Child, 4))
            result = MechanismHandler().restore((target,))
        self.assertEqual(result[0].status, RestoreStatus.RESTORED)
        self.assertIs(inspect.getattr_static(Dataset, "from_config"), descriptor)
        self.assertEqual(Child.from_config(3), (Child, 3))

    def test_inherited_method_does_not_become_an_owned_attribute(self):
        class Base:
            @classmethod
            def make(cls):
                return cls

        class Child(Base):
            pass

        self.module.Child = Child
        target = resolve_attribute("openwam_review_fixture.Child.make")
        set_attribute(target, lambda: None)
        self.assertEqual(MechanismHandler().restore((target,))[0].status, RestoreStatus.RESTORED)
        self.assertNotIn("make", vars(Child))
        self.assertIs(Child.make(), Child)

    def test_staticmethod_works_from_instances_after_rollback(self):
        class Owner:
            @staticmethod
            def value(number):
                return number

        self.module.Owner = Owner
        target = resolve_attribute("openwam_review_fixture.Owner.value")
        set_attribute(target, lambda number: -number)
        self.assertEqual(MechanismHandler().restore((target,))[0].status, RestoreStatus.RESTORED)
        self.assertEqual(Owner().value(5), 5)


class OpenWAMBoundaryTest(unittest.TestCase):
    def test_existing_queue_setting_wins_on_repeated_setup(self):
        function, _ = load_definition("hw_queues.py", "configure", requested=lambda: 16, os=os)
        torch = types.SimpleNamespace(version=types.SimpleNamespace(hip="test"), cuda=Mock())
        torch.cuda.is_initialized.return_value = True
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(os.environ, {"GPU_MAX_HW_QUEUES": "8"}, clear=True):
            function()
            function()
            self.assertEqual(os.environ["GPU_MAX_HW_QUEUES"], "8")
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "before the HIP"):
                function()
            self.assertNotIn("GPU_MAX_HW_QUEUES", os.environ)
        torch.cuda.is_initialized.return_value = False
        with patch.dict(sys.modules, {"torch": torch}), patch.dict(os.environ, {}, clear=True):
            function()
            self.assertEqual(os.environ["GPU_MAX_HW_QUEUES"], "16")

    def test_timer_update_keeps_other_timer_configuration(self):
        function, _ = load_definition("host_async.py", "configure_deepspeed")
        original = {"throughput": {"enabled": True, "synchronized": True, "custom": 7}, "other": {"enabled": True}}
        config = {"timers": original}
        function(config)
        self.assertEqual(config["timers"]["other"], {"enabled": True})
        self.assertEqual(config["timers"]["throughput"], {"enabled": False, "synchronized": False, "custom": 7})
        self.assertTrue(original["throughput"]["enabled"])

    def test_only_native_wan_vae_gets_scale_cache(self):
        class NativeVAE:
            pass

        native = types.ModuleType("openwam.model.video_backbone.wan.models.vae")
        native.WanVideoVAE = NativeVAE
        port = types.ModuleType("turbo_physai.optimizations.models.openwam.ports.vae")
        port.WanVideoVAE_prepare_scale_cache = Mock()
        function, _ = load_definition("__init__.py", "prepare_runtime_constants", enabled=lambda name: True)
        architecture = types.SimpleNamespace(device="device", video_backbone=types.SimpleNamespace(vae=object()))
        with patch.dict(sys.modules, {native.__name__: native, port.__name__: port}):
            function(architecture)
            port.WanVideoVAE_prepare_scale_cache.assert_not_called()
            vae = architecture.video_backbone.vae = NativeVAE()
            function(architecture)
            port.WanVideoVAE_prepare_scale_cache.assert_called_once_with(vae)


class DeferredLifetimeTest(unittest.TestCase):
    def make_gather(self):
        dist = Mock()
        dist.get_world_size.return_value = 1
        dist.get_global_rank.return_value = 0
        dist.get_process_group_ranks.return_value = [0]
        kind, namespace = load_definition(
            "deferred_gather.py", "DeferredGather", dist=dist,
            validate_zero=lambda opt: None, layer_stages=lambda arch: ({}, 0),
            step_with_gather=lambda opt, gather: Mock(), PENDING=None,
        )
        optimizer = types.SimpleNamespace(dp_process_group=object(), bit16_groups_flat=[], step=Mock())
        gather = kind(object(), optimizer)
        gather._check_plan = Mock()
        return gather, dist, namespace

    def test_constructor_allocates_no_group_and_exit_destroys_owned_group(self):
        gather, dist, _ = self.make_gather()
        original = gather.optimizer.step
        dist.new_group.assert_not_called()
        with gather:
            group = gather.group
            self.assertIs(gather.optimizer.step, gather.step)
        self.assertIs(gather.optimizer.step, original)
        self.assertIsNone(gather.group)
        dist.destroy_process_group.assert_called_once_with(group)

    def test_failed_entry_releases_group_without_patching_optimizer(self):
        gather, dist, _ = self.make_gather()
        original = gather.optimizer.step
        gather._check_plan.side_effect = ValueError("bad plan")
        with self.assertRaisesRegex(ValueError, "bad plan"):
            gather.__enter__()
        self.assertIs(gather.optimizer.step, original)
        self.assertIsNone(gather.group)
        dist.destroy_process_group.assert_called_once_with(dist.new_group.return_value)

    def test_wait_error_still_restores_optimizer_and_releases_group(self):
        gather, dist, namespace = self.make_gather()
        original = gather.optimizer.step
        gather.__enter__()
        namespace["PENDING"] = gather
        gather.finish = Mock(side_effect=RuntimeError("wait failed"))
        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            gather.__exit__(None, None, None)
        self.assertIs(gather.optimizer.step, original)
        self.assertIsNone(namespace["PENDING"])
        dist.destroy_process_group.assert_called_once()

    def test_partial_broadcast_failure_waits_for_submitted_work(self):
        gather, dist, namespace = self.make_gather()
        flat = Mock()
        gather.flats = [flat]
        gather.pieces = [types.SimpleNamespace(group=0, start=i, length=1, owner=0, stage=0) for i in range(2)]
        work = Mock()
        dist.broadcast.side_effect = [work, RuntimeError("launch failed")]
        with self.assertRaisesRegex(RuntimeError, "launch failed"):
            with gather:
                gather.gather([flat], [], [])
        work.wait.assert_called_once()
        self.assertIsNone(namespace["PENDING"])
        dist.destroy_process_group.assert_called_once()


if __name__ == "__main__":
    unittest.main()
