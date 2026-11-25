# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""maptrv2.sync_elimination: compare actual baseline methods, shapes and gradients.

CPU tests load unchanged methods from the clean upstream source with controlled
assignment targets and reduction losses. Full loss-stack and device cases are
marked explicitly; no performance collection is installed.
"""

import ast
import copy
import importlib
import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import types

import pytest
import yaml

torch = pytest.importorskip("torch")
mmcv = pytest.importorskip("mmcv")
from mmcv.runner import force_fp32
from mmcv.utils import TORCH_VERSION, digit_version

from turbo_physai.engine.checking.evidence import source_hash, ast_hash
from turbo_physai.optimizations.models.maptrv2 import catalog
from turbo_physai.optimizations.models.maptrv2 import sync_elimination as sync


ROOT = Path(__file__).resolve().parents[2]
GROUP = "maptrv2.sync_elimination"
SPECS = (
    ("maptr.dense_heads.maptrv2_head", "MapTRv2Head", "loss_single", "maptrv2_head_loss_single"),
    ("maptr.dense_heads.maptr_head", "MapTRHead", "loss_single", "maptr_head_loss_single"),
    ("bevformer.dense_heads.bevformer_head", "BEVFormerHead", "loss_single", "bevformer_head_loss_single"),
    ("bevformer.modules.encoder", "BEVFormerEncoder", "point_sampling", "bevformer_point_sampling"),
)


def reference_root():
    candidates = [Path(os.environ["TURBO_PHYSAI_MAPTRV2_BASELINE"])] if os.environ.get(
        "TURBO_PHYSAI_MAPTRV2_BASELINE") else []
    candidates += [ROOT.parent / "MapTrv2_base", ROOT.parent / "model/MapTrv2"]
    for path in candidates:
        if (path / "projects/mmdet3d_plugin/maptr/dense_heads/maptrv2_head.py").is_file():
            return path.resolve()
    pytest.skip("clean MapTRv2 sources required; set TURBO_PHYSAI_MAPTRV2_BASELINE")


def _reduce_mean(tensor):
    if not torch.distributed.is_initialized():
        return tensor
    value = tensor.clone()
    torch.distributed.all_reduce(value)
    return value / torch.distributed.get_world_size()


def install_reference_modules(monkeypatch, root=None):
    """Load original functions verbatim, without importing native model ops.

Function code filenames/line numbers point to the original files, so engine
source/AST evidence identifies the same functions as a full model import.
This is a source/contract harness, not a complete model-stack installation.
"""
    import numpy as np

    root = root or reference_root()
    modules = {}
    for relative, cls_name, method, replacement in SPECS:
        name = "projects.mmdet3d_plugin." + relative
        path = root / (name.replace(".", "/") + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls_name)
        original = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        module = types.ModuleType(name)
        module.__file__ = str(path)
        namespace = module.__dict__
        namespace.update(dict(torch=torch, np=np, F=torch.nn.functional,
                         force_fp32=force_fp32, TORCH_VERSION=TORCH_VERSION,
                         digit_version=digit_version, reduce_mean=_reduce_mean,
                         bbox_xyxy_to_cxcywh=lambda boxes: torch.cat(
                             ((boxes[..., :2]+boxes[..., 2:])/2,
                              boxes[..., 2:]-boxes[..., :2]), -1),
                         bbox_cxcywh_to_xyxy=lambda boxes: torch.cat(
                             (boxes[..., :2]-boxes[..., 2:]/2,
                              boxes[..., :2]+boxes[..., 2:]/2), -1)))
        if cls_name == "BEVFormerHead":
            util = root / "projects/mmdet3d_plugin/core/bbox/util.py"
            util_tree = ast.parse(util.read_text(encoding="utf-8"))
            helper = next(node for node in util_tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == "normalize_bbox")
            exec(compile(ast.Module(body=[helper], type_ignores=[]), str(util), "exec"), namespace)
        exec(compile(ast.Module(body=functions + [original], type_ignores=[]), str(path), "exec"), namespace)
        setattr(module, cls_name, type(cls_name, (torch.nn.Module,), {method: namespace[method]}))
        parts = name.split(".")
        for index in range(1, len(parts)):
            parent_name = ".".join(parts[:index])
            if parent_name not in modules:
                parent = types.ModuleType(parent_name)
                parent.__path__ = [str(root.joinpath(*parts[:index]))]
                modules[parent_name] = parent
                monkeypatch.setitem(sys.modules, parent_name, parent)
                if index > 1:
                    setattr(modules[".".join(parts[:index-1])], parts[index-1], parent)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        setattr(modules[".".join(parts[:-1])], parts[-1], module)
    return modules


@pytest.fixture
def baseline(monkeypatch):
    return install_reference_modules(monkeypatch)


class ReductionLoss(torch.nn.Module):
    """Controlled Torch loss: isolates the denominator and broadcasting contract."""
    def __init__(self, kind="regression", scale=1.0):
        super().__init__()
        self.kind, self.scale = kind, scale
        self.factors = []

    def forward(self, pred, target, weight=None, avg_factor=None):
        self.factors.append(avg_factor)
        if self.kind == "class":
            one_hot = torch.nn.functional.one_hot(target, pred.shape[-1]+1)[..., :-1].to(pred.dtype)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, one_hot, reduction="none")
            weight = weight.unsqueeze(-1)
        elif self.kind == "direction":
            loss = 1 - torch.nn.functional.cosine_similarity(pred, target, dim=-1)
        else:
            loss = (pred-target).abs()
        if weight is not None:
            loss = loss * weight
        return loss.sum() / avg_factor * self.scale


class TargetsHead(torch.nn.Module):
    def __init__(self, kind, positives, sync_cls, zero_weight, three_d=False):
        super().__init__()
        self.kind, self.positives = kind, positives
        self.cls_out_channels = 2
        self.bg_cls_weight = .2
        self.sync_cls_avg_factor = sync_cls
        self.num_pts_per_vec, self.num_pts_per_gt_vec = 3, 4
        self.dir_interval = 1
        self.z_cfg = {"gt_z_flag": three_d}
        self.pc_range = [-10., -10., -2., 10., 10., 2.]
        self.register_buffer("code_weights", torch.ones(10 if kind == "bevformer" else 4))
        self.loss_cls = ReductionLoss("class")
        self.loss_bbox = ReductionLoss(scale=0. if zero_weight else 1.)
        self.loss_iou = ReductionLoss(scale=0. if zero_weight else .3)
        self.loss_pts = ReductionLoss()
        self.loss_dir = ReductionLoss("direction", scale=.005)

    def get_targets(self, cls, boxes, *args):
        device = cls[0].device
        total = len(cls) * len(cls[0])
        per_image = []
        remaining = self.positives
        for scores in cls:
            count = len(scores)
            positive = min(count, remaining)
            remaining -= positive
            labels = torch.full((count,), self.cls_out_channels, dtype=torch.long, device=device)
            labels[:positive] = 0
            mask = torch.zeros(count, device=device)
            mask[:positive] = 1
            if self.kind == "bevformer":
                targets = torch.ones(count, 9, device=device)
                targets[:, :3] = .3
                weights = mask[:, None].expand(count, 10).clone()
                per_image.append((labels, torch.ones(count, device=device), targets, weights))
            else:
                targets = torch.tensor([-.5, -.4, .7, .8], device=device).expand(count, 4).clone()
                dimension = 3 if self.z_cfg["gt_z_flag"] else 2
                points = torch.linspace(.1, .9, count*4*dimension, device=device).reshape(count, 4, dimension)
                per_image.append((labels, torch.ones(count, device=device), targets,
                                  mask[:, None].expand(count, 4).clone(), points,
                                  mask[:, None, None].expand_as(points).clone()))
        fields = tuple([item[index] for item in per_image] for index in range(len(per_image[0])))
        return fields + (self.positives, total-self.positives)


def loss_inputs(kind, three_d=False, device="cpu"):
    generator = torch.Generator().manual_seed(7)
    def make(shape):
        return torch.rand(shape, generator=generator).to(device).requires_grad_()
    cls, boxes = make((2, 3, 2)), make((2, 3, 10 if kind == "bevformer" else 4))
    if kind == "bevformer":
        return [cls, boxes], [[], []]
    points = make((2, 3, 3, 3 if three_d else 2))
    return [cls, boxes, points], [[], [], []]


def compare_loss(original, replacement, kind, positives, sync_cls, zero_weight,
                 three_d=False, device="cpu", bg_weight=.2):
    original_head = TargetsHead(kind, positives, sync_cls, zero_weight, three_d).to(device)
    original_head.bg_cls_weight = bg_weight
    updated_head = copy.deepcopy(original_head)
    inputs, extra = loss_inputs(kind, three_d, device)
    updated_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    expected = original(original_head, *inputs, *extra)
    actual = replacement(updated_head, *updated_inputs, *extra)
    assert len(actual) == len(expected) == (2 if kind == "bevformer" else 5)
    for left, right in zip(expected, actual):
        assert left.shape == right.shape
        assert left.dtype == right.dtype and left.device == right.device
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    sum(value.sum() for value in expected).backward()
    sum(value.sum() for value in actual).backward()
    for left, right in zip(inputs, updated_inputs):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5)
    for name in (["loss_bbox"] if kind == "bevformer" else ["loss_bbox", "loss_iou", "loss_pts", "loss_dir"]):
        factor = getattr(updated_head, name).factors[-1]
        assert factor.ndim == 0 and factor.device == updated_inputs[0].device
        assert not factor.requires_grad


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("positives", [0, 1, 4])
@pytest.mark.parametrize("sync_cls", [False, True])
@pytest.mark.parametrize("zero_weight", [False, True])
def test_loss_shapes_values_and_gradients(baseline, index, positives, sync_cls, zero_weight):
    relative, cls, method, function = SPECS[index]
    original = getattr(baseline["projects.mmdet3d_plugin."+relative], cls).__dict__[method]
    compare_loss(original, getattr(sync, function),
                 ("maptrv2", "maptr", "bevformer")[index], positives, sync_cls, zero_weight)


def test_3d_points_path(baseline):
    original = baseline["projects.mmdet3d_plugin."+SPECS[0][0]].MapTRv2Head.loss_single
    compare_loss(original, sync.maptrv2_head_loss_single, "maptrv2", 3, True, False, three_d=True)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_zero_classification_factor_preserves_scalar_output(baseline, index):
    relative, cls, method, function = SPECS[index]
    original = getattr(baseline["projects.mmdet3d_plugin."+relative], cls).__dict__[method]
    compare_loss(original, getattr(sync, function),
                 ("maptrv2", "maptr", "bevformer")[index], 0, True, False, bg_weight=0.)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("fp16_enabled", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.hcu)])
def test_point_sampling_precision_mask_and_gradients(baseline, dtype, fp16_enabled, device):
    import numpy as np
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("real HCU/CUDA device required")
    original = baseline["projects.mmdet3d_plugin."+SPECS[3][0]].BEVFormerEncoder.point_sampling
    sampler = torch.nn.Module()
    sampler.fp16_enabled = fp16_enabled
    coords = torch.tensor([[[[.2, .3, .4], [.8, .7, .6]]]], dtype=dtype, device=device).requires_grad_()
    updated = coords.detach().clone().requires_grad_()
    metas = [dict(lidar2img=[np.eye(4)], img_shape=[(20, 30, 3)])]
    pc_range = [0., 0., 1., 10., 10., 5.]
    expected = original(sampler, coords, pc_range, metas)
    actual = sync.bevformer_point_sampling(sampler, updated, pc_range, metas)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert actual[0].dtype == torch.float32
    assert actual[1].dtype == torch.bool and torch.equal(actual[1], expected[1])
    expected[0].sum().backward()
    actual[0].sum().backward()
    torch.testing.assert_close(coords.grad, updated.grad, rtol=0, atol=0)


@pytest.mark.parametrize("version", ["1.7.1", "1.8.0"])
def test_point_sampling_nonfinite_mask_matches_baseline(baseline, version, monkeypatch):
    import numpy as np
    original = baseline["projects.mmdet3d_plugin."+SPECS[3][0]].BEVFormerEncoder.point_sampling
    monkeypatch.setitem(inspect.unwrap(original).__globals__, "TORCH_VERSION", version)
    sampler = torch.nn.Module()
    sampler.fp16_enabled = True
    coords = torch.tensor([[[[float("nan"), .3, .4], [float("inf"), .7, .6],
                            [float("-inf"), .3, .4], [0., .3, .4],
                            [1., .3, .4], [.2, 0., .4]]]])
    metas = [dict(lidar2img=[np.eye(4)], img_shape=[(20, 30, 3)])]
    args = (sampler, coords, [0., 0., 1., 10., 10., 5.], metas)
    expected, actual = original(*args), sync.bevformer_point_sampling(*args)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0, equal_nan=True)
    assert torch.equal(actual[1], expected[1])


def test_point_sampling_needs_no_mask_cleanup(monkeypatch):
    import numpy as np
    def unexpected(*args, **kwargs):
        pytest.fail("Boolean mask cleanup must not run")
    monkeypatch.setattr(torch, "nan_to_num", unexpected)
    monkeypatch.setattr(np, "nan_to_num", unexpected)
    sampler = torch.nn.Module()
    sampler.fp16_enabled = True
    _, mask = sync.bevformer_point_sampling(
        sampler, torch.tensor([[[[.2, .3, .4]]]]), [0., 0., 1., 10., 10., 5.],
        [dict(lidar2img=[np.eye(4)], img_shape=[(20, 30, 3)])])
    assert mask.dtype == torch.bool


@pytest.mark.hcu
@pytest.mark.model_deps
def test_real_bevformer_encoder_forward_removes_mask_cleanup(monkeypatch):
    import numpy as np
    if not torch.cuda.is_available():
        pytest.skip("real HCU/CUDA device required")
    root = reference_root()
    monkeypatch.syspath_prepend(str(root))
    importlib.import_module("projects.mmdet3d_plugin")
    from mmcv import Config
    from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
    config = Config.fromfile(str(root / "projects/configs/maptr/maptr_tiny_r50_24e_bevformer.py"))
    encoder_config = config.model.pts_bbox_head.transformer.encoder
    assert encoder_config.type == "BEVFormerEncoder"
    encoder = build_transformer_layer_sequence(encoder_config).cuda().eval()
    original = encoder.point_sampling
    calls = []
    def sampled(*args, **kwargs):
        calls.append(True)
        return sync.bevformer_point_sampling(encoder, *args, **kwargs)
    width = 4
    dim = config.model.pts_bbox_head.transformer.embed_dims
    generator = torch.Generator(device="cuda").manual_seed(47)
    query = torch.randn(width * width, 1, dim, device="cuda", generator=generator)
    features = torch.randn(6, width * width, 1, dim, device="cuda", generator=generator)
    projection = np.eye(4)
    projection[2, 3] = 10.
    kwargs = dict(bev_h=width, bev_w=width, bev_pos=torch.zeros_like(query),
                  spatial_shapes=torch.tensor([[width, width]], device="cuda"),
                  level_start_index=torch.tensor([0], device="cuda"),
                  shift=torch.zeros(1, 2, device="cuda"),
                  img_metas=[dict(lidar2img=[projection] * 6, img_shape=[(32, 32, 3)] * 6)])
    with torch.no_grad():
        expected = encoder(query, features, features, **kwargs)
        monkeypatch.setattr(encoder, "point_sampling", sampled)
        actual = encoder(query, features, features, **kwargs)
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    refs = encoder.get_reference_points(width, width, encoder.pc_range[5] - encoder.pc_range[2],
                                        encoder.num_points_in_pillar, device="cuda")
    counts = []
    for sampling in (original, encoder.point_sampling):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    profile_memory=True) as profile:
            sampling(refs, encoder.pc_range, kwargs["img_metas"])
            torch.cuda.synchronize()
        counts.append(sum(event.count for event in profile.key_averages()
                          if event.key == "aten::nan_to_num"))
    assert counts[0] > 0 and counts[1] == 0


def test_interfaces_and_documented_changes_only(baseline):
    for relative, cls, method, function in SPECS:
        original = getattr(baseline["projects.mmdet3d_plugin."+relative], cls).__dict__[method]
        replacement = getattr(sync, function)
        assert inspect.signature(original) == inspect.signature(replacement)
        before = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(original)))).body[0]
        after = ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(replacement)))).body[0]
        # Only test normalization/NaN edits after accounting for function name
        # and the local imports that preserve original helper lookup.
        after.name = before.name
        after.body = [node for node in after.body if not isinstance(node, ast.ImportFrom)]
        if method == "loss_single":
            item = next(node for node in ast.walk(before) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute) and node.func.attr == "item")
            item.func.attr = "reshape"
            item.args = [ast.Tuple(elts=[], ctx=ast.Load())]
        else:
            branch = next(node for node in before.body if isinstance(node, ast.If)
                          and "digit_version" in ast.unparse(node.test))
            index = before.body.index(branch)
            del before.body[index]
        assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)


def test_four_targets_share_an_atomic_group():
    declaration = catalog.SYNC_ELIMINATION
    assert declaration.group_id == GROUP and len(declaration.specs) == 4
    assert declaration.definition.compatibility_check is None
    for filename in ("recipe.yaml", "optimization.yaml"):
        value = yaml.safe_load((ROOT / "turbo_physai/optimizations/models/maptrv2/configs" / filename).read_text())
        entry = next(entry for entry in value["optimization_groups"] if entry["id"] == GROUP)
        assert isinstance(entry["enabled"], bool)


def test_packaged_trust_matches_each_original_method(baseline):
    path = ROOT / "turbo_physai/optimizations/models/maptrv2/configs/optimization.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    entry = next(item for item in config["optimization_groups"] if item["id"] == GROUP)
    for relative, cls, method, function in SPECS:
        module = baseline["projects.mmdet3d_plugin." + relative]
        original = getattr(module, cls).__dict__[method]
        target = module.__name__ + "." + cls + "." + method
        assert source_hash(original) in entry["trust"]["source_hashes"][target]
        assert ast_hash(original) in entry["trust"]["ast_hashes"][target]


@pytest.mark.parametrize("mode", ["enabled", "disabled", "old_torch", "fail_install"] +
                         [f"bad_trust_{index}" for index in range(4)])
def test_public_apply_is_atomic_or_keeps_all_originals(tmp_path, mode):
    script = r'''
import sys
from pathlib import Path
import pytest,yaml,mmcv.utils
import turbo_physai
from test.optimizations.test_maptrv2_sync_elimination import install_reference_modules,SPECS,GROUP
from turbo_physai.optimizations.models.maptrv2 import sync_elimination as sync

with pytest.MonkeyPatch.context() as patch:
    modules=install_reference_modules(patch,Path(sys.argv[3]))
    source=Path(sys.argv[1])
    config=yaml.safe_load(source.read_text(encoding='utf-8'))
    entry=next(item for item in config['optimization_groups'] if item['id']==GROUP)
    config['optimization_groups']=[entry]
    config['optimization_modules']=['turbo_physai.optimizations.models.maptrv2.catalog']
    mode=sys.argv[2]
    entry['enabled']=mode!='disabled'
    if mode=='old_torch': patch.setattr(mmcv.utils,'TORCH_VERSION','1.7.1')
    if mode.startswith('bad_trust_'):
        index=int(mode.rsplit('_',1)[1])
        relative,cls,method,_=SPECS[index]
        target='projects.mmdet3d_plugin.'+relative+'.'+cls+'.'+method
        for values in entry['trust'].values():
            values[target]=['invalid']
    if mode=='fail_install':
        from turbo_physai.engine.execution.replacements.replace import ReplaceHandler
        real_apply=ReplaceHandler.apply
        calls=[]
        def failing_apply(self,prepared):
            calls.append(prepared)
            result=real_apply(self,prepared)
            if len(calls)==3: raise RuntimeError('injected install failure')
            return result
        patch.setattr(ReplaceHandler,'apply',failing_apply)
    originals=[getattr(modules['projects.mmdet3d_plugin.'+relative],cls).__dict__[method]
               for relative,cls,method,function in SPECS]
    destination=Path(sys.argv[4])
    destination.write_text(yaml.safe_dump(config),encoding='utf-8')
    report=turbo_physai.apply(optimization_config_path=destination)
    expected=mode in ('enabled','old_torch')
    assert report.summary['applied']==int(expected),report
    if mode=='fail_install':
        assert len(calls)==3
        assert report.summary['rolled_back']==1,report
    for (relative,cls,method,function),original in zip(SPECS,originals):
        installed=getattr(modules['projects.mmdet3d_plugin.'+relative],cls).__dict__[method]
        assert installed is (getattr(sync,function) if expected else original)
'''
    config = ROOT / "turbo_physai/optimizations/models/maptrv2/configs/optimization.yaml"
    process = subprocess.run(
        [sys.executable, "-c", script, str(config), mode, str(reference_root()), str(tmp_path / "config.yaml")],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr


def test_all_four_replacements_restore_through_engine_handlers(baseline):
    from turbo_physai.engine.execution.replacements.replace import ReplaceHandler
    handler = ReplaceHandler()
    snapshots = []
    originals = []
    try:
        for spec in catalog.SYNC_ELIMINATION.specs:
            prepared = handler.prepare(spec, {})
            originals.append((prepared.targets[0], prepared.targets[0].original))
            snapshots.append(handler.snapshot(prepared))
            handler.apply(prepared)
        for target, original in originals:
            assert getattr(target.parent, target.attribute) is not original
    finally:
        for snapshot in reversed(snapshots):
            handler.restore(snapshot)
    for target, original in originals:
        assert getattr(target.parent, target.attribute) is original


def test_static_assigner_and_two_loss_rounds(baseline, monkeypatch):
    # The static-loss orchestration only needs this pure utility from MMDet.
    # Keep CPU coverage runnable without importing its native operator stack.
    import functools
    core = types.ModuleType("mmdet.core")
    def multi_apply(function, *args, **kwargs):
        function = functools.partial(function, **kwargs) if kwargs else function
        return tuple(map(list, zip(*map(function, *args))))
    core.multi_apply = multi_apply
    monkeypatch.setitem(sys.modules, "mmdet.core", core)
    from turbo_physai.optimizations.models.maptrv2.assigner import static_loss
    original = baseline["projects.mmdet3d_plugin."+SPECS[0][0]].MapTRv2Head.loss_single
    heads = [TargetsHead("maptrv2", 2, True, False), TargetsHead("maptrv2", 2, True, False)]
    for head, function in zip(heads, (original, sync.maptrv2_head_loss_single)):
        head.loss_single = types.MethodType(function, head)
        head.gt_shift_pts_pattern = "v2"
        head.aux_seg = {"use_aux_seg": False}
        head.fp16_enabled = True
    tensors, unused = loss_inputs("maptrv2")
    clones = [item.detach().clone().requires_grad_() for item in tensors]
    def predictions(values):
        return dict(all_cls_scores=torch.stack([values[0]]*6),
                    all_bbox_preds=torch.stack([values[1]]*6),
                    all_pts_preds=torch.stack([values[2]]*6),
                    enc_cls_scores=None, enc_bbox_preds=None, enc_pts_preds=None)
    vectors = [types.SimpleNamespace(bbox=torch.ones(1,4),
                                   shift_fixed_num_sampled_points_v2=torch.ones(1,3,4,2)) for _ in range(2)]
    labels = [torch.zeros(1, dtype=torch.long) for _ in range(2)]
    reference, updated = [], []
    for unused in range(2):
        reference += list(static_loss(heads[0], vectors, labels, None, None, predictions(tensors)).values())
        updated += list(static_loss(heads[1], vectors, labels, None, None, predictions(clones)).values())
    assert len(reference) == len(updated) == 60
    for before, after in zip(reference, updated):
        assert before.shape == after.shape
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    assert len(heads[1].loss_pts.factors) == 12
    assert all(factor.ndim == 0 for factor in heads[1].loss_pts.factors)
    sum(value.sum() for value in reference).backward()
    sum(value.sum() for value in updated).backward()
    for before, after in zip(tensors, clones):
        torch.testing.assert_close(before.grad, after.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.hcu
@pytest.mark.model_deps
def test_real_static_assigner_with_fp16_public_loss_inputs(monkeypatch):
    """Exercise the real target/matching path and public FP32 loss boundary."""
    if not torch.cuda.is_available():
        pytest.skip("real HCU/CUDA device required")
    device = torch.device("cuda")
    root = reference_root()
    monkeypatch.syspath_prepend(str(root))
    try:
        import projects.mmdet3d_plugin  # noqa: F401
        from mmcv import Config
        from mmdet.models.builder import build_head
    except Exception as exc:
        pytest.skip(f"MapTRv2 model dependencies are unavailable: {exc}")

    from turbo_physai.optimizations.models.maptrv2 import assigner

    config_path = root / "projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py"
    config = Config.fromfile(str(config_path))
    head_config = config.model.pts_bbox_head.copy()
    head_config.train_cfg = config.model.train_cfg.pts
    baseline_head = build_head(head_config).to(device)
    optimized_head = copy.deepcopy(baseline_head)
    baseline_head.fp16_enabled = True
    optimized_head.fp16_enabled = True
    optimized_head.assigner.assign = types.MethodType(
        assigner.assign, optimized_head.assigner
    )
    optimized_head._get_target_single = types.MethodType(
        assigner.get_target_single, optimized_head
    )
    optimized_head.loss_single = types.MethodType(
        sync.maptrv2_head_loss_single, optimized_head
    )
    optimized_head.loss = types.MethodType(assigner.static_loss, optimized_head)

    generator = torch.Generator().manual_seed(41)
    batch_size, queries, classes, points = 2, head_config.num_query, 3, 20
    predictions = []
    for _ in range(2):
        predictions.append({
            "all_cls_scores": torch.randn(
                1, batch_size, queries, classes, generator=generator
            ).to(device).half().requires_grad_(),
            "all_bbox_preds": torch.rand(
                1, batch_size, queries, 4, generator=generator
            ).to(device).half().requires_grad_(),
            "all_pts_preds": torch.rand(
                1, batch_size, queries, points, 2, generator=generator
            ).to(device).half().requires_grad_(),
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
            "seg": None,
            "pv_seg": None,
        })

    def make_vector(count, offset):
        boxes = torch.tensor(
            [[-5.0 + offset, -4.0, 3.0 + offset, 5.0]],
            dtype=torch.float32,
            device=device,
        ).expand(count, -1).clone()
        sampled = torch.linspace(0.1, 0.9, points, device=device).view(1, points, 1)
        sampled = torch.cat((sampled + offset / 100, sampled), dim=-1)
        sampled = sampled.expand(count, -1, -1).clone()
        shifted = sampled[:, None].expand(count, 3, points, 2).clone()
        return types.SimpleNamespace(
            bbox=boxes,
            fixed_num_sampled_points=sampled,
            shift_fixed_num_sampled_points_v2=shifted,
        )

    vectors = [make_vector(1, 0.0), make_vector(2, 1.0)]
    labels = [torch.tensor([0], device=device), torch.tensor([1, 2], device=device)]
    args = (vectors, labels, None, None)

    for round_index, source_predictions in enumerate(predictions):
        baseline_predictions = {
            key: value.detach().clone().requires_grad_()
            if isinstance(value, torch.Tensor) else value
            for key, value in source_predictions.items()
        }
        updated_predictions = {
            key: value.detach().clone().requires_grad_()
            if isinstance(value, torch.Tensor) else value
            for key, value in source_predictions.items()
        }
        assert baseline_predictions["all_cls_scores"].dtype == torch.float16
        expected = baseline_head.loss(*args, baseline_predictions)
        actual = optimized_head.loss(*args, updated_predictions)
        assert expected.keys() == actual.keys()
        for key in expected:
            assert expected[key].dtype == actual[key].dtype == torch.float32
            torch.testing.assert_close(expected[key], actual[key], rtol=1e-5, atol=1e-6)
        sum(expected.values()).backward()
        sum(actual.values()).backward()
        for key in ("all_cls_scores", "all_bbox_preds", "all_pts_preds"):
            torch.testing.assert_close(
                baseline_predictions[key].grad,
                updated_predictions[key].grad,
                rtol=1e-5,
                atol=1e-6,
            )


def _distributed_worker(rank, rendezvous, root):
    from datetime import timedelta
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank,
        world_size=2, timeout=timedelta(seconds=60),
    )
    try:
        with pytest.MonkeyPatch.context() as patch:
            modules = install_reference_modules(patch, Path(root))
            for index in range(3):
                relative, cls, method, function = SPECS[index]
                original = getattr(modules["projects.mmdet3d_plugin."+relative], cls).__dict__[method]
                compare_loss(original, getattr(sync, function),
                             ("maptrv2", "maptr", "bevformer")[index], 3*rank, True, False)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_unequal_positive_counts(tmp_path):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("CPU Gloo backend required")
    script = (
        "from test.optimizations.test_maptrv2_sync_elimination import _distributed_worker; "
        "import torch,sys; "
        "torch.multiprocessing.spawn(_distributed_worker,args=(sys.argv[1],sys.argv[2]),nprocs=2,join=True)"
    )
    # Both ranks are local; do not inherit a training-specific network interface.
    env = os.environ.copy()
    env["GLOO_SOCKET_IFNAME"] = "lo"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "rendezvous"), str(reference_root())],
        cwd=ROOT, capture_output=True, text=True, timeout=150, env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.hcu
@pytest.mark.model_deps
@pytest.mark.parametrize("index", [0, 1, 2])
def test_real_loss_stack_on_device(baseline, index):
    if not torch.cuda.is_available():
        pytest.skip("real HCU/CUDA device required")
    builder = pytest.importorskip("mmdet.models.builder")
    if index != 2:
        importlib.import_module("projects.mmdet3d_plugin.maptr.losses.map_loss")
    kind = ("maptrv2", "maptr", "bevformer")[index]
    original_head = TargetsHead(kind, 3, True, False).cuda()
    original_head.loss_cls = builder.build_loss(dict(type="FocalLoss", use_sigmoid=True, gamma=2., alpha=.25))
    original_head.loss_bbox = builder.build_loss(dict(type="L1Loss"))
    if index != 2:
        original_head.loss_iou = builder.build_loss(dict(type="GIoULoss"))
        original_head.loss_pts = builder.build_loss(dict(type="PtsL1Loss"))
        original_head.loss_dir = builder.build_loss(dict(type="PtsDirCosLoss"))
    updated_head = copy.deepcopy(original_head)
    tensors, extra = loss_inputs(kind, device="cuda")
    clones = [value.detach().clone().requires_grad_() for value in tensors]
    relative, cls, method, function = SPECS[index]
    original = getattr(baseline["projects.mmdet3d_plugin."+relative], cls).__dict__[method]
    before = original(original_head, *tensors, *extra)
    after = getattr(sync, function)(updated_head, *clones, *extra)
    for left, right in zip(before, after):
        assert left.shape == right.shape and left.dtype == right.dtype and left.device == right.device
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
    sum(item.sum() for item in before).backward()
    sum(item.sum() for item in after).backward()
    for left, right in zip(tensors, clones):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.hcu
@pytest.mark.parametrize("index", [0, 1, 2])
def test_device_loss_contract(baseline, index):
    if not torch.cuda.is_available():
        pytest.skip("real HCU/CUDA device required")
    relative, cls, method, function = SPECS[index]
    original = getattr(baseline["projects.mmdet3d_plugin."+relative], cls).__dict__[method]
    compare_loss(original, getattr(sync, function),
                 ("maptrv2", "maptr", "bevformer")[index], 3, True, False, device="cuda")
