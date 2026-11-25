# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""maptrv2.fused_adamw: CPU clock/state tests plus real Apex device tests.

The CPU test double models Apex's group clock; it is NOT a fused-kernel test.
"""

import copy
import io
import sys
import types

import pytest

from turbo_physai.optimizations.models.maptrv2 import optimizer as opt

torch = pytest.importorskip("torch")


class GroupStepAdam(torch.optim.Optimizer):
    """CPU model of non-capturable Apex, including its inactive-group clock."""

    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999),
                 eps=1e-8, adam_w_mode=True, weight_decay=0., amsgrad=False,
                 set_grad_none=True, capturable=False, master_weights=False):
        super().__init__(params, dict(lr=lr, bias_correction=bias_correction,
                                     betas=betas, eps=eps, weight_decay=weight_decay))
        self.set_grad_none = set_grad_none
        self.capturable = capturable
        self.master_weights = master_weights

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1
            step = group["step"]
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["exp_avg"].mul_(beta1).add_(parameter.grad, alpha=1-beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    parameter.grad, parameter.grad, value=1-beta2)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                denom = state["exp_avg_sq"].sqrt() / (1-beta2**step)**0.5 + group["eps"]
                parameter.addcdiv_(state["exp_avg"], denom,
                                  value=-group["lr"]/(1-beta1**step))


@pytest.fixture
def fused_class():
    return opt._make_fused_adamw(GroupStepAdam)


def _parameters(device="cpu"):
    return [torch.nn.Parameter(torch.tensor([0.3, -0.7], device=device)),
            torch.nn.Parameter(torch.tensor([0.9, 0.2], device=device))]


def _groups(parameters):
    return [dict(params=[parameters[0]], lr=6e-5),
            dict(params=[parameters[1]], lr=6e-4, weight_decay=0.02)]


def _update(optimizer, parameters, iteration):
    optimizer.zero_grad()
    loss = sum(((parameter - (iteration+1)*0.02)**2).sum() for parameter in parameters)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
    optimizer.step()


def _assert_state_equal(left, right):
    for identifier, state in left["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, right["state"][identifier][key],
                                       atol=2e-7, rtol=2e-6)


def _roundtrip(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=True)


def test_group_clock_contract_and_cache(fused_class):
    opt._validate_apex_contract(GroupStepAdam)
    assert fused_class is opt._make_fused_adamw(GroupStepAdam)


@pytest.mark.parametrize("source_fused", [False, True])
@pytest.mark.parametrize("destination_fused", [False, True])
def test_resume_and_rollback_match_continuous_adamw(fused_class, source_fused, destination_fused):
    baseline_params, params = _parameters(), _parameters()
    baseline = torch.optim.AdamW(_groups(baseline_params))
    source_cls = fused_class if source_fused else torch.optim.AdamW
    optimizer = source_cls(_groups(params))
    for iteration in range(5):
        _update(baseline, baseline_params, iteration)
        _update(optimizer, params, iteration)
    checkpoint = _roundtrip(optimizer.state_dict())
    untouched = copy.deepcopy(checkpoint)
    resumed_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    target_cls = fused_class if destination_fused else torch.optim.AdamW
    resumed = target_cls(_groups(resumed_params))
    resumed.load_state_dict(checkpoint)
    _assert_state_equal(checkpoint, untouched)
    assert checkpoint["param_groups"] == untouched["param_groups"]
    assert [g["lr"] for g in resumed.param_groups] == [6e-5, 6e-4]
    for iteration in range(5, 15):
        _update(baseline, baseline_params, iteration)
        _update(resumed, resumed_params, iteration)
    for left, right in zip(baseline_params, resumed_params):
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)
    _assert_state_equal(baseline.state_dict(), resumed.state_dict())
    assert "step" not in resumed.state_dict()["param_groups"][0]


def test_divergent_checkpoint_fails_without_mutating_optimizer(fused_class):
    params = _parameters()
    previous = torch.optim.AdamW(params)
    params[0].grad = torch.ones_like(params[0])
    params[1].grad = torch.ones_like(params[1])
    previous.step()
    params[1].grad = None
    previous.step()
    checkpoint = previous.state_dict()
    optimizer = fused_class(_parameters())
    with pytest.raises(ValueError, match="group 0: unequal parameter steps"):
        optimizer.load_state_dict(checkpoint)
    assert not optimizer.state and "step" not in optimizer.param_groups[0]


def test_raw_apex_checkpoint_can_be_loaded(fused_class):
    params = _parameters()
    optimizer = fused_class(_groups(params))
    _update(optimizer, params, 0)
    raw = torch.optim.Optimizer.state_dict(optimizer)
    for state in raw["state"].values():
        state.pop("step")
    restored = fused_class(_groups(_parameters()))
    restored.load_state_dict(raw)
    assert restored.param_groups[0]["step"] == 1
    assert all(state["step"] == 1 for state in restored.state.values())


def test_no_gradient_group_does_not_advance_and_frozen_parameter_can_resume(fused_class):
    params = _parameters()
    optimizer = fused_class(params)
    optimizer.step()
    assert optimizer.param_groups[0]["step"] == 0 and not optimizer.state
    params[0].grad = torch.ones_like(params[0])
    optimizer.step()
    assert optimizer.state[params[0]]["step"] == 1
    assert params[1] not in optimizer.state
    restored = fused_class(_parameters())
    restored.load_state_dict(optimizer.state_dict())
    assert restored.param_groups[0]["step"] == 1
    # Reject a previously unused parameter's late first gradient, rather than
    # update it with another parameter's bias-correction clock.
    params[1].grad = torch.ones_like(params[1])
    with pytest.raises(ValueError, match="unequal parameter steps"):
        optimizer.step()


def test_partial_grad_none_rejected_before_updates(fused_class):
    params = _parameters()
    optimizer = fused_class(params)
    _update(optimizer, params, 0)
    saved = [p.detach().clone() for p in params]
    params[1].grad = None
    with pytest.raises(ValueError, match="partial grad=None"):
        optimizer.step()
    for parameter, original in zip(params, saved):
        torch.testing.assert_close(parameter, original, atol=0, rtol=0)


@pytest.mark.parametrize("flag", ["amsgrad", "maximize", "capturable", "differentiable",
                                  "master_weights", "foreach", "fused"])
def test_unsupported_modes_fail(fused_class, flag):
    with pytest.raises(ValueError, match=flag):
        fused_class(_parameters(), **{flag: True})


def test_checkpoint_shape_and_modes_are_validated(fused_class):
    params = _parameters()
    source = torch.optim.AdamW(params)
    _update(source, params, 0)
    checkpoint = source.state_dict()
    checkpoint["state"][0]["exp_avg"] = torch.ones(10)
    with pytest.raises(ValueError, match="incompatible exp_avg"):
        fused_class(_parameters()).load_state_dict(checkpoint)


def test_sparse_gradients_and_non_fp32_parameters_fail(fused_class):
    params = _parameters()
    optimizer = fused_class(params)
    params[0].grad = torch.sparse_coo_tensor(
        torch.tensor([[0]]), torch.ones(1), size=(2,), check_invariants=True)
    with pytest.raises(ValueError, match="sparse"):
        optimizer.step()
    with pytest.raises(ValueError, match="FP32"):
        fused_class([torch.nn.Parameter(torch.ones(2, dtype=torch.float16))])


def test_invalid_clocks_and_group_conflicts_fail():
    for value in (True, -1, 1.5, float("inf"), float("nan"), torch.ones(2)):
        with pytest.raises(ValueError):
            opt._integer_step(value, "test")
    snapshot = dict(state={0: dict(step=2, exp_avg=torch.ones(1), exp_avg_sq=torch.ones(1))},
                    param_groups=[dict(params=[0], step=3)])
    with pytest.raises(ValueError, match="steps disagree"):
        opt._to_apex_state(snapshot)


def test_large_checkpoint_clock_does_not_round_on_rollback():
    snapshot = dict(state={0: dict(step=2**24+1, exp_avg=torch.ones(2), exp_avg_sq=torch.ones(2))},
                    param_groups=[dict(params=[0], step=2**24+1, lr=.001,
                                       betas=(.9, .999), eps=1e-8, weight_decay=.01)])
    portable = opt._to_adamw_state(snapshot)
    assert portable["state"][0]["step"].item() == 2**24+1
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], foreach=False)
    optimizer.load_state_dict(portable)
    assert next(iter(optimizer.state.values()))["step"].item() == 2**24+1


def test_unknown_apex_contract_is_rejected():
    class PerParameterClock:
        def step(self):
            state = {}
            state["step"] = 1
    with pytest.raises(RuntimeError, match="group-step"):
        opt._validate_apex_contract(PerParameterClock)


def test_zero_grad_and_amp_overflow(fused_class):
    params = _parameters()
    optimizer = fused_class(_groups(params))
    _update(optimizer, params, 0)
    optimizer.zero_grad(set_to_none=False)
    assert all(p.grad is not None and not p.grad.any() for p in params)
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in params)
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    scaler.scale(sum(p.sum()*float("inf") for p in params)).backward()
    previous = [p.detach().clone() for p in params]
    scaler.step(optimizer)
    scaler.update()
    assert optimizer.param_groups[0]["step"] == 1
    for p, before in zip(params, previous):
        torch.testing.assert_close(p, before, atol=0, rtol=0)
    optimizer.zero_grad()
    scaler.scale(sum((p**2).sum() for p in params)).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, 0.5)
    scaler.step(optimizer)
    scaler.update()
    assert optimizer.param_groups[0]["step"] == 2


@pytest.fixture
def registry(monkeypatch):
    class Registry:
        def __init__(self):
            self.entries = {"AdamW": torch.optim.AdamW}

        def get(self, name):
            return self.entries.get(name)

        def register_module(self, name, module):
            self.entries[name] = module

    registry = Registry()
    module = types.ModuleType("mmcv.runner.optimizer.builder")
    module.OPTIMIZERS = registry
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return registry


def test_builder_keeps_config_and_param_groups(monkeypatch, registry, fused_class):
    monkeypatch.setattr(opt, "_load_apex", lambda: GroupStepAdam)
    params = _parameters()
    seen = []

    def build(model, cfg):
        seen.append(cfg)
        config = dict(cfg)
        cls = registry.get(config.pop("type"))
        paramwise = config.pop("paramwise_cfg")
        assert paramwise == {"custom_keys": {"img_backbone": {"lr_mult": 0.1}}}
        return cls(_groups(model), **config)

    cfg = dict(type="AdamW", weight_decay=.01,
               paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=.1))))
    saved = copy.deepcopy(cfg)
    wrapper = opt.fused_adamw_builder_wrapper(build, {})
    optimizer = wrapper(params, cfg)
    assert isinstance(optimizer, fused_class) and cfg == saved
    assert registry.get("AdamW") is torch.optim.AdamW
    assert seen[0]["type"] == opt._NAME
    assert isinstance(wrapper(params, cfg), fused_class)
    assert opt.fused_adamw_builder_wrapper(wrapper, {}) is wrapper
    registry.entries[opt._NAME] = object()
    with pytest.raises(RuntimeError, match="registry conflict"):
        wrapper(params, cfg)


def test_other_optimizers_do_not_load_apex(monkeypatch):
    def forbidden():
        raise AssertionError("Apex must not be imported")
    monkeypatch.setattr(opt, "_load_apex", forbidden)
    cfg = {"type": "SGD", "lr": .1}
    wrapper = opt.fused_adamw_builder_wrapper(lambda model, config: config, {})
    assert wrapper(None, cfg) is cfg


def test_enabled_dependency_failure_is_explicit(monkeypatch, registry):
    def missing():
        raise RuntimeError("requires Apex with compiled amp_C kernels")
    monkeypatch.setattr(opt, "_load_apex", missing)
    wrapper = opt.fused_adamw_builder_wrapper(lambda *args: None, {})
    with pytest.raises(RuntimeError, match="requires Apex"):
        wrapper(None, {"type": "AdamW"})


@pytest.mark.model_deps
def test_real_mmcv_registry_and_paramwise_cfg(monkeypatch, fused_class):
    builder = pytest.importorskip("mmcv.runner.optimizer.builder")
    monkeypatch.setattr(opt, "_load_apex", lambda: GroupStepAdam)
    model = torch.nn.Module()
    model.img_backbone = torch.nn.Linear(2, 2)
    model.head = torch.nn.Linear(2, 1)
    config = dict(type="AdamW", lr=6e-4, weight_decay=.01,
                  paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=.1))))
    plain = builder.build_optimizer(model, config)
    wrapped = opt.fused_adamw_builder_wrapper(builder.build_optimizer, {})
    fused = wrapped(model, config)
    assert isinstance(fused, fused_class)
    assert config["type"] == "AdamW"
    for reference, actual in zip(plain.param_groups, fused.param_groups):
        assert reference["lr"] == actual["lr"]
        assert reference["weight_decay"] == actual["weight_decay"]
        assert reference["params"][0] is actual["params"][0]


@pytest.mark.hcu
@pytest.mark.model_deps
def test_real_apex_resume_rollback_and_amp():
    if not torch.cuda.is_available():
        pytest.skip("real HCU/CUDA device required")
    pytest.importorskip("apex.optimizers")
    adapter = opt._make_fused_adamw(opt._load_apex())
    baseline_params, params = _parameters("cuda"), _parameters("cuda")
    baseline = torch.optim.AdamW(_groups(baseline_params), foreach=False)
    optimizer = torch.optim.AdamW(_groups(params), foreach=False)
    for iteration in range(5):
        _update(baseline, baseline_params, iteration)
        _update(optimizer, params, iteration)
    fused = adapter(_groups(params))
    fused.load_state_dict(optimizer.state_dict())
    for iteration in range(5, 10):
        _update(baseline, baseline_params, iteration)
        _update(fused, params, iteration)
    resumed = adapter(_groups(params))
    resumed.load_state_dict(_roundtrip(fused.state_dict()))
    scaler = torch.amp.GradScaler("cuda", init_scale=8)
    resumed.zero_grad()
    scaler.scale(sum(p.sum()*float("inf") for p in params)).backward()
    before = [p.detach().clone() for p in params]
    clock = resumed.param_groups[0]["step"]
    scaler.step(resumed)
    scaler.update()
    assert resumed.param_groups[0]["step"] == clock
    for p, previous in zip(params, before):
        torch.testing.assert_close(p, previous, atol=0, rtol=0)
    rollback = torch.optim.AdamW(_groups(params), foreach=False)
    rollback.load_state_dict(resumed.state_dict())
    for iteration in range(10, 15):
        _update(baseline, baseline_params, iteration)
        _update(rollback, params, iteration)
    for left, right in zip(baseline_params, params):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
