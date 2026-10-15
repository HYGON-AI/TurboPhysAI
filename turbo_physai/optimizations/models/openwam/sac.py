# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

"""Standard PyTorch SAC policies for complete FFNs, attention and other ops.

Each checkpoint owns its standard forward/recompute modes. Only explicit eager
operator regions enter those modes; unselected compiled regions keep their path.
No activation cache, backward implementation or parameter replacement lives here.

Policies apply to the trailing sac_layers of the checkpointed layers.
Layers without checkpointing (checkpoint_skip_layers) keep their
activations through autograd, so SAC never caches them a second time.
"""

from turbo_physai.optimizations.models.openwam.settings import value

import json
import logging
import os
import re
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache

import torch
from torch.utils import checkpoint as torch_checkpoint
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from turbo_physai.optimizations.models.openwam import enabled
from turbo_physai.optimizations.models.openwam.runtime import (
    checkpoint_layers,
    checkpoint_skip_layers,
    integer,
)


FLAGS = ("sac_attention", "sac_ffn_full", "sac_gemm")
logger = logging.getLogger(__name__)
_session = ContextVar("openwam_sac_session", default=None)
_regions = ContextVar("openwam_sac_regions", default=())
_inside_mode = ContextVar("openwam_inside_sac_mode", default=False)
_native_mot = ContextVar("openwam_sac_native_mot_expression", default=False)
_counts = Counter()


def dispatcher(fn):
    """Let Dynamo inline a thin dispatcher, but never compile it as its own frame.

    A graph break inside an inlined dispatcher makes Dynamo compile the
    dispatcher separately, guarded on the callable argument. Every attention
    callable or per-layer module then recompiles it until recompile_limit.
    """
    fn.__openwam_dispatcher__ = True
    if not enabled("global_compile"):
        return fn
    try:
        from torch._dynamo.eval_frame import skip_code
    except ImportError:
        logger.warning(
            "[optimizations] torch._dynamo skip_code is unavailable; SAC dispatchers may recompile"
        )
        return fn
    skip_code(fn.__code__)
    return fn


_FA_OPS = frozenset(
    (
        "flash_attn._flash_attn_forward.default",
        "flash_attn2_c_op.varlen_fwd.default",
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten._scaled_dot_product_cudnn_attention.default",
        "aten._flash_attention_forward.default",
        "aten._efficient_attention_forward.default",
        "aten._scaled_dot_product_attention_math.default",
        "aten._scaled_dot_product_fused_attention_overrideable.default",
    )
)
_MATH_ATTENTION_OPS = frozenset(
    ("aten.bmm.default", "aten._safe_softmax.default", "aten._softmax.default")
)
_ATTENTION_OPS = _FA_OPS | _MATH_ATTENTION_OPS
_GEMM_OPS = frozenset(("aten.mm.default", "aten.addmm.default", "aten.bmm.default"))


def extras_enabled():
    return bool(value("sac_extra_ops", ""))


def active():
    return any(enabled(name) for name in FLAGS) or extras_enabled()


def ffn_boundary():
    return enabled("sac_ffn_full") or enabled("sac_gemm") or extras_enabled()


def attention_boundary():
    return enabled("sac_attention") or enabled("sac_gemm") or extras_enabled()


def projection_boundary():
    return enabled("sac_gemm") or extras_enabled()


@lru_cache(maxsize=16)
def _extra_ops(raw):
    result = set()
    for identifier in raw.split(","):
        identifier = identifier.strip()
        if not identifier:
            continue
        if not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*",
            identifier,
        ):
            raise ValueError(
                f"SAC extra op must be namespace.op.overload, got {identifier!r}"
            )
        namespace, name, overload = identifier.split(".")
        try:
            op = getattr(getattr(getattr(torch.ops, namespace), name), overload)
        except AttributeError as exc:
            raise ValueError(f"SAC extra op is unavailable: {identifier}") from exc
        if (
            op in getattr(torch_checkpoint, "SAC_IGNORED_OPS", ())
            or op._schema.is_mutable
        ):
            raise ValueError(
                f"SAC cannot cache a metadata or mutating op: {identifier}"
            )
        if identifier in _ATTENTION_OPS:
            raise ValueError(
                "Use sac_attention for attention operators, rather than SAC_EXTRA_OPS"
            )
        result.add(identifier)
    if raw.strip() and not result:
        raise ValueError("sac_extra_ops contains no operator identifiers")
    return frozenset(result)


def layers(total):
    """SAC layer count; unset covers every checkpointed layer."""
    checkpointed = checkpoint_layers(total)
    if value("sac_layers") is None:
        return checkpointed
    count = integer("sac_layers", checkpointed)
    if count > checkpointed:
        raise ValueError(
            f"sac_layers={count} exceeds checkpointed layers={checkpointed} "
            f"(model layers={total}, checkpoint_skip_layers={total - checkpointed})"
        )
    return count


def layer_range(total):
    """Half-open layer range [start, stop) covered by SAC: the checkpointed tail."""
    stop = checkpoint_layers(total)
    return stop - layers(total), stop


def validate(total=None):
    enabled("checkpoint_stats")
    if not active():
        return
    if total is not None:
        layers(total)
        _extra_ops(value("sac_extra_ops", ""))
    elif value("sac_layers") is not None:
        integer("sac_layers", 0)


def prepare(architecture):
    if not active():
        return
    from openwam.model.video_backbone.wan_backbone import WanBase

    backbone = architecture.video_backbone
    if (
        type(architecture).__name__ != "DualSystemSelfAttnArchitecture"
        or not isinstance(backbone, WanBase)
        or backbone.vace is not None
    ):
        raise ValueError(
            "Complete SAC currently requires native Wan dual_system/joint_self_attn without VACE"
        )
    validate(backbone.num_layers)
    for block in (*backbone.dit.blocks, *architecture.action_backbone.blocks):
        ffn = block.ffn
        modules = block.modules() if projection_boundary() else ffn.modules()
        for module in modules:
            if getattr(module, "_compiled_call_impl", None) is not None or hasattr(
                module.forward, "_torchdynamo_orig_callable"
            ):
                raise ValueError(
                    "Complete SAC requires operator regions without separate module/forward compilation"
                )
    if enabled("sac_attention"):
        from openwam.model.video_backbone.wan.models.dit import fused_backend_name

        backend_name = fused_backend_name()
        if backend_name not in ("flash_attention_2", "torch_sdpa"):
            raise ValueError(
                f"FA SAC does not yet cover backend {backend_name}; do not benchmark partial coverage"
            )
        if backend_name == "flash_attention_2":
            from flash_attn import flash_attn_interface as backend

            expected = (
                (
                    backend._wrapped_flash_attn_forward,
                    "flash_attn::_flash_attn_forward",
                ),
                (
                    backend._wrapped_flash_attn_varlen_forward,
                    "flash_attn2_c_op::varlen_fwd",
                ),
            )
            for entry, name in expected:
                op = getattr(entry, "default", entry)
                if getattr(getattr(op, "_schema", None), "name", None) != name:
                    raise ValueError(
                        f"FA SAC requires registered forward op {name}, got {entry}"
                    )
    start, stop = layer_range(backbone.num_layers)
    logger.info(
        "[optimizations] standard SAC flags=%s, layers=[%d, %d) of %d, no-checkpoint tail=%d, extra ops=%s",
        {name: enabled(name) for name in FLAGS},
        start,
        stop,
        backbone.num_layers,
        backbone.num_layers - stop,
        sorted(_extra_ops(value("sac_extra_ops", ""))),
    )


class _Policy:
    def __init__(self):
        self.fa = enabled("sac_attention")
        self.ffn = enabled("sac_ffn_full")
        self.gemm = enabled("sac_gemm")
        self.extra = _extra_ops(value("sac_extra_ops", ""))
        self.diagnostics = enabled("checkpoint_stats")
        self.seen_extra = set()
        self.seen_fa = False

    def __call__(self, context, op, *args, **kwargs):
        region_stack = _regions.get()
        name = str(op)
        attention = any(label.endswith("attention") for label in region_stack)
        ffn = any(label.endswith("ffn") for label in region_stack)
        save = (
            (self.fa and attention and name in _ATTENTION_OPS)
            or (self.ffn and ffn)
            or (self.gemm and name in _GEMM_OPS)
            # Padding plans intentionally take different paths on replay.
            # Extra policies apply to stable FFN/projection/norm/pointwise regions.
            or (not attention and name in self.extra)
        )
        if not context.is_recompute:
            if self.fa and attention and name in _ATTENTION_OPS:
                self.seen_fa = True
            if not attention and name in self.extra:
                self.seen_extra.add(name)
        if self.diagnostics:
            phase = "replay" if context.is_recompute else "forward"
            choice = "saved" if save else "computed"
            _counts[(phase, region_stack[-1], choice, name)] += 1
        return CheckpointPolicy.MUST_SAVE if save else CheckpointPolicy.PREFER_RECOMPUTE


@contextmanager
def _checkpoint_scope(mode, phase, policy):
    token = _session.set(mode)
    if enabled("checkpoint_stats"):
        _counts[(phase, "checkpoint", "entered", "layer")] += 1
        if mode is not None:
            _counts[(phase, "checkpoint", "selected", "layer")] += 1
    try:
        yield
        if phase == "forward" and policy is not None:
            if policy.fa and not policy.seen_fa:
                raise RuntimeError(
                    "FA SAC did not intercept an attention forward op in this layer; check the backend/path"
                )
            missing = policy.extra - policy.seen_extra
            if missing:
                raise RuntimeError(
                    f"SAC extra ops did not execute in this layer's instrumented regions: {sorted(missing)}"
                )
    finally:
        _session.reset(token)


def context_fn(layer_id, total):
    """Create fresh standard caches for the configured checkpointed layers."""
    start, stop = layer_range(total) if active() else (0, 0)
    selected = start <= layer_id < stop

    def contexts():
        if selected:
            policy = _Policy()
            forward, replay = create_selective_checkpoint_contexts(policy)
        else:
            forward = replay = None
            policy = None
        return _checkpoint_scope(forward, "forward", policy), _checkpoint_scope(
            replay, "replay", policy
        )

    return contexts


def native_mot_expression():
    return _native_mot.get()


def _invoke(region, fn, *args, **kwargs):
    mode = _session.get()
    # Norms moved out of GLOBAL keep the same native expression/backend choice.
    scope = value("compile_scope", "all").strip().lower()
    native = enabled("global_compile") and scope in {"mot", "all"}
    native_token = _native_mot.set(native)
    previous = _regions.get()
    region_stack = (
        previous
        if region == "attention" and previous and previous[-1].endswith("attention")
        else (*previous, region)
    )
    region_token = _regions.set(region_stack)
    inside = _inside_mode.get()
    mode_token = _inside_mode.set(inside or mode is not None)
    try:
        if mode is None or inside:
            result = fn(*args, **kwargs)
        else:
            with mode:
                result = fn(*args, **kwargs)
        # Eager projection/norm outputs cross graph breaks. Preserve the text
        # axis annotation on them; otherwise TEXT_TRIM again specializes L.
        if (
            native
            and region in {"projection", "norm", "pointwise"}
            and isinstance(result, torch.Tensor)
        ):
            source = next((arg for arg in args if isinstance(arg, torch.Tensor)), None)
            if source is not None and result.ndim == source.ndim:
                from torch._dynamo import mark_dynamic

                for axis in getattr(source, "_dynamo_dynamic_indices", ()):
                    if (
                        result.shape[axis] == source.shape[axis]
                        and result.shape[axis] > 1
                    ):
                        mark_dynamic(result, axis)
        return result
    finally:
        _inside_mode.reset(mode_token)
        _regions.reset(region_token)
        _native_mot.reset(native_token)


_eager_invoke = torch.compiler.disable(_invoke)


@dispatcher
def _call(region, fn, *args, **kwargs):
    # Always disabled: a region that reaches this frame eagerly, e.g. after a
    # recompile-limit fallback, must not let Dynamo trace the SAC region.
    return _eager_invoke(region, fn, *args, **kwargs)


@dispatcher
def run_attention(fn, *args, label="attention", **kwargs):
    if not attention_boundary():
        return fn(*args, **kwargs)
    return _call(label, fn, *args, **kwargs)


@dispatcher
def run_ffn(ffn, x, *, video=False):
    if video:
        from turbo_physai.optimizations.models.openwam.linear_bias2d import video_ffn

        return _call("video_ffn", video_ffn, ffn, x)
    return _call("action_ffn", ffn, x)


@dispatcher
def run_projection(linear, x):
    if not projection_boundary():
        return linear(x)
    return _call("projection", linear, x)


@dispatcher
def run_video_projection(linear, x):
    from turbo_physai.optimizations.models.openwam.linear_bias2d import _video_linear

    return _call("projection", _video_linear, linear, x)


@dispatcher
def run_norm(norm, x):
    if not extras_enabled():
        return norm(x)
    return _call("norm", norm, x)


@dispatcher
def run_pointwise(fn, *args):
    return _call("pointwise", fn, *args)


def begin_step(device):
    _counts.clear()
    torch.cuda.reset_peak_memory_stats(device)


def report_step(step, device, metrics, steps_per_sec, batch_size, opt_step):
    groups = {}
    for (phase, region, choice, op), count in sorted(_counts.items()):
        groups.setdefault(phase, {}).setdefault(region, {}).setdefault(choice, {})[
            op
        ] = count
    print(
        "[checkpoint_stats] "
        + json.dumps(
            {
                "rank": int(os.environ.get("RANK", "0")),
                "step": step,
                "optimizer_step": opt_step,
                "sac_flags": {name: enabled(name) for name in FLAGS},
                "sac_layers": value("sac_layers", "all"),
                "skip_layers": checkpoint_skip_layers(),
                "operators": groups,
                "s_microstep": 1.0 / steps_per_sec,
                "global_samples_s": steps_per_sec
                * batch_size
                * int(os.environ.get("WORLD_SIZE", "1")),
                "samples_s_gpu": steps_per_sec * batch_size,
                "loss_total": float(metrics["loss_total"]),
                "grad_norm": float(metrics["grad_norm"]),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            },
            sort_keys=True,
        ),
        flush=True,
    )
