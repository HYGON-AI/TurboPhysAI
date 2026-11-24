# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Scoped ACT Inductor score kernels for the supported FP32 layouts."""
import ast
import copy
from dataclasses import dataclass
import hashlib
import functools
import os
import re


PREFIX = "triton_red_fused__softmax__softmax_backward_data_native_dropout_backward"
REFERENCE_HEADS = 688
_PTRS = ("in_ptr0", "in_ptr1", "in_ptr2", "in_ptr3", "in_ptr4")
_TILES = ("xnumel", "r0_numel", "XBLOCK", "R0_BLOCK")
_INPLACE = ("in_out_ptr0", *_PTRS, *_TILES)
_OUT = (*_PTRS, "out_ptr0", "out_ptr2", *_TILES)
_SHAPES = {
    (902, 902): (813696, 813632),
    (100, 902): (90240, 90208),
    (100, 100): (10112, 10016),
    (102, 102): (10496, 10432),
}
# Body-only fingerprints from the ACT v8 Inductor cache. Batch-dependent row
# counts are normalized before comparison; device metadata is not included.
_BODIES = {
    (68800, 100, True): "f74d745efb1c18bc5bea188659d213313760e715b500f1ebfcda4f6371292a47",
    (68800, 100, False): "fda5c9157d1def6930634705e1c978a02ab6d3458d047757bb300736cbeaa076",
    (68800, 902, True): "58a2f3a4d045a32054c3b19f653f8d48228bf252fa9ff2fed85dba12d7d4f8d6",
    (70176, 102, True): "9815aaa2a15adebdb0182b83196d3b09974239fef0b8b7b1bfaaebae9d36f578",
    (70176, 102, False): "b5283bfcfd862fd4407d99f169a6b927289ab392ab6cd481855f8abc28dc8693",
    (620576, 902, True): "a6bcccbba65ec76eb749b82a9e3b274216c5cb35e98a85a7b70c9de225e7765d",
}


@dataclass(frozen=True)
class Case:
    queries: int
    keys: int
    mask_stride: int
    work_stride: int
    inplace: bool
    heads: int

    @property
    def rows(self):
        return self.queries * self.heads

    @property
    def label(self):
        return f"{self.queries}x{self.keys}_{'inplace' if self.inplace else 'out'}"


def _kernel_function(kernel_name, source):
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name == kernel_name]
    if len(nodes) != 1:
        raise ValueError(f"ACT softmax contract changed: function for {kernel_name}")
    return nodes[0]


def _shape(node):
    constants = {
        s.targets[0].id: s.value.value for s in node.body
        if isinstance(s, ast.Assign) and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Constant)
    }
    return constants.get("xnumel"), constants.get("r0_numel")


def _body_hash(node):
    return hashlib.sha256(
        ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False).encode()
    ).hexdigest()


def _normalized_body_hash(node, rows):
    clone = copy.deepcopy(node)
    for stmt in clone.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == 'xnumel'
                and isinstance(stmt.value, ast.Constant)):
            stmt.value.value = rows
    return _body_hash(clone)


def _queries(node, width):
    candidates = {q for q, w in _SHAPES if w == width}
    observed = {
        part.right.value for part in ast.walk(node)
        if isinstance(part, ast.BinOp) and isinstance(part.op, (ast.Mod, ast.FloorDiv))
        and isinstance(part.left, ast.Name) and part.left.id == 'xindex'
        and isinstance(part.right, ast.Constant) and part.right.value in candidates
    }
    if len(observed) == 1:
        return observed.pop()
    if not observed and len(candidates) == 1:
        return candidates.pop()
    raise ValueError(f'ACT score backward has ambiguous query length for {width}')


def _check_index_range(rows, queries, width, mask_stride, work_stride):
    heads = rows // queries
    if heads * max(queries * width, mask_stride, work_stride) >= 2**31:
        raise ValueError('ACT score kernel tensor indexing exceeds the supported 32-bit range')


def contract(kernel_name, source):
    node = _kernel_function(kernel_name, source)
    rows, width = _shape(node)
    if not isinstance(rows, int) or rows <= 0 or not isinstance(width, int):
        raise ValueError(f"ACT score backward does not support {rows} rows × {width}")
    queries = _queries(node, width)
    if (queries, width) not in _SHAPES or rows % queries:
        raise ValueError(f"ACT score backward does not support {rows} rows × {width}")
    signature = tuple(arg.arg for arg in node.args.args)
    if signature not in (_INPLACE, _OUT):
        raise ValueError(f"ACT softmax contract changed: signature for {kernel_name}")
    if "'grid_type': 'Grid1D'" not in source:
        raise ValueError(f"ACT softmax contract changed: grid for {kernel_name}")
    inplace = signature == _INPLACE
    reference_rows = REFERENCE_HEADS * queries
    actual = _normalized_body_hash(node, reference_rows)
    if actual != _BODIES.get((reference_rows, width, inplace)):
        raise ValueError(f"ACT softmax contract changed: body for {kernel_name} {rows}x{width}")
    mask_stride, work_stride = _SHAPES[queries, width]
    _check_index_range(rows, queries, width, mask_stride, work_stride)
    return Case(queries, width, mask_stride, work_stride, inplace, rows // queries), node


def rewrite_case(case, node, source):
    tile, warps = (4, 2) if case.keys == 902 else (16, 4)
    block = 1 << (case.keys - 1).bit_length()
    offset = f"{case.keys} * xindex + rindex"
    body = f"""    xnumel = {case.rows}
    r0_numel = {case.keys}
    tl.static_assert(R0_BLOCK == {block})
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rindex = tl.arange(0, R0_BLOCK)[None, :]
    valid_x = xindex < xnumel
    valid = valid_x & (rindex < r0_numel)
    head = xindex // {case.queries}
    off = {offset}
    work_off = off + {case.work_stride - case.queries * case.keys} * head
    shift = tl.load(in_ptr3 + xindex, valid_x, other=0.0)
    denom = tl.load(in_ptr4 + xindex, valid_x, other=1.0)
    scores = tl.load(in_ptr2 + off, valid, other=0.0, eviction_policy='evict_first')
    dy = tl.load(in_ptr0 + off, valid, other=0.0, eviction_policy='evict_first')
    keep = tl.load(in_ptr1 + off + {case.mask_stride - case.queries * case.keys} * head, valid, other=False, eviction_policy='evict_first')
    exp_score = libdevice.exp(scores - shift)
    prob = exp_score * (1.0 / denom)
    g = dy * (keep.to(tl.float32) * 1.1111111111111112)
    weighted = g * prob
    dot = tl.sum(tl.where(valid, weighted, 0.0), 1)[:, None]
    result = tl.fma(-prob, dot, weighted)
"""
    if case.inplace:
        body += "    tl.store(in_out_ptr0 + work_off, result, valid)\n"
    else:
        body += "    tl.store(out_ptr0 + work_off, weighted, valid)\n"
        body += "    tl.store(out_ptr2 + off, result, valid)\n"
    lines = source.splitlines(keepends=True)
    new_source = "".join(lines[:node.body[0].lineno - 1]) + body + "".join(lines[node.end_lineno:])
    fixed = {"XBLOCK": tile, "R0_BLOCK": block, "num_warps": warps, "num_stages": 1}
    return _fixed_decorator(new_source, node, repr(fixed))


_FORWARD = "triton_per_fused__softmax_native_dropout_prepare_softmax_online"
_FORWARD_ABI = (
    "in_ptr0", "in_ptr1", "out_ptr0", "out_ptr1", "out_ptr3", "out_ptr4",
    "load_seed_offset", "xnumel", "r0_numel", "XBLOCK",
)
_FORWARD_BODY = "656a7f62e2ee5c3d40cb33da76c3cfbfefa47ad29ad8862ca49f9bc90a7ee329"
def check_software_stack(context):
    from turbo_physai.engine.contracts import CompatibilityResult

    if os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
        return CompatibilityResult(True, reason="compilation disabled explicitly")
    try:
        from torch._inductor.async_compile import AsyncCompile
        from torch._inductor.runtime import triton_heuristics
        from torch._inductor.runtime.triton_helpers import libdevice, max2
        import triton
    except ImportError as error:
        return CompatibilityResult(False, reason=f"ACT score compiler support unavailable: {error}")
    available = all(callable(item) for item in (
        getattr(AsyncCompile, 'triton', None), getattr(triton_heuristics, 'fixed_config', None),
        getattr(triton, 'jit', None), max2,
    )) and hasattr(libdevice, 'exp')
    return CompatibilityResult(available, reason="ACT score compiler capabilities")


class ScoreForward:
    """The .run ABI used in ACT's generated Inductor graph."""

    def __init__(self, name="", trace=False):
        self._name = name
        self._trace = trace

    def run(self, x, seed, shift, denom, keep, prob, seed_offset, rows, width, *, stream):
        if width != 902 or rows <= 0 or rows % 902:
            raise ValueError(f"ACT score forward: unexpected launch shape {rows} × {width}")
        if (keep.stride(0), prob.stride(0)) != (813696, 813632):
            raise ValueError("ACT score forward: unexpected mask/probability stride")
        _check_index_range(rows, 902, width, 813696, 813632)
        import torch
        import triton

        from .compiled_attention_forward import dropout_linear, softmax_stats

        def launch():
            softmax_stats[(triton.cdiv(rows, 2),)](x, shift, denom, prob, rows, 2, num_warps=1)
            elems = rows * width
            dropout_linear[(triton.cdiv(elems, 1024),)](
                seed, keep, prob, seed_offset, elems, 1024, num_warps=4,
            )

        if torch.cuda.current_stream(x.device).cuda_stream == stream:
            launch()
        else:
            with torch.cuda.stream(torch.cuda.ExternalStream(stream, device=x.device)):
                launch()
        if self._trace:
            self._trace = False
            print(f"ACT_SCORE_KERNEL rank={os.getenv('RANK')} forward=902x902 kernel={self._name}", flush=True)


def _fixed_decorator(source, node, tile):
    dec = node.decorator_list[0]
    if not isinstance(dec, ast.Call) or ast.unparse(dec.func) not in (
        "triton_heuristics.reduction", "triton_heuristics.persistent_reduction",
    ):
        raise ValueError("ACT score backward: unexpected Inductor launch decorator")
    keep = [k for k in dec.keywords if k.arg in ("filename", "triton_meta", "inductor_meta")]
    if len(keep) != 3:
        raise ValueError("ACT score backward: unexpected Inductor launch metadata")
    args = []
    for k in keep:
        value = ast.get_source_segment(source, k.value)
        if k.arg == "inductor_meta":
            value += ' | {"coordinate_descent_tuning": False, "dynamic_scale_rblock": False}'
        args.append(f"{k.arg}={value}")
    replacement = "@triton_heuristics.fixed_config(" + tile + ", " + ", ".join(args) + ")\n"
    lines = source.splitlines(keepends=True)
    return "".join(lines[:dec.lineno - 1]) + replacement + "".join(lines[dec.end_lineno:])


def wrap_async_triton(original, options):
    if set(options) - {"backward"} or type(options.get("backward", True)) is not bool:
        raise ValueError("lerobot_act.compiled_attention_scores accepts a boolean backward option")
    if os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
        return original
    optimize_backward = options.get("backward", True)
    trace = os.environ.get("TURBO_PHYSAI_TRACE_ACT_SCORE_KERNELS") == "1"

    @functools.wraps(original)
    def wrapped(self, kernel_name, source_code, device_str="cuda"):
        if re.fullmatch(rf"{_FORWARD}_\d+", kernel_name):
            node = _kernel_function(kernel_name, source_code)
            shape = _shape(node)
            rows, width = shape
            if not isinstance(rows, int) or rows <= 0 or not isinstance(width, int):
                raise ValueError(f"ACT score forward does not support {shape} at {kernel_name}")
            queries = _queries(node, width)
            if rows % queries:
                raise ValueError(f"ACT score forward has inconsistent shape {shape} at {kernel_name}")
            if (queries, width) == (902, 902):
                if (device_str not in ("cuda", "hip")
                        or tuple(a.arg for a in node.args.args) != _FORWARD_ABI
                        or _normalized_body_hash(node, 620576) != _FORWARD_BODY):
                    raise ValueError(f"ACT score forward: unsupported 902 source or device at {kernel_name}")
                return ScoreForward(kernel_name, trace)
        if optimize_backward and re.fullmatch(rf"{PREFIX}_\d+", kernel_name):
            if device_str not in ("cuda", "hip"):
                raise ValueError(f"ACT score backward: unsupported device {device_str}")
            case, node = contract(kernel_name, source_code)
            source_code = rewrite_case(case, node, source_code)
            if trace:
                print(f"ACT_SCORE_KERNEL rank={os.getenv('RANK')} backward={case.label} kernel={kernel_name}",
                      flush=True)
        return original(self, kernel_name, source_code, device_str)

    return wrapped
