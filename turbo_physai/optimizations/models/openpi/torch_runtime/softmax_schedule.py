# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Retain the original online reduction, screen scheduling and output staging."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path

HEADER = (
    "import triton\nimport triton.language as tl\n"
    "from torch._inductor.runtime import triton_helpers\n"
    "from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math\n"
)

ONE_EXP = """
@triton.jit
def openpi_online_combine(lhs_max, lhs_sum, rhs_max, use_fast_math: tl.constexpr):
    out_max = triton_helpers.maximum(lhs_max, rhs_max)
    left_is_max = lhs_max == out_max
    delta = tl.where(left_is_max, rhs_max - out_max, lhs_max - out_max)
    scale = tl.where(out_max == float('-inf'), 1.0, libdevice.exp(delta))
    lhs_scale = tl.where(left_is_max, 1.0, scale)
    rhs_scale = tl.where(left_is_max, scale, 1.0)
    out_sum = tl.fma(lhs_sum, lhs_scale, rhs_scale)
    out_sum = tl.where(out_max == float('inf'), float('nan'), out_sum)
    return out_max, out_sum
"""


def build(source, directory):
    fn = ast.parse(source).body[0]
    assert [a.arg for a in fn.args.args][:5] == [
        "in_ptr0",
        "in_ptr1",
        "out_ptr0",
        "out_ptr1",
        "out_ptr2",
    ]
    loops = [n for n in fn.body if isinstance(n, ast.For)]
    assert len(loops) == 2
    first, second = loops
    reduction = next(
        n
        for n in fn.body
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "triton_helpers.online_softmax_reduce"
    )
    scheduled = copy.deepcopy(fn)
    scheduled.name += "_openpi_schedule"
    oneexp = copy.deepcopy(fn)
    oneexp.name += "_openpi_one_exp"
    for n in ast.walk(oneexp):
        if (
            isinstance(n, ast.Call)
            and ast.unparse(n.func) == "triton_helpers.online_softmax_combine"
        ):
            n.func = ast.Name(id="openpi_online_combine", ctx=ast.Load())
    phase1 = copy.deepcopy(fn)
    phase1.name += "_openpi_reduce"
    phase1.body = [copy.deepcopy(n) for n in fn.body if n is not second]
    phase2 = copy.deepcopy(fn)
    phase2.name += "_openpi_output"
    body = []
    for n in fn.body:
        if n is first:
            continue
        if n is reduction:
            body.extend(
                ast.parse(
                    "tmp7 = tl.load(out_ptr0 + x5 + 7840*x2, xmask)\n"
                    "tmp8 = tl.load(out_ptr1 + x5 + 7840*x2, xmask)"
                ).body
            )
            continue
        if (
            isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "tl.store"
        ):
            continue
        body.append(copy.deepcopy(n))
    phase2.body = body
    functions = [scheduled, phase1, phase2, oneexp]
    for f in functions:
        f.decorator_list = [ast.Name(id="triton.jit", ctx=ast.Load())]
        ast.fix_missing_locations(f)
    generated = HEADER + ONE_EXP + "\n\n".join(ast.unparse(f) for f in functions) + "\n"
    digest = hashlib.sha256(generated.encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("softmax_schedule_" + digest + ".py")
    path.write_text(generated)
    spec = importlib.util.spec_from_file_location(
        "openpi_softmax_schedule_" + digest, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [getattr(module, f.name) for f in functions], digest
