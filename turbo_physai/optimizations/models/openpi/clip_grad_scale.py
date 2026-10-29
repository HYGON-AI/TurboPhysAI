# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in clipping scaling: leave native norms, coefficients and grouping intact."""
import ast
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics


def _make_runtime(original):
    import torch
    import torch.nn.utils.clip_grad as module

    root = Path(os.environ["OPENPI_TORCH_TIMING_OUTPUT"])
    rank = int(os.environ["LOCAL_RANK"])
    policy = json.loads(
        (Path(__file__).parent / "torch_runtime" / "clip_scale_policy.json").read_text()
    )
    if (
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        != policy["framework_sha256"]
    ):
        raise RuntimeError("Clipping framework source differs from audited version")
    source = inspect.getsource(original)
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "torch._foreach_mul_"
    ]
    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args] == [
        "device_grads",
        "clip_coef_clamped.to(device)",
    ]
    calls[0].func = ast.Name(id="_openpi_vector_scale_", ctx=ast.Load())
    ast.fix_missing_locations(tree)
    state = dict(
        rank=rank,
        clip_calls=0,
        tensor_calls=0,
        elements=0,
        native_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        runtime_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        micro_passed=False,
    )

    def timed(fn):
        for _ in range(3):
            fn()
        pairs = []
        for _ in range(15):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            a.record()
            fn()
            b.record()
            pairs.append((a, b))
        torch.cuda.synchronize()
        return statistics.median(a.elapsed_time(b) for a, b in pairs)

    def verify(grads, coefficient):
        from .torch_runtime.clip_batched import scale_all

        def scale(tensor, coefficient):
            scale_all([tensor], coefficient)

        test = next(g for g in grads if 1048576 <= g.numel() <= 40000000)
        props = torch.cuda.get_device_properties(test.device)
        assert props.multi_processor_count == 64 and props.gcnArchName.startswith(
            "gfx938"
        )
        encoding = torch.arange(65536, device=test.device, dtype=torch.int32).to(
            torch.int16
        )
        all_values = encoding.view(torch.bfloat16)
        finite = all_values[torch.isfinite(all_values)]
        rows = []
        codes32 = (
            (torch.arange(65536, device=test.device, dtype=torch.int64) * 65537)
            .to(torch.int32)
            .view(torch.float32)
        )
        finite32 = codes32[torch.isfinite(codes32)]
        for value in [0.0, 1.0, 0.1, 0.4998, 0.9998, 0.0001]:
            coef = torch.tensor(value, device=test.device, dtype=coefficient.dtype)
            native, candidate = finite.clone(), finite.clone()
            torch._foreach_mul_([native], coef)
            scale(candidate, coef)
            unequal = int(
                torch.count_nonzero(
                    native.view(torch.int16) != candidate.view(torch.int16)
                )
            )
            rows.append(
                dict(
                    coefficient=value,
                    finite_bf16_encodings=finite.numel(),
                    unequal=unequal,
                )
            )
            assert unequal == 0, rows[-1]
            native32, candidate32 = finite32.clone(), finite32.clone()
            torch._foreach_mul_([native32], coef)
            scale(candidate32, coef)
            unequal32 = int(
                torch.count_nonzero(
                    native32.view(torch.int32) != candidate32.view(torch.int32)
                )
            )
            rows[-1].update(
                finite_fp32_samples=finite32.numel(), fp32_unequal=unequal32
            )
            assert unequal32 == 0, rows[-1]
        native, candidate = test.clone(), test.clone()
        torch._foreach_mul_([native], coefficient)
        scale(candidate, coefficient)
        unequal = int(
            torch.count_nonzero(native.view(torch.int16) != candidate.view(torch.int16))
        )
        assert unequal == 0, ("real gradient clipping mismatch", unequal)
        scratch = test.clone()
        report = dict(
            rows=rows,
            real_elements=test.numel(),
            real_unequal=unequal,
            coefficient_dtype=str(coefficient.dtype),
            coefficient=float(coefficient),
            native_ms=timed(lambda: torch._foreach_mul_([scratch], coefficient)),
            candidate_ms=timed(lambda: scale(scratch, coefficient)),
        )
        return report

    def vector_scale(grads, coefficient):
        from .torch_runtime.clip_batched import scale_all

        def scale(tensor, coefficient):
            scale_all([tensor], coefficient)

        eligible = [
            g
            for g in grads
            if g.dtype in [torch.bfloat16, torch.float32]
            and g.is_contiguous()
            and g.numel() > 0
            and g.numel() % (8 if g.dtype == torch.bfloat16 else 4) == 0
            and g.data_ptr() % 16 == 0
            and g.device.type == "cuda"
        ]
        if (
            coefficient.dtype != torch.float32
            or not coefficient.is_cuda
            or coefficient.numel() != 1
        ):
            return torch._foreach_mul_(grads, coefficient)
        if not eligible:
            return torch._foreach_mul_(grads, coefficient)
        props = torch.cuda.get_device_properties(eligible[0].device)
        if (
            props.multi_processor_count != 64
            or not getattr(props, "gcnArchName", "").startswith("gfx938")
        ):
            return torch._foreach_mul_(grads, coefficient)
        if not state["micro_passed"]:
            verify(eligible, coefficient)
            state["micro_passed"] = True
        eligible_ids = {id(g) for g in eligible}
        rest = [g for g in grads if id(g) not in eligible_ids]
        if rest:
            torch._foreach_mul_(rest, coefficient)
        scale_all(eligible, coefficient)
        state["clip_calls"] += 1
        state["tensor_calls"] += len(eligible)
        state["elements"] += sum(g.numel() for g in eligible)

    namespace = dict(vars(module), _openpi_vector_scale_=vector_scale)
    exec(compile(tree, "<openpi_clip_vector_scale>", "exec"), namespace)
    return namespace["_clip_grads_with_norm_"]


from .torch_runtime.installation import _prepare_runtime


def wrap_clip_grad_scale(original, options):
    del options
    runtime = None

    @wraps(original)
    def wrapped(*args, **kwargs):
        nonlocal runtime
        if runtime is None:
            _prepare_runtime()
            runtime = _make_runtime(original)
        return runtime(*args, **kwargs)

    return wrapped
