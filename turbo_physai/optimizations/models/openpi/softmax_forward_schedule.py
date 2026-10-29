# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Only native R0_BLOCK reduction grouping can be selected."""
import os
from pathlib import Path
import statistics


def make_run(original):
    import torch
    import triton
    from .torch_runtime.softmax_schedule import build

    root = Path(os.environ["OPENPI_TORCH_TIMING_OUTPUT"])
    rank = int(os.environ["LOCAL_RANK"])
    states = {}

    def run(self, *args, stream, **kw):
        name = self.inductor_meta.get("kernel_name", "")
        if (
            "prepare_softmax_online" not in name
            or len(args) != 7
            or args[-2:] != (250368, 978)
        ):
            return original(self, *args, stream=stream, **kw)
        identity = id(self)
        if identity not in states:
            original(self, *args, stream=stream, **kw)
            cfg = self.launchers[0].config
            assert (
                self.device_props.cc == "gfx938"
                and self.device_props.multi_processor_count == 64
            )
            assert torch.cuda.current_stream().cuda_stream == stream
            kernels, digest = build(
                self.fn.src, root / f"softmax_schedule_sources_rank{rank}"
            )
            golden = [a.clone() for a in args[2:5]]
            fusion = self.triton_meta.get("enable_fp_fusion", True)
            report = dict(
                source=self.fn.src,
                digest=digest,
                native_config=dict(kwargs=cfg.kwargs, warps=cfg.num_warps),
                results=[],
            )

            def launch(choice):
                opts = dict(
                    XBLOCK=choice["x"],
                    R0_BLOCK=cfg.kwargs["R0_BLOCK"],
                    num_warps=choice["w"],
                    num_stages=1,
                    enable_fp_fusion=fusion,
                )
                if choice["kind"] in ["schedule", "oneexp"]:
                    kernels[3 if choice["kind"] == "oneexp" else 0][
                        (triton.cdiv(250368, choice["x"]),)
                    ](*args, **opts)
                else:
                    kernels[1][(triton.cdiv(250368, choice["x"]),)](*args, **opts)
                    out = choice["out"]
                    kernels[2][(triton.cdiv(250368, out["x"]),)](
                        *args,
                        XBLOCK=out["x"],
                        R0_BLOCK=1024,
                        num_warps=out["w"],
                        num_stages=1,
                        enable_fp_fusion=fusion,
                    )

            def timed(fn):
                for _ in range(2):
                    fn()
                pairs = []
                for _ in range(5):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    a.record()
                    fn()
                    b.record()
                    pairs.append((a, b))
                torch.cuda.synchronize()
                return statistics.median(a.elapsed_time(b) for a, b in pairs)

            report["native_ms"] = timed(
                lambda: original(self, *args, stream=stream, **kw)
            )
            options = [dict(kind="oneexp", x=16, w=4)]
            for c in options:
                row = dict(choice=c)
                try:
                    launch(c)
                    unequal = [
                        int(torch.count_nonzero(g != a))
                        for g, a in zip(golden, args[2:5])
                    ]
                    row["unequal"] = unequal
                    if not any(unequal):
                        row["ms"] = timed(lambda: launch(c))
                except Exception as e:
                    row["error"] = str(e)[-1200:]
                report["results"].append(row)
            exact = [r for r in report["results"] if "ms" in r]
            best = min(exact, key=lambda r: r["ms"]) if exact else None
            report["best"] = best
            selected = (
                best["choice"]
                if best and best["ms"] < report["native_ms"] * 0.98
                else None
            )
            states[identity] = dict(
                kernels=kernels,
                choice=selected,
                fusion=fusion,
                rblock=cfg.kwargs["R0_BLOCK"],
                calls=0,
            )
            for a, g in zip(args[2:5], golden):
                a.copy_(g)
            return
        s = states[identity]
        c = s["choice"]
        if c is None:
            return original(self, *args, stream=stream, **kw)
        opts = dict(
            XBLOCK=c["x"],
            R0_BLOCK=s["rblock"],
            num_warps=c["w"],
            num_stages=1,
            enable_fp_fusion=s["fusion"],
        )
        s["calls"] += 1
        if c["kind"] in ["schedule", "oneexp"]:
            return s["kernels"][3 if c["kind"] == "oneexp" else 0][
                (triton.cdiv(250368, c["x"]),)
            ](*args, **opts)
        s["kernels"][1][(triton.cdiv(250368, c["x"]),)](*args, **opts)
        o = c["out"]
        return s["kernels"][2][(triton.cdiv(250368, o["x"]),)](
            *args,
            XBLOCK=o["x"],
            R0_BLOCK=1024,
            num_warps=o["w"],
            num_stages=1,
            enable_fp_fusion=s["fusion"],
        )

    return run
