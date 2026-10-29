# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""BF16 gradient scaling with the original scalar conversion and rounding."""
import ctypes

_lib = None
_cached = {}


def scale_all(tensors, coefficient):
    import torch

    global _lib
    assert (
        coefficient.dtype == torch.float32
        and coefficient.is_cuda
        and coefficient.numel() == 1
    )
    if _lib is None:
        from .hip_libraries import ensure_library

        _lib = ctypes.CDLL(str(ensure_library("clip_batched.hip", "libclip_batched")))
        _lib.openpi_clip_batch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
        _lib.openpi_clip_batch.restype = ctypes.c_int
        _lib.openpi_clip32_batch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2
        _lib.openpi_clip32_batch.restype = ctypes.c_int
    groups = {}
    for tensor in tensors:
        assert (
            tensor.dtype in [torch.bfloat16, torch.float32]
            and tensor.is_contiguous()
            and tensor.numel() % (8 if tensor.dtype == torch.bfloat16 else 4) == 0
            and tensor.data_ptr() % 16 == 0
        )
        # Exact sizes keep all CTAs useful and preserve independent tensor writes.
        groups.setdefault((tensor.dtype, tensor.numel()), []).append(tensor)
    for (dtype, n), rows in groups.items():
        key = (dtype, n)
        pointers = tuple(t.data_ptr() for t in rows)
        cached = _cached.get(key)
        if cached is None or cached[0] != pointers:
            host = torch.tensor(
                [[t.data_ptr(), n] for t in rows], dtype=torch.int64, pin_memory=True
            )
            metadata = host.to(rows[0].device, non_blocking=True)
            cached = (pointers, metadata)
            _cached[key] = cached
        pack = 8 if dtype == torch.bfloat16 else 4
        grid = min(
            (n + pack * 256 - 1) // (pack * 256), 4096 if n >= 100000000 else 1024
        )
        if len(rows) > 1:
            grid = min(grid, 64)
        function = (
            _lib.openpi_clip_batch
            if dtype == torch.bfloat16
            else _lib.openpi_clip32_batch
        )
        error = function(
            torch.cuda.current_stream().cuda_stream,
            cached[1].data_ptr(),
            coefficient.data_ptr(),
            len(rows),
            grid,
        )
        if error:
            raise RuntimeError("Batched gradient clipping HIP error " + str(error))
