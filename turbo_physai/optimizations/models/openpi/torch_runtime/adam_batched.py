# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Launch independent native-precision AdamW tensor updates in a shared grid."""
import ctypes
from pathlib import Path

_lib = None


def launch(metadata, scales, count, config, lr, b1, b2, wd, eps):
    import torch
    global _lib
    if _lib is None:
        from .hip_libraries import ensure_library

        _lib = ctypes.CDLL(str(ensure_library('adam_batched.hip', 'libadam_batched')))
        _lib.openpi_adam_batch.argtypes = ([ctypes.c_int]*4 + [ctypes.c_void_p]*3
                                         + [ctypes.c_int] + [ctypes.c_double]*5)
        _lib.openpi_adam_batch.restype = ctypes.c_int
    result = _lib.openpi_adam_batch(*config, torch.cuda.current_stream().cuda_stream,
                                    metadata.data_ptr(), scales.data_ptr(), count,
                                    lr, b1, b2, wd, eps)
    if result:
        raise RuntimeError('Batched AdamW HIP error '+str(result))


def prepare(rows, device):
    import torch
    data = [[*[t.data_ptr() for t in row], row[0].numel()] for row in rows]
    host = torch.tensor(data, dtype=torch.int64, pin_memory=True)
    metadata = host.to(device, non_blocking=True)
    scales = torch.empty((len(rows), 3), dtype=torch.float32, device=device)
    return metadata, scales
