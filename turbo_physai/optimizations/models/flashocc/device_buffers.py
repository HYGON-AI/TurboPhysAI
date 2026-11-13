# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Register view-transformer constants as non-persistent module buffers."""

import functools


_BUFFER_NAMES = ("grid_lower_bound", "grid_interval", "grid_size", "frustum")


def register_device_buffers_wrapper(original, options):
    del options
    import torch

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        for name in _BUFFER_NAMES:
            value = getattr(self, name, None)
            if not isinstance(value, torch.Tensor) or name in self._buffers:
                continue
            delattr(self, name)
            self.register_buffer(name, value, persistent=False)

        return result

    return wrapped
