# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""DataContainer support for the native DataLoader pin-memory thread.

Importing this module installs no patches and does not import Torch or MMCV.
The engine owns installation/restoration of the selected native entry point.
"""

import functools


def datacontainer_pin_wrapper(original, options):
    if options:
        raise ValueError("maptrv2.datacontainer_pin does not accept options")
    if getattr(original, "__turbo_physai_datacontainer_pin__", False):
        return original
    # The supplied v4 reference already patches this global entry point. Leave
    # it alone rather than adding a second full traversal to every batch.
    if (getattr(original, "__name__", "") == "_pin_memory_with_data_container"
            and "_pin_recursive" in getattr(original, "__globals__", {})):
        return original

    import torch
    from mmcv.parallel import DataContainer

    @functools.wraps(original)
    def pin_memory(data, *args, **kwargs):
        if isinstance(data, DataContainer):
            if not data.cpu_only:
                data._data = pin_memory(data.data, *args, **kwargs)
            return data
        if isinstance(data, torch.Tensor):
            if data.device.type != "cpu" or data.is_pinned():
                return data
        # Native recursion resolves the engine-installed global function, so
        # nested DataContainers are visited once and native container semantics
        # (including namedtuple and mapping subclasses) remain intact.
        return original(data, *args, **kwargs)

    pin_memory.__turbo_physai_datacontainer_pin__ = True
    return pin_memory
