# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Match the reference launcher's interface selection before NCCL initializes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import wraps


def _select_interface() -> str:
    explicit = os.environ.get("OPENPI_NETWORK_IFACE")
    if explicit:
        return explicit
    for candidate in ("ens14f0", "eth0"):
        if os.path.isdir(f"/sys/class/net/{candidate}"):
            return candidate
    return "lo"


def wrap_process_group_network(original, options: Mapping[str, object]):
    explicit_values = options.get("environment", {})
    values = dict(explicit_values) if isinstance(explicit_values, Mapping) else {}

    @wraps(original)
    def wrapped(*args, **kwargs):
        interface = _select_interface()
        os.environ["NCCL_SOCKET_IFNAME"] = str(values.get("NCCL_SOCKET_IFNAME", interface))
        os.environ["GLOO_SOCKET_IFNAME"] = str(values.get("GLOO_SOCKET_IFNAME", interface))
        for name, value in values.items():
            os.environ[str(name)] = str(value)
        return original(*args, **kwargs)

    return wrapped
