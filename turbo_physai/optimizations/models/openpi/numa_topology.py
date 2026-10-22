# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""HCU PCI-to-NUMA binding copied from the validated OpenPI launcher semantics."""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def parse_id_list(spec: str) -> set[int]:
    values: set[int] = set()
    for item in spec.strip().split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            values.update(range(int(start), int(end) + 1))
        else:
            values.add(int(item))
    return values


def format_id_list(values: set[int]) -> str:
    ordered = sorted(values)
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def load_hip_runtime() -> ctypes.CDLL:
    errors: list[str] = []
    for library in ("libamdhip64.so", "/opt/dtk/lib/libamdhip64.so"):
        try:
            return ctypes.CDLL(library)
        except OSError as exc:
            errors.append(f"{library}: {exc}")
    raise RuntimeError("cannot load HIP runtime: " + "; ".join(errors))


def visible_hcu_pci_devices() -> list[str]:
    hip = load_hip_runtime()
    hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    hip.hipGetDeviceCount.restype = ctypes.c_int
    hip.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    hip.hipDeviceGetPCIBusId.restype = ctypes.c_int
    count = ctypes.c_int()
    if hip.hipGetDeviceCount(ctypes.byref(count)) != 0:
        raise RuntimeError("hipGetDeviceCount failed")
    devices: list[str] = []
    for logical_device in range(count.value):
        buffer = ctypes.create_string_buffer(32)
        if hip.hipDeviceGetPCIBusId(buffer, len(buffer), logical_device) != 0:
            raise RuntimeError(f"hipDeviceGetPCIBusId({logical_device}) failed")
        pci = buffer.value.decode("ascii").lower()
        devices.append(f"0000:{pci}" if pci.count(":") == 1 else pci)
    return devices


def pci_topology(pci: str) -> tuple[int, set[int]]:
    path = Path("/sys/bus/pci/devices") / pci
    if not path.exists():
        raise RuntimeError(f"PCI device is absent from sysfs: {pci}")
    numa_node = int((path / "numa_node").read_text().strip())
    if numa_node < 0:
        raise RuntimeError(f"PCI device {pci} has no NUMA node")
    return numa_node, parse_id_list((path / "local_cpulist").read_text())


def thread_siblings(cpu: int) -> set[int]:
    return parse_id_list(
        Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text()
    )


def physical_core_units(
    cpus: set[int],
    sibling_lookup: Callable[[int], set[int]] = thread_siblings,
) -> list[set[int]]:
    remaining = set(cpus)
    units: list[set[int]] = []
    while remaining:
        cpu = min(remaining)
        unit = sibling_lookup(cpu) & cpus
        units.append(unit or {cpu})
        remaining -= units[-1]
    return sorted(units, key=lambda unit: min(unit))


def assigned_cpus(
    topology: Sequence[tuple[int, set[int]]],
    *,
    local_rank: int,
    allowed_cpus: set[int],
    siblings: Mapping[int, set[int]] | None = None,
    sibling_lookup: Callable[[int], set[int]] = thread_siblings,
) -> set[int]:
    if not 0 <= local_rank < len(topology):
        raise RuntimeError(f"LOCAL_RANK={local_rank} is outside the local world")
    numa_node, local_cpus = topology[local_rank]
    local_cpus = set(local_cpus) & allowed_cpus
    if not local_cpus:
        raise RuntimeError(
            f"HCU-local CPUs for NUMA node {numa_node} do not intersect the container cpuset"
        )
    peer_ranks = [rank for rank, (node, _) in enumerate(topology) if node == numa_node]
    peer_index = peer_ranks.index(local_rank)
    lookup = (lambda cpu: siblings[cpu]) if siblings is not None else sibling_lookup
    core_units = physical_core_units(local_cpus, lookup)
    begin = len(core_units) * peer_index // len(peer_ranks)
    end = len(core_units) * (peer_index + 1) // len(peer_ranks)
    units = core_units[begin:end]
    if not units:
        raise RuntimeError(f"NUMA node {numa_node} has too few physical cores for ranks {peer_ranks}")
    return set().union(*units)


def allowed_memory_nodes() -> set[int]:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("Mems_allowed_list:"):
            return parse_id_list(line.split(":", 1)[1])
    return set()


def prefer_numa_node(numa_node: int) -> bool:
    if numa_node not in allowed_memory_nodes():
        return False
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
        libnuma.numa_available.restype = ctypes.c_int
        libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
        libnuma.numa_set_preferred.restype = None
        if libnuma.numa_available() < 0:
            return False
        libnuma.numa_set_preferred(numa_node)
        return True
    except OSError:
        return False


@dataclass(frozen=True)
class BindingResult:
    local_rank: int
    pci: str
    numa_node: int
    cpus: set[int]
    memory_preferred: bool


def auto_bind_rank(
    *,
    environment: Mapping[str, str] | None = None,
    visible_devices: Callable[[], list[str]] = visible_hcu_pci_devices,
    lookup_topology: Callable[[str], tuple[int, set[int]]] = pci_topology,
    get_affinity: Callable[[], set[int]] = lambda: set(os.sched_getaffinity(0)),
    set_affinity: Callable[[set[int]], None] = lambda cpus: os.sched_setaffinity(0, cpus),
    sibling_lookup: Callable[[int], set[int]] = thread_siblings,
    prefer_memory: Callable[[int], bool] = prefer_numa_node,
) -> BindingResult:
    env = os.environ if environment is None else environment
    local_rank = int(env["LOCAL_RANK"])
    local_world_size = int(env["LOCAL_WORLD_SIZE"])
    devices = visible_devices()
    if len(devices) < local_world_size:
        raise RuntimeError(
            f"visible HCU count {len(devices)} is smaller than LOCAL_WORLD_SIZE={local_world_size}"
        )
    topology = [lookup_topology(pci) for pci in devices[:local_world_size]]
    numa_node, _ = topology[local_rank]
    cpus = assigned_cpus(
        topology,
        local_rank=local_rank,
        allowed_cpus=set(get_affinity()),
        sibling_lookup=sibling_lookup,
    )
    set_affinity(cpus)
    return BindingResult(
        local_rank=local_rank,
        pci=devices[local_rank],
        numa_node=numa_node,
        cpus=cpus,
        memory_preferred=prefer_memory(numa_node),
    )


def bind_current_rank_or_log() -> BindingResult | None:
    original_affinity = set(os.sched_getaffinity(0))
    try:
        result = auto_bind_rank()
        print(
            "[openpi-numa] "
            f"rank={result.local_rank} logical_hcu={result.local_rank} pci={result.pci} "
            f"numa={result.numa_node} cpus={format_id_list(result.cpus)} "
            f"memory_preferred={str(result.memory_preferred).lower()}",
            flush=True,
        )
        return result
    except Exception as exc:
        try:
            os.sched_setaffinity(0, original_affinity)
        except OSError:
            pass
        print(f"[openpi-numa] automatic binding disabled safely: {exc}", file=sys.stderr, flush=True)
        return None
