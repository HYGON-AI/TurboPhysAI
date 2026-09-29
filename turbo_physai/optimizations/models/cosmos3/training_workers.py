# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Reinstall data-only replacements in spawned DataLoader workers."""

from functools import partial, wraps
import importlib
import inspect
import sys

_PORT = "turbo_physai.optimizations.models.cosmos3"
_ACTION = "cosmos_framework.data.generator.action.datasets"
_REASONER = "cosmos_framework.data.generator.reasoner.video_decoder_qwen"
_INTERLEAVED = "cosmos_framework.data.generator.augmentors.interleaved_video_parsing"

# Keep this list limited to data methods: workers must not initialize devices,
# distributed process groups, or model optimization modules.
WORKER_REPLACEMENTS = (
    (_ACTION + ".cosmos3_action_lerobot", "BaseActionLeRobotDataset._get_dataset", "action_lerobot", "_get_dataset"),
    (_ACTION + ".cosmos3_action_lerobot", "BaseActionLeRobotDataset._register_sources", "action_lerobot", "_register_sources"),
    (_REASONER, "_video_decoder_qwen_func", "reasoner_video", "_video_decoder_qwen_func"),
    (_INTERLEAVED, "_create_video_decoder", "interleaved_video", "_create_video_decoder"),
    (_INTERLEAVED, "VideoTransferAlignedFullFramesParsing._probe_video_len", "interleaved_video", "_probe_video_len"),
)


def _owner(module, path):
    parts = path.split(".")
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


def initialize_worker(worker_id, *, replacements, original=None):
    for module_name, path, port, symbol in replacements:
        module = importlib.import_module(module_name)
        owner, name = _owner(module, path)
        replacement = getattr(importlib.import_module(f"{_PORT}.{port}"), symbol)
        setattr(owner, name, replacement)
    if original is not None:
        original(worker_id)


def wrap_dataloader_init(original, options):
    signature = inspect.signature(original)

    @wraps(original)
    def initialize(*args, **kwargs):
        selected = []
        for item in WORKER_REPLACEMENTS:
            module_name, path, _, _ = item
            module = sys.modules.get(module_name)
            if module is None:
                continue
            try:
                owner, name = _owner(module, path)
            except AttributeError:
                continue
            if getattr(getattr(owner, name, None), "__module__", "").startswith(_PORT + "."):
                selected.append(item)
        if selected:
            bound = signature.bind(*args, **kwargs)
            bound.arguments["worker_init_fn"] = partial(
                initialize_worker,
                replacements=tuple(selected),
                original=bound.arguments.get("worker_init_fn"),
            )
            return original(*bound.args, **bound.kwargs)
        return original(*args, **kwargs)

    return initialize
