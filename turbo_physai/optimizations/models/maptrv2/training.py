# Copyright 2018-2019 OpenMMLab. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Hygon.

import functools
import os


def _env_flag(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no"}


def _option_flag(options, key, env_name, default):
    if key in options:
        return bool(options[key])
    return _env_flag(env_name, default)


def _matmul_precision(options):
    value = options.get("matmul_precision")
    if value is None:
        value = os.getenv("TURBO_PHYSAI_MATMUL_PRECISION", "high")
    return str(value).strip().lower()


def _channels_last_scope(options):
    value = options.get("channels_last_scope")
    if value is None:
        value = os.getenv("TURBO_PHYSAI_CHANNELS_LAST_SCOPE", "model")
    scope = str(value).strip().lower()
    if scope not in {"model", "backbone"}:
        raise ValueError(
            "channels_last_scope must be 'model' or 'backbone', got "
            f"{value!r}")
    return scope


def _convert_to_channels_last(model, scope):
    import torch

    if scope == "model":
        model = model.to(memory_format=torch.channels_last)
    else:
        if not hasattr(model, "img_backbone"):
            return model, 0
        model.img_backbone = model.img_backbone.to(
            memory_format=torch.channels_last)
    converted = sum(
        1 for module in model.modules()
        if isinstance(module, torch.nn.Conv2d)
    )
    return model, converted


def _install_input_layout_hook(backbone):
    import torch

    def _pre_hook(_module, inputs):
        if not inputs:
            return None
        first = inputs[0]
        if not torch.is_tensor(first) or first.dim() != 4:
            return None
        if first.is_contiguous(memory_format=torch.channels_last):
            return None
        return (first.contiguous(memory_format=torch.channels_last),
                ) + tuple(inputs[1:])

    return backbone.register_forward_pre_hook(_pre_hook)


def training_runtime_wrapper(original, options):
    import torch

    options = dict(options)

    @functools.wraps(original)
    def wrapped(model, dataset, cfg, *args, **kwargs):
        if _option_flag(options, "fork_start_method",
                        "TURBO_PHYSAI_FORK_START_METHOD", True):
            start_method = str(
                options.get(
                    "start_method",
                    os.getenv("TURBO_PHYSAI_DATALOADER_START_METHOD", "fork"),
                )
            )
            if start_method:
                current = torch.multiprocessing.get_start_method(
                    allow_none=True)
                if current != start_method:
                    torch.multiprocessing.set_start_method(
                        start_method, force=current is not None)

        if _option_flag(options, "cudnn_benchmark",
                        "TURBO_PHYSAI_CUDNN_BENCHMARK", True):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        precision = _matmul_precision(options)
        if precision not in {"", "off", "none", "default"}:
            setter = getattr(torch, "set_float32_matmul_precision", None)
            if setter is not None:
                setter(precision)

        if _option_flag(options, "channels_last",
                        "TURBO_PHYSAI_CHANNELS_LAST", True):
            if hasattr(model, "img_backbone"):
                scope = _channels_last_scope(options)
                model, converted = _convert_to_channels_last(model, scope)
                _install_input_layout_hook(model.img_backbone)
                logger = getattr(cfg, "logger", None)
                if logger is not None:
                    logger.info(
                        "TurboPhysAI: channels-last %s enabled "
                        "(%d conv modules in channels-last)", scope, converted)

        return original(model, dataset, cfg, *args, **kwargs)

    return wrapped


def init_dist_fork_wrapper(original, options):
    """Default the start method to ``fork`` before ``init_dist`` picks ``spawn``.

    ``mmcv.runner.init_dist`` sets ``spawn`` when no start method exists yet.
    ``tools/test.py`` never reaches ``custom_train_detector``, so its evaluation
    ``Pool`` workers would re-import modules without the TurboPhysAI patches.
    The start method is only set while it is still unset: a value chosen by the
    user or an earlier caller is kept.
    """

    options = dict(options)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if _option_flag(options, "fork_start_method",
                        "TURBO_PHYSAI_FORK_START_METHOD", True):
            import torch

            start_method = str(
                options.get(
                    "start_method",
                    os.getenv("TURBO_PHYSAI_DATALOADER_START_METHOD", "fork"),
                )
            )
            if start_method and torch.multiprocessing.get_start_method(
                    allow_none=True) is None:
                torch.multiprocessing.set_start_method(start_method)
        return original(*args, **kwargs)

    return wrapped


def ddp_forward_compat_wrapper(original, options):
    """Support MMCV's DDP forward on newer PyTorch versions.

    ``MMDistributedDataParallel._run_ddp_forward`` in MMCV 1.6 reads
    ``self._use_replicated_tensor_module``, an attribute that newer Torch DDP
    no longer defines. Training bypasses it through ``train_step``, but the
    evaluation hook calls ``model(return_loss=False, ...)``, which reaches it
    and raises ``AttributeError``. Defaulting the attribute to ``False`` makes
    MMCV pick ``self.module``.
    """

    del options

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, "_use_replicated_tensor_module"):
            self._use_replicated_tensor_module = False
        return original(self, *args, **kwargs)

    return wrapped
