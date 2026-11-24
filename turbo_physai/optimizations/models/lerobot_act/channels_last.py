# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
"""Channels-last and optimizer-state helpers for ACT."""

import functools

import torch

_CONV_MARKER = '_turbo_act_reference_input_conv'


def channels_last_init_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.backbone.to(memory_format=torch.channels_last)
        self.encoder_img_feat_input_proj.to(memory_format=torch.channels_last)
        first_conv = next(module for module in self.backbone.modules() if isinstance(module, torch.nn.Conv2d))
        setattr(first_conv, _CONV_MARKER, True)
        from .pool_backward_gather import mark_act_pool

        mark_act_pool(self)
        return result

    return wrapped


def channels_last_input_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, input, *args, **kwargs):
        if getattr(self, _CONV_MARKER, False):
            input = input.to(memory_format=torch.channels_last)
        return original(self, input, *args, **kwargs)

    return wrapped


def _contiguous_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.contiguous()
    if isinstance(value, dict):
        return {key: _contiguous_tensors(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_contiguous_tensors(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_contiguous_tensors(item) for item in value)
    return value


def contiguous_optimizer_state_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(optimizer, save_dir, optim_state_dict=None):
        state = optimizer.state_dict() if optim_state_dict is None else optim_state_dict
        state = {key: value if key == 'param_groups' else _contiguous_tensors(value) for key, value in state.items()}
        return original(optimizer, save_dir, optim_state_dict=state)

    return wrapped
