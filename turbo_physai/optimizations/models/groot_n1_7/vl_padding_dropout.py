# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon from Isaac-GR00T, gr00t/model/gr00t_n1d7/gr00t_n1d7.py
# and gr00t/model/modules/dit.py.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-width VL attention with Dropout executed at the collator width."""

from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn

from .fixed_padding import get_collator_width


class CollatorWidthDropout(nn.Dropout):
    # Read the variable width only here, outside the fixed-width VL graphs.
    @torch.compiler.disable
    def forward(self, hidden_states):
        width = get_collator_width()
        if not self.training or width is None or width == hidden_states.shape[1] or self.p == 0:
            return super().forward(hidden_states)

        extra = hidden_states.shape[1] - width
        native = hidden_states[:, extra:, :].contiguous()
        dropped = super().forward(native)
        return F.pad(dropped, (0, 0, extra, 0))


def vl_dropout_init_wrapper(original, group_options):
    """Replace Dropout only in the VL transformer, including its attention and FFN."""
    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if any(block.pos_embed is not None for block in self.transformer_blocks):
            raise ValueError("Fixed-width VL Dropout requires VL positional embeddings to be disabled")
        for name, module in list(self.named_modules()):
            if type(module) is nn.Dropout:
                parent_path, _, attribute = name.rpartition(".")
                replacement = CollatorWidthDropout(p=module.p, inplace=module.inplace)
                replacement.train(module.training)
                setattr(self.get_submodule(parent_path), attribute, replacement)

    return initialize


def action_head_vl_dropout_wrapper(original, group_options):
    @wraps(original)
    def process_backbone_output(self, backbone_output):
        width = get_collator_width()
        vl = self.vl_self_attention
        if width is None or isinstance(vl, nn.Identity):
            return original(self, backbone_output)

        features = backbone_output["backbone_features"]
        padded_width = features.shape[1]
        if not 0 < width <= padded_width:
            raise ValueError(f"GR00T collator width {width} does not fit backbone width {padded_width}")
        keep = torch.arange(padded_width, device=features.device) >= padded_width - width
        mask = keep.view(1, 1, padded_width).expand(features.shape[0], 1, padded_width)
        backbone_output["backbone_features"] = masked_vl_forward(vl, self.vlln(features), mask)
        return backbone_output

    return process_backbone_output


def masked_vl_forward(vl, hidden_states, attention_mask):
    """Run the fixed-width path, with optional wrapping by the VL compile Group."""
    hidden_states = hidden_states.contiguous()
    for block in vl.transformer_blocks:
        hidden_states = block(hidden_states, attention_mask=attention_mask)
    return hidden_states
