# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
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
# Modified by Hygon.
#
# Encoder and decoder forward implementations are adapted from LeRobot
# https://github.com/huggingface/lerobot at 3f2179f3b69708b6ad009b2e7685dd9d05269ee1,
# src/lerobot/policies/act/modeling_act.py, licensed under Apache-2.0:
# http://www.apache.org/licenses/LICENSE-2.0
"""Compile the ACT visual backbone."""

import functools
import os

import torch


def backbone_compile_wrapper(original, options):
    del options
    if os.environ.get("TURBO_PHYSAI_DISABLE_TORCH_COMPILE") == "1":
        return original
    compiled = None

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        nonlocal compiled
        if compiled is None:
            compiled = torch.compile(original, dynamic=False, options={'triton.cudagraphs': False})
        return compiled(self, *args, **kwargs)

    return wrapped
