# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Nonblocking H2D configuration for ACT."""

from __future__ import annotations

import functools
import inspect
from dataclasses import replace


def nonblocking_h2d_wrapper(original, options):
    """Enable non-blocking H2D for ACT's default DataLoader configuration."""
    del options
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        from accelerate.utils import DataLoaderConfiguration

        bound = signature.bind(*args, **kwargs)
        config = bound.arguments.get('dataloader_config')
        bound.arguments['dataloader_config'] = (
            DataLoaderConfiguration(non_blocking=True)
            if config is None else replace(config, non_blocking=True)
        )
        return original(*bound.args, **bound.kwargs)

    return wrapped
