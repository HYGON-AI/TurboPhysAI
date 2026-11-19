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

"""DDP options for ACT."""

from __future__ import annotations

import functools


def ddp_options_wrapper(original, options):
    """Match reference DDP and apply explicitly selected ACT DDP options."""
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        kwargs['find_unused_parameters'] = False
        if 'static_graph' in options:
            kwargs['static_graph'] = options['static_graph']
        return original(*args, **kwargs)

    return wrapped
