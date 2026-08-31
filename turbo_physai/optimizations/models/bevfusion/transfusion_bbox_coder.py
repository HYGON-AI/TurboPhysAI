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

"""Lightweight installation hooks for the optimized TransFusionBBoxCoder."""

import functools


_RUNTIME_CLASS = None


def get_transfusion_bbox_coder_class():
    """Load the Torch/MMDetection-dependent coder only when a model is patched."""

    global _RUNTIME_CLASS
    if _RUNTIME_CLASS is None:
        from .transfusion_bbox_coder_runtime import TransFusionBBoxCoder

        _RUNTIME_CLASS = TransFusionBBoxCoder
    return _RUNTIME_CLASS


def transfusion_bbox_coder_class_wrapper(original, options):
    """Replace public class aliases with the complete optimized class."""

    del original, options
    return get_transfusion_bbox_coder_class()


def build_bbox_coder_wrapper(original, options):
    """Route string registry configs to the replacement nn.Module class."""

    del options

    @functools.wraps(original)
    def wrapped(cfg, *args, **kwargs):
        coder_type = cfg.get("type") if hasattr(cfg, "get") else None
        if coder_type == "TransFusionBBoxCoder":
            cfg = cfg.copy()
            cfg["type"] = get_transfusion_bbox_coder_class()
        return original(cfg, *args, **kwargs)

    return wrapped
