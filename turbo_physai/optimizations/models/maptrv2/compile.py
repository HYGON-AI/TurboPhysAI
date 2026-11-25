# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""``torch.compile`` wrappers for MapTRv2 hot paths.

The validated reference implementation carries nineteen active
``@torch.compile()`` hooks.  Ten of them sit on symbols that also exist in the
official baseline, and ``catalog.COMPILE`` lists exactly those ten:
``MapTRPerceptionTransformer.format_feats``, ``MapTRDecoder.forward``,
``MapTRv2.extract_img_feat``, ``LSSTransform.get_cam_feats``/``get_mlp_input``
and the five ``normalize_*``/``denormalize_*`` scalar transforms in
``maptrv2_head``.  Eight more hang off helpers the reference extracted out of
larger methods, so they have no baseline counterpart, and the nineteenth
(``MapTRAssigner.assign``) belongs to the ``maptrv2.assigner`` Group.

TurboPhysAI does not edit model sources, so the compile is applied here as a
declared wrapper.  The reference uses a bare ``@torch.compile()``; this package
defaults to ``max-autotune-no-cudagraphs`` and honours ``options["mode"]``.
"""

import functools
import os


def compile_wrapper(original, options):
    """Compile ``original`` with the reference ``max-autotune`` mode.

    Returns ``original`` unchanged when the global kill switch is set or when
    this interpreter cannot compile at all, so the Group stays usable on
    eager-only builds instead of failing while the wrapper is installed.
    """

    if os.getenv("TURBO_PHYSAI_DISABLE_TORCH_COMPILE", "0") == "1":
        return original

    from .compat import torch_compile_available

    if not torch_compile_available():
        return original

    import torch

    mode = options.get("mode", "max-autotune-no-cudagraphs")
    compiled = torch.compile(original, mode=mode)

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return compiled(*args, **kwargs)

    return wrapped


def dynamo_disable_wrapper(original, options):
    """Keep a Python/numpy dominated helper outside surrounding Dynamo graphs."""

    del options

    from .compat import dynamo_available

    if not dynamo_available():
        return original

    import torch

    return torch._dynamo.disable(original)
