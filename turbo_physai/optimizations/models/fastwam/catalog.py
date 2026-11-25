# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic FastWAM integration; importing this catalog needs no model or Torch."""

from ....engine.definitions import group, replace


_UPSTREAM = "fastwam.models.wan22"
_LOCAL = "turbo_physai.optimizations.models.fastwam"

VAE = group(
    "fastwam.vae",
    replace(f"{_UPSTREAM}.wan_video_vae.WanVideoVAE.encode", f"{_LOCAL}.vae.encode_videos"),
    replace(f"{_UPSTREAM}.wan_video_vae.VideoVAE38_.encode", f"{_LOCAL}.vae.encode_vae38"),
)

# The expert classes, their MoT consumer, and the loader must move together.
# In particular, replacing only loader.WanVideoDiT leaves the original class
# captured inside WAN22_MODEL_REGISTRY; use the adapted loader as well.
MODEL = group(
    "fastwam.model",
    replace(
        f"{_UPSTREAM}.wan_video_dit.WanVideoDiT",
        f"{_LOCAL}.wan_video_dit.WanVideoDiT",
        aliases=(f"{_UPSTREAM}.wan22.WanVideoDiT", f"{_UPSTREAM}.helpers.loader.WanVideoDiT"),
    ),
    replace(
        f"{_UPSTREAM}.action_dit.ActionDiT", f"{_LOCAL}.action_dit.ActionDiT",
        aliases=(f"{_UPSTREAM}.fastwam.ActionDiT",),
    ),
    replace(
        f"{_UPSTREAM}.mot.MoT", f"{_LOCAL}.mot.MoT",
        aliases=(f"{_UPSTREAM}.fastwam.MoT",),
    ),
    replace(
        f"{_UPSTREAM}.fastwam.FastWAM", f"{_LOCAL}.fastwam.FastWAM",
        aliases=(f"{_UPSTREAM}.fastwam_joint.FastWAM", f"{_UPSTREAM}.fastwam_optional_idm.FastWAM"),
    ),
    replace(
        f"{_UPSTREAM}.fastwam_joint.FastWAMJoint", f"{_LOCAL}.fastwam_joint.FastWAMJoint",
        aliases=(f"{_UPSTREAM}.fastwam_idm.FastWAMJoint",),
    ),
    replace(
        f"{_UPSTREAM}.fastwam_idm.FastWAMIDM", f"{_LOCAL}.fastwam_idm.FastWAMIDM",
        aliases=(f"{_UPSTREAM}.fastwam_optional_idm.FastWAMIDM",),
    ),
    replace(
        f"{_UPSTREAM}.fastwam_optional_idm.FastWAMOptionalIDM",
        f"{_LOCAL}.fastwam_optional_idm.FastWAMOptionalIDM",
    ),
    replace(f"{_UPSTREAM}.wan22.Wan22Core", f"{_LOCAL}.wan22.Wan22Core"),
    replace(
        f"{_UPSTREAM}.helpers.loader.load_wan22_ti2v_5b_components",
        f"{_LOCAL}.loader.load_wan22_ti2v_5b_components",
        aliases=(
            f"{_UPSTREAM}.fastwam.load_wan22_ti2v_5b_components",
            f"{_UPSTREAM}.wan22.load_wan22_ti2v_5b_components",
        ),
    ),
    depends_on=("fastwam.vae",),
)
