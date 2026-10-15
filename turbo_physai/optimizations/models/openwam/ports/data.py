# Modified by Hygon Information Technology Co., Ltd., 2026.
# Derived from OpenWAM main bfce20b575a9b34aea385030fcb00a9eaf61fcb5.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import types
import torch
import torch.nn as nn
import openwam.dataloader.robotwin as _source
from turbo_physai.optimizations.common.python.source_binding import bind_missing_globals


def _prompt_choices(instructions: dict, ep_file: str, task_name: str):
    """Raw prompt choices and whether the train reader samples them randomly."""
    ep_num = os.path.basename(ep_file).replace("episode", "").replace(".hdf5", "")
    key = f"episode{ep_num}.json"
    instr = instructions.get(key)
    if isinstance(instr, dict):
        pool = instr.get("seen") or instr.get("unseen") or []
        if pool:
            return pool, True
        base = instr.get("instruction")
    elif isinstance(instr, str):
        base = instr
    elif isinstance(instr, list) and instr:
        base = instr[0] if isinstance(instr[0], str) else str(instr[0])
    else:
        base = None
    if base is None:
        base = f"The bimanual robot is performing a {task_name} task."
    return [base], False


def _training_prompts(instructions: dict, ep_file: str, task_name: str) -> list[str]:
    """Every prompt the train reader can emit, without consuming its RNG."""
    choices, _ = _prompt_choices(instructions, ep_file, task_name)
    return [format_prompt_for_inference(value) for value in choices]


def _resolve_prompt(
    instructions: dict,
    ep_file: str,
    split: str,
    task_name: str,
) -> str:
    """Pure function: build the training-time prompt from raw state.

    Extracted from :meth:`RoboTwinDataset._get_prompt` so tests and
    downstream adapters can reproduce training-time prompts without having
    to instantiate a full dataset (no ``__new__`` + private-attr injection).

    Must stay byte-compatible with what training sees — deployment uses the
    same ``format_prompt_for_inference`` helper.

    Args:
        instructions: Mapping from ``episode<N>.json`` to a RoboTwin
            instruction dict (``{"seen": [...], "unseen": [...]}``) or a
            plain string / list.
        ep_file: Episode file name or full path (``episode<N>.hdf5``).
        split: ``"train"`` samples a random instruction from ``seen``;
            anything else deterministically picks the first entry.
        task_name: Falls back to ``f"... performing a {task_name} task."``
            when instructions don't supply a prompt for this episode.

    Returns:
        The final wrapped prompt string that the model sees.
    """
    choices, randomize = _prompt_choices(instructions, ep_file, task_name)
    base_prompt = (
        random.choice(choices) if split == "train" and randomize else choices[0]
    )
    return format_prompt_for_inference(base_prompt)


def RoboTwinDataset__build_sample(self, ep_idx: int, start: int) -> dict:
    """Assemble a single sample at (ep_idx, start).

    - ``num_frames``: raw HDF5 window length (state/action rate).
    - Video is subsampled by ``video_stride`` to ``num_video_frames``.
    - State/action stay at raw rate.
    - ``proprio = raw_actions[0:1]`` (time dim kept).
    - ``action = raw_actions[1:num_frames]`` (length ``num_frames-1``).
    """
    path = self._episode_files[ep_idx]
    ep_len = self._episode_lengths[ep_idx]

    raw_end = start + self._raw_window_len
    actual_raw_end = min(raw_end, ep_len)
    actual_raw_len = max(0, actual_raw_end - start)

    if actual_raw_len <= 0:
        raise IndexError(
            f"Window [{start}, {raw_end}) has no frames (ep_len={ep_len})."
        )
    if actual_raw_len < 2:
        raise IndexError(
            f"Window [{start}, {raw_end}) has no valid action label "
            f"(actual_raw_len={actual_raw_len}, ep_len={ep_len})."
        )

    # A stored VAE latent replaces JPEG decoding (vae_latent_cache).
    latent_store = getattr(self, "_latent_store", None)
    with h5py.File(path, "r") as f:
        if latent_store is not None:
            raw_frames = None
        elif self.multiview:
            raw_frames = self._read_multiview_frames(
                f, self.cameras, start, actual_raw_end
            )
        else:
            raw_frames = self._read_camera_frames(
                f, self.target_camera, start, actual_raw_end
            )
        raw_actions = self._read_raw_actions(f, start, actual_raw_end)

    # Pad to full window if the episode ended early
    if actual_raw_len < self._raw_window_len:
        pad_len = self._raw_window_len - actual_raw_len
        if raw_frames is not None:
            raw_frames = raw_frames + [raw_frames[-1]] * pad_len
        raw_actions = np.concatenate(
            [raw_actions, np.repeat(raw_actions[-1:], pad_len, axis=0)],
            axis=0,
        )

    # Normalize before splitting so proprio and action share the same space
    if self._normalizer is not None:
        raw_actions = self._normalizer.normalize(raw_actions)

    # Unify: scatter normalized (T, raw) -> (T, UNIFY_DIM) AFTER normalization
    # (so unify operates in normalized space; the inverse un-unify in
    # denormalize_action runs BEFORE unnormalize). _unify_dim_mask marks the
    # mapped slots; unmapped slots stay 0 and are masked out below.
    unify_dim_mask = None
    if self._unify_dst_index is not None:
        raw_actions, unify_dim_mask = map_to_unify(
            raw_actions.astype(np.float32), self._unify_dst_index, UNIFY_DIM
        )

    # Video: subsampled. State/action: raw rate.
    if raw_frames is None:
        sampled_video = None
    else:
        sampled_video = [raw_frames[i] for i in self._video_sample_indices]
        if not self.multiview:
            sampled_video = [
                crop_and_resize(frame, self.height, self.width)
                for frame in sampled_video
            ]

    proprio_np = raw_actions[0:1].astype(np.float32)
    action_np = raw_actions[1 : self.num_frames].astype(np.float32)

    video_mask = torch.tensor(
        [idx < actual_raw_len for idx in self._video_sample_indices],
        dtype=torch.bool,
    )
    # 2-D action_mask (num_action_steps, action_dim): time × dim validity.
    # Without unify, RoboTwin is bimanual / single-tensor — every valid
    # timestep has all dims real, so dim validity is all-True. With unify,
    # dim validity = the scattered slots (unify_dim_mask); unmapped slots are
    # masked out so they never enter the loss.
    time_validity = torch.tensor(
        [(t + 1) < actual_raw_len for t in range(self.num_action_steps)],
        dtype=torch.bool,
    )
    action_mask = (
        time_validity.unsqueeze(-1).expand(-1, self._action_dim_value).contiguous()
    )
    # 2-D proprio_mask (1, action_dim): all True when proprio is present.
    proprio_mask = torch.full(
        (1, self._action_dim_value),
        fill_value=bool(0 < actual_raw_len),
        dtype=torch.bool,
    )
    if unify_dim_mask is not None:
        dim_valid = torch.from_numpy(unify_dim_mask)  # (UNIFY_DIM,) bool
        action_mask = action_mask & dim_valid.unsqueeze(0)
        proprio_mask = proprio_mask & dim_valid.unsqueeze(0)

    prompt = self._get_prompt(ep_idx)

    ep_key = f"episode_{ep_idx}"
    ep_scene = self._scene_info.get(ep_key, {})
    ep_info = ep_scene.get("info", ep_scene)
    active_arm = ep_info.get("active_arm", "both")

    action_tensor = torch.from_numpy(action_np)
    proprio_tensor = torch.from_numpy(proprio_np)

    extra = {}
    if latent_store is not None:
        from turbo_physai.optimizations.models.openwam.latent_cache import (
            FIRST_FRAME_FROM_LATENTS,
        )

        extra = {
            "video_latents": latent_store(ep_idx, start),
            "video_size": (self.num_video_frames, self.height, self.width),
        }
    return {
        **extra,
        "video": sampled_video,
        "vace_video": None,
        "first_frame_image": [FIRST_FRAME_FROM_LATENTS]
        if sampled_video is None
        else [sampled_video[0]],
        "action": action_tensor,
        "action_mask": action_mask,
        "video_mask": video_mask,
        "proprio": proprio_tensor,
        "proprio_mask": proprio_mask,
        "prompt": prompt,
        "episode_index": ep_idx,
        "episode_path": path,
        "start_frame": start,
        "end_frame": min(raw_end, ep_len),
        "episode_length": ep_len,
        "task_name": self.task_name,
        "active_arm": active_arm,
    }


def MultiTaskRoboTwinDataset_from_config(cls, config, split: str = "train"):
    """Build from Hydra DictConfig or dict, optionally restricting tasks."""

    def _get(key, default=None):
        if hasattr(config, key):
            val = getattr(config, key)
            return default if val is None else val
        if hasattr(config, "get"):
            val = config.get(key, default)
            return default if val is None else val
        return default

    # Optional camera_layout may be a ListConfig; normalize to a plain list
    _cam_layout = _get("camera_layout", None)
    if _cam_layout is not None:
        _cam_layout = list(_cam_layout)

    _tasks = _get("tasks", None)
    if _tasks is not None:
        _tasks = list(_tasks)

    # normalize_mode can be null/None to disable
    _norm_mode = _get("normalize_mode", "min-max")
    if isinstance(_norm_mode, str) and _norm_mode.lower() in ("none", "null", ""):
        _norm_mode = None

    return cls(
        dataset_dir=_get("dataset_dir"),
        embodiment=_get("embodiment", "aloha-agilex"),
        variant=_get("variant", "both"),
        tasks=_tasks,
        normalization_stats_path=_get("normalization_stats_path", None),
        normalize_mode=_norm_mode,
        action_mode=_get("action_mode", "eef"),
        num_frames=int(_get("num_frames", 33)),
        height=int(_get("height", 384)),
        width=int(_get("width", 320)),
        split=split,
        target_camera=_get("target_camera", "head_camera"),
        window_stride=int(_get("window_stride", 1)),
        video_stride=int(_get("video_stride", 4)),
        multiview=bool(_get("multiview", True)),
        camera_layout=_cam_layout,
        backbone=_get("backbone", None),
        unify_action=bool(_get("unify_action", False)),
        unify_action_map=_get("unify_action_map", None),
        unify_state_map=_get("unify_state_map", None),
        color_jitter=_get("color_jitter", None),
    )


bind_missing_globals(globals(), _source)

# Preserve the bound classmethod interface without changing dataset identity.
MultiTaskRoboTwinDataset_from_config = types.MethodType(
    MultiTaskRoboTwinDataset_from_config, _source.MultiTaskRoboTwinDataset
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

_resolve_prompt_wrapper = _port_wrapper(
    _resolve_prompt, "openwam.dataloader.robotwin._resolve_prompt", "openwam.data"
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

RoboTwinDataset__build_sample_wrapper = _port_wrapper(
    RoboTwinDataset__build_sample,
    "openwam.dataloader.robotwin.RoboTwinDataset._build_sample",
    "openwam.data",
)

from turbo_physai.optimizations.models.openwam.settings import (
    port_wrapper as _port_wrapper,
)

MultiTaskRoboTwinDataset_from_config_wrapper = _port_wrapper(
    MultiTaskRoboTwinDataset_from_config,
    "openwam.dataloader.robotwin.MultiTaskRoboTwinDataset.from_config",
    "openwam.data",
)
