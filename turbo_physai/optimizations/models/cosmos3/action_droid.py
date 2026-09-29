# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from __future__ import annotations

import cosmos_framework.data.generator.action.datasets.droid_lerobot_dataset as _source
from ._source_binding import bind_missing_globals

def __init__(
    self,
    root: str = "/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_action_datasets/droid_plus_lerobot_640x360_20260412",
    fps: float = 15.0,
    chunk_length: int = 16,
    split_seed: int = 42,
    split_val_ratio: float = 0.03,
    split: str = "train",
    mode: str = "wam",
    pose_convention: PoseConvention = "backward_framewise",
    action_normalization: ActionNormalization | None = None,
    tolerance_s=2e-4,
    viewpoint: Viewpoint = "concat_view",
    use_success_only: bool = False,
    video_backend: str | None = None,
    video_mode: str | None = None,  # TODO (ychao): remove
    action_space: str = "midtrain",  # TODO (ychao): remove
    use_state: bool = False,
    use_filter_dict: bool = False,
    filter_dict_path: str | None = None,
    enable_fast_init: bool = False,
    max_num_history_actions: int = 0,
    use_image_augmentation: bool = False,
) -> None:
    """ """
    super(_source.DROIDLeRobotDataset, self).__init__(
        fps=fps,
        chunk_length=chunk_length,
        split_seed=split_seed,
        split_val_ratio=split_val_ratio,
        split=split,
        mode=mode,
        embodiment_type="droid_lerobot",
        viewpoint=viewpoint,
        pose_convention=pose_convention,
        rotation_format="rot6d",
        action_normalization=action_normalization,
        tolerance_s=tolerance_s,
        video_backend=video_backend,
        enable_fast_init=enable_fast_init,
    )
    self._use_success_only = use_success_only
    self._video_mode = video_mode
    self._action_space = action_space
    self._use_state = use_state
    self._use_filter_dict = use_filter_dict
    self._filter_dict_path = filter_dict_path or _FILTER_DICT_PATH
    self._max_num_history_actions = max_num_history_actions
    self._use_image_augmentation = use_image_augmentation
    if max_num_history_actions > 0 and action_space not in ("midtrain", "joint_pos"):
        raise ValueError(
            f"max_num_history_actions is only supported with action_space='midtrain' or 'joint_pos', got {action_space!r}"
        )

    self._is_val_temp_seg = split == "val_temp_seg"
    self._to_opencv = _DROID_TO_OPENCV

    version = os.path.basename(root)
    if version not in LEROBOT_ROOTS:
        # Compatibility fallback: if root is a performance subset or custom folder
        # (e.g. Cosmos3-DROID-perf64) containing a success split, map to standard 640x360 schema.
        if (
            version == "Cosmos3-DROID-perf64"
            or "perf" in version.lower()
            or (os.path.isdir(root) and os.path.isdir(os.path.join(root, "success")))
        ) and "droid_plus_lerobot_640x360_20260412" in LEROBOT_ROOTS:
            version = "droid_plus_lerobot_640x360_20260412"

    try:
        lerobot_roots = LEROBOT_ROOTS[version]
        self._image_features = IMAGE_FEATURES[version]
        self._state_features = STATE_FEATURES[version]
        self._action_features = ACTION_FEATURES[version]
        self._is_flat_action = IS_FLAT_ACTION[version]
        self._has_multi_language_annotations = HAS_MULTI_LANGUAGE_ANNOTATIONS[version]
        self._is_gripper_action_flipped = IS_GRIPPER_ACTION_FLIPPED[version]
    except KeyError as e:
        raise ValueError(f"Unknown version: {version!r}. Supported: {list(LEROBOT_ROOTS.keys())}") from e

    if self._use_success_only and lerobot_roots:
        lerobot_roots = [x for x in lerobot_roots if x.split("/", 1)[0] == "success"]

    self._all_shard_roots = [os.path.join(root, x) for x in lerobot_roots] if lerobot_roots else [root]

    observation_ts = [i * self._dt for i in range(0, self._chunk_length + 1)]
    action_ts = [i * self._dt for i in range(0, self._chunk_length)]
    if self._max_num_history_actions > 0 and self._action_space in ("midtrain", "joint_pos"):
        observation_ts_ext = [i * self._dt for i in range(-self._max_num_history_actions, self._chunk_length + 1)]
        action_ts_ext = [i * self._dt for i in range(-self._max_num_history_actions, self._chunk_length)]
    else:
        observation_ts_ext = observation_ts
        action_ts_ext = action_ts
    self._delta_timestamps: dict[str, list[float]] = {
        self._state_features: observation_ts_ext,
        self._action_features: action_ts_ext,
    }
    if self._viewpoint in ("wrist_view", "concat_view"):
        self._delta_timestamps[self._image_features["wrist"]] = observation_ts
    if self._viewpoint in ("third_person_view", "concat_view"):
        self._delta_timestamps[self._image_features["left"]] = observation_ts
        self._delta_timestamps[self._image_features["right"]] = observation_ts
    if self._action_space == "joint_pos":
        self._delta_timestamps[_JOINT_ACTION_FEATURE] = action_ts
        if self._use_state or self._max_num_history_actions > 0:
            self._delta_timestamps[_JOINT_STATE_FEATURE] = observation_ts_ext
            self._delta_timestamps[_GRIPPER_STATE_FEATURE] = observation_ts_ext
    if self._use_state and self._action_space != "joint_pos":
        self._delta_timestamps[_GRIPPER_STATE_FEATURE] = observation_ts

    if self._use_filter_dict:
        with open(self._filter_dict_path) as f:
            self._filter_dict = json.load(f)

    self._image_augmentor: T.Compose | None = None

    # Eager source registration. i4 defers this to its own dataloader's
    # ActionUnifiedIterableDataset.assign_worker(); cosmos-framework instead
    # drives the dataset through ActionIterableShuffleDataset (block-striding,
    # which needs the full flat index present in every worker), so we build
    # the index at construction time here. Metadata-only (LeRobotDatasetMetadata:
    # info.json + episodes.parquet + tasks.parquet); the heavy per-shard
    # LeRobotDataset video readers stay lazy behind the LRU in _get_dataset.
    self._register_sources()


bind_missing_globals(globals(), _source)
