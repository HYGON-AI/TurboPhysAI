# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable PyTorch reference for deformable feature aggregation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _evaluate(
    feature_maps: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_starts: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Aggregate sampled multi-camera features with grouped weights."""

    batch, _, channels = feature_maps.shape
    num_cameras, num_levels, _ = spatial_shapes.shape
    _, num_anchors, num_points, location_cameras, coordinates = (
        sampling_locations.shape
    )
    weight_shape = attention_weights.shape
    num_groups = weight_shape[-1]

    if location_cameras != num_cameras or coordinates != 2:
        raise ValueError("sampling locations are inconsistent with spatial shapes")
    if weight_shape[:5] != (
        batch,
        num_anchors,
        num_points,
        num_cameras,
        num_levels,
    ):
        raise ValueError("attention weights have an incompatible shape")
    if channels % num_groups:
        raise ValueError("feature channels must be divisible by weight groups")

    channels_per_group = channels // num_groups
    output = feature_maps.new_zeros(
        batch, num_anchors, num_groups, channels_per_group
    )
    shape_values = spatial_shapes.detach().cpu().tolist()
    start_values = level_starts.detach().cpu().tolist()

    for camera in range(num_cameras):
        normalized_locations = sampling_locations[:, :, :, camera]
        sampling_grid = normalized_locations.mul(2).sub(1)
        valid = (
            normalized_locations.gt(0)
            & normalized_locations.lt(1)
        ).all(dim=-1)

        for level in range(num_levels):
            height, width = (int(value) for value in shape_values[camera][level])
            start = int(start_values[camera][level])
            end = start + height * width
            feature_level = (
                feature_maps[:, start:end]
                .reshape(batch, height, width, channels)
                .permute(0, 3, 1, 2)
            )
            sampled = F.grid_sample(
                feature_level,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.permute(0, 2, 3, 1).reshape(
                batch,
                num_anchors,
                num_points,
                num_groups,
                channels_per_group,
            )
            level_weights = attention_weights[:, :, :, camera, level]
            weighted = sampled * level_weights.unsqueeze(-1)
            weighted = weighted * valid.unsqueeze(-1).unsqueeze(-1)
            output = output + weighted.sum(dim=2)

    return output.reshape(batch, num_anchors, channels)


def deformable_aggregation_reference(
    feature_maps: np.ndarray,
    spatial_shapes: np.ndarray,
    level_starts: np.ndarray,
    sampling_locations: np.ndarray,
    attention_weights: np.ndarray,
) -> np.ndarray:
    """Return the CPU reference output as a NumPy array."""

    with torch.no_grad():
        output = _evaluate(
            torch.from_numpy(feature_maps),
            torch.from_numpy(spatial_shapes),
            torch.from_numpy(level_starts),
            torch.from_numpy(sampling_locations),
            torch.from_numpy(attention_weights),
        )
    return output.numpy()


def deformable_aggregation_reference_gradients(
    feature_maps: np.ndarray,
    spatial_shapes: np.ndarray,
    level_starts: np.ndarray,
    sampling_locations: np.ndarray,
    attention_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate a unit-sum output instead of maintaining a second oracle."""

    feature_tensor = torch.from_numpy(feature_maps).requires_grad_()
    location_tensor = torch.from_numpy(sampling_locations).requires_grad_()
    weight_tensor = torch.from_numpy(attention_weights).requires_grad_()
    output = _evaluate(
        feature_tensor,
        torch.from_numpy(spatial_shapes),
        torch.from_numpy(level_starts),
        location_tensor,
        weight_tensor,
    )
    gradients = torch.autograd.grad(
        output,
        (feature_tensor, location_tensor, weight_tensor),
        grad_outputs=torch.ones_like(output),
    )
    return tuple(gradient.detach().numpy() for gradient in gradients)
