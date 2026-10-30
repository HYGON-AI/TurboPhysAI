"""Flattened batched pillar scatter replacement."""

from functools import wraps

import torch


def flat_pillar_scatter_wrapper(original, options):
    """Replace the per-sample mask/stack loop with one flat scatter."""
    @wraps(original)
    def wrapped(self, voxel_features, coors, batch_size):
        if voxel_features.ndim != 2 or coors.ndim != 2 or coors.shape[1] < 4:
            return original(self, voxel_features, coors, batch_size)

        n, channels = voxel_features.shape
        height, width = self.ny, self.nx
        if channels != self.in_channels or coors.shape[0] != n:
            return original(self, voxel_features, coors, batch_size)

        spatial_size = height * width
        batch = coors[:, 0].to(torch.long)
        spatial = coors[:, 2].to(torch.long) * width + coors[:, 3].to(torch.long)
        # The upstream path assigns, rather than adds, duplicate coordinates.
        destinations = batch * spatial_size + spatial
        if (bool(((batch < 0) | (batch >= batch_size) | (spatial < 0)
                  | (spatial >= spatial_size)).any())
                or torch.unique(destinations).numel() != n):
            return original(self, voxel_features, coors, batch_size)
        channel = torch.arange(channels, device=voxel_features.device)
        indices = ((batch[:, None] * channels + channel[None, :]) * spatial_size
                   + spatial[:, None]).reshape(-1)
        canvas = voxel_features.new_zeros(batch_size * channels * spatial_size)
        canvas.scatter_add_(0, indices, voxel_features.reshape(-1))
        return canvas.view(batch_size, channels, height, width)

    return wrapped
