"""Compile voxel-coordinate/index preparation for LSSViewTransformer."""

import functools

import torch


def voxel_pooling_pre_core(coor, grid_lower_bound, grid_interval, grid_size):
    batch, cameras, depth, height, width, _ = coor.shape
    num_points = batch * cameras * depth * height * width
    ranks_depth = torch.arange(
        num_points, dtype=torch.int32, device=coor.device
    )
    ranks_feat = torch.arange(
        num_points // depth, dtype=torch.int32, device=coor.device
    )
    ranks_feat = ranks_feat.reshape(batch, cameras, 1, height, width)
    ranks_feat = ranks_feat.expand(
        batch, cameras, depth, height, width
    ).flatten()
    voxel_coor = ((coor - grid_lower_bound) / grid_interval).long().view(
        num_points, 3
    )
    batch_idx = torch.arange(
        batch, dtype=voxel_coor.dtype, device=coor.device
    ).reshape(batch, 1).expand(batch, num_points // batch).reshape(num_points, 1)
    voxel_coor = torch.cat((voxel_coor, batch_idx), 1)
    kept = (
        (voxel_coor[:, 0] >= 0)
        & (voxel_coor[:, 0] < grid_size[0])
        & (voxel_coor[:, 1] >= 0)
        & (voxel_coor[:, 1] < grid_size[1])
        & (voxel_coor[:, 2] >= 0)
        & (voxel_coor[:, 2] < grid_size[2])
    )
    return voxel_coor, ranks_depth, ranks_feat, kept


def voxel_pooling_prepare_wrapper(original, options):
    compile_kwargs = {}
    if "mode" in options:
        compile_kwargs["mode"] = options["mode"]
    compiled_pre_core = torch.compile(voxel_pooling_pre_core, **compile_kwargs)

    @functools.wraps(original)
    def wrapped(self, coor):
        core = compiled_pre_core if coor.device.type == "cuda" else voxel_pooling_pre_core
        voxel_coor, ranks_depth, ranks_feat, kept = core(
            coor, self.grid_lower_bound, self.grid_interval, self.grid_size
        )
        if len(kept) == 0:
            return None, None, None, None, None
        voxel_coor, ranks_depth, ranks_feat = (
            voxel_coor[kept], ranks_depth[kept], ranks_feat[kept]
        )
        ranks_bev = voxel_coor[:, 3] * (
            self.grid_size[2] * self.grid_size[1] * self.grid_size[0]
        )
        ranks_bev += voxel_coor[:, 2] * (
            self.grid_size[1] * self.grid_size[0]
        )
        ranks_bev += voxel_coor[:, 1] * self.grid_size[0] + voxel_coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = (
            ranks_bev[order], ranks_depth[order], ranks_feat[order]
        )
        starts_mask = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool
        )
        starts_mask[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(starts_mask)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return (
            ranks_bev.int().contiguous(), ranks_depth.int().contiguous(),
            ranks_feat.int().contiguous(), interval_starts.int().contiguous(),
            interval_lengths.int().contiguous(),
        )

    return wrapped
