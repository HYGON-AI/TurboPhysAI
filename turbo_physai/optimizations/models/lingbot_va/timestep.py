# Adapted from Robbyant/lingbot-va, commit 7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb.
# Original copyright: 2024-2025 The Robbyant Team Authors. All rights reserved.
import functools
import torch


def timestep_wrapper(original, options):
    if options:
        raise ValueError('timestep_dedup currently accepts no options')
    @functools.wraps(original)
    def forward(self, timesteps, H, W, dtype, action_mode=False):
        ph, pw = (1, 1) if action_mode else self.patch_size[1:3]
        repeats = (H // ph) * (W // pw)
        if repeats <= 1 or timesteps.ndim != 2:
            return original(self, timesteps, H, W, dtype, action_mode=action_mode)
        embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, projection = embedder(timesteps, dtype=dtype)
        return (torch.repeat_interleave(temb, repeats, dim=1),
                torch.repeat_interleave(projection.unflatten(2, (6, -1)), repeats, dim=1))
    return forward
