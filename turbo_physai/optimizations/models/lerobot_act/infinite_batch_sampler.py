# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep ACT batch sharding continuous while preserving epoch and resume order."""

import functools

from lerobot.datasets.sampler import EpisodeAwareSampler


def _episode_sampler(loader):
    batch_sampler = loader.batch_sampler
    sampler = getattr(batch_sampler, 'sampler', None)
    if isinstance(sampler, EpisodeAwareSampler):
        return sampler
    return getattr(getattr(batch_sampler, 'batch_sampler', None), 'sampler', None)


def supports_resume_epoch(self, epoch):
    """Restrict the global Accelerate hook to ACT's sampler."""
    del epoch
    return isinstance(_episode_sampler(self), EpisodeAwareSampler)


def supports_continuous_batches(self):
    """Leave BatchSamplerShard users with other samplers unchanged."""
    return isinstance(getattr(self.batch_sampler, 'sampler', None), EpisodeAwareSampler)


def preserve_resume_epoch_wrapper(original, options):
    """Preserve a checkpoint's sampler state across Accelerate's initial set_epoch."""
    del options

    @functools.wraps(original)
    def wrapped(self, epoch):
        sampler = _episode_sampler(self)
        state = sampler.state_dict()
        unsharded = getattr(self.batch_sampler, 'sampler', None) is sampler
        result = original(self, epoch)
        if ((epoch == 0 and (state['epoch'] != 0 or state['start_index'] != 0))
                or (unsharded and state['epoch'] > epoch)):
            sampler.load_state_dict(state)
        return result

    return wrapped


def continuous_batches_wrapper(original, options):
    """Chain complete sharded epochs, including their original padding batches."""
    del options

    @functools.wraps(original)
    def wrapped(self):
        sampler = self.batch_sampler.sampler
        if self.split_batches or not self.even_batches or self.drop_last:
            raise ValueError('ACT continuous batches requires split_batches=False, even_batches=True, drop_last=False')
        if len(self) == 0:
            raise ValueError('ACT continuous batches requires nonempty epochs')

        state = sampler.state_dict()
        start = state['start_index']
        if start:
            global_batch = self.batch_size * self.num_processes
            if start % global_batch:
                raise ValueError('ACT resume sample offset must be divisible by global batch size')
            sampler.load_state_dict({'epoch': state['epoch'], 'start_index': 0})
            first_epoch = iter(original(self))
            for _ in range(start // global_batch):
                try:
                    next(first_epoch)
                except StopIteration as error:
                    raise ValueError('ACT resume offset exceeds sampler epoch') from error
            yield from first_epoch
        while True:
            yield from original(self)

    return wrapped
