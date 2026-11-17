# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep ACT batch sharding continuous while preserving epoch and resume order."""

import functools

from lerobot.datasets.sampler import EpisodeAwareSampler


def preserve_resume_epoch_wrapper(original, options):
    """Preserve a checkpoint's sampler state across Accelerate's initial set_epoch."""
    del options

    @functools.wraps(original)
    def wrapped(self, epoch):
        shard = self.batch_sampler
        sampler = getattr(getattr(shard, 'batch_sampler', None), 'sampler', None)
        if not isinstance(sampler, EpisodeAwareSampler):
            raise TypeError('ACT resume requires EpisodeAwareSampler in DataLoaderShard')
        state = sampler.state_dict()
        result = original(self, epoch)
        if epoch == 0 and (state['epoch'] != 0 or state['start_index'] != 0):
            sampler.load_state_dict(state)
        return result

    return wrapped


def continuous_batches_wrapper(original, options):
    """Chain complete sharded epochs, including their original padding batches."""
    del options

    @functools.wraps(original)
    def wrapped(self):
        sampler = self.batch_sampler.sampler
        if not isinstance(sampler, EpisodeAwareSampler):
            raise TypeError('ACT continuous batches requires EpisodeAwareSampler')
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
