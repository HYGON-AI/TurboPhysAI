# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized shuffled ACT sampling, preserving sampler order and resume state."""

import functools

import numpy as np
import torch


def vectorized_sampler_wrapper(original, options):
    del options

    @functools.wraps(original)
    def wrapped(self, epoch, start):
        if not self.shuffle:
            raise ValueError('ACT sampler requires shuffle=True, got shuffle=False')
        if not isinstance(start, int):
            raise TypeError(f'ACT sampler start must be a Python int, got {type(start).__name__}: {start!r}')
        if start < 0:
            raise ValueError(f'ACT sampler start must be non-negative, got {start}')

        order = torch.randperm(self._num_frames, generator=self._epoch_generator(epoch))
        positions = order.numpy()[start:]

        episodes = np.searchsorted(self._cum_lengths, positions, side="right")
        previous = np.concatenate((np.zeros(1, dtype=self._cum_lengths.dtype), self._cum_lengths[:-1]))
        indices = (self._starts[episodes] + positions - previous[episodes]).tolist()
        mapping = self._absolute_to_relative
        if mapping is not None:
            try:
                indices = [mapping[index] for index in indices]
            except KeyError as error:
                raise KeyError(
                    f'ACT sampler absolute frame index {error.args[0]} is missing from the dataset mapping'
                ) from error
        yield from indices

    return wrapped
