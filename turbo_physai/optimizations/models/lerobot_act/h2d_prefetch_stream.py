# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

"""Copy-stream prefetch for the ACT Accelerate training DataLoader."""

import torch
from accelerate.data_loader import synchronize_rng_states
from accelerate.utils import send_to_device


def _record_batch(batch, stream):
    if isinstance(batch, torch.Tensor):
        if batch.device.type == 'cuda':
            batch.record_stream(stream)
    elif isinstance(batch, dict):
        for value in batch.values():
            _record_batch(value, stream)
    elif isinstance(batch, (list, tuple)):
        for value in batch:
            _record_batch(value, stream)


def supports_prefetch_dataloader(self):
    """Use the copy-stream path only for pinned, nonblocking HCU loaders."""
    return bool(self.device is not None and self.device.type == 'cuda'
                and self.pin_memory and self._non_blocking)


def prefetch_dataloader_iter(self):
    """Preserve Accelerate's one-batch lookahead while moving it on a copy stream."""
    if self.rng_types is not None:
        synchronize_rng_states(self.rng_types, self.synchronized_generator)
    self.begin()
    try:
        self.set_epoch(self.iteration)
        data_iter = iter(self.base_dataloader)
        try:
            source = next(data_iter)
        except StopIteration:
            return

        copy_stream = torch.cuda.Stream(device=self.device)
        pending_sources = []

        def enqueue(source_batch):
            with torch.cuda.stream(copy_stream):
                device_batch = send_to_device(source_batch, self.device, non_blocking=True)
                event = torch.cuda.Event()
                event.record(copy_stream)
            pending_sources.append((source_batch, event))
            return device_batch, event

        def ready(device_batch, event):
            consumer = torch.cuda.current_stream(self.device)
            consumer.wait_event(event)
            _record_batch(device_batch, consumer)
            return device_batch

        try:
            current, current_event = enqueue(source)
            batch_index = 0
            while True:
                pending_sources[:] = [(batch, event) for batch, event in pending_sources if not event.query()]
                self._update_state_dict()
                try:
                    next_source = next(data_iter)
                except StopIteration:
                    self.end_of_dataloader = True
                    self._update_state_dict()
                    if batch_index >= self.skip_batches:
                        yield ready(current, current_event)
                    break
                next_batch, next_event = enqueue(next_source)
                if batch_index >= self.skip_batches:
                    yield ready(current, current_event)
                batch_index += 1
                current, current_event = next_batch, next_event
        finally:
            # Keep pinned sources alive until outstanding copies complete, even on early exit.
            copy_stream.synchronize()
        self.iteration += 1
    finally:
        self.end()
