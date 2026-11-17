# Adapted from Isaac-GR00T, SPDX-License-Identifier: Apache-2.0
"""Pad GR00T text inputs to width 156 at the trainer input boundary."""

from collections.abc import MutableMapping
from copy import copy
from functools import wraps

import torch
import torch.nn.functional as F


_ALLOWED_FIELDS = frozenset({
    "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state",
    "action", "action_mask", "state_mask", "embodiment_id",
})


def fixed_padding_wrapper(original, group_options):
    """Wrap input preparation without changing the tokenizer or model."""
    @wraps(original)
    def prepare(self, inputs):
        if (not isinstance(inputs, MutableMapping) or set(inputs) != {"inputs"}
                or not isinstance(inputs["inputs"], MutableMapping)):
            raise ValueError("Unsupported batch structure for fixed sequence padding")
        values = inputs["inputs"]
        unexpected = set(values) - _ALLOWED_FIELDS
        if unexpected:
            raise ValueError(f"Unsupported batch fields for fixed sequence padding: {sorted(unexpected)}")
        ids, mask = values.get("input_ids"), values.get("attention_mask")
        if (not isinstance(ids, torch.Tensor) or not isinstance(mask, torch.Tensor)
                or ids.ndim != 2 or ids.shape != mask.shape or ids.device.type != "cpu"
                or mask.device.type != "cpu" or ids.shape[0] == 0):
            raise ValueError("Expected matching nonempty CPU rank-2 input_ids and attention_mask")
        if not torch.all((mask == 0) | (mask == 1)) or torch.any(mask[:, 1:] < mask[:, :-1]):
            raise ValueError("Expected a binary left-padded attention mask")
        if ids.shape[1] > 156 or int(mask.sum(dim=1).max().item()) > 156:
            raise ValueError("Sequence exceeds fixed maximum 156")
        try:
            pad_token_id = self.data_collator.processor.tokenizer.pad_token_id
        except AttributeError as exc:
            raise ValueError("Fixed padding requires a tokenizer pad token") from exc
        if type(pad_token_id) is not int:
            raise ValueError("Fixed padding requires an integer tokenizer pad token")
        extra = 156 - ids.shape[1]
        padded_values = copy(values)
        padded_values["input_ids"] = F.pad(ids, (extra, 0), value=pad_token_id)
        padded_values["attention_mask"] = F.pad(mask, (extra, 0), value=0)
        padded_inputs = copy(inputs)
        padded_inputs["inputs"] = padded_values
        return original(self, padded_inputs)
    return prepare
