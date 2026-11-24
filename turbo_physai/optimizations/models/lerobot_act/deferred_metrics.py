# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""Defer ACT loss metric scalar reads to the logging boundary."""

import functools

import torch
import torch.nn.functional as F


def _require_group_options(options):
    if options:
        raise ValueError("Enable or disable the whole ACT async scalars Group; individual options are unsupported.")


def deferred_metrics_policy_forward(self, batch):
    """Retain the ACT loss calculation but defer scalar reads to logging."""
    from lerobot.utils.constants import ACTION, OBS_IMAGES

    if self.config.image_features:
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

    actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)
    abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction='none')
    valid_mask = ~batch['action_is_pad'].unsqueeze(-1)
    num_valid = valid_mask.sum() * abs_err.shape[-1]
    l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
    loss_dict = {'l1_loss': l1_loss.detach()}
    if self.config.use_vae and log_sigma_x2_hat is not None:
        mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
        loss_dict['kld_loss'] = mean_kld.detach()
        loss = l1_loss + mean_kld * self.config.kl_weight
    else:
        loss = l1_loss
    return loss, loss_dict


def deferred_metrics_policy_wrapper(original, options):
    del original
    _require_group_options(options)
    return deferred_metrics_policy_forward


def deferred_metrics_wandb_wrapper(original, options):
    _require_group_options(options)

    @functools.wraps(original)
    def wrapped(self, values, *args, **kwargs):
        values = dict(values)
        for key in ("l1_loss", "kld_loss"):
            if isinstance(values.get(key), torch.Tensor):
                values[key] = values[key].item()
        return original(self, values, *args, **kwargs)

    return wrapped
