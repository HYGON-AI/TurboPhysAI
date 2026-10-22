# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable external aliases for official OpenPI targets.

The two PI0 compile groups need independent Turbo configuration while both
intercepting the same official ``PI0Pytorch._apply_checkpoint`` method.  An
alias of the official class gives Turbo a second dotted target without
editing the official checkout; both paths resolve to the same class object.
"""

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch


PI0PytorchImageAlias = PI0Pytorch
