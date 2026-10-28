# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Install OpenPI compatibility shims during Python startup."""

from turbo_physai.optimizations.models.openpi.runtime_compatibility import apply_compatibility_shims

apply_compatibility_shims()
