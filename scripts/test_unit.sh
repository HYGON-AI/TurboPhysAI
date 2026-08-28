#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python -m pytest -m "not hcu" \
  test/engine \
  test/test_runner.py \
  test/optimizations
