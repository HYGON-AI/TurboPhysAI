#!/usr/bin/env bash
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python -m coverage erase
python -m coverage run -m unittest discover -s test/engine
python -m coverage report
