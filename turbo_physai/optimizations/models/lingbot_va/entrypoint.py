# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the same imported training module that LingBot-VA Groups patch.

Executing ``-m train`` would define a second Trainer in ``__main__`` after
bootstrap has patched ``train.Trainer``. Keep one module identity instead.
The upstream repository's ``wan_va`` directory must be on ``PYTHONPATH``
before bootstrap; the example launcher sets it from ``LINGBOT_VA_ROOT``.
"""

from importlib import import_module


def main() -> None:
    training = import_module("train")
    # Match the upstream __main__ block, including its logger initialization.
    training.init_logger()
    training.main()


if __name__ == "__main__":
    main()
