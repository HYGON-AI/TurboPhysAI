# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify every shipped Config without importing model or Catalog modules."""

import unittest

from turbo_physai.engine.config.generation_record import PACKAGE_ROOT, verify_generated


# Follow the imported package so wheel CI also checks the packaged receipts.
OPTIMIZATION_ROOT = PACKAGE_ROOT / "optimizations"


class GeneratedConfigsTest(unittest.TestCase):
    def _verify_configs(self, category):
        configs = sorted((OPTIMIZATION_ROOT / category).glob("**/configs/optimization.yaml"))
        self.assertTrue(configs, f"No shipped Configs found under {category}")
        for config in configs:
            with self.subTest(config=str(config.relative_to(OPTIMIZATION_ROOT))):
                verify_generated(config)

    def test_common_configs_match_generation_records(self):
        self._verify_configs("common")

    def test_model_configs_match_generation_records(self):
        self._verify_configs("models")


if __name__ == "__main__":
    unittest.main()
