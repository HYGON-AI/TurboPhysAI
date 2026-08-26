# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Create a minimal external optimization development project."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional

from ..engine.errors import OptimizationConfigError


_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _names(name: str) -> tuple:
    value = name.strip()
    if not _NAME_PATTERN.fullmatch(value):
        raise OptimizationConfigError(
            "optimization name must start with a letter and contain only "
            "letters, numbers, '-' or '_'"
        )
    slug = value.lower().replace("-", "_")
    package = (
        slug if slug.endswith("_optimization") else f"{slug}_optimization"
    )
    model = slug[: -len("_optimization")] if slug.endswith("_optimization") else slug
    return value, model, package


def _files(display_name: str, model: str, package: str) -> Dict[Path, str]:
    recipe_name = "recipe.yaml"
    return {
        Path("README.md"): f"""# {display_name} TurboPhysAI 优化开发工程

该目录由 `turbo-physai optimization init` 生成。初始配方只继承
`common.hcu.base` 公共基础优化，不包含任何模型优化。

## 开发步骤

1. 在 `{package}/replacements.py` 或同包其他文件中实现优化代码。
2. 在 `{package}/catalog.py` 中用 `group/replace/wrap` 声明优化。
3. 将需要启用的 Group ID 写入 `configs/{recipe_name}` 的 `optimization_groups`。
4. 在模型基线仓库干净且 commit 正确时生成最终 YAML：

```bash
turbo-physai optimization generate \\
  --recipe configs/{recipe_name} \\
  --repo /path/to/model/repository \\
  --commit <validated_commit> \\
  --output configs/optimization.yaml
```

5. 使用最终 YAML 启动原有训练命令：

```bash
turbo-physai run \
  --optimization-config configs/optimization.yaml \
  python tools/train.py <原训练参数>
```

开发环境可执行 `python -m unittest discover -s tests` 验证基础结构。
如需让任意工作目录都能导入本优化包，可执行 `python -m pip install -e .`。
""",
        Path("pyproject.toml"): f"""[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "turbo-physai-{model.replace('_', '-')}-optimization"
version = "0.1.0"
description = "{display_name} optimizations developed with TurboPhysAI"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["."]
include = ["{package}*"]
""",
        Path(package) / "__init__.py": (
            f'"""{display_name} optimization package."""\n'
        ),
        Path(package) / "catalog.py": f'''"""{display_name} optimization declarations.

Add model-specific Groups here. Importing this module registers the declarations
with TurboPhysAI; the generated OptimizationConfig loads it through optimization_modules.
"""

from turbo_physai import group, replace, wrap  # noqa: F401


# Add validated model-specific Group declarations below.
''',
        Path(package) / "replacements.py": f'''"""{display_name} optimization implementations.

Keep this module empty until a real optimization has been implemented and
validated against the model baseline.
"""
''',
        Path("configs") / recipe_name: f"""schema_version: turbophysai/optimization-config/v1
kind: OptimizationConfig

metadata:
  id: model.{model}.development.hcu
  version: "0.1.0"
  description: {display_name} optimization development config

model:
  name: {display_name}

optimization_modules:
  - {package}.catalog

extends:
  - common.hcu.base

compatibility: {{}}

# Add validated model-specific Group IDs here.
optimization_groups: []
""",
        Path("tests") / "test_catalog.py": f"""import importlib
import unittest


class CatalogTest(unittest.TestCase):
    def test_catalog_can_be_imported(self):
        importlib.import_module("{package}.catalog")


if __name__ == "__main__":
    unittest.main()
""",
    }


def create_optimization_project(
    name: str,
    output: Optional[Path] = None,
) -> Path:
    """Create a blank optimization project and return its absolute path."""

    display_name, model, package = _names(name)
    target = (
        Path(output)
        if output is not None
        else Path.cwd() / package
    ).expanduser().resolve()
    if target.exists():
        raise OptimizationConfigError(
            f"refusing to overwrite existing path: {target}"
        )

    target.mkdir(parents=True)
    for relative_path, content in _files(display_name, model, package).items():
        path = target / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return target


__all__ = ["create_optimization_project"]
