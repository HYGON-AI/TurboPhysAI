# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from .base import HandlerError, MechanismHandler, PreparedReplacement, ResolvedAttribute
from .import_replace import ImportReplaceHandler
from .import_alias import ImportAliasHandler
from .optional_import import OptionalImportHandler
from .registry_override import RegistryOverrideHandler
from .replace import ReplaceHandler
from .wrapper import WrapperHandler


def default_handlers():
    return {
        ReplaceHandler.mechanism: ReplaceHandler(),
        WrapperHandler.mechanism: WrapperHandler(),
        ImportReplaceHandler.mechanism: ImportReplaceHandler(),
        ImportAliasHandler.mechanism: ImportAliasHandler(),
        OptionalImportHandler.mechanism: OptionalImportHandler(),
        RegistryOverrideHandler.mechanism: RegistryOverrideHandler(),
    }


__all__ = [
    "HandlerError",
    "MechanismHandler",
    "PreparedReplacement",
    "ResolvedAttribute",
    "WrapperHandler",
    "ImportReplaceHandler",
    "ImportAliasHandler",
    "OptionalImportHandler",
    "RegistryOverrideHandler",
    "ReplaceHandler",
    "default_handlers",
]
