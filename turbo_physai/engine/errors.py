# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Optional


class TurboPhysAIError(RuntimeError):
    """Base class for public TurboPhysAI errors."""


class OptimizationConfigError(TurboPhysAIError):
    pass


class OptimizationConfigNotFoundError(OptimizationConfigError):
    pass


class RuntimeConfigError(TurboPhysAIError):
    pass


class ReportCarryingError(TurboPhysAIError):
    """An error raised after a best-effort report has been produced."""

    def __init__(self, message: str, *, report: Any = None) -> None:
        super().__init__(message)
        self.report = report


class OptimizationExecutionError(ReportCarryingError):
    pass


class OptimizationRollbackError(OptimizationExecutionError):
    pass


class ReportWriteError(ReportCarryingError):
    def __init__(
        self, message: str, *, report: Any = None, cause: Optional[BaseException] = None
    ) -> None:
        super().__init__(message, report=report)
        self.__cause__ = cause
