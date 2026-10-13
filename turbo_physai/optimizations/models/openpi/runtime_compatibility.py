"""Compatibility shims that remain active independently of optimization groups."""

from __future__ import annotations

import datetime as _datetime
import os
from pathlib import Path
import shutil
import tempfile


def _seed_external_tokenizer_cache() -> None:
    """Seed the user-provided tokenizer into OpenPI runtime cache."""
    data_home = os.environ.get("OPENPI_DATA_HOME")
    if not data_home:
        return

    default_assets_dir = Path(__file__).resolve().parents[5] / "static_data" / "assets"
    source = (
        Path(os.environ.get("OPENPI_ASSETS_DIR", default_assets_dir)).expanduser()
        / "paligemma_tokenizer.model"
    )
    target = (
        Path(data_home).expanduser()
        / "big_vision"
        / "paligemma_tokenizer.model"
    )
    if target.is_file():
        return
    if not source.is_file():
        raise FileNotFoundError(f"OpenPI tokenizer is missing from external assets: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=".paligemma_tokenizer.", dir=target.parent
    )
    os.close(fd)
    try:
        shutil.copyfile(source, temporary_path)
        os.chmod(temporary_path, 0o644)
        try:
            # Link atomically so concurrent torchrun ranks never observe a
            # partial file and an existing user-provided tokenizer is kept.
            os.link(temporary_path, target)
        except FileExistsError:
            pass
    finally:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass


def apply_compatibility_shims() -> None:
    """Apply only compatibility fixes needed by this Python/data runtime."""
    _seed_external_tokenizer_cache()
    # Python 3.10 does not expose datetime.UTC (added in Python 3.11).
    if not hasattr(_datetime, "UTC"):
        _datetime.UTC = _datetime.timezone.utc

    # Older LeRobot parquet metadata can contain _type="List". Newer datasets
    # releases represent the same feature with Sequence but omit the old alias.
    try:
        from datasets.features import features as _features

        feature_types = getattr(_features, "_FEATURE_TYPES", None)
        sequence_type = getattr(_features, "Sequence", None)
        if (
            isinstance(feature_types, dict)
            and "List" not in feature_types
            and sequence_type is not None
        ):
            feature_types["List"] = sequence_type
    except Exception:
        # Non-training interpreters may not have datasets installed.
        pass
