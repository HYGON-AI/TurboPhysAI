# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Identity checks must distinguish unreadable code from changed code."""

import hashlib
import inspect
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from turbo_physai.engine.checking import context, evidence
from turbo_physai.engine.config.reference_extraction import direct_reference_lines


class EvidenceBoundariesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_module_hashes_ignore_formatting_only_for_ast(self):
        path = self.root / "module.py"
        module = types.ModuleType("evidence_module")
        module.__file__ = str(path)
        path.write_text("def f(x):\n    return x + 1\n")
        source, syntax = evidence.source_hash(module), evidence.ast_hash(module)
        path.write_text("# comment\ndef f(x):\n    return x+1\n")
        self.assertNotEqual(evidence.source_hash(module), source)
        self.assertEqual(evidence.ast_hash(module), syntax)
        path.write_text("def f(x):\n    return x + 2\n")
        self.assertNotEqual(evidence.ast_hash(module), syntax)
        path.write_text("def broken(")
        self.assertIsNone(evidence.ast_hash(module))
        self.assertIsNotNone(evidence.source_hash(module))
        path.write_bytes(b"\xff")
        self.assertIsNone(evidence.source_hash(module))
        path.unlink()
        self.assertIsNone(evidence.source_hash(module))

    def test_native_hash_tracks_artifact_bytes_and_unavailable_files(self):
        native = object()
        module = types.ModuleType("native_evidence")
        module.__file__ = str(self.root / "extension.so")
        path = Path(module.__file__)
        path.write_bytes(b"first binary")
        with patch.object(inspect, "getmodule", return_value=module):
            self.assertEqual(
                evidence.source_hash(native),
                "artifact-v1:" + hashlib.sha256(b"first binary").hexdigest(),
            )
            path.write_bytes(b"second binary")
            self.assertEqual(
                evidence.source_hash(native),
                "artifact-v1:" + hashlib.sha256(b"second binary").hexdigest(),
            )
            self.assertIsNone(evidence.ast_hash(native))
            path.unlink()
            self.assertIsNone(evidence.source_hash(native))
            module.__file__ = str(self.root / "module.py")
            self.assertIsNone(evidence.source_hash(native))
            del module.__file__
            self.assertIsNone(evidence.source_hash(native))
        self.assertIsNone(evidence.source_hash(property(len)))
        self.assertIsNone(evidence.ast_hash(property()))

    def test_target_context_handles_native_missing_and_untracked_sources(self):
        self.assertIsNone(context.detect_target_context("native.len", len).source_file)

        def target():
            pass

        path = self.root / "target.py"
        with patch.object(inspect, "getsourcefile", return_value=str(path)):
            target_context = context.detect_target_context("target", target)
            self.assertEqual(target_context.source_file, str(path.resolve()))
            self.assertIsNone(target_context.repository_root)
            path.write_text("pass")
            target_context = context.detect_target_context("target", target)
            self.assertIsNone(target_context.repository_root)
        with patch.object(
            inspect, "getsourcefile", side_effect=ValueError("unwrap loop")
        ):
            self.assertIsNone(
                context.detect_target_context("target", target).source_file
            )

    def test_reference_inspection_tolerates_missing_or_invalid_source(self):
        for path in ("builtins.len", "nonexistent_evidence_module.fn"):
            self.assertEqual(direct_reference_lines(path, ["some.op"]), ())
        with patch(
            "turbo_physai.engine.config.reference_extraction.resolve_replacement",
            return_value=lambda: None,
        ), patch.object(inspect, "getsourcelines", return_value=(["pass"], 1)):
            with patch.object(inspect, "getsourcefile", return_value=None):
                self.assertEqual(direct_reference_lines("ignored", ["some.op"]), ())
            source = self.root / "broken.py"
            with patch.object(inspect, "getsourcefile", return_value=str(source)):
                self.assertEqual(direct_reference_lines("ignored", ["some.op"]), ())
                source.write_text("def broken(")
                self.assertEqual(direct_reference_lines("ignored", ["some.op"]), ())
