"""Offline checks for the code release boundary (D-R4).

The release artifact is defined by an explicit allowlist, not by excluding
known-bad files.  These tests verify that no forbidden content is tracked in
Git, that untracking never removed a local original, and that the artifact
assembled from the allowlist carries no full text, vector database, production
database, runtime binary or credential-shaped text.

Shared rules live in ``tests/release_scope.py``; the distribution set is
assembled in ``tests/test_release_scope.py``.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import release_scope as rs  # noqa: E402

ROOT = rs.ROOT
ALLOWLIST = rs.ALLOWLIST

# Local-only data whose removal from the index must never touch the disk copy.
UNTRACKED_RUNTIME_FILES = (
    "corpus_raw/PMID_31096736/article.xml",
    "corpus_raw/PMID_31096736/source.json",
    "直肠癌文献爬取/tasks.sqlite",
    "直肠癌文献爬取/reports/download_report.csv",
    "直肠癌文献爬取/reports/run_history.csv",
)

class ReleaseBoundaryTests(unittest.TestCase):
    def test_no_forbidden_path_is_tracked(self):
        if not rs.git_available():
            self.skipTest("no local Git repository: the index cannot be inspected here")
        offenders = {
            name: rs.forbidden_hits(name)
            for name in rs.tracked_files()
            if rs.forbidden_hits(name)
        }
        # ledger_manifest, legacy_pdf_corpus and runtime_reports stay tracked on
        # purpose (local metadata inputs); the allowlist keeps them out of the
        # release.  Everything else must be untracked.
        unexpected = {
            name: labels for name, labels in offenders.items()
            if "ledger_manifest" not in labels
            and "legacy_pdf_corpus" not in labels
            and "runtime_reports" not in labels
        }
        self.assertEqual(unexpected, {}, f"forbidden content is still tracked: {unexpected}")

    def test_untracked_runtime_files_are_still_on_disk(self):
        if not rs.git_available():
            self.skipTest("no local Git repository: the index cannot be inspected here")
        tracked = rs.tracked_files()
        for relative in UNTRACKED_RUNTIME_FILES:
            with self.subTest(relative=relative):
                self.assertNotIn(relative, tracked)
        if not any((ROOT / relative).exists() for relative in UNTRACKED_RUNTIME_FILES):
            self.skipTest("local originals are not present in this checkout: "
                          "on-disk retention cannot be verified here")
        for relative in UNTRACKED_RUNTIME_FILES:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).exists(),
                                "untracking must never delete the local original")

    def test_gitignore_covers_raw_corpus_and_production_database(self):
        text = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for expected in ("/corpus_raw/", "tasks.sqlite", "*.sqlite-wal"):
            self.assertIn(expected, text)

    def test_release_artifact_contains_no_forbidden_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "release"
            target.mkdir()
            copied = rs.build_release_dir(target)
            self.assertTrue(copied, "the allowlist matched no files")
            self.assertEqual(rs.scan_release(target.resolve(), copied), [])

    def test_scan_rejects_secret_content_and_non_allowlisted_paths_without_echoing_values(self):
        fake_token = "sk-proj-" + "Z" * 30
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp).resolve()
            (target / "README.md").write_text(
                fake_token + '\n{"NCBI_API_KEY": "' + "Q" * 36 + '"}', encoding="utf-8",
            )
            outside = "unexpected-" + fake_token + ".txt"
            (target / outside).touch()
            self.assertCountEqual(rs.scan_release(target, ["README.md", outside]), [
                "file #1: openai_style_key at line 1", "file #1: configured_token at line 2",
                "file #2: outside_allowlist",
            ])

    def test_allowlist_names_the_local_only_data(self):
        text = ALLOWLIST.read_text(encoding="utf-8")
        for expected in ("corpus_raw/", "tasks.sqlite", "chroma_db", "backups/"):
            self.assertIn(expected, text)


class VerifiedCheckoutTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.allowlist = self.root / ALLOWLIST.relative_to(ROOT)
        self.allowlist.parent.mkdir()
        relative = self.allowlist.relative_to(self.root).as_posix()
        self.allowlist.write_bytes(f"```\nREADME.md\n{relative}\n```\n".encode("utf-8"))
        self.readme = self.root / "README.md"
        self.output = self.root / "output"
        self.git("init", "-q")
        self.git("config", "core.autocrlf", "false")

    def git(self, *args):
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def scan(self, *args):
        stdout = StringIO()
        with patch.multiple(rs, ROOT=self.root, ALLOWLIST=self.allowlist), \
                patch.object(sys, "argv", ["release_scope.py", "--verify-checkout", *args]), \
                redirect_stdout(stdout):
            code = rs.main()
        return code, stdout.getvalue()

    def test_same_path_with_staged_secret_and_clean_worktree_is_rejected(self):
        fake_token = "sk-proj-" + "Z" * 30
        self.readme.write_bytes((fake_token + "\n").encode())
        self.git("add", ".")
        self.readme.write_bytes(b"clean\n")

        for args in ((), ("--output", str(self.output))):
            with self.subTest(args=args):
                code, report = self.scan(*args)
                self.assertEqual(code, 1, report)
                self.assertIn("worktree_content_differs_from_index", report)
                self.assertIn("openai_style_key at line 1", report)
                self.assertNotIn(fake_token, report)
                self.assertFalse(self.output.exists())

    def test_clean_checkout_exports_the_exact_staged_bytes(self):
        self.readme.write_bytes(b"clean\n")
        self.git("add", ".")
        self.git("config", "core.autocrlf", "true")
        self.readme.write_bytes(b"clean\r\n")

        code, report = self.scan("--output", str(self.output))
        self.assertEqual(code, 0, report)
        self.assertIn("PASS release scan: 2 files, 0 findings", report)
        self.assertEqual((self.output / "README.md").read_bytes(), b"clean\n")
        self.assertEqual((self.output / self.allowlist.relative_to(self.root)).read_bytes(),
                         self.allowlist.read_bytes())


if __name__ == "__main__":
    sys.exit(unittest.main())
