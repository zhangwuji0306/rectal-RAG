"""Distribution-set and Git-history checks for the release boundary (D-R4).

Two different things are checked here:

* the **distribution set** - the files a code release would carry, assembled
  from ``docs/发布范围-allowlist.txt`` - contains no corpus full text, no vector
  store, no production database, no runtime report or driver binary and no
  credential-shaped text;
* the **reachable Git history** - what has ever been committed - so that a
  history-bearing publication can be planned with concrete paths and commits
  instead of an assumption.

Only read-only Git commands are used (``ls-files``, ``log``, ``ls-tree``,
``status``).  Nothing here pushes, rewrites, deletes or untracks anything, and
the remotely stored history is not contacted: the only remote evidence is the
remote-tracking ref that an earlier fetch recorded locally.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import release_scope as rs  # noqa: E402

ROOT = rs.ROOT

# Forbidden categories that must not be tracked in the index at all.
NEVER_TRACKED_LABELS = (
    "raw_fulltext_xml",
    "corpus_markdown",
    "mac_corpus_copy",
    "normalized_corpus",
    "vector_store",
    "production_database",
    "bulk_metadata_export",
    "model_cache",
    "browser_driver",
    "zotero_metadata",
    "virtualenv",
    "release_artifacts",
)
# Local inputs still tracked and excluded from the release by the allowlist.
TRACKED_BUT_NOT_RELEASED_PATHS = (
    "index/manifest.json",
    "papers/索引信息.csv",
)

# Inspected history: every forbidden path ever committed, with the commit that
# added it and whether it is still reachable from the current HEAD, still in the
# index, and still on disk.  Update this ledger only after re-inspecting history.
ROOT_COMMIT = "ba1f70cf7605f9ec1a68f0b91a3364b75fdaab8c"
SNAPSHOT_COMMIT = "0a41189b76a0f4e53da689192fb0dfb3114a02a0"
CORPUS_XML_COMMIT = "ce5c1e737ea744a2807fa94503f7eb0a66147ff7"
LICENSE_REPORT_COMMIT = "8e1e4808a0599f59c1d35a941458cc469144ba2a"

HISTORY_LEDGER: dict[str, dict] = {
    "corpus_raw/PMID_31096736/article.xml": {
        "labels": ("raw_fulltext_xml",), "in_head": True, "in_index": False,
        "added_by": (CORPUS_XML_COMMIT,),
    },
    "corpus_raw/PMID_31096736/source.json": {
        "labels": ("raw_fulltext_xml",), "in_head": True, "in_index": False,
        "added_by": (CORPUS_XML_COMMIT,),
    },
    "index/manifest.json": {
        "labels": ("ledger_manifest",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "papers/索引信息.csv": {
        "labels": ("legacy_pdf_corpus",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/focus/focus_lnm.sqlite": {
        "labels": ("production_database",), "in_head": False, "in_index": False,
        "added_by": (SNAPSHOT_COMMIT,),
    },
    "直肠癌文献爬取/focus/focus_lnm_pmc.sqlite": {
        "labels": ("production_database",), "in_head": False, "in_index": False,
        "added_by": (SNAPSHOT_COMMIT,),
    },
    "直肠癌文献爬取/pdfs_merged/summary.txt": {
        "labels": ("legacy_pdf_corpus",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/pdfs_merged/语料元数据.csv": {
        "labels": ("legacy_pdf_corpus",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/pubmed-RectalNeop-set.nbib": {
        "labels": ("bulk_metadata_export",), "in_head": True, "in_index": False,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/reports/archive_pre2000.csv": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/reports/batch810.csv": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/reports/download_report.csv": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": False,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/reports/pmc_batches.csv": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/reports/run_history.csv": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": False,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/reports/sample_1000.csv": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/reports/sample_400.csv": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/reports/sample_50.csv": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/reports/source_licenses.csv": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": True,
        "added_by": (LICENSE_REPORT_COMMIT,),
    },
    "直肠癌文献爬取/reports/summary.txt": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/reports/template_merged.csv": {
        "labels": ("runtime_reports",), "in_head": True, "in_index": True,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64.zip": {
        "labels": ("browser_driver",), "in_head": True, "in_index": False,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64/LICENSE.chromedriver": {
        "labels": ("browser_driver",), "in_head": True, "in_index": False,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64/THIRD_PARTY_NOTICES.chromedriver": {
        "labels": ("browser_driver",), "in_head": True, "in_index": False,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64/chromedriver.exe": {
        "labels": ("browser_driver",), "in_head": True, "in_index": False,
        "added_by": (ROOT_COMMIT,),
    },
    "直肠癌文献爬取/tasks.sqlite": {
        "labels": ("production_database",), "in_head": True, "in_index": False,
        "added_by": (ROOT_COMMIT,),
    },
}

# Reviewed untracking set; ignored local originals remain before and after commit.
STAGED_REMOVALS = (
    ".claude/tmp/audit.py",
    ".claude/tmp/audit2.py",
    ".claude/tmp/bench_convert.py",
    ".claude/tmp/debug_csv.py",
    ".claude/tmp/debug_prescan.py",
    ".claude/tmp/scan_pages.py",
    ".claude/tmp/page_scan.json",
    ".claude/tmp/verify_t2.py",
    ".claude/conversion-tracker.json",
    "直肠癌文献爬取/reports/archive_pre2000.csv",
    "直肠癌文献爬取/reports/batch810.csv",
    "直肠癌文献爬取/reports/pmc_batches.csv",
    "直肠癌文献爬取/reports/sample_1000.csv",
    "直肠癌文献爬取/reports/sample_400.csv",
    "直肠癌文献爬取/reports/sample_50.csv",
    "直肠癌文献爬取/reports/source_licenses.csv",
    "直肠癌文献爬取/reports/summary.txt",
    "直肠癌文献爬取/reports/template_merged.csv",
    "直肠癌文献爬取/pdfs_merged/summary.txt",
    "直肠癌文献爬取/pdfs_merged/语料元数据.csv",
    "corpus_raw/PMID_31096736/article.xml",
    "corpus_raw/PMID_31096736/source.json",
    "直肠癌文献爬取/pubmed-RectalNeop-set.nbib",
    "直肠癌文献爬取/reports/download_report.csv",
    "直肠癌文献爬取/reports/run_history.csv",
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64.zip",
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64/LICENSE.chromedriver",
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64/THIRD_PARTY_NOTICES.chromedriver",
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64/chromedriver.exe",
    "直肠癌文献爬取/tasks.sqlite",
)

# Forbidden content in the tree of the locally recorded origin/main ref.  This
# is the state an earlier fetch recorded for the remote main branch; the remote
# itself was never contacted.
RECORDED_ORIGIN_MAIN_TREE = (
    "index/manifest.json",
    "papers/索引信息.csv",
    "直肠癌文献爬取/pdfs_merged/summary.txt",
    "直肠癌文献爬取/pdfs_merged/语料元数据.csv",
    "直肠癌文献爬取/pubmed-RectalNeop-set.nbib",
    "直肠癌文献爬取/reports/archive_pre2000.csv",
    "直肠癌文献爬取/reports/batch810.csv",
    "直肠癌文献爬取/reports/download_report.csv",
    "直肠癌文献爬取/reports/pmc_batches.csv",
    "直肠癌文献爬取/reports/run_history.csv",
    "直肠癌文献爬取/reports/sample_1000.csv",
    "直肠癌文献爬取/reports/sample_400.csv",
    "直肠癌文献爬取/reports/sample_50.csv",
    "直肠癌文献爬取/reports/source_licenses.csv",
    "直肠癌文献爬取/reports/summary.txt",
    "直肠癌文献爬取/reports/template_merged.csv",
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64.zip",
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64/LICENSE.chromedriver",
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64/THIRD_PARTY_NOTICES.chromedriver",
    "直肠癌文献爬取/scripts/drivers/chromedriver-win64/chromedriver.exe",
    "直肠癌文献爬取/tasks.sqlite",
)

# The release must stay a code archive, not a packaged working directory.
RELEASE_MAX_FILES = 400
RELEASE_MAX_BYTES = 16 * 1024 * 1024


def require_git(test: unittest.TestCase) -> None:
    if not rs.git_available():
        test.skipTest("no local Git repository in this directory: "
                      "history checks need the workspace checkout")


def require_local_originals(test: unittest.TestCase, paths: tuple[str, ...]) -> None:
    if not any((ROOT / path).exists() for path in paths):
        test.skipTest("local originals are not present in this checkout: "
                      "on-disk retention cannot be verified here")


class ReleaseSetTests(unittest.TestCase):
    """The distribution set assembled from the allowlist."""

    def test_release_set_is_allowlisted_and_stays_a_code_archive(self):
        entries = rs.release_files("worktree")
        self.assertTrue(entries, "the allowlist selected no file")
        strays = [name for name in entries if not rs.matches_allowlist(name)]
        self.assertEqual(strays, [], "release set contains non-allowlisted files")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "release"
            target.mkdir()
            copied = rs.build_release_dir(target)
            size = rs.tree_size(target)
        self.assertGreater(len(copied), 50, "release set looks truncated")
        self.assertLessEqual(len(copied), RELEASE_MAX_FILES,
                             "release set grew towards a full working directory")
        self.assertLessEqual(size, RELEASE_MAX_BYTES,
                             "release set grew beyond a code archive")
        for expected in (
            "AGENTS.md",
            ".github/workflows/public-ci.yml",
            "index/build_index.py",
            "tests/run_public_tests.py",
            "tests/run_local_acceptance.py",
            "tests/suite_scope.py",
            "docs/数据合规与发布清单.md",
            "docs/发布范围-allowlist.txt",
        ):
            self.assertIn(expected, copied, f"{expected} is missing from the release set")

    def test_release_set_has_no_forbidden_path(self):
        offenders = {
            name: rs.forbidden_hits(name)
            for name in rs.release_files("worktree")
            if rs.forbidden_hits(name)
        }
        self.assertEqual(offenders, {}, f"forbidden content in the release set: {offenders}")

    def test_release_set_has_no_credential_or_private_path(self):
        self.assertEqual(rs.scan_release(ROOT, rs.release_files("worktree")), [])

    def test_only_exemptions_carry_a_pdf_or_env_file(self):
        entries = [name for name in rs.release_files("worktree")
                   if name.lower().endswith(".pdf") or Path(name).name.startswith(".env")]
        self.assertEqual(sorted(entries), sorted(rs.EXEMPT_FILES),
                         "an unexpected PDF or environment file entered the release set")
        fixture = ROOT / rs.FIXTURE_PDF
        self.assertTrue(fixture.is_file(), "the synthetic PDF fixture is missing")
        self.assertLessEqual(fixture.stat().st_size, rs.FIXTURE_PDF_MAX_BYTES,
                             "the PDF fixture is no longer a tiny synthetic file")
        self.assertEqual(fixture.read_bytes()[:5], b"%PDF-",
                         "the fixture does not look like a PDF")
        self.assertFalse(rs.is_ignored(rs.FIXTURE_PDF),
                         "the PDF fixture must be explicitly un-ignored")
        self.assertTrue(rs.is_ignored("corpus_raw/PMID_31096736/article.xml"),
                        "raw corpus full text must be ignored")
        self.assertTrue(rs.is_ignored(".env"), "real environment files must be ignored")

    def test_tracked_release_set_is_a_subset_of_the_working_tree(self):
        require_git(self)
        tracked = set(rs.release_files("tracked"))
        worktree = set(rs.release_files("worktree"))
        self.assertTrue(tracked, "no tracked file matches the allowlist")
        missing = sorted(tracked - worktree)
        self.assertEqual(missing, [], f"tracked release files are absent on disk: {missing}")
        self.assertTrue(tracked <= worktree)


class TrackedBoundaryTests(unittest.TestCase):
    """What the Git index and the ignore rules say about local-only data."""

    def test_must_never_be_tracked_categories_are_untracked(self):
        require_git(self)
        offenders = {}
        for name in rs.tracked_files():
            labels = [label for label in rs.forbidden_hits(name)
                      if label in NEVER_TRACKED_LABELS]
            if labels:
                offenders[name] = labels
        self.assertEqual(offenders, {},
                         f"local-only data is tracked in Git: {offenders}")
        tracked_excluded = {name for name in rs.tracked_files() if rs.forbidden_hits(name)}
        unreviewed = sorted(tracked_excluded - set(TRACKED_BUT_NOT_RELEASED_PATHS))
        self.assertEqual(unreviewed, [],
                         "forbidden content became tracked outside the reviewed "
                         "tracked-but-excluded set; review the allowlist")

    def test_staged_removals_keep_the_local_originals(self):
        require_git(self)
        staged_forbidden = {name for name in rs.staged_deletions()
                            if rs.forbidden_hits(name)}
        for relative in STAGED_REMOVALS:
            with self.subTest(relative=relative):
                self.assertNotIn(relative, rs.tracked_files(),
                                 "the file is still tracked in the index")
                self.assertTrue(rs.is_ignored(relative),
                                "an ignored file needs an ignore rule after untracking")
        expected_pending = set(STAGED_REMOVALS) & set(rs.head_tree())
        expected_pending = {name for name in expected_pending if rs.forbidden_hits(name)}
        self.assertEqual(sorted(staged_forbidden), sorted(expected_pending),
                         "the staged removal set of forbidden paths changed; "
                         "re-inspect the index")
        require_local_originals(self, STAGED_REMOVALS)
        for relative in STAGED_REMOVALS:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).exists(),
                                "untracking must never delete the local original")


class HistoryTests(unittest.TestCase):
    """What a history-bearing publication would carry."""

    @classmethod
    def setUpClass(cls):
        if not rs.git_available():
            raise unittest.SkipTest("no local Git repository in this directory")
        cls.entries = rs.history_entries()
        cls.added = {}
        for commit, path in rs.history_additions():
            cls.added.setdefault(path, []).append(commit)
        cls.head = set(rs.head_tree())
        cls.index = set(rs.tracked_files())

    def inspected(self) -> dict[str, dict]:
        found: dict[str, dict] = {}
        for _commit, path in self.entries:
            labels = tuple(rs.forbidden_hits(path))
            if not labels:
                continue
            found[path] = {
                "labels": labels,
                "in_head": path in self.head,
                "in_index": path in self.index,
                "added_by": tuple(self.added.get(path, ())),
            }
        return found

    def test_history_scan_covers_every_reachable_commit(self):
        require_git(self)
        commits = {commit for commit, _ in self.entries}
        listed = {name for name in rs.git("log", "--all", "--pretty=%H").splitlines() if name}
        self.assertTrue(commits <= listed, "history scan produced unknown commits")
        self.assertIn(ROOT_COMMIT, commits, "the root import commit was not inspected")
        self.assertIn(SNAPSHOT_COMMIT, commits,
                      "the local snapshot commit was not inspected")
        self.assertGreater(len(commits), 90, "the history scan skipped most commits")
        self.assertEqual(rs.tags(), [], "new tags appeared: inspect their trees too")

    def test_history_ledger_matches_the_inspected_history(self):
        require_git(self)
        found = self.inspected()
        self.assertEqual(sorted(found), sorted(HISTORY_LEDGER),
                         "the set of forbidden paths ever committed changed; "
                         "re-inspect the history and update the ledger")
        for path, expected in HISTORY_LEDGER.items():
            with self.subTest(path=path):
                self.assertEqual(found[path]["labels"], expected["labels"])
                self.assertEqual(found[path]["added_by"], expected["added_by"])

    def test_forbidden_paths_never_re_enter_head_or_index(self):
        """Cleanup may shrink the state; nothing may silently come back."""
        require_git(self)
        found = self.inspected()
        for path, expected in HISTORY_LEDGER.items():
            with self.subTest(path=path):
                if not expected["in_head"]:
                    self.assertFalse(found[path]["in_head"],
                                     "a path that left HEAD came back")
                if not expected["in_index"]:
                    self.assertFalse(found[path]["in_index"],
                                     "an untracked forbidden path was tracked again")

    def test_history_ledger_paths_still_on_disk_were_never_deleted(self):
        require_git(self)
        require_local_originals(self, tuple(HISTORY_LEDGER))
        for path, expected in HISTORY_LEDGER.items():
            with self.subTest(path=path):
                if expected["in_head"]:
                    self.assertTrue(
                        (ROOT / path).exists(),
                        "a tracked-then-untracked local original is missing on disk",
                    )

    def test_recorded_origin_main_tree_is_still_forbidden_content(self):
        """The locally recorded remote ref, not a live check of the remote."""
        require_git(self)
        refs = rs.recorded_remote_refs()
        sha = refs.get("refs/remotes/origin/main")
        if not sha:
            self.skipTest("no recorded origin/main ref in this checkout")
        tree = rs.git("ls-tree", "-r", "--name-only", sha).splitlines()
        found = sorted({name for name in tree if rs.forbidden_hits(name)})
        self.assertEqual(found, sorted(RECORDED_ORIGIN_MAIN_TREE),
                         "the locally recorded origin/main tree changed; re-inspect "
                         "before planning a history-bearing publication")


if __name__ == "__main__":
    sys.exit(unittest.main())
