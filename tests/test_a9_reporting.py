"""Offline regression tests for the A9 read-only unified report."""
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "直肠癌文献爬取" / "scripts" / "a9_reporting.py"
spec = importlib.util.spec_from_file_location("a9_reporting_test", SCRIPT)
reporting = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(reporting)


class A9ReportingTests(unittest.TestCase):
    @staticmethod
    def make_renderable_report():
        return {
            "snapshot_at": "t",
            "sqlite": {"schema_version": 2, "task_count": 0, "fetch_attempts": 0, "source_candidates": 0},
            "corpus": {"corpus_target": 0, "metadata_ready": 0, "fulltext_ready": 0, "metadata_only": 0, "retryable_error": 0, "archived": 0, "excluded": 0, "pending": 0},
            "metadata_coverage": {name: {"count": 0, "denominator": 0, "percent": 0.0} for name in ("title", "doi", "pmcid", "metadata_updated_at")},
            "source_successes": {name: {"canonical_attempts": 0, "canonical_unique_pmids": 0, "legacy_done_tasks": 0} for name in reporting.SOURCE_BUCKETS},
            "errors": {"canonical_fetch_attempts": {name: 0 for name in reporting.REQUIRED_ERROR_BUCKETS}, "legacy_last_error_heuristic": {name: 0 for name in reporting.REQUIRED_ERROR_BUCKETS}},
            "files": {"raw_fulltext": 0, "normalized_documents": 0, "converted_markdown_archive": 0, "conversion_failed": 0, "flash_failed_pdfs": 0, "legacy_pdf_corpus": {"files": 0, "valid": 0}},
            "index": {"manifest": {"file_count": 0, "total_chunks": 0}, "chroma": {"count": None}},
            "cross_checks": {"sqlite_status_sum_matches_tasks": True, "target_plus_archived_equals_tasks": True, "legacy_done_vs_pdf": {"done_tasks": 0, "legacy_pdf_files": 0, "valid_legacy_pdfs": 0, "done_without_pdf": 0, "pdf_without_done": 0}, "manifest_files_vs_converted_archive": {"manifest_files": 0, "converted_markdown_archive": 0, "equal": True}, "manifest_chunks_vs_chroma": {"manifest_chunks": 0, "chroma_count": None, "equal": None}},
            "gates": {"sqlite_authority": "PASS", "provenance_tables_actually_used": "NOT SATISFIED", "http_error_semantics": "PASS", "pmcid_metadata_refresh": "PASS", "official_resolvers_and_validation": "PASS", "fixture_isolation": "PASS"},
        }

    def make_db(self, path):
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE tasks (
                pmid INTEGER PRIMARY KEY, doi TEXT, pmc TEXT, pmcid TEXT,
                title TEXT, status TEXT, route TEXT, content_path TEXT,
                metadata_updated_at TEXT, license TEXT
            );
            CREATE TABLE fetch_attempts (
                attempt_id TEXT PRIMARY KEY, pmid INTEGER, source TEXT,
                outcome TEXT, error_class TEXT, retryable INTEGER
            );
            CREATE TABLE source_candidates (
                candidate_id INTEGER PRIMARY KEY, pmid INTEGER, source TEXT
            );
            CREATE TABLE schema_version (version INTEGER);
            INSERT INTO schema_version VALUES (2);
        """)
        conn.executemany(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (1, "10.1/one", "PMC1", "PMC1", "One", "pending", None, None, "t", "CC-BY"),
                (2, "10.1/two", None, None, "Two", "done", "europepmc:pdf", "two.pdf", None, "unverified"),
                (3, None, None, None, "Three", "failed", "legacy", None, None, "unverified"),
                (4, None, None, None, "Four", "archived", None, None, None, "unverified"),
            ],
        )
        conn.executemany(
            "INSERT INTO fetch_attempts VALUES (?,?,?,?,?,?)",
            [("a1", 1, "EuropePMC", "success", None, 0),
             ("a2", 3, "PMC_AWS", "failure", "server_error", 1)],
        )
        conn.execute("INSERT INTO source_candidates VALUES (1,1,'EuropePMC')")
        conn.commit()
        conn.close()

    def test_report_separates_canonical_and_legacy_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            self.make_db(db)
            pdf_dir = root / "pdfs"
            pdf_dir.mkdir()
            (pdf_dir / "PMID_2.pdf").write_bytes(b"%PDF-1.7\n" + b"x" * 32 + b"\n%%EOF\n")
            raw = root / "raw"
            (raw / "PMID_1").mkdir(parents=True)
            (raw / "PMID_1" / "article.xml").write_text("<article />", encoding="utf-8")
            (raw / "PMID_2").mkdir()
            (raw / "PMID_2" / "article.pdf").write_bytes(b"pdf")
            (raw / "PMID_3").mkdir()
            (raw / "PMID_3" / "article.md").write_text("# not an A5 artifact", encoding="utf-8")
            converted = root / "converted" / "已入库"
            converted.mkdir(parents=True)
            (converted / "one.md").write_text("# one", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": {"one.md": {"chunks": 1}}, "total_chunks": 1}), encoding="utf-8")
            tracker = root / "tracker.json"
            tracker.write_text(json.dumps({"converted": {"one": {}}, "failed": {"bad": {}}}), encoding="utf-8")
            result = reporting.build_report(
                db_path=db, pdf_dir=pdf_dir, raw_root=raw,
                normalized_root=root / "normalized", converted_root=root / "converted",
                tracker_path=tracker, manifest_path=manifest,
                chroma_dir=root / "missing-chroma", read_chroma=False,
                snapshot_at="2026-09-20T00:00:00+00:00",
            )
            self.assertEqual(result["corpus"]["corpus_target"], 3)
            self.assertEqual(result["corpus"]["metadata_only"], 1)
            self.assertEqual(result["corpus"]["retryable_error"], 1)
            self.assertEqual(result["corpus"]["archived"], 1)
            self.assertEqual(result["metadata_coverage"]["doi"]["count"], 2)
            self.assertEqual(result["source_successes"]["Europe PMC"]["canonical_attempts"], 1)
            self.assertEqual(result["source_successes"]["Europe PMC"]["legacy_done_tasks"], 1)
            self.assertEqual(result["errors"]["canonical_fetch_attempts"]["server_error"], 1)
            self.assertEqual(result["files"]["converted_markdown_archive"], 1)
            self.assertEqual(result["files"]["raw_fulltext"], 2)
            self.assertEqual(result["files"]["conversion_failed"], 1)

    def test_markdown_contains_gate_and_cross_check_sections(self):
        report = self.make_renderable_report()
        text = reporting.render_markdown(report)
        self.assertIn("- SQLite: not provided", text)
        self.assertIn("## Acquisition Gate", text)
        self.assertIn("## 交叉核对", text)
        self.assertIn("NOT SATISFIED", text)

    def test_markdown_provenance_pass_uses_current_production_evidence(self):
        report = self.make_renderable_report()
        report["sqlite"]["fetch_attempts"] = 1
        report["sqlite"]["source_candidates"] = 1
        report["gates"]["provenance_tables_actually_used"] = "PASS"
        text = reporting.render_markdown(report)
        self.assertIn("两表均有真实记录", text)
        self.assertNotIn("生产 SQLite 尚未产生 A1–A5 provenance rows", text)

    def test_markdown_empty_data_sources_uses_defaults(self):
        report = self.make_renderable_report()
        report["data_sources"] = {}
        text = reporting.render_markdown(report)
        self.assertIn("- SQLite: not provided", text)
        self.assertIn("- Canonical raw fulltext: not provided", text)
        self.assertIn("- Chroma: not provided", text)

    def test_cli_defaults_use_project_root_canonical_paths(self):
        args = reporting._build_parser().parse_args(["--skip-chroma"])
        self.assertEqual(Path(args.raw_root), ROOT / "corpus_raw")
        self.assertEqual(Path(args.normalized_root), ROOT / "normalized")
        self.assertNotEqual(Path(args.raw_root), ROOT / "直肠癌文献爬取" / "corpus_raw")
        self.assertNotEqual(Path(args.normalized_root), ROOT / "直肠癌文献爬取" / "normalized")


if __name__ == "__main__":
    unittest.main()
