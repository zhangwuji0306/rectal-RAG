# -*- coding: utf-8 -*-
"""Offline regression tests for the A-R4 reporting truth requirements.

The report must separate scheduling status from content readiness, compute its
gates instead of hardcoding PASS, report per-format counts and provider states,
and distinguish offline unit tests, one real single-article run and a
full-corpus rollout.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "直肠癌文献爬取" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import a1_schema as schema  # noqa: E402
import content_validation as validation  # noqa: E402
import ingest_pmids as ingest  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "a9_reporting_ar4", SCRIPTS / "a9_reporting.py"
)
reporting = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reporting)


class AR4ReportingTruthTests(unittest.TestCase):
    """Build a synthetic canonical snapshot with deliberately mixed states."""

    def make_db(self, path: Path) -> None:
        schema.migrate_to_v2(path)
        validation.ensure_validation_schema(path)
        ingest.ingest_pmids(
            [1101, 1102, 1103, 1104],
            source="ar4-test",
            batch_id="ar4-test-batch",
            db_path=path,
        )
        conn = sqlite3.connect(path)
        conn.executemany(
            "UPDATE tasks SET title=?, title_norm=?, year=2024, status=? WHERE pmid=?",
            [
                ("Bound fulltext ready", "bound fulltext ready", "fulltext_ready", 1101),
                ("Metadata only with body", "metadata only with body", "metadata_only", 1102),
                ("Stale verdict", "stale verdict", "metadata_only", 1103),
                ("Pending task", "pending task", "pending", 1104),
            ],
        )
        # 1101: verdict bound to the current content version.
        conn.execute(
            """UPDATE tasks SET content_status='fulltext_ready', content_format='xml',
               content_path=?, content_sha256='sha-a', validated_content_path=?,
               validated_content_sha256='sha-a', file_valid=1, bibliographic_match=1,
               metadata_updated_at='2026-09-20T00:00:00+00:00' WHERE pmid=1101""",
            ("PMID_1101/article.xml", "PMID_1101/article.xml"),
        )
        # 1102: the A9 review's case - validated body while scheduling says
        # metadata_only; it must not be reported as "no usable body".
        conn.execute(
            """UPDATE tasks SET content_status='fulltext_ready', content_format='xml',
               content_path='PMID_1102/article.xml', content_sha256='sha-b',
               validated_content_path='PMID_1102/article.xml',
               validated_content_sha256='sha-b', file_valid=1, bibliographic_match=1
               WHERE pmid=1102"""
        )
        # 1103: a verdict left behind by an older content version.
        conn.execute(
            """UPDATE tasks SET content_status='fulltext_ready', content_format='xml',
               content_path='PMID_1103/article.xml', content_sha256='sha-c-new',
               validated_content_path='PMID_1103/article.xml',
               validated_content_sha256='sha-c-old', file_valid=1,
               bibliographic_match=1 WHERE pmid=1103"""
        )
        # 1104: a fetched PDF artifact without any verdict.
        conn.execute(
            """UPDATE tasks SET content_status='fetched', content_format='pdf',
               content_path='PMID_1104/article.pdf', content_sha256='sha-d'
               WHERE pmid=1104"""
        )
        conn.commit()
        conn.close()

    def make_workspace(self, root: Path) -> dict:
        db = root / "tasks.sqlite"
        self.make_db(db)
        raw = root / "corpus_raw"
        for pmid in (1101, 1102, 1103):
            directory = raw / f"PMID_{pmid}"
            directory.mkdir(parents=True)
            (directory / "article.xml").write_text("<article />", encoding="utf-8")
        (raw / "PMID_1104").mkdir()
        (raw / "PMID_1104" / "article.pdf").write_bytes(b"%PDF-1.7")
        normalized = root / "normalized"
        normalized.mkdir()
        for pmid in (1102, 1103):
            (normalized / f"PMID_{pmid}.md").write_text(
                f'---\npmid: {pmid}\ndoi: "10.1000/ar4.{pmid}"\nsource: "PMC_AWS"\n'
                'license: "CC BY 4.0"\ncontent_sha256: "sha"\n---\n\n# article\n',
                encoding="utf-8",
            )
        converted = root / "converted"
        (converted / "已入库").mkdir(parents=True)
        (converted / "已入库" / "one.md").write_text("# one", encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps({"files": {"PMID_1102.md": {"chunks": 3}}, "total_chunks": 3}),
            encoding="utf-8",
        )
        tracker = root / "tracker.json"
        tracker.write_text(json.dumps({"converted": {}, "failed": {}}), encoding="utf-8")
        pdf_dir = root / "pdfs_merged"
        pdf_dir.mkdir()
        (pdf_dir / "PMID_1101.pdf").write_bytes(b"%PDF-1.7\n" + b"x" * 64 + b"\n%%EOF\n")
        return {
            "db_path": db,
            "pdf_dir": pdf_dir,
            "raw_root": raw,
            "normalized_root": normalized,
            "converted_root": converted,
            "tracker_path": tracker,
            "manifest_path": manifest,
            "chroma_dir": root / "missing-chroma",
        }

    def build(self, paths: dict) -> dict:
        return reporting.build_report(
            **paths,
            read_chroma=False,
            snapshot_at="2026-09-22T00:00:00+00:00",
        )

    def test_fulltext_ready_is_reported_as_two_separate_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            report = self.build(paths)
            # Scheduling status stays the tasks.status count.
            self.assertEqual(report["corpus"]["fulltext_ready"], 1)
            readiness = report["readiness"]
            self.assertEqual(
                readiness["content_readiness"]["validated_body_bound_to_current_content"],
                2,
            )
            self.assertEqual(
                readiness["content_readiness"]["verdict_without_content_binding"], 1
            )
            self.assertEqual(
                readiness["content_readiness"]["verdict_without_content_binding_pmids"],
                [1103],
            )
            self.assertEqual(readiness["metadata_only"]["total"], 2)
            self.assertEqual(readiness["metadata_only"]["with_validated_body"], 1)
            self.assertEqual(readiness["metadata_only"]["without_validated_body"], 1)
            self.assertIn("不相加", readiness["dimension_note"])
            self.assertEqual(
                readiness["normalized_readiness"]["indexed_normalized"], "NOT CHECKED"
            )
            self.assertEqual(readiness["indexed_readiness"]["manifest_pmid_named_entries"], 1)
            self.assertEqual(readiness["indexed_readiness"]["bound_pmids_present_in_manifest"], 1)

    def test_per_format_pipeline_counts_fetched_validated_normalized_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            report = self.build(paths)
            pipeline = report["format_pipeline"]
            self.assertEqual(pipeline["xml"]["fetched"], 3)
            self.assertEqual(pipeline["xml"]["validated"], 2)
            self.assertEqual(pipeline["xml"]["normalized"], 2)
            self.assertEqual(pipeline["xml"]["indexed"], 1)
            self.assertEqual(pipeline["pdf"]["fetched"], 1)
            self.assertEqual(pipeline["pdf"]["validated"], 0)
            self.assertEqual(pipeline["txt"]["fetched"], 0)

    def test_legacy_pdf_corpus_is_not_reparsed_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            with mock.patch.object(
                reporting, "_pdf_inventory", side_effect=AssertionError("must not re-parse")
            ):
                report = self.build(paths)
            self.assertIsNone(report["files"]["legacy_pdf_corpus"]["valid"])
            self.assertFalse(report["files"]["legacy_pdf_corpus"]["scanned"])
            self.assertFalse(
                report["cross_checks"]["legacy_done_vs_pdf"]["content_scanned"]
            )
            self.assertEqual(report["files"]["legacy_pdf_corpus"]["files"], 1)

    def test_gates_are_computed_results_with_scope_and_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            empty = self.build(paths)
            self.assertEqual(empty["gates"]["provenance_tables_actually_used"], "NOT SATISFIED")
            self.assertEqual(empty["gates"]["content_readiness_evidence"], "PASS")
            self.assertEqual(empty["gates"]["pmc_aws_inventory"], "NOT SATISFIED")

            conn = sqlite3.connect(paths["db_path"])
            conn.execute(
                "INSERT INTO fetch_attempts (attempt_id, pmid, source, started_at, "
                "outcome, error_class) VALUES (?,?,?,?,?,?)",
                ("ar4-attempt", 1101, "EuropePMC", "2026-09-21T00:00:00+00:00", "failure", "timeout"),
            )
            conn.execute(
                "INSERT INTO source_candidates (pmid, source, url, format) VALUES (?,?,?,?)",
                (1101, "EuropePMC", "https://europe.example/x.xml", "xml"),
            )
            conn.commit()
            conn.close()
            populated = self.build(paths)
            self.assertEqual(populated["gates"]["provenance_tables_actually_used"], "PASS")
            for name, value in populated["gates"].items():
                with self.subTest(gate=name):
                    self.assertIn(value, reporting.GATE_RESULTS)
                    detail = populated["gates_detail"][name]
                    self.assertTrue(detail["scope"])
                    self.assertTrue(detail["evidence"])
                    self.assertTrue(detail["checked"])

    def test_gates_change_with_the_evidence_instead_of_staying_constant(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            first = self.build(paths)
            conn = sqlite3.connect(paths["db_path"])
            conn.execute("UPDATE tasks SET content_sha256='sha-x' WHERE pmid IN (1101, 1102)")
            conn.commit()
            conn.close()
            second = self.build(paths)
            self.assertEqual(first["gates"]["content_readiness_evidence"], "PASS")
            self.assertEqual(second["gates"]["content_readiness_evidence"], "NOT SATISFIED")
            self.assertEqual(second["readiness"]["content_readiness"]["validated_body_bound_to_current_content"], 0)
            self.assertEqual(second["readiness"]["content_readiness"]["verdict_without_content_binding"], 3)
            self.assertNotEqual(first["gates"], second["gates"])

    def test_source_states_distinguish_not_configured_failed_and_no_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            conn = sqlite3.connect(paths["db_path"])
            conn.execute(
                "INSERT INTO source_candidates (pmid, source, url, format) VALUES (?,?,?,?)",
                (1101, "EuropePMC", "https://europe.example/x.xml", "xml"),
            )
            conn.execute(
                "INSERT INTO fetch_attempts (attempt_id, pmid, source, started_at, "
                "outcome, error_class) VALUES (?,?,?,?,?,?)",
                ("ar4-unpaywall", 1102, "Unpaywall", "2026-09-21T00:00:00+00:00", "failure", "timeout"),
            )
            conn.commit()
            conn.close()
            report = self.build(paths)
            queries = report["source_queries"]
            self.assertEqual(queries["Europe PMC"]["outcome"], "已产生候选")
            self.assertEqual(queries["Unpaywall"]["outcome"], "调用失败")
            self.assertEqual(queries["PMC AWS"]["outcome"], "配置缺失")
            self.assertEqual(queries["Other"]["outcome"], "未查询")
            self.assertFalse(report["provider_configuration"]["pmc_aws_inventory_configured"])

    def test_a2_refresh_coverage_is_reported_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            report = self.build(paths)
            refresh = report["a2_refresh"]
            self.assertEqual(refresh["metadata_updated_at"]["count"], 1)
            self.assertEqual(refresh["metadata_updated_at"]["denominator"], 4)
            self.assertIn("不代表全库已刷新", refresh["note"])

    def test_verification_scope_separates_tests_single_article_and_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            report = self.build(paths)
            scopes = report["verification_scope"]
            self.assertEqual(set(scopes), set(reporting.VERIFICATION_SCOPES))
            self.assertEqual(scopes["offline_unit_tests"]["result"], "NOT CHECKED")
            self.assertEqual(scopes["single_article_real_run"]["result"], "OBSERVED")
            self.assertEqual(scopes["single_article_real_run"]["observed"], 2)
            self.assertEqual(scopes["full_corpus_rollout"]["result"], "PARTIAL")
            for scope in scopes.values():
                self.assertTrue(scope["evidence"])

    def test_render_markdown_exposes_the_new_dimensions_and_no_hardcoded_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            text = reporting.render_markdown(self.build(paths))
            self.assertIn("准备度维度", text)
            self.assertIn("调度状态 vs 正文就绪", text)
            self.assertIn("格式维度统计", text)
            self.assertIn("来源查询状态", text)
            self.assertIn("验证范围", text)
            self.assertIn("NOT CHECKED", text)
            self.assertIn("中有可用已验证正文", text)
            self.assertIn("未配置本地 inventory", text)
            self.assertIn("未解析内容", text)
            # Every gate row carries scope and evidence columns.
            gate_rows = [line for line in text.splitlines() if line.startswith("| ") and "| Gate |" not in line]
            self.assertTrue(any("适用范围" in line for line in text.splitlines()))
            self.assertTrue(gate_rows)

    def test_index_mismatch_is_reported_as_fail_and_unavailable_as_not_checked(self):
        manifest = {"file_count": 2, "total_chunks": 10}
        self.assertEqual(
            reporting._index_gate(manifest, {"count": 10, "error": None}), "PASS"
        )
        self.assertEqual(
            reporting._index_gate(manifest, {"count": 3, "error": None}), "FAIL"
        )
        self.assertEqual(
            reporting._index_gate(manifest, {"count": None, "error": "directory_missing"}),
            "NOT CHECKED",
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            report = reporting.build_report(
                **paths, read_chroma=False, snapshot_at="2026-09-22T00:00:00+00:00"
            )
            # Chroma was not read, so the report must not claim the index agrees.
            self.assertEqual(report["gates"]["index_consistency"], "NOT CHECKED")
            text = reporting.render_markdown(report)
            self.assertIn("索引账实一致（manifest vs Chroma） | NOT CHECKED", text)

    def test_rollout_coverage_uses_the_corpus_target_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.make_workspace(Path(tmp))
            conn = sqlite3.connect(paths["db_path"])
            conn.execute(
                """INSERT INTO tasks (pmid, title, year, status, metadata_updated_at)
                   VALUES (1105, 'Archived out-of-scope task', 1998, 'archived',
                           '2026-09-21T00:00:00+00:00')"""
            )
            conn.commit()
            conn.close()
            report = self.build(paths)
            target = report["corpus"]["corpus_target"]
            self.assertEqual(target, 4)
            self.assertEqual(report["sqlite"]["task_count"], 5)
            # The archived task is outside the corpus target: no coverage figure
            # in this report may use the 5-task total as its denominator.
            self.assertEqual(
                report["metadata_coverage"]["metadata_updated_at"]["denominator"], target
            )
            self.assertEqual(
                report["a2_refresh"]["metadata_updated_at"]["denominator"], target
            )
            rollout = report["verification_scope"]["full_corpus_rollout"]["metadata_updated_at"]
            self.assertEqual(rollout["denominator"], target)
            self.assertEqual(rollout["count"], 1)
            self.assertIn("分母=corpus target", report["verification_scope"]["full_corpus_rollout"]["evidence"])

    def test_cli_writes_a_json_report_of_the_same_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = self.make_workspace(root)
            md_path = root / "report.md"
            json_path = root / "report.json"
            exit_code = reporting.main(
                [
                    "--db", str(paths["db_path"]),
                    "--pdf-dir", str(paths["pdf_dir"]),
                    "--raw-root", str(paths["raw_root"]),
                    "--normalized-root", str(paths["normalized_root"]),
                    "--converted-root", str(paths["converted_root"]),
                    "--tracker", str(paths["tracker_path"]),
                    "--manifest", str(paths["manifest_path"]),
                    "--chroma-dir", str(paths["chroma_dir"]),
                    "--skip-chroma",
                    "--snapshot-at", "2026-09-22T06:00:00+00:00",
                    "--out", str(md_path),
                    "--json-out", str(json_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["snapshot_at"], "2026-09-22T06:00:00+00:00")
            self.assertEqual(payload["sqlite"]["task_count"], 4)
            self.assertEqual(
                payload["readiness"]["content_readiness"][
                    "validated_body_bound_to_current_content"
                ],
                2,
            )
            self.assertEqual(payload["gates"]["index_consistency"], "NOT CHECKED")
            self.assertEqual(
                set(payload), set(reporting.build_report(
                    **paths, read_chroma=False, snapshot_at="2026-09-22T06:00:00+00:00"
                ))
            )
            self.assertIn("准备度维度", md_path.read_text(encoding="utf-8"))

    def test_report_still_renders_a_minimal_hand_built_payload(self):
        report = {
            "snapshot_at": "t",
            "sqlite": {"schema_version": 2, "task_count": 0, "fetch_attempts": 0, "source_candidates": 0},
            "corpus": {"corpus_target": 0, "metadata_ready": 0, "fulltext_ready": 0, "metadata_only": 0, "retryable_error": 0, "archived": 0, "excluded": 0, "pending": 0},
            "metadata_coverage": {name: {"count": 0, "denominator": 0, "percent": 0.0} for name in ("title", "doi", "pmcid", "metadata_updated_at")},
            "source_successes": {name: {"canonical_attempts": 0, "canonical_unique_pmids": 0, "legacy_done_tasks": 0} for name in reporting.SOURCE_BUCKETS},
            "errors": {"canonical_fetch_attempts": {name: 0 for name in reporting.REQUIRED_ERROR_BUCKETS}, "legacy_last_error_heuristic": {name: 0 for name in reporting.REQUIRED_ERROR_BUCKETS}},
            "files": {"raw_fulltext": 0, "normalized_documents": 0, "converted_markdown_archive": 0, "conversion_failed": 0, "flash_failed_pdfs": 0, "legacy_pdf_corpus": {"files": 0, "valid": None}},
            "index": {"manifest": {"file_count": 0, "total_chunks": 0}, "chroma": {"count": None}},
            "cross_checks": {"sqlite_status_sum_matches_tasks": True, "target_plus_archived_equals_tasks": True, "legacy_done_vs_pdf": {"done_tasks": 0, "legacy_pdf_files": 0, "valid_legacy_pdfs": None, "done_without_pdf": 0, "pdf_without_done": 0}, "manifest_files_vs_converted_archive": {"manifest_files": 0, "converted_markdown_archive": 0, "equal": True}, "manifest_chunks_vs_chroma": {"manifest_chunks": 0, "chroma_count": None, "equal": None}},
            "gates": {},
        }
        text = reporting.render_markdown(report)
        self.assertIn("NOT CHECKED", text)
        self.assertIn("## Acquisition Gate", text)


if __name__ == "__main__":
    unittest.main()
