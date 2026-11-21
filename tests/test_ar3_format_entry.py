# -*- coding: utf-8 -*-
"""Offline regression tests for A-R3: one format-consistent A5→A6→A7 entry.

Covers the candidate loop (a wrong first candidate falls back to the correct
one, unsupported formats are skipped instead of fetched), the explicit support
matrix, and the normalized-ingestion adapter that deduplicates against the
existing converted corpus without touching the agreed B3 source list.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "直肠癌文献爬取" / "scripts"
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(SCRIPTS))

import a1_schema as schema  # noqa: E402
import content_validation as validation  # noqa: E402
import fetch_fulltext as fetch  # noqa: E402
import fulltext_client as client_module  # noqa: E402
import ingest_pmids as ingest  # noqa: E402
import normalized_ingest  # noqa: E402
import oa_resolver as oa  # noqa: E402

OK_XML = (FIXTURES / "a5_article_ok.xml").read_bytes()
WRONG_XML = (FIXTURES / "a6_article_wrong_identity.xml").read_bytes()
B3_SOURCES = ROOT / "index" / "b3-sources.json"


class AR3FormatEntryTests(unittest.TestCase):
    pmid = 5001

    def make_db(self, path: Path) -> None:
        schema.migrate_to_v2(path)
        validation.ensure_validation_schema(path)
        ingest.ingest_pmids(
            [self.pmid], source="ar3-test", batch_id="ar3-test-batch", db_path=path
        )
        conn = sqlite3.connect(path)
        conn.execute(
            """UPDATE tasks SET pmcid='PMC5001', doi='10.1000/a5',
               title='A5 full-text test article', year=2021, authors='Walker, Anna K',
               first_author_family='walker', status='oa_resolved' WHERE pmid=?""",
            (self.pmid,),
        )
        conn.commit()
        conn.close()

    def task(self, path: Path) -> dict:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            return dict(
                conn.execute("SELECT * FROM tasks WHERE pmid=?", (self.pmid,)).fetchone()
            )
        finally:
            conn.close()

    def attempts(self, path: Path) -> list[dict]:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute("SELECT * FROM fetch_attempts")]
        finally:
            conn.close()

    @staticmethod
    def response(body: bytes):
        return client_module.HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/octet-stream"},
            body=body,
            url="https://mock.example/response",
        )

    @staticmethod
    def sequence_transport(items):
        pending = list(items)
        calls: list[str] = []

        def transport(url, **_kwargs):
            calls.append(url)
            return pending.pop(0)

        return transport, calls

    def make_client(self, db: Path, transport):
        return client_module.UnifiedHttpClient(
            settings=client_module.RetrySettings(max_attempts=1),
            transport=transport,
            sleeper=lambda _seconds: None,
            random_value=lambda: 0.0,
            db_path=db,
        )

    def acquire(self, db: Path, raw_root: Path, client, candidates):
        return fetch.acquire_one(
            self.pmid,
            db_path=db,
            raw_root=raw_root,
            candidates=candidates,
            client=client,
        )

    # --------------------------------------------------------- candidate loop

    def test_first_candidate_fails_validation_and_the_second_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            candidates = [
                {
                    "source": oa.PMC_AWS, "format": "xml", "pmcid": "PMC5001",
                    "license": "CC BY 4.0", "url": "https://aws.example/PMC5001.xml",
                },
                {
                    "source": oa.EUROPE_PMC, "format": "xml", "pmcid": "PMC5001",
                    "license": "CC BY 4.0",
                    "url": "https://europe.example/PMC5001/fullTextXML",
                },
            ]
            transport, calls = self.sequence_transport(
                [self.response(WRONG_XML), self.response(OK_XML)]
            )
            result = self.acquire(db, raw_root, self.make_client(db, transport), candidates)
            self.assertEqual(result["outcome"], "raw_fetched")
            self.assertEqual(result["source"], oa.EUROPE_PMC)
            self.assertEqual(len(calls), 2)
            # The first candidate's real failure evidence is kept.
            first_failure = result["failures"][0]
            self.assertEqual(first_failure["stage"], "content_validation")
            self.assertEqual(first_failure["source"], oa.PMC_AWS)
            self.assertEqual(first_failure["error_class"], "identifier_mismatch")
            self.assertEqual(first_failure["file_valid"], 1)
            self.assertEqual(first_failure["bibliographic_match"], 0)
            # One attempt row per real HTTP request, no fabricated attempts.
            self.assertEqual(len(self.attempts(db)), 2)
            row = self.task(db)
            self.assertEqual(row["file_valid"], 1)
            self.assertEqual(row["bibliographic_match"], 1)
            self.assertEqual(row["content_status"], "fulltext_ready")
            self.assertEqual(
                row["validated_content_sha256"],
                hashlib.sha256(OK_XML).hexdigest(),
            )

    def test_unsupported_candidates_are_skipped_and_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            candidates = [
                {
                    "source": oa.PMC_AWS, "format": "txt", "pmcid": "PMC5001",
                    "license": "CC BY 4.0", "url": "https://aws.example/PMC5001.txt",
                },
                {
                    "source": oa.PUBLISHER, "format": "xml", "pmcid": "PMC5001",
                    "license": "CC BY 4.0", "url": "https://publisher.example/a.xml",
                },
            ]
            transport, calls = self.sequence_transport([self.response(OK_XML)])
            result = self.acquire(db, raw_root, self.make_client(db, transport), candidates)
            self.assertEqual(result["outcome"], "raw_fetched")
            self.assertEqual(result["source"], oa.PUBLISHER)
            self.assertEqual(calls, ["https://publisher.example/a.xml"])
            self.assertEqual(result["failures"][0]["stage"], "format_gate")
            self.assertEqual(result["failures"][0]["attempts"], 0)
            self.assertEqual(len(self.attempts(db)), 1)

    def test_only_unsupported_formats_report_no_ingestable_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            candidates = [
                {
                    "source": oa.PMC_AWS, "format": "txt", "pmcid": "PMC5001",
                    "license": "CC BY 4.0", "url": "https://aws.example/PMC5001.txt",
                },
                {
                    "source": oa.PUBLISHER, "format": "html", "pmcid": "PMC5001",
                    "license": "CC BY 4.0", "url": "https://publisher.example/a.html",
                },
            ]
            transport, calls = self.sequence_transport([])
            result = self.acquire(db, raw_root, self.make_client(db, transport), candidates)
            self.assertEqual(result["outcome"], "no_ingestable_candidate")
            self.assertEqual(result["attempts"], 0)
            self.assertEqual(calls, [])
            self.assertEqual(self.attempts(db), [])
            self.assertEqual(
                [failure["stage"] for failure in result["failures"]],
                ["format_gate", "format_gate"],
            )
            self.assertIsNone(self.task(db)["file_valid"])
            self.assertFalse((raw_root / f"PMID_{self.pmid}").exists())

    def test_http_200_with_invalid_content_is_not_an_ingestable_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            candidates = [
                {
                    "source": oa.PMC_AWS, "format": "xml", "pmcid": "PMC5001",
                    "license": "CC BY 4.0", "url": "https://aws.example/PMC5001.xml",
                }
            ]
            transport, _calls = self.sequence_transport(
                [self.response(b"<article>bogus</article>")]
            )
            result = self.acquire(db, raw_root, self.make_client(db, transport), candidates)
            self.assertEqual(result["outcome"], "fetched_unvalidated")
            self.assertTrue(result["raw_written"])
            self.assertEqual(result["failures"][0]["stage"], "content_validation")
            row = self.task(db)
            self.assertEqual(row["content_status"], "fetched")
            self.assertEqual(row["file_valid"], 0)
            self.assertNotEqual(row["status"], "fulltext_ready")
            self.assertTrue(
                Path(result["failures"][0]["artifact_path"]).is_file()
            )


class AR3NormalizedIngestTests(unittest.TestCase):
    """The adapter plans, deduplicates and stages without overwriting."""

    @staticmethod
    def write_normalized(path: Path, pmid: str, doi: str | None = "10.1000/x"):
        lines = ["---", f"pmid: {pmid}"]
        if doi:
            lines.append(f'doi: "{doi}"')
        lines.append('source: "PMC_AWS"')
        lines.append('license: "CC BY 4.0"')
        lines.append('content_sha256: "sha"')
        lines += ["---", "", f"# Article {pmid}", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_normalized_versus_existing_converted_for_the_same_pmid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = root / "normalized"
            normalized.mkdir()
            self.write_normalized(normalized / "PMID_7001.md", "7001", "10.1000/one")
            archived = root / "converted" / "已入库"
            archived.mkdir(parents=True)
            existing = archived / "PMID_7001.md"
            existing.write_text("# existing converted body\n", encoding="utf-8")
            plan = normalized_ingest.plan_ingestion(
                normalized, converted_roots=[root / "converted", archived],
                out_root=root / "staging",
            )
            entry = plan["entries"][0]
            self.assertEqual(entry.action, "skip_existing_converted")
            self.assertEqual(entry.preferred_source, "converted")
            self.assertEqual(entry.preferred_path, str(existing))
            self.assertEqual(entry.existing_converted, [str(existing)])
            self.assertEqual(plan["summary"]["stage"], 0)
            applied = normalized_ingest.apply_plan(plan)
            self.assertEqual(applied["summary"]["staged"], 0)
            self.assertFalse((root / "staging").exists())
            self.assertEqual(existing.read_text(encoding="utf-8"), "# existing converted body\n")

    def test_missing_metadata_is_reported_without_blocking_pmid_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = root / "normalized"
            normalized.mkdir()
            (normalized / "untitled.md").write_text("# no front matter\n", encoding="utf-8")
            (normalized / "PMID_7002.md").write_text(
                '---\npmid: 7002\n---\n\n# Article 7002\n', encoding="utf-8"
            )
            plan = normalized_ingest.plan_ingestion(
                normalized, converted_roots=[root / "converted"], out_root=root / "staging"
            )
            by_pmid = {entry.pmid: entry for entry in plan["entries"]}
            self.assertEqual(by_pmid[None].action, "missing_identity")
            self.assertIn("PMID/DOI", by_pmid[None].reason)
            # A-R3: without a canonical task database nothing can be verified,
            # so the document stays a candidate and carries no ingestable
            # meaning even though it is copied to staging for inspection.
            self.assertEqual(by_pmid["7002"].action, "stage_candidate")
            self.assertFalse(by_pmid["7002"].ingestable)
            self.assertEqual(by_pmid["7002"].verification, "no_database")
            self.assertEqual(
                tuple(by_pmid["7002"].missing_front_matter), ("doi", "source", "license")
            )

    def test_repeated_runs_are_idempotent_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = root / "normalized"
            normalized.mkdir()
            self.write_normalized(normalized / "PMID_7003.md", "7003")
            staging = root / "staging"
            plan = normalized_ingest.plan_ingestion(
                normalized, converted_roots=[root / "converted"], out_root=staging
            )
            first = normalized_ingest.apply_plan(plan)
            self.assertEqual(first["summary"]["staged"], 1)
            target = staging / "PMID_7003.md"
            self.assertTrue(target.is_file())
            published = target.read_bytes()

            second = normalized_ingest.apply_plan(plan)
            self.assertEqual(second["summary"]["staged"], 0)
            self.assertEqual(second["summary"]["skipped_unchanged"], 1)
            self.assertEqual(target.read_bytes(), published)

            target.write_text("# operator edit that must not be clobbered\n", encoding="utf-8")
            third = normalized_ingest.apply_plan(plan)
            self.assertEqual(third["summary"]["conflicts"], 1)
            self.assertEqual(
                target.read_text(encoding="utf-8"),
                "# operator edit that must not be clobbered\n",
            )

    def test_legacy_mapping_matches_converted_names_and_prefers_the_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = root / "normalized"
            normalized.mkdir()
            self.write_normalized(normalized / "PMID_7004.md", "7004")
            converted = root / "converted"
            archived = converted / "已入库"
            archived.mkdir(parents=True)
            legacy = converted / "Smith 2020 - A legacy name.md"
            legacy.write_text("# legacy\n", encoding="utf-8")
            archived_named = archived / "PMID_7004.md"
            archived_named.write_text("# archived PMID named\n", encoding="utf-8")
            mapping = root / "mapping_pmid.json"
            mapping.write_text(
                json.dumps({"files": [{"filename": legacy.name, "pmid": "7004", "doi": "10.1000/x"}]}),
                encoding="utf-8",
            )
            plan = normalized_ingest.plan_ingestion(
                normalized,
                converted_roots=[converted, archived],
                mapping_paths=[mapping],
                out_root=root / "staging",
            )
            entry = plan["entries"][0]
            self.assertEqual(entry.action, "skip_existing_converted")
            self.assertEqual(entry.preferred_path, str(archived_named))
            self.assertEqual(
                sorted(entry.existing_converted), sorted([str(legacy), str(archived_named)])
            )

    def test_staged_copy_is_the_preferred_source_when_no_converted_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            normalized = root / "normalized"
            normalized.mkdir()
            source = self.write_normalized(normalized / "PMID_7005.md", "7005")
            plan = normalized_ingest.plan_ingestion(
                normalized, converted_roots=[root / "converted"], out_root=root / "staging"
            )
            entry = plan["entries"][0]
            self.assertEqual(entry.preferred_source, "normalized")
            self.assertEqual(entry.preferred_path, str(source))
            normalized_ingest.apply_plan(plan)
            self.assertEqual(
                (root / "staging" / "PMID_7005.md").read_bytes(), source.read_bytes()
            )

    def test_adapter_reports_the_a6_binding_and_leaves_b3_sources_alone(self):
        before = B3_SOURCES.read_bytes() if B3_SOURCES.is_file() else None
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            schema.migrate_to_v2(db)
            validation.ensure_validation_schema(db)
            ingest.ingest_pmids(
                [7006], source="ar3-adapter", batch_id="ar3-adapter-batch", db_path=db
            )
            conn = sqlite3.connect(db)
            conn.execute(
                """UPDATE tasks SET content_status='fulltext_ready', file_valid=1,
                   bibliographic_match=1, content_path='PMID_7006/article.xml',
                   content_sha256='sha-6', validated_content_path='PMID_7006/article.xml',
                   validated_content_sha256='sha-6' WHERE pmid=7006"""
            )
            conn.commit()
            conn.close()
            normalized = root / "normalized"
            normalized.mkdir()
            self.write_normalized(normalized / "PMID_7006.md", "7006")
            plan = normalized_ingest.plan_ingestion(
                normalized,
                converted_roots=[root / "converted"],
                out_root=root / "staging",
                db_path=db,
            )
            self.assertEqual(plan["entries"][0].identity["binding"], "bound")
            normalized_ingest.apply_plan(plan)
        after = B3_SOURCES.read_bytes() if B3_SOURCES.is_file() else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
