# -*- coding: utf-8 -*-
"""Offline regression tests for A-R1: a verdict must bind the validated content.

The scenarios come from the review's offline reproduction: replacing the raw
file after A6 passed must not let A7 emit a normalized document, a failed
normalization must never overwrite an existing one, and the verdict commit must
be refused when the task pointer moved on while the bytes were being read.
"""

from __future__ import annotations

import hashlib
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
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

import a1_schema as schema  # noqa: E402
import content_validation as validation  # noqa: E402
import fetch_fulltext as fetch  # noqa: E402
import fulltext_client as client_module  # noqa: E402
import ingest_pmids as ingest  # noqa: E402
import normalization  # noqa: E402
import oa_resolver as oa  # noqa: E402

OK_XML = (FIXTURES / "a6_article_ok.xml").read_bytes()
WRONG_XML = (FIXTURES / "a6_article_wrong_identity.xml").read_bytes()
PDF_FIXTURE = (FIXTURES / "a7_article.pdf").read_bytes()


class AR1ContentBindingTests(unittest.TestCase):
    pmid = 6001
    title = "Neoadjuvant chemoradiotherapy for locally advanced rectal cancer"
    doi = "10.1000/a6.fixture"
    pmcid = "PMC6001"
    authors = "Smith, Jane Q; Jones, Robert"

    # ---------------------------------------------------------------- helpers

    def make_db(self, path: Path) -> None:
        schema.migrate_to_v2(path)
        validation.ensure_validation_schema(path)
        ingest.ingest_pmids(
            [self.pmid], source="ar1-test", batch_id="ar1-test-batch", db_path=path
        )
        conn = sqlite3.connect(path)
        conn.execute(
            """UPDATE tasks SET doi=?, pmcid=?, title=?, year=2020,
               journal='Journal of A6 Synthetic Fixtures', authors=?,
               first_author='Smith, Jane Q', first_author_family='smith',
               status='fetching', content_status='fetched' WHERE pmid=?""",
            (self.doi, self.pmcid, self.title, self.authors, self.pmid),
        )
        conn.commit()
        conn.close()

    def publish(self, db: Path, raw_root: Path, body: bytes, *, fmt: str = "xml"):
        """Publish raw bytes exactly as A5 does, including verdict invalidation."""

        candidate = {
            "source": oa.PMC_AWS,
            "url": f"https://aws.example/PMC{self.pmid}.{fmt}",
            "format": fmt,
            "pmcid": self.pmcid,
            "license": "CC BY 4.0",
            "version": "publishedVersion",
        }
        response = client_module.HttpResponse(
            status_code=200,
            headers={"Content-Type": "application/octet-stream"},
            body=body,
            url=str(candidate["url"]),
        )
        artifact = fetch.RawCorpusWriter(raw_root).write(
            pmid=self.pmid, task={"doi": self.doi}, candidate=candidate, response=response
        )
        fetch._update_task_content(
            self.pmid, candidate, artifact, db_path=db, worker_id=None
        )
        return artifact

    def task_row(self, db: Path) -> dict:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            return dict(
                conn.execute("SELECT * FROM tasks WHERE pmid=?", (self.pmid,)).fetchone()
            )
        finally:
            conn.close()

    def validate(self, db: Path, raw_root: Path, *, at=None) -> dict:
        return validation.validate_task_content(
            self.pmid,
            db_path=db,
            raw_root=raw_root,
            validated_at=at or "2026-09-22T00:00:00+00:00",
        )

    # ------------------------------------------------------- verdict identity

    def test_verdict_persists_the_identity_of_the_validated_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            artifact = self.publish(db, raw_root, OK_XML)
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 1)
            row = self.task_row(db)
            self.assertEqual(row["validated_content_path"], str(artifact.path))
            self.assertEqual(
                row["validated_content_sha256"], hashlib.sha256(OK_XML).hexdigest()
            )
            self.assertEqual(row["content_sha256"], row["validated_content_sha256"])
            self.assertTrue(validation.content_binding(row).bound)
            # A second identical validation stays idempotent.
            again = self.validate(db, raw_root)
            self.assertEqual(again["file_valid"], 1)
            self.assertEqual(self.task_row(db)["validated_content_sha256"], row["validated_content_sha256"])

    def test_binding_helper_explains_every_unbound_case(self):
        base = {
            "file_valid": 1,
            "bibliographic_match": 1,
            "content_path": "a/article.xml",
            "content_sha256": "sha-1",
            "validated_content_path": "a/article.xml",
            "validated_content_sha256": "sha-1",
        }
        self.assertTrue(validation.content_binding(base).bound)
        self.assertTrue(
            validation.content_binding(base, actual_sha256="sha-1").bound
        )
        cases = {
            "flags": {**base, "bibliographic_match": 0},
            "sha disagreement": {**base, "content_sha256": "sha-2"},
            "no sha at all": {**base, "content_sha256": None, "validated_content_sha256": None},
            "actual bytes differ": None,
            "path moved": {**base, "content_path": "b/article.xml"},
        }
        for name, row in cases.items():
            with self.subTest(case=name):
                if row is None:
                    binding = validation.content_binding(base, actual_sha256="sha-9")
                else:
                    binding = validation.content_binding(row)
                self.assertFalse(binding.bound)
                self.assertTrue(binding.reason)

    # ---------------------------------------------------- replaced raw content

    def test_replaced_raw_file_blocks_normalization_and_keeps_the_old_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            normalized_root = root / "normalized"
            self.make_db(db)
            artifact = self.publish(db, raw_root, OK_XML)
            self.assertEqual(self.validate(db, raw_root)["file_valid"], 1)
            first = normalization.normalize_one(
                self.pmid, db_path=db, raw_root=raw_root, normalized_root=normalized_root
            )
            output = Path(first["output_path"])
            published = output.read_bytes()
            self.assertIn(self.title.encode("utf-8"), published)

            # The review's reproduction: replace the file behind the pointer.
            artifact.path.write_bytes(WRONG_XML)
            with self.assertRaises(normalization.NormalizationError) as caught:
                normalization.normalize_one(
                    self.pmid, db_path=db, raw_root=raw_root, normalized_root=normalized_root
                )
            self.assertIn("validated content version", str(caught.exception))
            self.assertEqual(output.read_bytes(), published)

    def test_same_path_changed_content_after_a5_replacement_is_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            normalized_root = root / "normalized"
            self.make_db(db)
            self.publish(db, raw_root, OK_XML)
            self.assertEqual(self.validate(db, raw_root)["file_valid"], 1)

            replacement = self.publish(db, raw_root, WRONG_XML)
            row = self.task_row(db)
            self.assertEqual(row["content_sha256"], replacement.sha256)
            self.assertIsNone(row["file_valid"])
            self.assertIsNone(row["bibliographic_match"])
            self.assertIsNone(row["validated_content_path"])
            self.assertIsNone(row["validated_content_sha256"])
            self.assertEqual(row["content_status"], "fetched")
            self.assertFalse(validation.content_binding(row).bound)
            with self.assertRaises(normalization.NormalizationError):
                normalization.normalize_one(
                    self.pmid, db_path=db, raw_root=raw_root, normalized_root=normalized_root
                )
            self.assertFalse(list(normalized_root.glob("*.md")) if normalized_root.is_dir() else [])

    def test_verdict_commit_is_refused_when_another_fetch_replaced_the_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            self.publish(db, raw_root, OK_XML)
            original_match = validation.match_bibliographic
            replacement = fetch.RawCorpusWriter(raw_root)

            def concurrent_fetch(evidence, task):
                # Simulate another acquisition committing new content between the
                # read and the verdict commit.
                new_bytes = WRONG_XML
                candidate = {
                    "source": oa.PMC_AWS,
                    "url": "https://aws.example/PMC6001.xml",
                    "format": "xml",
                    "license": "CC BY 4.0",
                }
                response = client_module.HttpResponse(
                    status_code=200,
                    headers={},
                    body=new_bytes,
                    url="https://aws.example/PMC6001.xml",
                )
                artifact = replacement.write(
                    pmid=self.pmid, task={"doi": self.doi}, candidate=candidate, response=response
                )
                fetch._update_task_content(
                    self.pmid, candidate, artifact, db_path=db, worker_id=None
                )
                return original_match(evidence, task)

            with mock.patch.object(
                validation, "match_bibliographic", side_effect=concurrent_fetch
            ):
                result = self.validate(db, raw_root)

            self.assertEqual(result["outcome"], validation.OUTCOME_CONTENT_CHANGED)
            self.assertTrue(result["content_changed"])
            self.assertEqual(result["file_valid"], 0)
            self.assertIsNone(result["bibliographic_match"])
            self.assertEqual(result["validation_error"], validation.ERROR_CONTENT_CHANGED)
            self.assertIn("content version changed", result["validation_detail"])
            row = self.task_row(db)
            self.assertNotEqual(row["file_valid"], 1)
            self.assertIsNone(row["validated_content_sha256"])
            self.assertNotEqual(row["content_status"], "fulltext_ready")
            self.assertFalse(validation.content_binding(row).bound)

    # ------------------------------------------------------------ A7 consumption

    def test_normalization_parses_the_verified_bytes_not_the_live_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            artifact = self.publish(db, raw_root, OK_XML)
            self.assertEqual(self.validate(db, raw_root)["file_valid"], 1)
            original_parser = normalization.parse_jats_xml

            def swap_then_parse(data):
                # Anything that re-reads the path after verification would now
                # parse the wrong article.
                artifact.path.write_bytes(WRONG_XML)
                return original_parser(data)

            with mock.patch.object(
                normalization, "parse_jats_xml", side_effect=swap_then_parse
            ):
                result = normalization.normalize_one(
                    self.pmid,
                    db_path=db,
                    raw_root=raw_root,
                    normalized_root=root / "normalized",
                )
            text = Path(result["output_path"]).read_text(encoding="utf-8")
            self.assertIn(self.title, text)
            self.assertEqual(
                result["content_sha256"], hashlib.sha256(OK_XML).hexdigest()
            )
            self.assertEqual(artifact.path.read_bytes(), WRONG_XML)

    def test_pdf_conversion_is_bound_to_the_verified_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            # The tiny PDF fixture has too little extractable text for A6, so the
            # validated verdict is written directly; A6's PDF path is covered by
            # test_a6_content_validation.py. What is under test here is that the
            # converter consumes the verified bytes, not the live path.
            artifact = self.publish(db, raw_root, PDF_FIXTURE, fmt="pdf")
            digest = hashlib.sha256(PDF_FIXTURE).hexdigest()
            conn = sqlite3.connect(db)
            conn.execute(
                """UPDATE tasks SET content_status='fulltext_ready', content_format='pdf',
                   content_path=?, content_sha256=?, validated_content_path=?,
                   validated_content_sha256=?, file_valid=1, bibliographic_match=1
                   WHERE pmid=?""",
                (str(artifact.path), digest, str(artifact.path), digest, self.pmid),
            )
            conn.commit()
            conn.close()
            seen: list[tuple[Path, bytes]] = []

            def fake_converter(source_path):
                path = Path(source_path)
                seen.append((path, path.read_bytes()))
                return "# Same PubMed ID 7011\n\n## Abstract\nVerified PDF body.\n"

            with mock.patch.object(
                normalization, "convert_pdf_with_mineru", side_effect=fake_converter
            ):
                result = normalization.normalize_one(
                    self.pmid,
                    db_path=db,
                    raw_root=raw_root,
                    normalized_root=root / "normalized",
                )
            self.assertTrue(Path(result["output_path"]).is_file())
            self.assertEqual(len(seen), 1)
            converted_path, converted_bytes = seen[0]
            self.assertEqual(converted_bytes, PDF_FIXTURE)
            self.assertNotEqual(converted_path, artifact.path)


# The five A6 columns the production snapshot had before A-R1 landed. A legacy
# database is modelled by adding exactly these and no binding columns.
LEGACY_A6_COLUMNS = {
    name: definition
    for name, definition in validation.A6_VALIDATION_COLUMNS.items()
    if name not in ("validated_content_path", "validated_content_sha256")
}
BINDING_COLUMNS = ("validated_content_path", "validated_content_sha256")


def publish_raw_artifact(db: Path, raw_root: Path, pmid: int, body: bytes, *, fmt: str = "xml"):
    """Publish raw bytes the way A5 does, for an arbitrary task."""

    candidate = {
        "source": oa.PMC_AWS,
        "url": f"https://aws.example/PMC{pmid}.{fmt}",
        "format": fmt,
        "pmcid": f"PMC{pmid}",
        "license": "CC BY 4.0",
        "version": "publishedVersion",
    }
    response = client_module.HttpResponse(
        status_code=200,
        headers={"Content-Type": "application/octet-stream"},
        body=body,
        url=str(candidate["url"]),
    )
    artifact = fetch.RawCorpusWriter(raw_root).write(
        pmid=pmid, task={"doi": "10.1000/ar1.legacy"}, candidate=candidate, response=response
    )
    fetch._update_task_content(pmid, candidate, artifact, db_path=db, worker_id=None)
    return artifact


class AR1HistoricalMigrationTests(unittest.TestCase):
    """A-R1: migrating a database that already holds old-format verdicts.

    The pre-A-R1 production snapshot has the five original A6 columns and rows
    verified before the binding columns existed. The additive migration may only
    add columns: a historical row must end up either provable through the A5
    ``content_sha256`` fallback or explicitly pending verification. It is never
    filled in just so that it can pass.
    """

    pmids = (6101, 6102, 6103, 6104)

    def make_legacy_db(self, path: Path, pmids: tuple[int, ...] | None = None) -> None:
        """Create a database shaped exactly like the pre-migration snapshot."""

        schema.migrate_to_v2(path)
        ingest.ingest_pmids(
            list(pmids or self.pmids),
            source="ar1-legacy",
            batch_id="ar1-legacy-batch",
            db_path=path,
        )
        conn = sqlite3.connect(path)
        for name, definition in LEGACY_A6_COLUMNS.items():
            conn.execute(f'ALTER TABLE tasks ADD COLUMN "{name}" {definition}')
        conn.commit()
        conn.close()

    def columns(self, path: Path) -> set[str]:
        conn = sqlite3.connect(path)
        try:
            return {row[1] for row in conn.execute('PRAGMA table_info("tasks")')}
        finally:
            conn.close()

    def rows(self, path: Path) -> dict[int, dict]:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            return {
                int(row["pmid"]): dict(row)
                for row in conn.execute("SELECT * FROM tasks ORDER BY pmid")
            }
        finally:
            conn.close()

    def counts(self, path: Path, sql: str) -> int:
        conn = sqlite3.connect(path)
        try:
            return int(conn.execute(sql).fetchone()[0])
        finally:
            conn.close()

    def seed_legacy_rows(self, db: Path, root: Path) -> dict[str, Path]:
        """Write three historical verdict rows plus one never-verified row."""

        artifact_dir = root / "raw"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        verified_file = artifact_dir / "verified.xml"
        verified_file.write_bytes(OK_XML)
        digest = hashlib.sha256(OK_XML).hexdigest()
        conn = sqlite3.connect(db)
        # 6101: verified before A-R1, provable through the A5 content_sha256.
        conn.execute(
            """UPDATE tasks SET status='metadata_only', content_status='fulltext_ready',
               content_format='xml', content_path=?, content_sha256=?, file_valid=1,
               bibliographic_match=1, validated_at='2026-09-19T00:00:00+00:00',
               validation_method='pmid_pmcid_exact' WHERE pmid=6101""",
            (str(verified_file), digest),
        )
        # 6102: the same verdict with no recorded content version at all.
        conn.execute(
            """UPDATE tasks SET status='metadata_only', content_status='fulltext_ready',
               content_format='xml', content_path=?, content_sha256=NULL, file_valid=1,
               bibliographic_match=1, validated_at='2026-09-19T00:00:00+00:00',
               validation_method='pmid_pmcid_exact' WHERE pmid=6102""",
            (str(verified_file),),
        )
        # 6103: a recorded digest that does not describe the file on disk.
        conn.execute(
            """UPDATE tasks SET status='metadata_only', content_status='fulltext_ready',
               content_format='xml', content_path=?, content_sha256=?, file_valid=1,
               bibliographic_match=1, validated_at='2026-09-19T00:00:00+00:00',
               validation_method='pmid_pmcid_exact' WHERE pmid=6103""",
            (str(verified_file), "deadbeef" * 8),
        )
        # 6104: fetched content without any verdict at all.
        conn.execute(
            """UPDATE tasks SET status='pending', content_status='fetched',
               content_format='xml', content_path=?, content_sha256=?
               WHERE pmid=6104""",
            (str(verified_file), digest),
        )
        conn.commit()
        conn.close()
        return {"verified_file": verified_file, "digest": digest}

    # ------------------------------------------------------------- migration

    def test_migration_on_a_legacy_db_adds_only_the_binding_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            self.make_legacy_db(db)
            self.seed_legacy_rows(db, root)
            before = self.rows(db)
            self.assertEqual(
                self.columns(db) & set(validation.A6_VALIDATION_COLUMNS),
                set(LEGACY_A6_COLUMNS),
            )

            result = validation.ensure_validation_schema(db)

            self.assertEqual(set(result["added_columns"]), set(BINDING_COLUMNS))
            self.assertEqual(set(self.columns(db)) & set(validation.A6_VALIDATION_COLUMNS), set(validation.A6_VALIDATION_COLUMNS))
            after = self.rows(db)
            self.assertEqual(sorted(after), sorted(before))
            for pmid, row in before.items():
                with self.subTest(pmid=pmid):
                    for field in (*LEGACY_A6_COLUMNS, "status", "content_status",
                                  "content_path", "content_sha256"):
                        self.assertEqual(after[pmid][field], row[field])
            # The migration is idempotent and never rewrites it on a second run.
            again = validation.ensure_validation_schema(db)
            self.assertEqual(again["added_columns"], [])
            self.assertEqual(self.rows(db), after)

    def test_migration_never_fills_binding_fields_to_obtain_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            self.make_legacy_db(db)
            self.seed_legacy_rows(db, root)

            validation.ensure_validation_schema(db)

            self.assertEqual(
                self.counts(
                    db,
                    "SELECT COUNT(*) FROM tasks WHERE validated_content_path IS NOT NULL "
                    "OR validated_content_sha256 IS NOT NULL",
                ),
                0,
            )
            rows = self.rows(db)
            # Byte-level check on the recorded pointer: only the row whose
            # recorded digest actually describes the file passes, so the
            # migration fabricated no PASS for the other historical rows.
            provable = []
            for pmid, row in rows.items():
                path = Path(row["content_path"]) if row.get("content_path") else None
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None
                if actual and validation.content_binding(row, actual_sha256=actual).bound:
                    provable.append(pmid)
            self.assertEqual(provable, [6101])
            self.assertEqual(rows[6102]["validated_content_sha256"], None)
            self.assertEqual(rows[6103]["validated_content_sha256"], None)

    def test_legacy_row_without_a_provable_content_version_stays_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            self.make_legacy_db(db)
            seeded = self.seed_legacy_rows(db, root)
            validation.ensure_validation_schema(db)
            row = self.rows(db)[6102]

            binding = validation.content_binding(row)
            self.assertFalse(binding.bound)
            self.assertIn("sha256", binding.reason)
            self.assertEqual(binding.expected_sha256, None)
            # Nothing in the migrated row claims a verified content version.
            self.assertIsNone(row["validated_content_path"])
            self.assertIsNone(row["validated_content_sha256"])
            # Even with the bytes in hand the row cannot be proven: the previous
            # verdict was computed against an unknown content version.
            self.assertFalse(
                validation.content_binding(
                    row, actual_sha256=seeded["digest"]
                ).bound
            )

    def test_legacy_row_with_mismatching_bytes_is_refused_not_rebound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            self.make_legacy_db(db)
            seeded = self.seed_legacy_rows(db, root)
            validation.ensure_validation_schema(db)
            row = self.rows(db)[6103]

            # Recorded-identity check: the row still looks bound, which is why
            # the report says it does not re-read corpus bytes.
            self.assertTrue(validation.content_binding(row).bound)
            # The byte-level check the A7 entry point performs refuses it, and
            # the migration did not paper over the difference.
            byte_binding = validation.content_binding(
                row, actual_sha256=seeded["digest"]
            )
            self.assertFalse(byte_binding.bound)
            self.assertIn("实际内容 sha256", byte_binding.reason)
            self.assertIsNone(row["validated_content_sha256"])

    def test_revalidating_a_legacy_row_creates_an_honest_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            pmid = 6001  # the article the synthetic JATS fixture declares
            self.make_legacy_db(db, pmids=(pmid,))
            conn = sqlite3.connect(db)
            conn.execute(
                "UPDATE tasks SET title=?, doi=?, pmcid=?, year=2020, "
                "first_author_family='smith', first_author='Smith, Jane Q' WHERE pmid=?",
                (
                    AR1ContentBindingTests.title,
                    AR1ContentBindingTests.doi,
                    AR1ContentBindingTests.pmcid,
                    pmid,
                ),
            )
            conn.commit()
            conn.close()
            artifact = publish_raw_artifact(db, raw_root, pmid, OK_XML)
            before = self.rows(db)[pmid]
            self.assertIsNone(before.get("validated_content_path"))
            self.assertIsNone(before.get("file_valid"))

            result = validation.validate_task_content(
                pmid,
                db_path=db,
                raw_root=raw_root,
                validated_at="2026-09-22T12:00:00+00:00",
            )

            self.assertEqual(result["outcome"], validation.OUTCOME_VALIDATED)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 1)
            row = self.rows(db)[pmid]
            # A real re-validation recomputes the digest from the bytes it read.
            self.assertEqual(
                row["validated_content_sha256"],
                hashlib.sha256(artifact.path.read_bytes()).hexdigest(),
            )
            self.assertEqual(row["validated_content_path"], str(artifact.path))
            self.assertEqual(row["validated_at"], "2026-09-22T12:00:00+00:00")
            self.assertTrue(validation.content_binding(row).bound)
            self.assertTrue(
                validation.content_binding(
                    row, actual_sha256=hashlib.sha256(OK_XML).hexdigest()
                ).bound
            )


if __name__ == "__main__":
    unittest.main()
