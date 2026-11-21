"""Offline regression tests for A6 content validation and bibliographic match."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "直肠癌文献爬取" / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(SCRIPTS))

import a1_schema as schema  # noqa: E402
import content_validation as validation  # noqa: E402
import fetch_fulltext as fetch  # noqa: E402
import fulltext_client as client_module  # noqa: E402
import ingest_pmids as ingest  # noqa: E402
import oa_resolver as oa  # noqa: E402
import state_machine as state  # noqa: E402


OK_XML = (FIXTURES / "a6_article_ok.xml").read_text(encoding="utf-8")
WRONG_XML = (FIXTURES / "a6_article_wrong_identity.xml").read_text(encoding="utf-8")
SMALL_XML = (FIXTURES / "a6_article_small_body.xml").read_text(encoding="utf-8")
NAMESPACE_XML = (FIXTURES / "a6_article_namespace.xml").read_text(encoding="utf-8")
REFERENCE_ONLY_XML = (FIXTURES / "a6_article_reference_only.xml").read_text(encoding="utf-8")

PDF_FILLER = (
    "Locally advanced rectal cancer is treated with neoadjuvant chemoradiotherapy.",
    "Magnetic resonance imaging defines the clinical stage before any treatment decision.",
)


class A6ContentValidationTests(unittest.TestCase):
    pmid = 6001
    title = "Neoadjuvant chemoradiotherapy for locally advanced rectal cancer"
    doi = "10.1000/a6.fixture"
    pmcid = "PMC6001"
    authors = "Smith, Jane Q; Jones, Robert"

    # ---------------------------------------------------------------- helpers

    def make_db(self, path: Path, *, status: str = "metadata_only", **overrides) -> None:
        """Create a temporary A1/A2-shaped task row for the fixture PMID."""

        schema.migrate_to_v2(path)
        ingest.ingest_pmids(
            [self.pmid], source="a6-test", batch_id="a6-test-batch", db_path=path
        )
        fields = {
            "doi": self.doi,
            "pmcid": self.pmcid,
            "title": self.title,
            "year": 2020,
            "journal": "Journal of A6 Synthetic Fixtures",
            "authors": self.authors,
            "first_author": "Smith, Jane Q",
            "first_author_family": "smith",
            "status": status,
            "content_status": "fetched",
        }
        fields.update(overrides)
        assignments = ", ".join(f"{name}=?" for name in fields)
        conn = sqlite3.connect(path)
        conn.execute(
            f"UPDATE tasks SET {assignments} WHERE pmid=?",
            (*fields.values(), self.pmid),
        )
        conn.commit()
        conn.close()

    def publish_raw(
        self, db: Path, raw_root: Path, body: bytes, *, fmt: str = "xml"
    ) -> fetch.RawArtifact:
        """Publish raw bytes through A5's writer and mirror its task pointer."""

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
            pmid=self.pmid,
            task={"doi": self.doi},
            candidate=candidate,
            response=response,
        )
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE tasks SET content_status='fetched', content_source=?, "
            "content_format=?, content_path=?, content_sha256=?, retrieved_at=? "
            "WHERE pmid=?",
            (
                candidate["source"],
                artifact.format,
                str(artifact.path),
                artifact.sha256,
                artifact.retrieved_at,
                self.pmid,
            ),
        )
        conn.commit()
        conn.close()
        return artifact

    @staticmethod
    def _content_stream(lines):
        operations = ["BT", "/F1 10 Tf", "12 TL", "36 756 Td"]
        for index, line in enumerate(lines):
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            if index:
                operations.append("T*")
            operations.append(f"({escaped}) Tj")
        operations.append("ET")
        return "\n".join(operations).encode("latin-1", "replace")

    def write_pdf(
        self,
        path: Path,
        *,
        lines=(),
        metadata=None,
        blank: bool = False,
    ) -> Path:
        """Write a real, parser-readable PDF fixture with no external tools."""

        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        page = writer.pages[0]
        if not blank:
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {
                            NameObject("/F1"): DictionaryObject(
                                {
                                    NameObject("/Type"): NameObject("/Font"),
                                    NameObject("/Subtype"): NameObject("/Type1"),
                                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                                }
                            )
                        }
                    )
                }
            )
            stream = DecodedStreamObject()
            stream.set_data(self._content_stream(list(lines)))
            page.replace_contents(stream)
        if metadata:
            writer.add_metadata(metadata)
        with path.open("wb") as handle:
            writer.write(handle)
        return path

    def pdf_lines(
        self,
        *,
        title=None,
        authors="Jane Q Smith, Robert Jones",
        doi=None,
        pmid=None,
        pmcid=None,
        year=None,
        repeats=14,
    ):
        lines = [self.title if title is None else title, authors]
        identity = []
        if doi:
            identity.append(f"doi:{doi}")
        if pmid:
            identity.append(f"PMID: {pmid}")
        if pmcid:
            identity.append(f"PMCID: {pmcid}")
        if identity:
            lines.append("   ".join(identity))
        if year:
            lines.append(f"Published in {year}")
        for _ in range(repeats):
            lines.extend(PDF_FILLER)
        return lines

    @staticmethod
    def valid_pdf_metadata(title=None, author="Jane Q Smith", created="D:20200101000000Z"):
        metadata = {"/Producer": "a6-offline-fixture", "/CreationDate": created}
        if title:
            metadata["/Title"] = title
        if author:
            metadata["/Author"] = author
        return metadata

    def task_row(self, db: Path, pmid: int | None = None) -> dict:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            return dict(
                conn.execute(
                    "SELECT * FROM tasks WHERE pmid=?", (pmid or self.pmid,)
                ).fetchone()
            )
        finally:
            conn.close()

    @staticmethod
    def attempts(db: Path) -> list[dict]:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            return [
                dict(row)
                for row in conn.execute("SELECT * FROM fetch_attempts ORDER BY attempt_id")
            ]
        finally:
            conn.close()

    @staticmethod
    def columns(db: Path) -> set[str]:
        conn = sqlite3.connect(db)
        try:
            return {row[1] for row in conn.execute("PRAGMA table_info(tasks)")}
        finally:
            conn.close()

    @staticmethod
    def xml_without_identifiers(xml: str = OK_XML) -> str:
        return re.sub(r"\s*<article-id[^>]*>[^<]*</article-id>", "", xml)

    @staticmethod
    def xml_without_contrib(xml: str = OK_XML) -> str:
        return re.sub(r"<contrib-group>.*?</contrib-group>", "", xml, flags=re.S)

    def validate(self, db: Path, raw_root: Path, *, pmid: int | None = None, at="2026-09-19T00:00:00+00:00"):
        return validation.validate_task_content(
            pmid or self.pmid, db_path=db, raw_root=raw_root, validated_at=at
        )

    # ------------------------------------------------------- XML: happy path

    def test_valid_xml_with_matching_identity_becomes_fulltext_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            artifact = self.publish_raw(db, raw_root, OK_XML.encode("utf-8"))
            result = self.validate(db, raw_root)
            self.assertEqual(result["outcome"], validation.OUTCOME_VALIDATED)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 1)
            self.assertEqual(result["validation_method"], validation.MATCH_METHOD_IDENTIFIER)
            self.assertIsNone(result["validation_error"])
            self.assertGreaterEqual(result["text_chars"], validation.LIMITS.min_xml_body_chars)
            self.assertEqual(result["identifiers"]["pmid"], str(self.pmid))
            self.assertEqual(result["identifiers"]["doi"], self.doi)
            row = self.task_row(db)
            self.assertEqual(row["file_valid"], 1)
            self.assertEqual(row["bibliographic_match"], 1)
            self.assertEqual(row["content_status"], "fulltext_ready")
            self.assertEqual(row["validated_at"], "2026-09-19T00:00:00+00:00")
            self.assertIsNone(row["validation_error"])
            self.assertTrue(artifact.path.is_file())

    def test_namespaced_xml_uses_local_names_for_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(db, raw_root, NAMESPACE_XML.encode("utf-8"))
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 1)
            self.assertEqual(result["validation_method"], validation.MATCH_METHOD_IDENTIFIER)
            self.assertEqual(result["identifiers"]["pmid"], str(self.pmid))
            self.assertEqual(result["identifiers"]["pmcid"], self.pmcid)
            self.assertEqual(result["content_status"], "fulltext_ready")

    def test_reference_only_identifiers_are_not_used_as_article_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(db, raw_root, REFERENCE_ONLY_XML.encode("utf-8"))
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["identifiers"], {})
            self.assertEqual(result["bibliographic_match"], 1)
            self.assertEqual(result["validation_method"], validation.MATCH_METHOD_DESCRIPTIVE)
            self.assertEqual(result["content_status"], "fulltext_ready")

    def test_xml_without_strong_identifiers_matches_on_title_author_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(
                db, raw_root, self.xml_without_identifiers().encode("utf-8")
            )
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 1)
            self.assertEqual(result["validation_method"], validation.MATCH_METHOD_DESCRIPTIVE)
            self.assertEqual(result["identifiers"], {})
            self.assertEqual(self.task_row(db)["content_status"], "fulltext_ready")

    def test_body_surname_without_file_author_metadata_is_not_author_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            variant = self.xml_without_contrib(self.xml_without_identifiers())
            variant = variant.replace(
                "Locally advanced rectal cancer is routinely treated",
                "Smith is mentioned in the body. Locally advanced rectal cancer is routinely treated",
                1,
            )
            self.publish_raw(db, raw_root, variant.encode("utf-8"))
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertIsNone(result["bibliographic_match"])
            self.assertIn("file author metadata missing", result["validation_detail"])
            self.assertEqual(result["content_status"], "fetched")

    def test_pure_title_only_match_stays_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            variant = self.xml_without_contrib(self.xml_without_identifiers())
            variant = re.sub(r"<pub-date.*?</pub-date>", "", variant, flags=re.S)
            self.publish_raw(db, raw_root, variant.encode("utf-8"))
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertIsNone(result["bibliographic_match"])
            self.assertIn("author", result["validation_detail"])
            self.assertIn("year", result["validation_detail"])
            self.assertEqual(result["content_status"], "fetched")

    def test_title_only_match_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            variant = self.xml_without_identifiers()
            variant = re.sub(
                r"<contrib-group>.*?</contrib-group>",
                '<contrib-group><contrib contrib-type="author"><name>'
                "<surname>Nguyen</surname><given-names>Thu</given-names>"
                "</name></contrib></contrib-group>",
                variant,
                flags=re.S,
            )
            variant = variant.replace("<year>2020</year>", "<year>2015</year>")
            self.publish_raw(db, raw_root, variant.encode("utf-8"))
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 0)
            self.assertEqual(result["validation_error"], validation.ERROR_METADATA_MISMATCH)
            self.assertIn("author", result["validation_detail"])
            self.assertIn("year", result["validation_detail"])
            row = self.task_row(db)
            self.assertEqual(row["file_valid"], 1)
            self.assertEqual(row["bibliographic_match"], 0)
            self.assertEqual(row["content_status"], "fetched")
            self.assertIsNone(row["validation_method"])

    def test_identifier_conflict_is_not_overridden_by_title_similarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(db, raw_root, WRONG_XML.encode("utf-8"))
            similarity = validation.title_similarity(
                "Neoadjuvant chemoradiotherapy for locally advanced rectal cancer",
                self.title,
            )
            self.assertEqual(similarity, 1.0)
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 0)
            self.assertEqual(
                result["validation_error"], validation.ERROR_IDENTIFIER_MISMATCH
            )
            self.assertIn("PMID conflict", result["validation_detail"])
            self.assertEqual(self.task_row(db)["content_status"], "fetched")

    def test_valid_xml_without_identity_evidence_stays_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(
                db,
                raw_root,
                self.xml_without_contrib(self.xml_without_identifiers()).encode("utf-8"),
            )
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertIsNone(result["bibliographic_match"])
            self.assertIsNone(result["validation_error"])
            self.assertIn("insufficient comparable evidence", result["validation_detail"])
            row = self.task_row(db)
            self.assertEqual(row["file_valid"], 1)
            self.assertIsNone(row["bibliographic_match"])
            self.assertNotEqual(row["content_status"], "fulltext_ready")

    # ------------------------------------------------------ XML: reject paths

    def test_xml_below_minimum_body_length_is_invalid_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(db, raw_root, SMALL_XML.encode("utf-8"))
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 0)
            self.assertIsNone(result["bibliographic_match"])
            self.assertEqual(
                result["validation_error"], validation.ERROR_CONTENT_TOO_SMALL
            )
            row = self.task_row(db)
            self.assertEqual(row["file_valid"], 0)
            self.assertIsNone(row["bibliographic_match"])
            self.assertEqual(row["content_status"], "fetched")

    def test_unparseable_xml_is_invalid_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            truncated = OK_XML.encode("utf-8")[: int(len(OK_XML) * 0.45)]
            self.publish_raw(db, raw_root, truncated)
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 0)
            self.assertIsNone(result["bibliographic_match"])
            self.assertEqual(result["validation_error"], validation.ERROR_INVALID_XML)

    def test_missing_article_title_is_invalid_xml(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            variant = re.sub(r"<title-group>.*?</title-group>", "", OK_XML, flags=re.S)
            self.publish_raw(db, raw_root, variant.encode("utf-8"))
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 0)
            self.assertEqual(result["validation_error"], validation.ERROR_INVALID_XML)
            self.assertIn("article-title", result["validation_detail"])

    # ------------------------------------------------------------------- PDF

    def test_valid_pdf_with_matching_identity_becomes_fulltext_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            pdf = self.write_pdf(
                Path(tmp) / "article.pdf",
                lines=self.pdf_lines(doi=self.doi, pmid=self.pmid, pmcid=self.pmcid, year=2020),
                metadata=self.valid_pdf_metadata(title=self.title),
            )
            self.publish_raw(db, raw_root, pdf.read_bytes(), fmt="pdf")
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 1)
            self.assertEqual(result["validation_method"], validation.MATCH_METHOD_IDENTIFIER)
            self.assertEqual(result["page_count"], 1)
            self.assertGreaterEqual(result["text_chars"], validation.LIMITS.min_pdf_text_chars)
            self.assertEqual(self.task_row(db)["content_status"], "fulltext_ready")

    def test_truncated_pdf_is_rejected_as_invalid_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            pdf = self.write_pdf(
                Path(tmp) / "article.pdf",
                lines=self.pdf_lines(doi=self.doi, pmid=self.pmid),
                metadata=self.valid_pdf_metadata(title=self.title),
            )
            data = pdf.read_bytes()
            self.publish_raw(db, raw_root, data[: int(len(data) * 0.55)], fmt="pdf")
            result = self.validate(db, raw_root)
            self.assertEqual(result["outcome"], validation.OUTCOME_VALIDATED)
            self.assertEqual(result["file_valid"], 0)
            self.assertIsNone(result["bibliographic_match"])
            self.assertEqual(result["validation_error"], validation.ERROR_INVALID_PDF)
            row = self.task_row(db)
            self.assertEqual(row["file_valid"], 0)
            self.assertIsNone(row["bibliographic_match"])
            self.assertEqual(row["content_status"], "fetched")

    def test_valid_pdf_from_another_article_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            pdf = self.write_pdf(
                Path(tmp) / "other.pdf",
                lines=self.pdf_lines(
                    title="A different article about rectal imaging biomarkers",
                    authors="Thu Nguyen, Ravi Patel",
                    doi="10.9999/other.fixture",
                    pmid=9999,
                    pmcid="PMC9999",
                    year=2015,
                ),
                metadata=self.valid_pdf_metadata(
                    title="A different article about rectal imaging biomarkers",
                    author="Thu Nguyen",
                    created="D:20150101000000Z",
                ),
            )
            self.publish_raw(db, raw_root, pdf.read_bytes(), fmt="pdf")
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 0)
            self.assertEqual(
                result["validation_error"], validation.ERROR_IDENTIFIER_MISMATCH
            )
            row = self.task_row(db)
            self.assertEqual(row["file_valid"], 1)
            self.assertEqual(row["bibliographic_match"], 0)
            self.assertNotEqual(row["content_status"], "fulltext_ready")

    def test_pdf_without_extractable_text_is_content_too_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            pdf = self.write_pdf(
                Path(tmp) / "blank.pdf",
                blank=True,
                metadata=self.valid_pdf_metadata(title=self.title),
            )
            self.publish_raw(db, raw_root, pdf.read_bytes(), fmt="pdf")
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 0)
            self.assertIsNone(result["bibliographic_match"])
            self.assertEqual(
                result["validation_error"], validation.ERROR_CONTENT_TOO_SMALL
            )
            self.assertEqual(result["page_count"], 1)

    def test_pdf_text_is_read_from_a_real_parser(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf = self.write_pdf(
                Path(tmp) / "article.pdf",
                lines=self.pdf_lines(doi=self.doi, pmid=self.pmid, pmcid=self.pmcid),
                metadata=self.valid_pdf_metadata(title=self.title),
            )
            evidence = validation.validate_pdf_bytes(pdf.read_bytes())
            self.assertTrue(evidence.file_valid)
            self.assertEqual(evidence.page_count, 1)
            self.assertEqual(evidence.identifiers["pmid"], str(self.pmid))
            self.assertEqual(evidence.identifiers["pmcid"], self.pmcid)
            self.assertEqual(evidence.identifiers["doi"], self.doi)
            self.assertEqual(evidence.year, 2020)
            self.assertIn("smith", evidence.author_families)
            self.assertEqual(evidence.title_candidates[0], self.title)
            # The recorded artifact is a genuine PDF the stdlib-independent
            # parser can reopen, not just a file with a .pdf suffix.
            self.assertEqual(len(PdfReader(str(pdf)).pages), 1)

    # ------------------------------------------------- persistence behaviour

    def test_file_valid_and_bibliographic_match_are_persisted_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            cases = (
                ("a6_ok", OK_XML.encode("utf-8"), (1, 1), "fulltext_ready"),
                ("a6_wrong", WRONG_XML.encode("utf-8"), (1, 0), "fetched"),
                ("a6_small", SMALL_XML.encode("utf-8"), (0, None), "fetched"),
            )
            for label, body, expected, content_status in cases:
                with self.subTest(case=label):
                    self.publish_raw(db, raw_root, body)
                    result = self.validate(db, raw_root)
                    self.assertEqual(
                        (result["file_valid"], result["bibliographic_match"]), expected
                    )
                    row = self.task_row(db)
                    self.assertEqual(
                        (row["file_valid"], row["bibliographic_match"]), expected
                    )
                    self.assertEqual(row["content_status"], content_status)

    def test_fulltext_ready_is_cleared_when_the_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(db, raw_root, OK_XML.encode("utf-8"))
            self.assertEqual(self.validate(db, raw_root)["content_status"], "fulltext_ready")
            # Replace the artifact without letting A5's writer reset the state,
            # so the A6 gate alone has to clear the stale ready state.
            self.publish_raw(db, raw_root, WRONG_XML.encode("utf-8"))
            conn = sqlite3.connect(db)
            conn.execute(
                "UPDATE tasks SET content_status='fulltext_ready' WHERE pmid=?",
                (self.pmid,),
            )
            conn.commit()
            conn.close()
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 0)
            self.assertEqual(result["content_status"], "fetched")
            self.assertNotEqual(self.task_row(db)["content_status"], "fulltext_ready")

    def test_txt_and_html_artifacts_stay_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(db, raw_root, b"plain text body" * 200, fmt="txt")
            conn = sqlite3.connect(db)
            conn.execute(
                "UPDATE tasks SET content_status='fulltext_ready' WHERE pmid=?",
                (self.pmid,),
            )
            conn.commit()
            conn.close()
            result = self.validate(db, raw_root)
            self.assertEqual(result["outcome"], validation.OUTCOME_UNSUPPORTED_FORMAT)
            self.assertIsNone(result["file_valid"])
            self.assertIsNone(result["bibliographic_match"])
            row = self.task_row(db)
            self.assertIsNone(row["file_valid"])
            self.assertIsNone(row["bibliographic_match"])
            self.assertEqual(row["content_status"], "fetched")
            self.assertIsNone(row["validation_method"])

    def test_missing_raw_artifact_persists_failure_and_clears_stale_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            conn = sqlite3.connect(db)
            conn.execute(
                "UPDATE tasks SET content_status='fulltext_ready' WHERE pmid=?",
                (self.pmid,),
            )
            conn.commit()
            conn.close()
            result = self.validate(db, raw_root)
            self.assertEqual(result["outcome"], validation.OUTCOME_NO_RAW_CONTENT)
            self.assertTrue(result["written"])
            self.assertEqual(result["file_valid"], 0)
            self.assertIsNone(result["bibliographic_match"])
            self.assertEqual(result["validated_at"], "2026-09-19T00:00:00+00:00")
            self.assertEqual(result["content_status"], "fetched")
            row = self.task_row(db)
            self.assertEqual(row["file_valid"], 0)
            self.assertIsNone(row["bibliographic_match"])
            self.assertEqual(row["validated_at"], "2026-09-19T00:00:00+00:00")
            self.assertEqual(row["content_status"], "fetched")
            self.assertEqual(self.attempts(db), [])

    def test_raw_artifact_is_found_without_a_recorded_content_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(db, raw_root, OK_XML.encode("utf-8"))
            conn = sqlite3.connect(db)
            conn.execute(
                "UPDATE tasks SET content_path=NULL, content_sha256=NULL WHERE pmid=?",
                (self.pmid,),
            )
            conn.commit()
            conn.close()
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 1)
            self.assertEqual(result["bibliographic_match"], 1)

    def test_recorded_sha256_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            artifact = self.publish_raw(db, raw_root, OK_XML.encode("utf-8"))
            artifact.path.write_bytes(OK_XML.encode("utf-8") + b"<!--tampered-->")
            result = self.validate(db, raw_root)
            self.assertEqual(result["file_valid"], 0)
            self.assertIsNone(result["bibliographic_match"])
            self.assertEqual(result["validation_error"], validation.ERROR_PARSER)
            self.assertIn("sha256", result["validation_detail"])

    def test_repeated_validation_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(db, raw_root, OK_XML.encode("utf-8"))
            first = self.validate(db, raw_root)
            row_first = self.task_row(db)
            second = self.validate(db, raw_root)
            row_second = self.task_row(db)
            self.assertEqual(first["file_valid"], second["file_valid"])
            self.assertEqual(first["content_status"], second["content_status"])
            self.assertEqual(row_first, row_second)
            self.assertEqual(self.attempts(db), [])
            self.assertEqual(row_second["attempt_count"], 0)

    def test_failed_validation_keeps_raw_files_and_canonical_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            artifact = self.publish_raw(db, raw_root, WRONG_XML.encode("utf-8"))
            directory = artifact.path.parent
            before_files = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(directory.iterdir())
            }
            protected = (
                "doi",
                "pmcid",
                "title",
                "year",
                "authors",
                "first_author",
                "first_author_family",
                "journal",
                "license",
                "status",
                "content_path",
                "content_sha256",
                "content_source",
                "content_format",
                "attempt_count",
            )
            before_task = {name: self.task_row(db)[name] for name in protected}
            result = self.validate(db, raw_root)
            self.assertEqual(result["bibliographic_match"], 0)
            after_files = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(directory.iterdir())
            }
            self.assertEqual(before_files, after_files)
            self.assertTrue(artifact.path.is_file())
            after_task = self.task_row(db)
            self.assertEqual(before_task, {name: after_task[name] for name in protected})
            self.assertEqual(self.attempts(db), [])

    def test_local_validation_never_writes_fetch_attempts_or_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(db, raw_root, OK_XML.encode("utf-8"))
            result = self.validate(db, raw_root)
            self.assertEqual(result["content_status"], "fulltext_ready")
            row = self.task_row(db)
            self.assertEqual(row["status"], "metadata_only")
            self.assertEqual(row["attempt_count"], 0)
            self.assertEqual(self.attempts(db), [])
            # A3 stays the owner of tasks.status and re-derives the same value.
            self.assertEqual(
                state.derive_task_status(self.pmid, db_path=db), "metadata_only"
            )
            self.assertTrue(
                set(validation.A6_ERROR_CLASSES) <= set(state.ERROR_CLASSES)
            )

    # -------------------------------------------------------------- migration

    def make_legacy_db(self, path: Path, rows) -> None:
        conn = sqlite3.connect(path)
        conn.execute(
            """CREATE TABLE tasks (
                pmid INTEGER PRIMARY KEY, doi TEXT, pmc TEXT, year INTEGER,
                title TEXT, title_norm TEXT, status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0, route TEXT, last_error TEXT,
                pdf_path TEXT, license TEXT NOT NULL DEFAULT 'unverified',
                updated_at TEXT
            )"""
        )
        for row in rows:
            conn.execute("INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
        conn.commit()
        conn.close()

    def test_additive_migration_adds_columns_without_losing_legacy_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            self.make_legacy_db(
                db,
                [
                    (
                        123,
                        "10.1/x",
                        "PMC123",
                        2020,
                        "Legacy title",
                        "legacy title",
                        "done",
                        4,
                        "europepmc",
                        "old error",
                        "x.pdf",
                        "CC-BY",
                        "2026-09-01 01:02:03",
                    )
                ],
            )
            result = validation.ensure_validation_schema(db)
            self.assertEqual(
                set(result["added_columns"]), set(validation.A6_VALIDATION_COLUMNS)
            )
            self.assertTrue(set(validation.A6_VALIDATION_COLUMNS) <= self.columns(db))
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            try:
                legacy = conn.execute(
                    "SELECT pmid,doi,pmc,status,attempts,pdf_path,license,updated_at "
                    "FROM tasks"
                ).fetchone()
                self.assertEqual(
                    tuple(legacy),
                    (
                        123,
                        "10.1/x",
                        "PMC123",
                        "done",
                        4,
                        "x.pdf",
                        "CC-BY",
                        "2026-09-01 01:02:03",
                    ),
                )
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 1)
                self.assertEqual(
                    tuple(
                        conn.execute(
                            "SELECT file_valid,bibliographic_match,validated_at,"
                            "validation_method,validation_error FROM tasks"
                        ).fetchone()
                    ),
                    (None, None, None, None, None),
                )
                # A6 columns are additive: A1's schema version and verifier hold.
                self.assertEqual(conn.execute("SELECT version FROM schema_version").fetchone()[0], 2)
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 2)
                self.assertTrue(schema.verify_schema_v2(conn)["verified"])
            finally:
                conn.close()
            again = validation.ensure_validation_schema(db)
            self.assertEqual(again["added_columns"], [])

    def test_rollback_drops_only_validation_columns_and_guards_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            self.publish_raw(db, raw_root, OK_XML.encode("utf-8"))
            self.validate(db, raw_root)
            with self.assertRaises(validation.ValidationSchemaError):
                validation.rollback_validation_schema(db)
            result = validation.rollback_validation_schema(db, force=True)
            self.assertTrue(result["rolled_back"])
            self.assertEqual(
                set(result["dropped_columns"]), set(validation.A6_VALIDATION_COLUMNS)
            )
            remaining = self.columns(db)
            self.assertFalse(set(validation.A6_VALIDATION_COLUMNS) & remaining)
            self.assertTrue({"pmid", "status", "content_status"} <= remaining)
            row = self.task_row(db)
            self.assertEqual(row["title"], self.title)
            self.assertEqual(row["status"], "metadata_only")

    def test_cli_migrate_validate_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            raw_root = Path(tmp) / "corpus_raw"
            self.make_db(db)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                migrate_code = validation.run_cli(["migrate", "--db", str(db)])
            self.assertEqual(migrate_code, 0)
            self.assertTrue(
                set(validation.A6_VALIDATION_COLUMNS)
                <= set(json.loads(stream.getvalue())["columns"])
            )
            self.publish_raw(db, raw_root, OK_XML.encode("utf-8"))
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                exit_code = validation.run_cli(
                    ["validate", str(self.pmid), "--db", str(db), "--raw-root", str(raw_root)]
                )
            self.assertEqual(exit_code, 0)
            payload = json.loads(stream.getvalue())
            self.assertEqual((payload["file_valid"], payload["bibliographic_match"]), (1, 1))
            self.publish_raw(db, raw_root, WRONG_XML.encode("utf-8"))
            with contextlib.redirect_stdout(io.StringIO()):
                reject_code = validation.run_cli(
                    ["validate", str(self.pmid), "--db", str(db), "--raw-root", str(raw_root)]
                )
            self.assertEqual(reject_code, 1)

    # ------------------------------------------------------------ unit rules

    def test_normalization_and_threshold_contract(self):
        self.assertEqual(
            validation.normalize_doi("https://doi.org/10.1000/A6.Fixture."),
            "10.1000/a6.fixture",
        )
        self.assertEqual(
            validation.normalize_doi("doi: 10.1000/A6.Fixture)"), "10.1000/a6.fixture"
        )
        self.assertEqual(
            validation.normalize_doi("See https://dx.doi.org/10.1000/a6.fixture for details"),
            "10.1000/a6.fixture",
        )
        self.assertEqual(validation.normalize_doi(None), None)
        self.assertEqual(validation.normalize_doi("  "), None)
        self.assertIsNone(validation.normalize_doi("not a doi"))
        self.assertEqual(validation.normalize_pmid("no digits here"), None)
        self.assertEqual(validation.normalize_pmid(" PMID: 006001 "), "6001")
        self.assertEqual(validation.normalize_pmcid("pmc 6001"), "PMC6001")
        self.assertTrue(validation.families_match("van der berg", "berg"))
        self.assertTrue(validation.families_match("O'Brien", "obrien"))
        self.assertFalse(validation.families_match("smith", "jones"))
        self.assertEqual(
            validation.title_similarity(self.title, self.title.upper()), 1.0
        )
        self.assertLess(
            validation.title_similarity(self.title, "An unrelated editorial about imaging"),
            validation.LIMITS.min_title_similarity,
        )
        self.assertTrue(
            set(validation.A6_ERROR_CLASSES) <= set(state.ERROR_CLASSES)
        )
        self.assertEqual(
            {
                validation.MATCH_METHOD_IDENTIFIER,
                validation.MATCH_METHOD_DOI,
                validation.MATCH_METHOD_DESCRIPTIVE,
            },
            set(validation.A6_MATCH_METHODS),
        )
        self.assertEqual(
            validation.LIMITS.min_title_similarity, validation.ValidationLimits().min_title_similarity
        )
        self.assertGreater(validation.LIMITS.min_xml_body_chars, 0)
        self.assertGreater(validation.LIMITS.min_pdf_text_chars, 0)


if __name__ == "__main__":
    unittest.main()
