"""Offline regression tests for the A7 normalization layer."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "直肠癌文献爬取" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import a1_schema as schema  # noqa: E402
import content_validation as validation  # noqa: E402
import ingest_pmids as ingest  # noqa: E402
import normalization  # noqa: E402


FIXTURES = ROOT / "tests" / "fixtures"
XML_FIXTURE = FIXTURES / "a7_article.xml"
TXT_FIXTURE = FIXTURES / "a7_article.txt"
PDF_FIXTURE = FIXTURES / "a7_article.pdf"
IDENTITY_FIELDS = ("pmid", "doi", "pmcid", "title", "year", "journal")
METADATA = {
    "pmid": 7001,
    "doi": "10.1000/a7.fixture",
    "pmcid": "PMC7001",
    "title": "A7 normalization fixture article",
    "year": 2021,
    "journal": "Journal of A7 Fixtures",
    "license": "CC BY 4.0",
}
MINERU_MARKDOWN = """# A7 normalization fixture article

## Abstract
MinerU extracted abstract text.

## Introduction
MinerU extracted introduction text.

## Methods
MinerU extracted methods text.

## Results
MinerU extracted results text.

## Discussion
MinerU extracted discussion text.

## Conclusion
MinerU extracted conclusion text.

## Table 1 Baseline characteristics
| Feature | Count |
| --- | ---: |
| Patients | 12 |

## Figure 1 Workflow overview
Figure caption text is retained by MinerU.

## Supplementary information
MinerU supplementary text.

## References
- Example A. A cited article.
"""


class FakeMinerUConverter:
    def __init__(self, markdown: str = MINERU_MARKDOWN):
        self.markdown = markdown
        self.calls: list[Path] = []

    def __call__(self, source_path: Path) -> str:
        self.calls.append(source_path)
        return self.markdown


def read_front_matter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, raw, _ = text.split("---\n", 2)
    result: dict[str, object] = {}
    for line in raw.splitlines():
        key, value = line.split(":", 1)
        value = value.strip()
        if value.startswith('"'):
            result[key] = json.loads(value)
        elif value == "":
            result[key] = None
        else:
            result[key] = int(value) if value.isdigit() else value
    return result


class A7NormalizationTests(unittest.TestCase):
    def normalize(
        self,
        fixture: Path,
        root: Path,
        *,
        mineru_converter: normalization.MinerUConverter | None = None,
        **metadata: object,
    ) -> Path:
        result = normalization.normalize_artifact(
            fixture,
            {**METADATA, **metadata},
            output_dir=root,
            mineru_converter=mineru_converter,
        )
        output = Path(result["output_path"])
        self.assertEqual(output.name, "PMID_7001.md")
        self.assertTrue(output.is_file())
        return output

    def test_jats_preserves_front_matter_sections_tables_figures_supplement_and_refs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.normalize(XML_FIXTURE, Path(tmp))
            text = output.read_text(encoding="utf-8")
            front = read_front_matter(output)
            self.assertEqual(set(front), set(normalization.FRONT_MATTER_FIELDS))
            self.assertEqual({key: front[key] for key in IDENTITY_FIELDS},
                             {key: METADATA[key] for key in IDENTITY_FIELDS})
            self.assertEqual(front["content_sha256"], hashlib.sha256(XML_FIXTURE.read_bytes()).hexdigest())
            for heading in (
                "# A7 normalization fixture article",
                "## Abstract",
                "## Introduction",
                "## Methods",
                "## Results",
                "## Discussion",
                "## Conclusion",
                "## Tables",
                "### Table 1 Baseline characteristics",
                "## Figures",
                "### Figure 1 Workflow overview",
                "## Supplementary information",
                "## References",
            ):
                self.assertIn(heading, text)
            self.assertIn("| Feature | Count |", text)
            self.assertIn("Workflow overview", text)
            self.assertIn("Example A. A cited article", text)

    def test_txt_uses_the_same_stable_heading_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = self.normalize(TXT_FIXTURE, Path(tmp), source="fixture-txt")
            text = output.read_text(encoding="utf-8")
            self.assertIn("## Abstract", text)
            self.assertIn("## Introduction", text)
            self.assertIn("## Methods", text)
            self.assertIn("## Results", text)
            self.assertIn("## Discussion", text)
            self.assertIn("## Conclusion", text)
            self.assertIn("## Tables", text)
            self.assertIn("## Figures", text)
            self.assertIn("## Supplementary information", text)
            self.assertIn("## References", text)
            self.assertIn("| Patients | 12 |", text)
            self.assertEqual(read_front_matter(output)["source"], "fixture-txt")

    def test_pdf_uses_injected_mineru_adapter_and_shared_text_normalizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            converter = FakeMinerUConverter()
            output = self.normalize(
                PDF_FIXTURE,
                Path(tmp),
                source="fixture-pdf",
                mineru_converter=converter,
            )
            text = output.read_text(encoding="utf-8")
            front = read_front_matter(output)
            self.assertEqual(converter.calls, [PDF_FIXTURE])
            self.assertEqual(front["pmid"], METADATA["pmid"])
            self.assertEqual(front["doi"], METADATA["doi"])
            self.assertEqual(front["source"], "fixture-pdf")
            self.assertEqual(front["content_sha256"], hashlib.sha256(PDF_FIXTURE.read_bytes()).hexdigest())
            self.assertIn("MinerU extracted abstract text", text)
            self.assertIn("# A7 normalization fixture article", text)
            for heading in (
                "## Abstract",
                "## Introduction",
                "## Methods",
                "## Results",
                "## Discussion",
                "## Conclusion",
                "## Tables",
                "### Table 1 Baseline characteristics",
                "## Figures",
                "## Supplementary information",
                "## References",
            ):
                self.assertIn(heading, text)
            self.assertIn("| Feature | Count |", text)
            self.assertIn("Figure caption text is retained by MinerU", text)

    def test_default_mineru_adapter_uses_existing_flash_command_without_network(self):
        calls: list[tuple[list[str], Path]] = []

        def fake_run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
            calls.append((command, cwd))
            output_dir = Path(command[command.index("-o") + 1])
            (output_dir / f"{PDF_FIXTURE.stem}.md").write_text(
                MINERU_MARKDOWN,
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch.object(normalization, "_run_mineru_command", side_effect=fake_run):
            markdown = normalization.convert_pdf_with_mineru(PDF_FIXTURE)

        self.assertIn("## Table 1 Baseline characteristics", markdown)
        self.assertEqual(len(calls), 1)
        command, cwd = calls[0]
        self.assertEqual(command[:2], ["mineru-open-api", "flash-extract"])
        self.assertIn("--language", command)
        self.assertEqual(command[command.index("--language") + 1], "en")
        self.assertEqual(cwd, normalization.PROJECT)

    def test_identity_metadata_is_consistent_across_xml_txt_and_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = [
                self.normalize(XML_FIXTURE, root / "xml", source="fixture-xml"),
                self.normalize(TXT_FIXTURE, root / "txt", source="fixture-txt"),
                self.normalize(
                    PDF_FIXTURE,
                    root / "pdf",
                    source="fixture-pdf",
                    mineru_converter=FakeMinerUConverter(),
                ),
            ]
            fronts = [read_front_matter(output) for output in outputs]
            for key in IDENTITY_FIELDS:
                self.assertEqual({front[key] for front in fronts}, {METADATA[key]})
            self.assertEqual({front["license"] for front in fronts}, {"CC BY 4.0"})
            self.assertEqual({front["source"] for front in fronts}, {"fixture-xml", "fixture-txt", "fixture-pdf"})
            self.assertEqual(len({front["content_sha256"] for front in fronts}), 3)

    def test_mineru_markdown_adapter_writes_the_same_output_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "PMID_7001.md"
            source.write_text(TXT_FIXTURE.read_text(encoding="utf-8").replace("A7 normalization fixture article", "# A7 normalization fixture article", 1), encoding="utf-8")
            output = normalization.normalize_mineru_markdown(
                source, METADATA, output_dir=Path(tmp) / "normalized"
            )
            self.assertEqual(output["format"], "md")
            normalized_text = Path(output["output_path"]).read_text(encoding="utf-8")
            self.assertIn("# A7 normalization fixture article", normalized_text)
            self.assertNotIn("# # A7 normalization fixture article", normalized_text)
            self.assertIn("## Introduction", normalized_text)

    def test_pmid_adapter_requires_the_a6_ready_gate_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            raw_root = root / "corpus_raw"
            target = raw_root / "PMID_7001"
            target.mkdir(parents=True)
            article = target / "article.xml"
            shutil.copyfile(XML_FIXTURE, article)
            digest = hashlib.sha256(article.read_bytes()).hexdigest()
            (target / "source.json").write_text(
                json.dumps({
                    "PMID": 7001,
                    "DOI": METADATA["doi"],
                    "PMCID": METADATA["pmcid"],
                    "source": "fixture-xml",
                    "license": METADATA["license"],
                    "format": "xml",
                    "sha256": digest,
                }),
                encoding="utf-8",
            )
            schema.migrate_to_v2(db)
            validation.ensure_validation_schema(db)
            ingest.ingest_pmids([7001], source="a7-test", batch_id="a7-test-batch", db_path=db)
            conn = sqlite3.connect(db)
            conn.execute(
                "UPDATE tasks SET doi=?, pmcid=?, title=?, year=?, journal=?, license=?, "
                "content_status=?, content_format=?, content_path=?, content_sha256=?, "
                "validated_content_path=?, validated_content_sha256=?, "
                "file_valid=?, bibliographic_match=? WHERE pmid=?",
                (METADATA["doi"], METADATA["pmcid"], METADATA["title"], METADATA["year"],
                 METADATA["journal"], METADATA["license"], "fulltext_ready", "xml",
                 str(article), digest, str(article), digest, 1, 1, 7001),
            )
            conn.commit()
            conn.execute("UPDATE tasks SET content_status='fetched' WHERE pmid=7001")
            conn.commit()
            conn.close()
            with self.assertRaises(normalization.NormalizationError):
                normalization.normalize_one(7001, db_path=db, raw_root=raw_root, normalized_root=root / "normalized")
            conn = sqlite3.connect(db)
            conn.execute("UPDATE tasks SET content_status='fulltext_ready' WHERE pmid=7001")
            conn.commit()
            conn.close()
            result = normalization.normalize_one(
                7001, db_path=db, raw_root=raw_root, normalized_root=root / "normalized"
            )
            self.assertTrue(Path(result["output_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
