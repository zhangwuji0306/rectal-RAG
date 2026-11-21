"""Offline regression tests for the B3 passage representation v2.

Covers the representation helpers shared by every writer, the isolation of the
staging paths from production, and the chunking behaviour whose 2026-09-04 fix
explains why the planned v2 chunk count exceeds the historical production
manifest.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "index"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b3_fakes  # noqa: E402
import rebuild_b3  # noqa: E402
from build_index import (  # noqa: E402
    CHUNKING_CONFIG,
    EMBEDDING_TEXT_VERSION,
    MIN_CHUNK_TOKENS,
    SECTION_TYPES,
    build_embedding_text,
    chunk_markdown,
    chunk_metadata,
    paper_id_for,
    prepare_source,
    representation_signature,
    section_type,
)


class B3PassageRepresentationTests(unittest.TestCase):
    def test_section_type_covers_required_vocabulary(self):
        examples = {
            "Abstract": "abstract",
            "1 Introduction": "introduction",
            "Materials and Methods": "methods",
            "Results > Primary outcome": "results",
            "Discussion": "discussion",
            "Conclusions": "conclusion",
            "References": "references",
            "Supplementary Materials": "supplement",
            "": "other",
        }
        for section, expected in examples.items():
            with self.subTest(section=section):
                self.assertEqual(section_type(section), expected)
        self.assertEqual(set(SECTION_TYPES), {
            "abstract", "introduction", "methods", "results", "discussion",
            "conclusion", "references", "supplement", "other",
        })

    def test_embedding_text_contains_context_and_raw_chunk(self):
        raw_chunk = "The observed association was statistically significant."
        embedded = build_embedding_text(
            "MRI biomarkers in rectal cancer", "Results > Survival", raw_chunk
        )

        self.assertTrue(embedded.startswith("Title: MRI biomarkers in rectal cancer\n"))
        self.assertIn("Section: Results > Survival", embedded)
        self.assertTrue(embedded.endswith(f"Content:\n{raw_chunk}"))
        self.assertEqual(raw_chunk, "The observed association was statistically significant.")

    def test_metadata_has_paper_and_representation_fields(self):
        meta = {"title": "A paper", "doi": "10.1000/ABC"}
        result = chunk_metadata(meta, "PMID_123.md", "Methods", 4, False)

        self.assertEqual(result["paper_id"], "doi:10.1000/abc")
        self.assertEqual(result["section_type"], "methods")
        self.assertEqual(result["embedding_text_version"], EMBEDDING_TEXT_VERSION)
        self.assertEqual(result["source"], "PMID_123.md")
        self.assertEqual(result["chunk"], 4)

    def test_paper_id_falls_back_to_source_without_doi(self):
        self.assertEqual(
            paper_id_for("legacy paper.md", {"doi": ""}),
            "source:legacy paper",
        )

    def test_prepared_rows_keep_raw_text_and_embed_the_context(self):
        text = "# Results\n\n" + b3_fakes.paragraph("r", 1000) + "\n"
        prepared = prepare_source(
            "PMID_9.md", "a" * 64, text, b3_fakes.FakeTokenizer(),
            {"title": "A title"}, False,
        )

        self.assertGreater(len(prepared["ids"]), 1)
        self.assertEqual(len(prepared["ids"]), len(prepared["documents"]))
        self.assertEqual(len(prepared["ids"]), len(prepared["embedding_docs"]))
        for index, embedding_text in enumerate(prepared["embedding_docs"]):
            self.assertIn("Title: A title\n", embedding_text)
            self.assertIn(f"Content:\n{prepared['documents'][index]}", embedding_text)
            self.assertTrue(prepared["metadatas"][index]["section_type"] == "results")
        self.assertEqual(prepared["ids"][0], "aaaaaaaaaaaa-000")

    def test_representation_signature_tracks_every_embedding_input(self):
        base = representation_signature({"title": "A"})
        self.assertEqual(base, representation_signature({"title": "A"}))
        self.assertNotEqual(base, representation_signature({"title": "B"}))
        self.assertEqual(CHUNKING_CONFIG["max_tokens"], 800)

    def test_staging_paths_are_isolated_from_production(self):
        production_db = ROOT / "index" / "chroma_db"
        production_manifest = ROOT / "index" / "manifest.json"
        paths = rebuild_b3.DEFAULT_PATHS
        self.assertNotEqual(paths.staging_db.resolve(), production_db.resolve())
        self.assertNotEqual(paths.manifest.resolve(), production_manifest.resolve())
        self.assertNotEqual(paths.progress.resolve(), production_manifest.resolve())
        self.assertNotEqual(paths.sources.resolve(), production_manifest.resolve())

    def test_continuous_source_schema_has_no_batch_assignment(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            paths = b3_fakes.make_paths(tmp)
            b3_fakes.write_corpus(paths.archive, {"PMID_7.md": "# A\n\n" + b3_fakes.paragraph("x", 900)})
            payload = b3_fakes.plan_sources(paths)
        self.assertTrue(payload["sources"])
        for item in payload["sources"]:
            self.assertNotIn("batch_id", item)
            self.assertIn("canonical_source", item)
            self.assertIn("estimated_chunks", item)
        self.assertNotIn("batches", payload)

    def test_overlong_paragraph_is_split_without_losing_its_tail(self):
        """The chunker must keep every token of an over-long unit.

        The pre-2026-09-04 implementation truncated such a unit to an estimated
        character budget, which is why the historical production manifest
        (72,344 chunks) is smaller than the planned v2 rebuild (79,239 chunks)
        for sources indexed before that fix.
        """
        tail = "TAILMARKER"
        paragraph = b3_fakes.paragraph("word", 3000) + tail
        chunks = chunk_markdown(paragraph, b3_fakes.FakeTokenizer())

        self.assertGreater(len(chunks), 3)
        joined = "".join(chunk["text"] for chunk in chunks)
        self.assertIn(tail, joined, "the end of an over-long paragraph is kept")
        self.assertGreaterEqual(
            max(len(chunk["text"]) for chunk in chunks), MIN_CHUNK_TOKENS
        )


if __name__ == "__main__":
    unittest.main()
