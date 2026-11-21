# -*- coding: utf-8 -*-
"""A-R3 closure: verified XML -> A6 -> A7 -> normalized -> temporary index -> retrieval.

The synthetic fixtures are generated here; nothing reads the real corpus, the
production index or the production task database.  The temporary index pair is
built under a throwaway directory, and the test asserts that the fixed B3 corpus
list and the ``converted/`` documents are untouched.

The retrieval-closure class needs the real local bge-m3 model (CPU) and is
skipped with an explicit reason when the model cache is missing; the gate and
conflict classes are pure synthetic and run everywhere.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "直肠癌文献爬取" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "index"))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b3_fakes  # noqa: E402
import build_index  # noqa: E402
import index_common  # noqa: E402
import ingest_normalized  # noqa: E402
import normalized_ingest  # noqa: E402

import a1_schema as schema  # noqa: E402
import content_validation as validation  # noqa: E402
import ingest_pmids as ingest  # noqa: E402
import normalization  # noqa: E402

B3_SOURCES = ROOT / "index" / "b3-sources.json"
MODEL_DIR = ROOT / ".model-cache" / "BAAI__bge-m3"

BODY_A = (
    "Total neoadjuvant therapy has become a common strategy for locally advanced "
    "rectal cancer, and the choice between induction chemotherapy and consolidation "
    "chemotherapy remains an open question in everyday practice. This synthetic "
    "paragraph exists only to give the fixture a body that exceeds the centrally "
    "defined minimum length so that the content validator classifies the artifact as "
    "parseable rather than truncated. "
)
BODY_B = (
    "Preoperative magnetic resonance imaging derived radiomics may discriminate "
    "mucinous from non mucinous rectal adenocarcinoma, and the reported area under "
    "the curve varies widely between single centre studies. This synthetic paragraph "
    "carries no clinical meaning and is repeated with slightly different wording so "
    "that the fixture body stays above the minimum length required by the validator. "
)


def jats_article(*, pmid, pmcid, doi, title, surname, given, year, journal,
                 sections):
    """A minimal JATS article that satisfies the A6 content and identity rules."""
    body = "\n".join(
        f"""    <sec id="s{index}">
      <title>{heading}</title>
      <p>{text}</p>
    </sec>"""
        for index, (heading, text) in enumerate(sections, start=1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<article article-type="research-article" dtd-version="1.2">
  <front>
    <journal-meta>
      <journal-title-group><journal-title>{journal}</journal-title></journal-title-group>
    </journal-meta>
    <article-meta>
      <article-id pub-id-type="pmid">{pmid}</article-id>
      <article-id pub-id-type="pmc">{pmcid}</article-id>
      <article-id pub-id-type="doi">{doi}</article-id>
      <title-group><article-title>{title}</article-title></title-group>
      <contrib-group>
        <contrib contrib-type="author">
          <name><surname>{surname}</surname><given-names>{given}</given-names></name>
        </contrib>
      </contrib-group>
      <pub-date pub-type="epub"><year>{year}</year><month>06</month></pub-date>
      <abstract><p>Synthetic fixture used only by the offline A-R3 closure tests.</p></abstract>
    </article-meta>
  </front>
  <body>
{body}
  </body>
</article>
"""


def article_xml(*, pmid, doi, title, surname="Walker", given="Anna K", year=2021,
                body=BODY_A, headings=("Introduction", "Methods", "Results",
                                       "Discussion"), pmcid=None,
                journal="Journal of A-R3 Synthetic Fixtures"):
    """A JATS article whose declared identity matches the task metadata."""
    return jats_article(
        pmid=pmid, pmcid=pmcid or f"PMC{pmid}", doi=doi, title=title,
        surname=surname, given=given, year=year, journal=journal,
        sections=[(heading, body) for heading in headings],
    )


class Fixture:
    """One task database, raw artifact directory and normalized directory."""

    def __init__(self, root):
        self.root = Path(root)
        self.db = self.root / "tasks.sqlite"
        self.raw_root = self.root / "corpus_raw"
        self.normalized_root = self.root / "normalized"
        self.converted = self.root / "converted"
        self.staging_db = self.root / "chroma_db.normalized-staging"
        self.manifest = self.root / "manifest.normalized-staging.json"
        self.normalized_root.mkdir(parents=True, exist_ok=True)
        self.converted.mkdir(parents=True, exist_ok=True)

    def add_task(self, *, pmid, pmcid, doi, title, surname, given, year, article):
        schema.migrate_to_v2(self.db)
        validation.ensure_validation_schema(self.db)
        ingest.ingest_pmids(
            [pmid], source="ar3-closure", batch_id=f"ar3-closure-{pmid}",
            db_path=self.db,
        )
        conn = sqlite3.connect(self.db)
        conn.execute(
            """UPDATE tasks SET pmcid=?, doi=?, title=?, year=?, authors=?,
               first_author_family=?, status='oa_resolved' WHERE pmid=?""",
            (pmcid, doi, title, year, f"{surname}, {given}", surname.casefold(), pmid),
        )
        conn.commit()
        conn.close()
        directory = self.raw_root / f"PMID_{pmid}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "article.xml").write_bytes(article.encode("utf-8"))

    def artifact_sha256(self, pmid):
        return hashlib.sha256(
            (self.raw_root / f"PMID_{pmid}" / "article.xml").read_bytes()
        ).hexdigest()

    def target(self, **overrides):
        values = {
            "normalized_root": self.normalized_root,
            "staging_db": self.staging_db,
            "manifest": self.manifest,
            "converted_roots": (self.converted,),
            "mapping_paths": (),
            "task_db": self.db,
        }
        values.update(overrides)
        return ingest_normalized.default_target(**values)


def make_case(root, pmid, *, title, doi, article=None, **metadata):
    """Create a task, its raw artifact, its A6 verdict and its normalized file."""
    article = article or article_xml(pmid=pmid, doi=doi, title=title, **{
        key: value for key, value in metadata.items()
        if key in {"surname", "given", "year", "body", "headings", "pmcid", "journal"}
    })
    fixture = Fixture(root)
    fixture.add_task(
        pmid=pmid, pmcid=metadata.get("pmcid") or f"PMC{pmid}", doi=doi,
        title=title, surname=metadata.get("surname", "Walker"),
        given=metadata.get("given", "Anna K"), year=metadata.get("year", 2021),
        article=article,
    )
    verdict, normalized = validate_and_normalize(fixture, pmid)
    return fixture, verdict, normalized, article


TITLE_A = "Consolidation chemotherapy after induction treatment in rectal cancer"

ARTICLE_A = article_xml(pmid=8101, doi="10.1000/norm-a", title=TITLE_A)
ARTICLE_B = article_xml(
    pmid=8102, doi="10.1000/norm-b",
    title="Preoperative radiomics for mucinous rectal adenocarcinoma",
    surname="Ibrahim", given="Yusuf", year=2022, body=BODY_B,
    headings=("Introduction", "Methods", "Results"),
)


def validate_and_normalize(fixture, pmid):
    verdict = validation.validate_task_content(
        pmid, db_path=fixture.db, raw_root=fixture.raw_root
    )
    if verdict.get("content_status") != validation.CONTENT_STATUS_READY:
        return verdict, None
    result = normalization.normalize_one(
        pmid, db_path=fixture.db, raw_root=fixture.raw_root,
        normalized_root=fixture.normalized_root,
    )
    return verdict, result


class NormalizedGateTests(unittest.TestCase):
    """The ingestable decision itself, without any model or index build."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def plan(self, fixture):
        return normalized_ingest.plan_ingestion(
            fixture.normalized_root,
            converted_roots=[fixture.converted],
            out_root=self.root / "staging",
            db_path=fixture.db,
        )

    def test_verified_document_is_the_only_ingestable_kind(self):
        fixture, verdict, normalized, article = make_case(
            self.root, 8101, title=TITLE_A, doi="10.1000/norm-a"
        )
        self.assertEqual(verdict["outcome"], "validated")
        self.assertEqual(verdict["content_status"], "fulltext_ready")
        self.assertEqual(verdict["file_valid"], 1)
        self.assertEqual(verdict["bibliographic_match"], 1)
        self.assertIsNotNone(normalized)

        plan = self.plan(fixture)
        entry = plan["entries"][0]
        self.assertEqual(entry.verification, "verified")
        self.assertTrue(entry.ingestable)
        self.assertEqual(entry.action, normalized_ingest.STAGE_ACTION)
        self.assertEqual(entry.identity["binding"], "bound")
        self.assertEqual(entry.identity["content_sha256"],
                         fixture.artifact_sha256(8101))
        self.assertEqual(fixture.artifact_sha256(8101),
                         hashlib.sha256(article.encode("utf-8")).hexdigest())

    def test_content_version_mismatch_is_only_a_candidate(self):
        fixture, _, normalized, _ = make_case(
            self.root, 8103, title=TITLE_A, doi="10.1000/norm-a"
        )
        self.assertIsNotNone(normalized)
        # The normalized document now claims a content version that is not the
        # one the canonical task still validates.
        output = Path(normalized["output_path"])
        output.write_text(
            output.read_text(encoding="utf-8").replace(
                f'content_sha256: "{normalized["content_sha256"]}"',
                'content_sha256: "not-the-validated-version"',
            ),
            encoding="utf-8",
        )
        plan = self.plan(fixture)
        entry = plan["entries"][0]
        self.assertEqual(entry.verification, "content_version_mismatch")
        self.assertFalse(entry.ingestable)
        self.assertEqual(entry.action, normalized_ingest.CANDIDATE_ACTION)

        applied = normalized_ingest.apply_plan(plan, require_verified=True)
        self.assertEqual(applied["summary"]["staged"], 0)
        self.assertEqual(applied["summary"]["refused_unverified"], 1)
        self.assertFalse((self.root / "staging").exists(),
                         "a refused candidate must not be copied in production mode")

        permissive = normalized_ingest.apply_plan(plan)
        self.assertEqual(permissive["summary"]["staged"], 1,
                         "the library-level copy still stages candidates for inspection")

    def test_front_matter_identity_conflict_is_refused(self):
        fixture, _, normalized, _ = make_case(
            self.root, 8104, title=TITLE_A, doi="10.1000/norm-a"
        )
        output = Path(normalized["output_path"])
        output.write_text(
            output.read_text(encoding="utf-8").replace(
                'doi: "10.1000/norm-a"', 'doi: "10.1000/somebody-else"'
            ),
            encoding="utf-8",
        )
        plan = self.plan(fixture)
        entry = plan["entries"][0]
        self.assertEqual(entry.verification, "identity_conflict")
        self.assertFalse(entry.ingestable)
        self.assertIn("DOI", entry.verification_reason)

    def test_validation_failure_produces_no_normalized_document(self):
        wrong = article_xml(
            pmid=9999, doi="10.9999/other",
            title="An unrelated synthetic article about something else entirely",
            surname="Nobody", given="No One", year=1999,
        )
        fixture, verdict, normalized, _ = make_case(
            self.root, 8105, title="Synthetic article 8105",
            doi="10.1000/norm-8105", article=wrong,
        )
        self.assertIsNone(normalized, "a failed A6 must not produce a normalized document")
        self.assertEqual(verdict["file_valid"], 1)
        self.assertEqual(verdict["bibliographic_match"], 0)
        self.assertNotEqual(verdict.get("content_status"), "fulltext_ready")
        self.assertFalse(any(fixture.normalized_root.glob("*.md")))

        plan = self.plan(fixture)
        self.assertEqual(plan["entries"], [])
        self.assertEqual(plan["summary"]["stage"], 0)

    def test_matching_content_under_two_names_is_one_source_with_an_alias(self):
        fixture, _, normalized, _ = make_case(
            self.root, 8106, title=TITLE_A, doi="10.1000/norm-a"
        )
        original = Path(normalized["output_path"])
        copy = fixture.normalized_root / "PMID_8106_copy.md"
        copy.write_bytes(original.read_bytes())
        plan = self.plan(fixture)
        self.assertEqual(len(plan["entries"]), 2)
        self.assertTrue(all(entry.ingestable for entry in plan["entries"]))
        selected, report = ingest_normalized.select_documents(fixture.target())
        self.assertEqual(report["selected_sources"], 2)
        self.assertEqual(report["canonical_sources"], 1)
        self.assertEqual(report["alias_sources"], 1)
        canonical = next(item for item in selected if item["is_canonical"])
        alias = next(item for item in selected if not item["is_canonical"])
        self.assertEqual(canonical["source"], "PMID_8106.md")
        self.assertEqual(alias["dup_of"], "PMID_8106.md")

    def test_same_paper_with_different_content_requires_an_explicit_choice(self):
        fixture, _, normalized, _ = make_case(
            self.root, 8107, title=TITLE_A, doi="10.1000/norm-a"
        )
        original = Path(normalized["output_path"])
        other = fixture.normalized_root / "PMID_8107_other_source.md"
        other.write_text(
            original.read_text(encoding="utf-8").replace(
                "Total neoadjuvant therapy", "A different rendering of"
            ),
            encoding="utf-8",
        )
        with self.assertRaises(ingest_normalized.NormalizedIndexError) as caught:
            ingest_normalized.select_documents(fixture.target())
        self.assertIn("--prefer", str(caught.exception))

        selected, report = ingest_normalized.select_documents(
            fixture.target(), prefer="PMID_8107.md"
        )
        self.assertEqual(report["selected_sources"], 1)
        self.assertEqual(selected[0]["source"], "PMID_8107.md")

    def test_production_and_b3_targets_are_refused(self):
        fixture, _, _, _ = make_case(
            self.root, 8108, title=TITLE_A, doi="10.1000/norm-a"
        )
        for overrides in (
            {"staging_db": ROOT / "index" / "chroma_db"},
            {"manifest": ROOT / "index" / "manifest.json"},
            {"staging_db": ROOT / "index" / "chroma_db.v2-staging"},
            {"manifest": ROOT / "index" / "manifest.b3.partial.json"},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ingest_normalized.NormalizedIndexError):
                    ingest_normalized.validate_target(fixture.target(**overrides))

    def test_apply_never_overwrites_an_existing_staged_document(self):
        fixture, _, _, _ = make_case(
            self.root, 8109, title=TITLE_A, doi="10.1000/norm-a"
        )
        staging = self.root / "staging"
        plan = normalized_ingest.plan_ingestion(
            fixture.normalized_root, converted_roots=[fixture.converted],
            out_root=staging, db_path=fixture.db,
        )
        first = normalized_ingest.apply_plan(plan, require_verified=True)
        self.assertEqual(first["summary"]["staged"], 1)
        target = staging / "PMID_8109.md"
        published = target.read_bytes()
        self.assertGreater(len(published), 0)

        second = normalized_ingest.apply_plan(plan, require_verified=True)
        self.assertEqual(second["summary"]["staged"], 0)
        self.assertEqual(second["summary"]["skipped_unchanged"], 1)

        target.write_text("# operator edit\n", encoding="utf-8")
        third = normalized_ingest.apply_plan(plan, require_verified=True)
        self.assertEqual(third["summary"]["conflicts"], 1)
        self.assertEqual(target.read_text(encoding="utf-8"), "# operator edit\n")


class NormalizedIndexRerunTests(unittest.TestCase):
    """Synthetic reruns prove the isolated ledger and rows move together."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.fixture = Fixture(self.root)
        self.backends = self._new_backends()
        self.addCleanup(self._cleanup)

    def _new_backends(self):
        return b3_fakes.FakeBackends(
            SimpleNamespace(staging_db=self.fixture.staging_db)
        )

    def _cleanup(self):
        if getattr(self, "backends", None) is not None:
            self.backends.close()
        try:
            self._tmp.cleanup()
        except OSError:  # pragma: no cover - Windows handle latency
            pass

    def _add_paper(self, pmid, title=None):
        title = title or f"Synthetic normalized paper {pmid}"
        self.fixture.add_task(
            pmid=pmid,
            pmcid=f"PMC{pmid}",
            doi=f"10.1000/norm-{pmid}",
            title=title,
            surname="Walker",
            given="Anna K",
            year=2021,
            article=article_xml(
                pmid=pmid,
                doi=f"10.1000/norm-{pmid}",
                title=title,
            ),
        )
        verdict, normalized = validate_and_normalize(self.fixture, pmid)
        self.assertEqual(verdict["content_status"], "fulltext_ready")
        self.assertIsNotNone(normalized)
        return Path(normalized["output_path"])

    def _build(self):
        return ingest_normalized.build(
            self.fixture.target(), device="cpu", backends=self.backends
        )

    def _manifest(self):
        return json.loads(self.fixture.manifest.read_text(encoding="utf-8"))

    def _ready(self):
        return json.loads(
            Path(str(self.fixture.manifest) + ".ready.json").read_text(
                encoding="utf-8"
            )
        )

    def test_selection_shrink_and_all_invalid_remove_old_rows(self):
        first = self._add_paper(8201)
        second = self._add_paper(8202)
        self._build()

        second.unlink()
        report = self._build()
        manifest = self._manifest()
        self.assertEqual(set(manifest["files"]), {first.name})
        self.assertEqual(report["reused_sources"], 1)
        self.assertEqual(
            set(self.backends.collection.get(include=["metadatas"])["ids"]),
            {
                row_id
                for row_id in self.backends.collection.get(
                    where={"source": first.name}, include=["metadatas"]
                )["ids"]
            },
        )
        self.assertEqual(self._ready()["corpus_source_count"], 1)

        first.unlink()
        self._build()
        manifest = self._manifest()
        self.assertEqual(manifest["files"], {})
        self.assertEqual(manifest["total_chunks"], 0)
        self.assertEqual(self.backends.collection.count(), 0)
        self.assertEqual(self._ready()["corpus_source_count"], 0)

    def test_canonical_change_removes_old_source_rows(self):
        original = self._add_paper(8203)
        self._build()
        alias = self.fixture.normalized_root / "PMID_8203_copy.md"
        alias.write_bytes(original.read_bytes())
        self._build()

        original.unlink()
        self._build()
        manifest = self._manifest()
        self.assertEqual(set(manifest["files"]), {alias.name})
        self.assertNotIn("dup_of", manifest["files"][alias.name])
        rows = self.backends.collection.get(include=["metadatas"])
        self.assertEqual({meta["source"] for meta in rows["metadatas"]}, {alias.name})

    def test_current_manifest_with_missing_rows_is_reencoded(self):
        source = self._add_paper(8204)
        self._build()
        collection = self.backends.collection
        rows = collection.get(where={"source": source.name}, include=["metadatas"])
        collection.delete(ids=[rows["ids"][0]])
        report = self._build()
        self.assertEqual(report["encoded_sources"], 1)
        self.assertEqual(
            len(collection.get(where={"source": source.name}, include=["metadatas"])["ids"]),
            self._manifest()["files"][source.name]["chunks"],
        )

    def test_interrupted_write_leaves_partial_and_next_run_repairs_it(self):
        source = self._add_paper(8205)
        crashing = b3_fakes.CrashingCollection(
            self.backends.collection, fail_on_upsert=1
        )
        self.backends.wrap_collection(crashing)
        with self.assertRaises(RuntimeError):
            self._build()
        self.assertEqual(self._manifest()["status"], "partial")
        self.assertFalse(Path(str(self.fixture.manifest) + ".ready.json").exists())

        self.backends.close()
        self.backends = self._new_backends()
        report = self._build()
        self.assertEqual(report["encoded_sources"], 1)
        self.assertEqual(self._manifest()["status"], "complete")
        self.assertTrue(Path(str(self.fixture.manifest) + ".ready.json").exists())
        self.assertEqual(self._manifest()["files"][source.name]["chunks"],
                         self._ready()["total_chunks"])


@unittest.skipUnless(MODEL_DIR.is_dir(), f"local model cache missing: {MODEL_DIR}")
class NormalizedIndexClosureTests(unittest.TestCase):
    """The real chain: verified XML -> normalized -> index -> retrieval.

    Uses the real local bge-m3 model on CPU so the retrieval result is a real
    semantic lookup and not a stub.  One build is shared by the whole class.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.fixture = Fixture(cls.root)
        # Two papers so the retrieval result has to discriminate between them.
        cls.fixture.add_task(
            pmid=8101, pmcid="PMC8101", doi="10.1000/norm-a",
            title=TITLE_A,
            surname="Walker", given="Anna K", year=2021, article=ARTICLE_A,
        )
        cls.fixture.add_task(
            pmid=8102, pmcid="PMC8102", doi="10.1000/norm-b",
            title="Preoperative radiomics for mucinous rectal adenocarcinoma",
            surname="Ibrahim", given="Yusuf", year=2022, article=ARTICLE_B,
        )
        cls.verdicts = {}
        for pmid in (8101, 8102):
            verdict, normalized = validate_and_normalize(cls.fixture, pmid)
            cls.verdicts[pmid] = (verdict, normalized)
        cls.source_bytes_before = b3_sources_bytes()
        cls.converted_before = sorted(
            path.name for path in cls.fixture.converted.iterdir()
        )
        cls.report = ingest_normalized.build(cls.fixture.target(), device="cpu")
        cls.manifest = json.loads(cls.fixture.manifest.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        try:
            import rag_core

            rag_core.release_client(cls.fixture.staging_db)
        except Exception:  # noqa: BLE001 - best effort handle release
            pass
        b3_fakes.release_chroma(cls.fixture.staging_db)
        try:
            cls._tmp.cleanup()
        except OSError:  # pragma: no cover - Windows handle latency
            pass

    def test_both_validated_documents_reached_the_temporary_index(self):
        for pmid in (8101, 8102):
            verdict, normalized = self.verdicts[pmid]
            self.assertEqual(verdict["outcome"], "validated")
            self.assertEqual(verdict["content_status"], "fulltext_ready")
            self.assertIsNotNone(normalized)
        self.assertEqual(self.report["canonical_sources"], 2)
        self.assertEqual(self.report["encoded_sources"], 2)
        self.assertEqual(
            set(self.manifest["files"]), {"PMID_8101.md", "PMID_8102.md"}
        )
        self.assertEqual(self.manifest["status"], "complete")
        self.assertEqual(
            self.manifest["total_chunks"],
            sum(entry["chunks"] for entry in self.manifest["files"].values()),
        )
        self.assertGreater(self.manifest["total_chunks"], 0)

    def test_stored_rows_use_the_shared_passage_v2_representation(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        client = chromadb.PersistentClient(
            path=str(self.fixture.staging_db),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_collection("papers")
        rows = collection.get(include=["metadatas", "documents"])
        self.assertEqual(len(rows["ids"]), self.manifest["total_chunks"])
        bodies = {}
        for pmid in (8101, 8102):
            body = ingest_normalized.strip_front_matter(
                (self.fixture.normalized_root / f"PMID_{pmid}.md").read_text(
                    encoding="utf-8"
                )
            )
            bodies[f"PMID_{pmid}.md"] = body
        for metadata, document in zip(rows["metadatas"], rows["documents"]):
            self.assertEqual(
                metadata["embedding_text_version"],
                build_index.EMBEDDING_TEXT_VERSION,
            )
            self.assertEqual(metadata["passage_origin"], "normalized")
            self.assertEqual(metadata["section_type"],
                             build_index.section_type(metadata["section"]))
            self.assertTrue(metadata["validated_content_sha256"])
            self.assertFalse(document.startswith("Title: "),
                             "the stored passage stays the raw chunk text")
            self.assertIn(document, bodies[metadata["source"]],
                          "every stored passage is a verbatim chunk of its document")
        b3_fakes.release_chroma(self.fixture.staging_db)

    def test_the_same_body_chunks_identically_to_a_converted_document(self):
        from transformers import AutoTokenizer

        source = self.fixture.normalized_root / "PMID_8101.md"
        body = ingest_normalized.strip_front_matter(
            source.read_text(encoding="utf-8")
        )
        prepared = build_index.prepare_source(
            "PMID_8101.md",
            index_common.sha256_file(source),
            body,
            AutoTokenizer.from_pretrained(str(MODEL_DIR)),
            ingest_normalized.document_meta(
                {"title": "Consolidation chemotherapy after induction treatment in "
                          "rectal cancer"},
                "PMID_8101.md",
            ),
            False,
        )
        self.assertIsNotNone(prepared)
        self.assertEqual(len(prepared["ids"]),
                         self.manifest["files"]["PMID_8101.md"]["chunks"])
        for value in prepared["embedding_docs"]:
            self.assertTrue(value.startswith(
                "Title: Consolidation chemotherapy after induction treatment in "
                "rectal cancer\nSection: "
            ))

    def test_retrieval_returns_the_normalized_passage(self):
        import rag_core

        hits = rag_core.retrieve(
            "induction chemotherapy versus consolidation chemotherapy for locally "
            "advanced rectal cancer",
            k=3, db_dir=self.fixture.staging_db,
            manifest_path=self.fixture.manifest, model_dir=MODEL_DIR,
            device="cpu",
        )
        self.assertTrue(hits, "the temporary normalized index must be queryable")
        self.assertEqual(hits[0]["source"], "PMID_8101.md")
        self.assertIn("consolidation", hits[0]["snippet"].lower())
        self.assertEqual(hits[0]["doi"], "10.1000/norm-a")
        self.assertEqual(
            hits[0]["embedding_text_version"], build_index.EMBEDDING_TEXT_VERSION
        )
        b3_fakes.release_chroma(self.fixture.staging_db)

    def test_rebuild_is_idempotent_and_reuses_every_committed_source(self):
        before = json.loads(self.fixture.manifest.read_text(encoding="utf-8"))
        report = ingest_normalized.build(self.fixture.target(), device="cpu")
        after = json.loads(self.fixture.manifest.read_text(encoding="utf-8"))
        self.assertEqual(report["encoded_sources"], 0)
        self.assertEqual(report["reused_sources"], 2)
        self.assertEqual(after["total_chunks"], before["total_chunks"])
        self.assertEqual(set(after["files"]), set(before["files"]))
        b3_fakes.release_chroma(self.fixture.staging_db)

    def test_the_fixed_b3_corpus_and_converted_are_untouched(self):
        self.assertEqual(b3_sources_bytes(), self.source_bytes_before)
        self.assertEqual(
            sorted(path.name for path in self.fixture.converted.iterdir()),
            self.converted_before,
        )
        self.assertEqual(self.converted_before, [])

    def test_the_ready_marker_describes_the_normalized_corpus_not_production(self):
        marker = json.loads(
            Path(str(self.fixture.manifest) + ".ready.json").read_text(encoding="utf-8")
        )
        self.assertEqual(marker["status"], "ready")
        self.assertEqual(marker["db"], str(self.fixture.staging_db))
        self.assertEqual(marker["corpus_source_count"], 2)
        self.assertEqual(marker["total_chunks"], self.manifest["total_chunks"])
        converted_fingerprint = index_common.corpus_fingerprint(
            index_common.hashed_corpus(
                index_common.markdown_files(
                    ROOT / "converted", ROOT / "converted" / "已入库"
                )
            )
        )
        self.assertNotEqual(
            marker["corpus_fingerprint"], converted_fingerprint,
            "a normalized pair must not look publishable to the production entry point",
        )


def b3_sources_bytes():
    return B3_SOURCES.read_bytes() if B3_SOURCES.is_file() else None


if __name__ == "__main__":
    unittest.main()
