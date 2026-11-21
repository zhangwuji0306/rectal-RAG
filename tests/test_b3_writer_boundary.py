"""Offline regression tests for the index write boundary and writer consistency.

Covers remediation items R3 (publication entry points), R4 (uniform writer
representation) and R5 (staging-capable evaluation).  Everything runs against
temporary directories; the production index is only ever read.
"""

import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "index"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b3_fakes  # noqa: E402
import build_index  # noqa: E402
import index_common  # noqa: E402
import publish_index  # noqa: E402
import rag_core  # noqa: E402


def make_manifest(directory, files, model="BAAI/bge-m3"):
    """Create a manifest plus a matching Chroma collection in ``directory``."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    db = Path(directory)
    db.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(db), settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection("papers",
                                                 metadata={"hnsw:space": "cosine"})
    ids, documents, metadatas, embeddings = [], [], [], []
    entries = {}
    for source, chunks in files.items():
        digest = index_common.sha256_file(
            _write_source(directory, source)
        ) if chunks else "0" * 64
        entries[source] = {
            "sha256": digest,
            "chunks": chunks,
            "mac": "unknown",
            "repr": "0" * 64,
            "embedding_text_version": build_index.EMBEDDING_TEXT_VERSION,
        }
        for index in range(chunks):
            ids.append(f"row-{source}-{index}")
            documents.append(f"passage {index} of {source}")
            metadatas.append({
                "source": source,
                "chunk": index,
                "embedding_text_version": build_index.EMBEDDING_TEXT_VERSION,
            })
            embeddings.append([1.0, float(index + 1)])
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas,
                       embeddings=embeddings)
    manifest = {
        "version": 2,
        "status": "complete",
        "model": model,
        "embedding_text_version": build_index.EMBEDDING_TEXT_VERSION,
        "files": entries,
        "total_chunks": sum(files.values()),
        "built_at": "test",
    }
    manifest_path = db.parent / f"{db.name}.manifest.json"
    index_common.write_json_atomic(manifest_path, manifest)
    return db, manifest_path, manifest


def _write_source(directory, name):
    path = Path(directory).parent / "corpus" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {name}\n\nbody of {name}\n", encoding="utf-8")
    return path


def write_marker(manifest_path, manifest, db, corpus_fingerprint="fp"):
    marker = {
        "status": "ready",
        "manifest_sha256": index_common.sha256_file(manifest_path),
        "total_chunks": manifest["total_chunks"],
        "corpus_fingerprint": corpus_fingerprint,
    }
    index_common.write_json_atomic(
        Path(str(manifest_path) + ".ready.json"), marker
    )
    return marker


class WriteBoundaryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        b3_fakes.release_chroma(self.root)
        try:
            self._tmp.cleanup()
        except PermissionError:  # pragma: no cover
            pass

    def parser(self):
        return argparse.ArgumentParser()

    def resolve(self, argv, index_dir=None):
        parser = build_index.build_parser()
        args = parser.parse_args(argv)
        with mock.patch.object(build_index, "INDEX_DIR", index_dir or self.root), \
                mock.patch.object(build_index, "CHROMA_DIR", ROOT / "index" / "chroma_db"), \
                mock.patch.object(build_index, "MANIFEST_PATH", ROOT / "index" / "manifest.json"):
            return build_index.resolve_run_paths(args, parser)

    def test_force_with_limit_is_rejected(self):
        with self.assertRaises(SystemExit), \
                contextlib.redirect_stderr(io.StringIO()):
            build_index.main(["--force", "--limit", "1"])

    def test_half_specified_custom_paths_are_rejected(self):
        for argv in (["--db", str(self.root / "db")],
                     ["--manifest", str(self.root / "m.json")]):
            with self.subTest(argv=argv):
                with self.assertRaises(SystemExit) as caught, \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.resolve(argv)
                self.assertEqual(caught.exception.code, 2)

    def test_custom_paths_may_not_alias_production(self):
        production_db = ROOT / "index" / "chroma_db"
        aliases = [
            str(production_db),
            str(production_db / ".." / "chroma_db"),
            str(production_db).upper(),
        ]
        for alias in aliases:
            with self.subTest(alias=alias):
                with self.assertRaises(SystemExit), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.resolve(["--db", alias,
                                  "--manifest", str(self.root / "m.json")])

    def test_custom_pair_is_accepted_and_not_staged(self):
        run_db, run_manifest, staged = self.resolve([
            "--db", str(self.root / "db"),
            "--manifest", str(self.root / "m.json"),
        ])
        self.assertFalse(staged)
        self.assertEqual(run_db, self.root / "db")
        self.assertEqual(run_manifest, self.root / "m.json")

    def test_plain_force_builds_a_staging_pair_under_the_index_dir(self):
        run_db, run_manifest, staged = self.resolve(["--force"], index_dir=self.root)
        self.assertTrue(staged)
        self.assertTrue(str(run_db).startswith(str(self.root)))
        self.assertNotEqual(run_db, ROOT / "index" / "chroma_db")
        self.assertNotEqual(run_manifest, ROOT / "index" / "manifest.json")

    def test_incremental_default_run_targets_production(self):
        run_db, run_manifest, staged = self.resolve([])
        self.assertFalse(staged)
        self.assertEqual(run_db, ROOT / "index" / "chroma_db")
        self.assertEqual(run_manifest, ROOT / "index" / "manifest.json")

    def test_partial_corpus_coverage_is_not_a_complete_rebuild(self):
        manifest = {"files": {"a.md": {"chunks": 1}, "b.md": {"chunks": 1}}}
        self.assertTrue(build_index.verify_complete_corpus(manifest, {"a.md", "b.md"}))
        with self.assertRaises(build_index.RepresentationMismatch):
            build_index.verify_complete_corpus(manifest, {"a.md", "b.md", "c.md"})
        with self.assertRaises(build_index.RepresentationMismatch):
            build_index.verify_complete_corpus(manifest, {"a.md"})


class RepresentationGuardTests(unittest.TestCase):
    def test_legacy_manifest_refuses_a_v2_write(self):
        manifest = {"files": {"a.md": {"chunks": 5}}}
        with self.assertRaises(build_index.RepresentationMismatch):
            build_index.assert_representation_compatible(manifest, {}, "target")

    def test_rows_without_v2_metadata_refuse_a_v2_write(self):
        manifest = {
            "files": {"a.md": {"chunks": 5}},
            "embedding_text_version": build_index.EMBEDDING_TEXT_VERSION,
        }
        with self.assertRaises(build_index.RepresentationMismatch):
            build_index.assert_representation_compatible(
                manifest, {None: 5}, "target"
            )
        with self.assertRaises(build_index.RepresentationMismatch):
            build_index.assert_representation_compatible(
                manifest, {"passage-v1": 5}, "target"
            )

    def test_v2_pair_is_accepted_and_an_empty_index_is_allowed(self):
        manifest = {
            "files": {"a.md": {"chunks": 5}},
            "embedding_text_version": build_index.EMBEDDING_TEXT_VERSION,
        }
        self.assertTrue(build_index.assert_representation_compatible(
            manifest, {build_index.EMBEDDING_TEXT_VERSION: 5}, "target"
        ))
        self.assertTrue(build_index.assert_representation_compatible({}, {}, "target"))

    def test_title_change_invalidates_a_content_identical_entry(self):
        meta = {"title": "Original title"}
        signature = build_index.representation_signature(meta)
        entry = build_index.manifest_entry("a.md", "a" * 64, 3, meta)
        self.assertTrue(build_index.entry_is_current(entry, "a" * 64, signature))
        renamed = build_index.representation_signature({"title": "Renamed title"})
        self.assertFalse(build_index.entry_is_current(entry, "a" * 64, renamed))
        self.assertFalse(build_index.entry_is_current(entry, "b" * 64, signature))

    def test_representation_version_is_part_of_the_entry(self):
        entry = build_index.manifest_entry("a.md", "a" * 64, 3, {"title": "t"})
        self.assertEqual(entry["embedding_text_version"],
                         build_index.EMBEDDING_TEXT_VERSION)
        self.assertIn("repr", entry)


class IngestorRepresentationTests(unittest.TestCase):
    """R4: every writer must use the same passage-v2 representation."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        b3_fakes.release_chroma(self.root)
        try:
            self._tmp.cleanup()
        except PermissionError:  # pragma: no cover
            pass

    def test_shared_write_path_encodes_v2_text_and_stores_v2_metadata(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        db = self.root / "chroma_db"
        client = chromadb.PersistentClient(
            path=str(db), settings=ChromaSettings(anonymized_telemetry=False)
        )
        collection = client.get_or_create_collection("papers",
                                                     metadata={"hnsw:space": "cosine"})
        text = "# Methods\n\n" + b3_fakes.paragraph("m", 900) + "\n"
        sha = "c" * 64
        meta = {"title": "A rectal cancer paper", "mac": "yes", "doi": "10.1/x"}
        prepared = build_index.prepare_source(
            "paper.md", sha, text, b3_fakes.FakeTokenizer(), meta, False
        )
        captured = {}

        class FakeModel:
            device = "cpu"

            def encode(self, texts, **_kwargs):
                captured["texts"] = list(texts)
                return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

        entry = build_index.encode_and_store(collection, FakeModel(), prepared, 48)

        self.assertTrue(captured["texts"], "the writer must encode the v2 text")
        for value in captured["texts"]:
            self.assertTrue(value.startswith(
                "Title: A rectal cancer paper\nSection: Methods\nContent:\n"
            ))
        rows = collection.get(include=["metadatas", "documents"])
        for metadata, document in zip(rows["metadatas"], rows["documents"]):
            self.assertEqual(metadata["embedding_text_version"],
                             build_index.EMBEDDING_TEXT_VERSION)
            self.assertEqual(metadata["paper_id"], "doi:10.1/x")
            self.assertEqual(metadata["section_type"], "methods")
            self.assertTrue(document in text)
        self.assertEqual(entry["chunks"], len(captured["texts"]))
        self.assertEqual(entry["embedding_text_version"],
                         build_index.EMBEDDING_TEXT_VERSION)
        self.assertIn("repr", entry)
        stored = collection.get(include=["documents"])["documents"]
        self.assertFalse(any(value.startswith("Title: ") for value in stored),
                         "the stored passage must stay the raw chunk text")
        b3_fakes.release_chroma(db)

    def test_live_ingestor_no_longer_encodes_raw_chunk_text(self):
        source = (ROOT / ".claude" / "batch_convert.py").read_text(encoding="utf-8")
        self.assertNotIn("model.encode(docs", source)
        self.assertIn("bi.prepare_source(", source)
        self.assertIn("bi.encode_and_store(", source)

    def test_plain_builder_write_path_is_the_same_function(self):
        source = (ROOT / "index" / "build_index.py").read_text(encoding="utf-8")
        self.assertIn("prepared = prepare_source(", source)
        self.assertEqual(source.count("build_embedding_text(meta.get"), 1,
                         "the embedding text is built in exactly one place")


class PublishTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        b3_fakes.release_chroma(self.root)
        try:
            self._tmp.cleanup()
        except PermissionError:  # pragma: no cover
            pass

    def pair(self, name, files):
        directory = self.root / name
        db, manifest_path, manifest = make_manifest(directory, files)
        # Release the Chroma handles: publication renames the directory.
        b3_fakes.release_chroma(db)
        return db, manifest_path, manifest

    def test_publish_requires_a_ready_marker(self):
        staging_db, staging_manifest, _ = self.pair("staging", {"a.md": 2})
        production_db, production_manifest, _ = self.pair("production", {"old.md": 1})
        with self.assertRaises(publish_index.PublishRefused):
            publish_index.publish(staging_db, staging_manifest, production_db,
                                  production_manifest,
                                  journal_path=self.root / "journal.json",
                                  manifest_backup_dir=self.root / "backups")
        self.assertEqual(
            json.loads(production_manifest.read_text(encoding="utf-8"))["files"].keys(),
            {"old.md"},
        )

    def test_publish_refuses_a_manifest_that_does_not_cover_the_corpus(self):
        staging_db, staging_manifest, manifest = self.pair("staging", {"a.md": 2})
        write_marker(staging_manifest, manifest, staging_db, corpus_fingerprint="fp")
        production_db, production_manifest, _ = self.pair("production", {"old.md": 1})
        with self.assertRaises(publish_index.PublishRefused) as caught:
            publish_index.publish(staging_db, staging_manifest, production_db,
                                  production_manifest,
                                  journal_path=self.root / "journal.json",
                                  corpus_names={"a.md", "b.md"},
                                  corpus_fingerprint="fp",
                                  manifest_backup_dir=self.root / "backups")
        self.assertIn("does not cover the corpus", str(caught.exception))

    def test_publish_refuses_a_corpus_that_changed_after_the_build(self):
        staging_db, staging_manifest, manifest = self.pair("staging", {"a.md": 2})
        write_marker(staging_manifest, manifest, staging_db, corpus_fingerprint="old")
        production_db, production_manifest, _ = self.pair("production", {"old.md": 1})
        with self.assertRaises(publish_index.PublishRefused) as caught:
            publish_index.publish(staging_db, staging_manifest, production_db,
                                  production_manifest,
                                  journal_path=self.root / "journal.json",
                                  corpus_names={"a.md"}, corpus_fingerprint="new",
                                  manifest_backup_dir=self.root / "backups")
        self.assertIn("corpus changed", str(caught.exception))

    def test_publish_swaps_the_pair_and_retains_a_rollback_copy(self):
        staging_db, staging_manifest, manifest = self.pair("staging", {"a.md": 2})
        write_marker(staging_manifest, manifest, staging_db)
        production_db, production_manifest, old = self.pair("production", {"old.md": 1})
        journal_path = self.root / "journal.json"

        journal = publish_index.publish(
            staging_db, staging_manifest, production_db, production_manifest,
            journal_path=journal_path, manifest_backup_dir=self.root / "backups",
        )

        self.assertEqual(journal["state"], "completed")
        self.assertFalse(staging_db.exists())
        self.assertTrue(production_db.is_dir())
        published = json.loads(production_manifest.read_text(encoding="utf-8"))
        self.assertEqual(set(published["files"]), {"a.md"})
        self.assertTrue(Path(journal["previous_db"]).is_dir(),
                        "the previous production database is retained")
        self.assertTrue(Path(journal["previous_manifest"]).is_file())
        b3_fakes.release_chroma(production_db)
        preflight = rag_core.index_preflight(db_dir=production_db,
                                             manifest_path=production_manifest)
        self.assertEqual(preflight["collection_count"], 2)

    def test_publish_rolls_back_when_the_manifest_replace_fails(self):
        staging_db, staging_manifest, manifest = self.pair("staging", {"a.md": 2})
        write_marker(staging_manifest, manifest, staging_db)
        production_db, production_manifest, _ = self.pair("production", {"old.md": 1})
        journal_path = self.root / "journal.json"
        real_replace = publish_index.os.replace

        def failing_replace(source, target):
            if str(target).endswith("manifest.json"):
                raise OSError("injected manifest replace failure")
            return real_replace(source, target)

        with mock.patch.object(publish_index.os, "replace", side_effect=failing_replace):
            with self.assertRaises(OSError):
                publish_index.publish(
                    staging_db, staging_manifest, production_db, production_manifest,
                    journal_path=journal_path,
                    manifest_backup_dir=self.root / "backups",
                )

        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        self.assertEqual(journal["state"], "rolled-back")
        self.assertTrue(production_db.is_dir(), "production database restored")
        self.assertTrue(staging_db.is_dir(), "staging database restored")
        restored = json.loads(production_manifest.read_text(encoding="utf-8"))
        self.assertEqual(set(restored["files"]), {"old.md"})
        summary = rag_core.index_preflight(db_dir=production_db,
                                           manifest_path=production_manifest)
        self.assertEqual(summary["collection_count"], 1)
        b3_fakes.release_chroma(production_db)

    def test_recover_restores_an_interrupted_swap(self):
        staging_db, staging_manifest, manifest = self.pair("staging", {"a.md": 2})
        write_marker(staging_manifest, manifest, staging_db)
        production_db, production_manifest, _ = self.pair("production", {"old.md": 1})
        journal_path = self.root / "journal.json"
        previous_db = self.root / "production.previous-test"
        backup_manifest = self.root / "backups" / "manifest-pre-publish-test.json"
        backup_manifest.parent.mkdir(parents=True, exist_ok=True)
        backup_manifest.write_text(production_manifest.read_text(encoding="utf-8"),
                                   encoding="utf-8")
        # Interrupted after moving the production directory, before promotion.
        b3_fakes.rename_directory(production_db, previous_db)
        index_common.write_json_atomic(journal_path, {
            "state": "in-progress",
            "steps": ["production-db-moved"],
            "staging_db": str(staging_db),
            "staging_manifest": str(staging_manifest),
            "production_db": str(production_db),
            "production_manifest": str(production_manifest),
            "previous_db": str(previous_db),
            "previous_manifest": str(backup_manifest),
        })

        journal = publish_index.recover(journal_path)

        self.assertEqual(journal["state"], "rolled-back")
        self.assertTrue(production_db.is_dir())
        self.assertFalse(previous_db.exists())
        self.assertEqual(
            rag_core.index_preflight(db_dir=production_db,
                                     manifest_path=production_manifest)["collection_count"],
            1,
        )
        b3_fakes.release_chroma(production_db)

    def test_dry_run_publishes_nothing(self):
        staging_db, staging_manifest, manifest = self.pair("staging", {"a.md": 2})
        write_marker(staging_manifest, manifest, staging_db)
        production_db, production_manifest, _ = self.pair("production", {"old.md": 1})
        journal_path = self.root / "journal.json"

        journal = publish_index.publish(
            staging_db, staging_manifest, production_db, production_manifest,
            journal_path=journal_path, dry_run=True,
            manifest_backup_dir=self.root / "backups",
        )

        self.assertEqual(journal["state"], "dry-run")
        self.assertTrue(staging_db.is_dir())
        self.assertFalse(journal_path.exists())
        self.assertEqual(
            json.loads(production_manifest.read_text(encoding="utf-8"))["files"].keys(),
            {"old.md"},
        )

    def test_staging_and_production_must_not_nest(self):
        parent, manifest_path, _ = self.pair("parent", {"a.md": 1})
        with self.assertRaises(publish_index.PublishRefused):
            publish_index._validate_targets(parent / "child", manifest_path,
                                            parent, self.root / "m.json")


class EvaluatorManifestTests(unittest.TestCase):
    """R5: staging must be evaluable with its own paired manifest."""

    def test_evaluator_forwards_manifest_path(self):
        import evaluate_retrieval

        qrels = [{
            "query_id": "q1",
            "query": "mucinous adenocarcinoma prognosis",
            "split": "dev",
            "judgements": [{"doi": "10.1/a", "relevance": 3}],
        }]
        captured = {}

        def fake_retrieve(query, **kwargs):
            captured["query"] = query
            captured["kwargs"] = kwargs
            return [{"doi": "10.1/a", "source": "a.md", "title": "A", "year": "2020",
                     "journal": "J", "authors": "X", "mac": "yes", "section": "Results",
                     "score": 1.0}]

        qrels_path = ROOT / "index" / "_tmp_eval_qrels.jsonl"
        try:
            qrels_path.write_text(
                "\n".join(json.dumps(row) for row in qrels), encoding="utf-8"
            )
            argv = ["evaluate_retrieval.py", "--qrels", str(qrels_path),
                    "--db", "staging-db", "--manifest", "staging-manifest.json"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(evaluate_retrieval, "retrieve_literature",
                                      side_effect=fake_retrieve), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                evaluate_retrieval.main()
        finally:
            qrels_path.unlink(missing_ok=True)

        self.assertEqual(captured["kwargs"]["db_dir"], Path("staging-db"))
        self.assertEqual(captured["kwargs"]["manifest_path"], Path("staging-manifest.json"))
        result = json.loads(output.getvalue())
        self.assertEqual(result["splits"]["overall"]["total_queries"], 1)


if __name__ == "__main__":
    unittest.main()
