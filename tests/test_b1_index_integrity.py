"""Focused offline regression tests for B1 index integrity preflight."""

import json
import gc
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "index"))

import build_index  # noqa: E402
import chromadb  # noqa: E402
import rag_core  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402
from chromadb.api.shared_system_client import SharedSystemClient  # noqa: E402


class B1IndexIntegrityTests(unittest.TestCase):
    def tearDown(self):
        rag_core._MODEL_CACHE.clear()
        rag_core._COLLECTION_CACHE.clear()

    def test_current_index_passes_preflight(self):
        result = rag_core.index_preflight()

        self.assertEqual(result["collection"], "papers")
        self.assertEqual(result["collection_count"], 72344)
        self.assertEqual(result["source_count"], 2492)
        self.assertEqual(result["manifest_file_count"], 2499)
        self.assertEqual(result["manifest_total_chunks"], 72344)
        self.assertRegex(result["model_fingerprint"], r"^[0-9a-f]{64}$")

    def test_build_fails_when_manifest_exists_but_chroma_db_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_dir = root / "index"
            index_dir.mkdir()
            manifest_path = index_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "model": build_index.MODEL_NAME,
                        "files": {},
                        "total_chunks": 0,
                    }
                ),
                encoding="utf-8",
            )
            missing_db = index_dir / "chroma_db"
            converted = root / "converted"
            archive = converted / "已入库"
            converted.mkdir()
            archive.mkdir()

            with mock.patch.object(build_index, "INDEX_DIR", index_dir), \
                    mock.patch.object(build_index, "MANIFEST_PATH", manifest_path), \
                    mock.patch.object(build_index, "CHROMA_DIR", missing_db), \
                    mock.patch.object(build_index, "CONVERTED", converted), \
                    mock.patch.object(build_index, "ARCHIVE", archive), \
                    mock.patch.object(build_index, "LOCAL_MODEL_DIR", ROOT / ".model-cache" / "BAAI__bge-m3"), \
                    mock.patch.object(sys, "argv", ["build_index.py"]):
                with self.assertRaises(rag_core.IndexIntegrityError):
                    build_index.main()

            self.assertFalse(missing_db.exists(), "preflight must not create a DB")

    def test_db_and_collection_cache_keys_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_a = root / "db-a"
            db_b = root / "db-b"
            client_a = chromadb.PersistentClient(
                path=str(db_a), settings=ChromaSettings(anonymized_telemetry=False)
            )
            client_b = chromadb.PersistentClient(
                path=str(db_b), settings=ChromaSettings(anonymized_telemetry=False)
            )
            collection_a = client_a.get_or_create_collection("papers")
            collection_b = client_b.get_or_create_collection("papers")
            collection_a.add(
                ids=["a-1"], documents=["database A"],
                embeddings=[[1.0, 0.0, 0.0]], metadatas=[{"source": "A.md"}],
            )
            collection_b.add(
                ids=["b-1"], documents=["database B"],
                embeddings=[[0.0, 1.0, 0.0]], metadatas=[{"source": "B.md"}],
            )

            from_a = rag_core._get_collection(db_a)
            from_b = rag_core._get_collection(db_b)
            self.assertIsNot(from_a, from_b)
            self.assertEqual(from_a.get()["documents"], ["database A"])
            self.assertEqual(from_b.get()["documents"], ["database B"])

            other = client_a.get_or_create_collection("other")
            other.add(
                ids=["o-1"], documents=["other collection"],
                embeddings=[[0.0, 0.0, 1.0]], metadatas=[{"source": "O.md"}],
            )
            from_other = rag_core._get_collection(db_a, "other")
            self.assertIsNot(from_a, from_other)
            self.assertEqual(from_other.get()["documents"], ["other collection"])

            with mock.patch.object(rag_core, "SentenceTransformer", side_effect=lambda *a, **k: object()):
                model_a = rag_core._get_model(root / "model-a", "cpu", db_a, "papers")
                model_b = rag_core._get_model(root / "model-b", "cpu", db_b, "papers")
                self.assertIsNot(model_a, model_b)

            # PersistentClient keeps SQLite handles until its object graph is
            # collected on Windows; release all cached/client references before
            # TemporaryDirectory performs cleanup.
            rag_core._COLLECTION_CACHE.clear()
            del from_a, from_b, from_other, collection_a, collection_b, other
            del client_a, client_b
            for identifier in list(SharedSystemClient._identifier_to_system):
                if str(db_a) in identifier or str(db_b) in identifier:
                    del SharedSystemClient._identifier_to_system[identifier]
            gc.collect()


if __name__ == "__main__":
    unittest.main()
