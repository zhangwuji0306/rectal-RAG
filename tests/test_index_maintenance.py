"""Synthetic integrity, live-ingest, and manifest reconciliation checks."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "index"))
sys.path.insert(0, str(ROOT / "tests"))

import b3_fakes  # noqa: E402
import build_index  # noqa: E402
import ingest_normalized  # noqa: E402
import index_common  # noqa: E402
import chromadb  # noqa: E402
import rag_core  # noqa: E402
import reconcile_manifest  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")


class _SyntheticBuildModel:
    def __init__(self):
        self.device = "cpu"
        self.tokenizer = b3_fakes.FakeTokenizer()
        self.calls = []

    def __getitem__(self, _index):
        return SimpleNamespace(tokenizer=self.tokenizer)

    def encode(self, texts, **_kwargs):
        texts = list(texts)
        self.calls.append(texts)
        return [[1.0, 0.0] for _ in texts]


class IndexMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(os.environ["TEMP"]).resolve()

    def workspace(self):
        temporary = tempfile.TemporaryDirectory(prefix="b1-index-", dir=self.temp_root)
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name).resolve()
        self.assertTrue(path.is_relative_to(self.temp_root))
        return path

    def model_dir(self, workspace):
        path = workspace / ".model-cache" / "BAAI__bge-m3"
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text('{"fixture": true}\n', encoding="utf-8")
        return path

    def manifest_base(self, model_dir, files, total_chunks):
        return {
            "version": 1,
            "model": build_index.MODEL_NAME,
            "embedding_text_version": build_index.EMBEDDING_TEXT_VERSION,
            "chunking": dict(build_index.CHUNKING_CONFIG),
            "model_fingerprint": rag_core.model_fingerprint(model_dir),
            "files": files,
            "total_chunks": total_chunks,
            "built_at": "fixture",
        }

    def seed_pair(self, workspace, documents, actual_counts, ledger_counts,
                  aliases=None, row_versions=None, total_override=None):
        index_dir = workspace / "index"
        db_dir = index_dir / "chroma_db"
        manifest_path = index_dir / "manifest.json"
        def release_clients():
            rag_core.release_client(db_dir)
            b3_fakes.release_chroma(db_dir)
        self.addCleanup(release_clients)
        converted = workspace / "converted"
        archive = converted / "已入库"
        model_dir = self.model_dir(workspace)
        archive.mkdir(parents=True, exist_ok=True)
        for name, text in documents.items():
            (archive / name).write_text(text, encoding="utf-8")

        files = {}
        aliases = aliases or {}
        for name, count in ledger_counts.items():
            text = documents[name]
            meta = {"title": Path(name).stem, "mac": "unknown"}
            files[name] = build_index.manifest_entry(
                name,
                hashlib.sha256((archive / name).read_bytes()).hexdigest(),
                count,
                meta,
                dup_of=aliases.get(name),
            )
        ledger_total = sum(ledger_counts.values()) if total_override is None else total_override
        _write_json(manifest_path, self.manifest_base(model_dir, files, ledger_total))

        def version_for(name, index):
            configured = (row_versions or {}).get(name, build_index.EMBEDDING_TEXT_VERSION)
            if isinstance(configured, (list, tuple)):
                return configured[index]
            return configured

        rows = [
            (name, i, version_for(name, i))
            for name, count in actual_counts.items()
            for i in range(count)
        ]
        if rows:
            client = chromadb.PersistentClient(
                path=str(db_dir), settings=ChromaSettings(anonymized_telemetry=False)
            )
            collection = client.get_or_create_collection(
                "papers", metadata={"hnsw:space": "cosine"}
            )
            collection.add(
                ids=[f"{name}-{i}" for name, i, _ in rows],
                documents=[f"stored passage {name} {i}" for name, i, _ in rows],
                embeddings=[[1.0, 0.0] for _ in rows],
                metadatas=[
                    {"source": name, "mac": "unknown", "embedding_text_version": version}
                    for name, _, version in rows
                ],
            )
            del collection, client
            b3_fakes.release_chroma(db_dir)
        return {
            "root": workspace,
            "index": index_dir,
            "db": db_dir,
            "manifest": manifest_path,
            "converted": converted,
            "archive": archive,
            "model": model_dir,
        }


    def without_persistent_database(self, pair, state):
        db = pair["db"]
        if state == "missing-directory":
            db.rename(pair["root"] / "saved-db")
            return
        shutil.rmtree(db)
        db.mkdir(parents=True)
        if state == "residual-file-only":
            (db / "chroma.sqlite3-wal").write_bytes(b"synthetic leftover")

    def write_synthetic_metadata(self, root, source_names):
        mapping_dir = root / ".claude" / "tmp"
        files = [
            {"filename": name, "title": Path(name).stem}
            for name in source_names
        ]
        _write_json(mapping_dir / "mapping.json", {"files": files})
        _write_json(mapping_dir / "mapping_pmid.json", {"files": []})
        return mapping_dir

    def cli_workspace(self, workspace, encode_enabled=False):
        """Copy only the real entry point and its importable local code to a fixture."""
        for relative in (".claude", "index", "converted/已入库", "papers", "flash-failed"):
            (workspace / relative).mkdir(parents=True, exist_ok=True)
        (workspace / ".claude" / "conversion-tracker.json").write_text(
            '{"converted": {}, "failed": {}}\n', encoding="utf-8"
        )
        shutil.copy2(ROOT / ".claude" / "batch_convert.py", workspace / ".claude" / "batch_convert.py")
        for name in ("build_index.py", "rag_core.py", "file_lock.py", "index_common.py"):
            shutil.copy2(ROOT / "index" / name, workspace / "index" / name)

        stubs = workspace / "test-stubs"
        stubs.mkdir()
        (stubs / "torch.py").write_text(
            """class _Cuda:\n"""
            """    @staticmethod\n"""
            """    def is_available(): return False\n"""
            """    @staticmethod\n"""
            """    def empty_cache(): pass\n"""
            """cuda = _Cuda()\n"""
            """def set_num_threads(_value): pass\n"""
            """def get_num_threads(): return 1\n""",
            encoding="utf-8",
        )
        if encode_enabled:
            encode_body = """        with open(os.environ['RAG_FAKE_ENCODE_LOG'], 'a', encoding='utf-8') as log:
            log.write(str(len(texts)) + chr(10))
        return [[1.0, 0.0] for _ in texts]
"""
        else:
            encode_body = """        Path(os.environ['RAG_FAKE_ENCODE_LOG']).write_text('called', encoding='utf-8')
        raise AssertionError('unchanged synthetic sources must not be encoded')
"""
        stub_source = (
            """import os
from pathlib import Path
from types import SimpleNamespace
class _Tokenizer:
    def encode(self, text, add_special_tokens=False): return list(range(len(text)))
    def decode(self, ids, **kwargs): return 'x' * len(ids)
class SentenceTransformer:
    def __init__(self, model, device='cpu'): self.device = device
    def __getitem__(self, _index): return SimpleNamespace(tokenizer=_Tokenizer())
    def encode(self, texts, **kwargs):
""" + encode_body
        )
        (stubs / "sentence_transformers.py").write_text(stub_source, encoding="utf-8")
        return stubs

    def run_live_cli(self, workspace, stubs):
        environment = dict(os.environ)
        environment.update({
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": str(stubs),
            "RAG_FAKE_ENCODE_LOG": str(workspace / "encode-called.txt"),
            "TEMP": str(self.temp_root),
            "TMP": str(self.temp_root),
        })
        return subprocess.run(
            [sys.executable, str(workspace / ".claude" / "batch_convert.py"), "--ingest-only"],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=environment,
            timeout=90,
        )

    def run_build(self, workspace, forbid_encoding=False):
        converted = workspace / "converted"
        archive = converted / "已入库"
        index_dir = workspace / "index"
        db_dir = index_dir / "chroma_db"
        manifest_path = index_dir / "manifest.json"
        model_dir = self.model_dir(workspace)
        model = _SyntheticBuildModel()
        if forbid_encoding:
            model.encode = mock.Mock(side_effect=AssertionError("unexpected encoding"))
        with mock.patch.multiple(
            build_index,
            ROOT=workspace,
            INDEX_DIR=index_dir,
            CHROMA_DIR=db_dir,
            MANIFEST_PATH=manifest_path,
            CONVERTED=converted,
            ARCHIVE=archive,
            LOCAL_MODEL_DIR=model_dir,
        ), mock.patch.object(build_index, "SentenceTransformer", return_value=model):
            build_index.main(["--device", "cpu"])
        b3_fakes.release_chroma(db_dir)
        return model

    def assert_build_rows_match_ledger(self, workspace):
        manifest_path = workspace / "index" / "manifest.json"
        db_dir = workspace / "index" / "chroma_db"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        client = chromadb.PersistentClient(
            path=str(db_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        collection = client.get_collection("papers")
        try:
            rows = collection.get(include=["documents", "metadatas"])
            count = collection.count()
            expected_sources = {
                source for source, entry in manifest["files"].items()
                if entry.get("chunks", 0)
            }
            self.assertEqual(count, manifest["total_chunks"])
            self.assertEqual(
                count,
                sum(entry.get("chunks", 0) for entry in manifest["files"].values()),
            )
            self.assertEqual(len(rows["ids"]), count)
            self.assertEqual(
                {meta["source"] for meta in rows["metadatas"]}, expected_sources
            )
            for source, entry in manifest["files"].items():
                source_rows = collection.get(
                    where={"source": source}, include=["documents", "metadatas"]
                )
                self.assertEqual(len(source_rows["ids"]), entry.get("chunks", 0), source)
        finally:
            del collection, client
            b3_fakes.release_chroma(db_dir)
        return manifest, rows

    def normalized_inputs(self, workspace, documents):
        normalized_root = workspace / "normalized-inputs"
        normalized_root.mkdir(parents=True, exist_ok=True)
        index_dir = workspace / "index"
        index_dir.mkdir(parents=True, exist_ok=True)
        model_dir = self.model_dir(workspace)
        selected = []
        front_by_body = {}
        for name, body in documents.items():
            path = normalized_root / name
            path.write_text(body, encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            selected.append({
                "source": name,
                "path": path,
                "entry": None,
                "is_canonical": True,
                "dup_of": None,
                "sha256": digest,
            })
            front_by_body[body] = {"title": Path(name).stem, "doi": "", "mac": "no"}
        target = ingest_normalized.NormalizedTarget(
            normalized_root=normalized_root,
            staging_db=index_dir / "normalized-chroma",
            manifest=index_dir / "normalized-manifest.json",
            converted_roots=(),
            mapping_paths=(),
            task_db=None,
            model_dir=model_dir,
        )
        report = {
            "normalized_root": str(normalized_root),
            "task_db": None,
            "planned_documents": len(selected),
            "ingestable_documents": len(selected),
            "selected_sources": len(selected),
            "canonical_sources": len(selected),
            "alias_sources": 0,
            "conflicts": [],
            "not_ingestable": [],
        }
        return target, selected, report, front_by_body

    def run_normalized(self, target, selected, report, front_by_body, backends):
        def parse_body(text):
            return front_by_body[text], text
        with mock.patch.object(
            ingest_normalized, "select_documents", return_value=(selected, report)
        ), mock.patch.object(
            ingest_normalized, "normalized_front_and_body", side_effect=parse_body
        ):
            return ingest_normalized.build(
                target, device="cpu", backends=backends
            )

    def normalized_rows(self, backends):
        rows = backends.collection.get(
            include=["documents", "metadatas", "embeddings"]
        )
        return sorted(
            (
                row_id,
                document,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                tuple(float(value) for value in embedding),
            )
            for row_id, document, metadata, embedding in zip(
                rows["ids"], rows["documents"], rows["metadatas"], rows["embeddings"]
            )
        )

    @contextlib.contextmanager
    def patched_reconciler(self, pair):
        with contextlib.ExitStack() as stack:
            for name, value in (
                ("ROOT", pair["root"]),
                ("CONVERTED", pair["converted"]),
                ("ARCHIVE", pair["archive"]),
                ("INDEX_DIR", pair["index"]),
                ("CHROMA_DIR", pair["db"]),
                ("MANIFEST_PATH", pair["manifest"]),
                ("BACKUP_DIR", pair["root"] / "backups"),
                ("MODEL_DIR", pair["model"]),
            ):
                stack.enter_context(mock.patch.object(reconcile_manifest, name, value))
            stack.enter_context(mock.patch.object(sys, "argv", ["reconcile_manifest.py", "--no-backup"]))
            yield

    def test_preflight_rejects_missing_database_and_collection_without_initializing(self):
        workspace = self.workspace()
        pair = self.seed_pair(
            workspace,
            {"PMID_1.md": "# A\n\nfixture body"},
            {"PMID_1.md": 1},
            {"PMID_1.md": 1},
        )
        manifest_before = pair["manifest"].read_bytes()
        pair["db"].rename(pair["root"] / "saved-db")
        with self.assertRaisesRegex(rag_core.IndexIntegrityError, "DB directory is missing"):
            rag_core.index_preflight(
                pair["db"], pair["model"], manifest_path=pair["manifest"]
            )
        self.assertFalse(pair["db"].exists())
        self.assertEqual(pair["manifest"].read_bytes(), manifest_before)

        other_db = pair["root"] / "other-db"
        client = chromadb.PersistentClient(
            path=str(other_db), settings=ChromaSettings(anonymized_telemetry=False)
        )
        client.get_or_create_collection("other")
        del client
        b3_fakes.release_chroma(other_db)
        with self.assertRaisesRegex(rag_core.IndexIntegrityError, "collection is missing"):
            rag_core.index_preflight(
                other_db, pair["model"], manifest_path=pair["manifest"]
            )
        check = chromadb.PersistentClient(
            path=str(other_db), settings=ChromaSettings(anonymized_telemetry=False)
        )
        with self.assertRaises(Exception):
            check.get_collection("papers")
        del check
        b3_fakes.release_chroma(other_db)
        self.assertEqual(pair["manifest"].read_bytes(), manifest_before)


        for state in ("empty-directory", "residual-file-only"):
            with self.subTest(state=state):
                workspace = self.workspace()
                broken = self.seed_pair(
                    workspace,
                    {"PMID_2.md": "# B\n\nfixture body"},
                    {"PMID_2.md": 1},
                    {"PMID_2.md": 1},
                )
                self.without_persistent_database(broken, state)
                before = broken["manifest"].read_bytes()
                with self.assertRaisesRegex(
                    rag_core.IndexIntegrityError, "persistent state is missing or empty"
                ):
                    rag_core.index_write_preflight(
                        broken["db"], broken["model"],
                        manifest_path=broken["manifest"],
                    )
                self.assertFalse((broken["db"] / "chroma.sqlite3").exists())
                self.assertEqual(broken["manifest"].read_bytes(), before)

    def test_live_cli_initializes_only_empty_target_and_second_run_preflights(self):
        workspace = self.workspace()
        stubs = self.cli_workspace(workspace)
        self.model_dir(workspace)
        db = workspace / "index" / "chroma_db"
        manifest = workspace / "index" / "manifest.json"
        self.assertFalse(db.exists())
        self.assertFalse(manifest.exists())

        first = self.run_live_cli(workspace, stubs)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        initial = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(initial["embedding_text_version"], "passage-v2")
        self.assertEqual(initial["chunking"], build_index.CHUNKING_CONFIG)
        self.assertRegex(initial["model_fingerprint"], r"^[0-9a-f]{64}$")
        self.assertTrue(db.is_dir())

        second = self.run_live_cli(workspace, stubs)
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertIn("[INGEST] ready", second.stdout)
        self.assertFalse((workspace / "encode-called.txt").exists())

    def test_live_duplicate_canonical_is_order_independent_and_build_stable(self):
        content = (
            "# Synthetic duplicate" + chr(10) + chr(10)
            + "A synthetic passage with enough content to make one stable chunk."
        )
        for arrival_order in (
            ("legacy.md", "PMID_8101.md"),
            ("PMID_8101.md", "legacy.md"),
        ):
            with self.subTest(arrival_order=arrival_order):
                workspace = self.workspace()
                stubs = self.cli_workspace(workspace, encode_enabled=True)
                model_dir = self.model_dir(workspace)
                converted = workspace / "converted"
                for name in arrival_order:
                    source = converted / name
                    source.write_text(content, encoding="utf-8")
                    completed = self.run_live_cli(workspace, stubs)
                    self.assertEqual(
                        completed.returncode, 0, completed.stdout + completed.stderr
                    )

                manifest_path = workspace / "index" / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                canonical = "PMID_8101.md"
                alias = "legacy.md"
                self.assertGreater(manifest["files"][canonical]["chunks"], 0)
                self.assertEqual(manifest["files"][alias]["chunks"], 0)
                self.assertEqual(manifest["files"][alias]["dup_of"], canonical)
                self.assertEqual(
                    manifest["total_chunks"], manifest["files"][canonical]["chunks"]
                )
                archive = converted / "已入库"
                self.assertEqual((archive / canonical).read_text(encoding="utf-8"), content)
                self.assertEqual((archive / alias).read_text(encoding="utf-8"), content)

                db_dir = workspace / "index" / "chroma_db"
                client = chromadb.PersistentClient(
                    path=str(db_dir), settings=ChromaSettings(anonymized_telemetry=False)
                )
                collection = client.get_collection("papers")
                rows = collection.get(include=["metadatas"])
                self.assertEqual(collection.count(), manifest["total_chunks"])
                self.assertEqual(
                    {metadata["source"] for metadata in rows["metadatas"]},
                    {canonical},
                )
                initial_count = collection.count()
                del collection, client
                b3_fakes.release_chroma(db_dir)

                completed = self.run_live_cli(workspace, stubs)
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )
                client = chromadb.PersistentClient(
                    path=str(db_dir), settings=ChromaSettings(anonymized_telemetry=False)
                )
                collection = client.get_collection("papers")
                self.assertEqual(collection.count(), initial_count)
                del collection, client
                b3_fakes.release_chroma(db_dir)

                with mock.patch.multiple(
                    build_index,
                    ROOT=workspace,
                    INDEX_DIR=workspace / "index",
                    CHROMA_DIR=db_dir,
                    MANIFEST_PATH=manifest_path,
                    CONVERTED=converted,
                    ARCHIVE=archive,
                    LOCAL_MODEL_DIR=model_dir,
                ):
                    build_index.main([])
                client = chromadb.PersistentClient(
                    path=str(db_dir), settings=ChromaSettings(anonymized_telemetry=False)
                )
                collection = client.get_collection("papers")
                after_build = collection.get(include=["metadatas"])
                self.assertEqual(collection.count(), initial_count)
                self.assertEqual(
                    {metadata["source"] for metadata in after_build["metadatas"]},
                    {canonical},
                )
                del collection, client
                b3_fakes.release_chroma(db_dir)

    def test_incremental_build_promotes_old_alias_when_canonical_changes(self):
        workspace = self.workspace()
        self.model_dir(workspace)
        converted = workspace / "converted"
        converted.mkdir(parents=True, exist_ok=True)
        old = "# Shared\n\nShared original evidence passage for the synthetic source."
        changed = "# Changed\n\nChanged canonical evidence passage with distinct content."
        (converted / "A.md").write_text(old, encoding="utf-8")
        (converted / "B.md").write_text(old, encoding="utf-8")
        self.run_build(workspace)
        initial, _ = self.assert_build_rows_match_ledger(workspace)
        self.assertGreater(initial["files"]["A.md"]["chunks"], 0)
        self.assertEqual(initial["files"]["B.md"]["chunks"], 0)

        (converted / "A.md").write_text(changed, encoding="utf-8")
        self.run_build(workspace)
        manifest, rows = self.assert_build_rows_match_ledger(workspace)
        self.assertGreater(manifest["files"]["A.md"]["chunks"], 0)
        self.assertGreater(manifest["files"]["B.md"]["chunks"], 0)
        documents_by_source = {}
        for document, metadata in zip(rows["documents"], rows["metadatas"]):
            documents_by_source.setdefault(metadata["source"], []).append(document)
        self.assertTrue(any("Changed canonical evidence" in doc for doc in documents_by_source["A.md"]))
        self.assertTrue(any("Shared original evidence" in doc for doc in documents_by_source["B.md"]))
        self.assertEqual(set(manifest["files"]), {"A.md", "B.md"})

        rerun = self.run_build(workspace, forbid_encoding=True)
        self.assertEqual(rerun.calls, [])
        self.assert_build_rows_match_ledger(workspace)

    def test_incremental_build_promotes_changed_alias_and_cleans_new_duplicate_rows(self):
        workspace = self.workspace()
        self.model_dir(workspace)
        converted = workspace / "converted"
        converted.mkdir(parents=True, exist_ok=True)
        shared = "# Shared\n\nShared evidence body with enough synthetic content."
        changed_alias = "# B changed\n\nChanged alias evidence body is now independent."
        (converted / "A.md").write_text(shared, encoding="utf-8")
        (converted / "B.md").write_text(shared, encoding="utf-8")
        self.run_build(workspace)

        (converted / "B.md").write_text(changed_alias, encoding="utf-8")
        self.run_build(workspace)
        manifest, rows = self.assert_build_rows_match_ledger(workspace)
        self.assertGreater(manifest["files"]["A.md"]["chunks"], 0)
        self.assertGreater(manifest["files"]["B.md"]["chunks"], 0)
        self.assertEqual({meta["source"] for meta in rows["metadatas"]}, {"A.md", "B.md"})
        self.assertTrue(any(
            "Changed alias evidence" in doc
            for doc, meta in zip(rows["documents"], rows["metadatas"])
            if meta["source"] == "B.md"
        ))

        (converted / "B.md").write_text(shared, encoding="utf-8")
        self.run_build(workspace)
        manifest, rows = self.assert_build_rows_match_ledger(workspace)
        self.assertEqual(manifest["files"]["B.md"]["chunks"], 0)
        self.assertEqual(manifest["files"]["B.md"]["dup_of"], "A.md")
        self.assertEqual({meta["source"] for meta in rows["metadatas"]}, {"A.md"})
        self.assertEqual(len(rows["ids"]), manifest["files"]["A.md"]["chunks"])

    def test_incremental_build_transfers_canonical_rows_and_archive_move_is_idempotent(self):
        workspace = self.workspace()
        self.model_dir(workspace)
        converted = workspace / "converted"
        converted.mkdir(parents=True, exist_ok=True)
        archive = converted / "已入库"
        content = "# Preferred source\n\nSynthetic canonical transfer passage."
        (converted / "legacy.md").write_text(content, encoding="utf-8")
        self.run_build(workspace)

        (converted / "PMID_8301.md").write_text(content, encoding="utf-8")
        self.run_build(workspace)
        manifest, rows = self.assert_build_rows_match_ledger(workspace)
        self.assertEqual(manifest["files"]["legacy.md"]["chunks"], 0)
        self.assertEqual(manifest["files"]["legacy.md"]["dup_of"], "PMID_8301.md")
        self.assertGreater(manifest["files"]["PMID_8301.md"]["chunks"], 0)
        self.assertEqual({meta["source"] for meta in rows["metadatas"]}, {"PMID_8301.md"})

        archive.mkdir(parents=True, exist_ok=True)
        for name in ("legacy.md", "PMID_8301.md"):
            (converted / name).replace(archive / name)
        model = self.run_build(workspace, forbid_encoding=True)
        self.assertEqual(model.calls, [])
        self.assert_build_rows_match_ledger(workspace)

    def test_live_change_promotes_archived_alias_to_independent_source(self):
        workspace = self.workspace()
        stubs = self.cli_workspace(workspace, encode_enabled=True)
        self.model_dir(workspace)
        converted = workspace / "converted"
        archive = converted / "已入库"
        shared = "# Shared\n\nShared evidence body for two arriving sources."
        for name in ("A.md", "B.md"):
            (converted / name).write_text(shared, encoding="utf-8")
            completed = self.run_live_cli(workspace, stubs)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

        changed = "# A changed\n\nChanged A evidence body; B must be promoted too."
        (archive / "A.md").write_text(changed, encoding="utf-8")
        completed = self.run_live_cli(workspace, stubs)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        manifest_path = workspace / "index" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        db_dir = workspace / "index" / "chroma_db"
        client = chromadb.PersistentClient(
            path=str(db_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        collection = client.get_collection("papers")
        rows = collection.get(include=["documents", "metadatas"])
        self.assertEqual({meta["source"] for meta in rows["metadatas"]}, {"A.md", "B.md"})
        self.assertEqual(collection.count(), manifest["total_chunks"])
        self.assertGreater(manifest["files"]["A.md"]["chunks"], 0)
        self.assertGreater(manifest["files"]["B.md"]["chunks"], 0)
        self.assertTrue(any(
            "Changed A evidence" in doc
            for doc, meta in zip(rows["documents"], rows["metadatas"])
            if meta["source"] == "A.md"
        ))
        self.assertTrue(any(
            "Shared evidence body" in doc
            for doc, meta in zip(rows["documents"], rows["metadatas"])
            if meta["source"] == "B.md"
        ))
        del collection, client
        b3_fakes.release_chroma(db_dir)

    def test_live_merge_cleans_rows_from_source_that_becomes_alias(self):
        workspace = self.workspace()
        stubs = self.cli_workspace(workspace, encode_enabled=True)
        self.model_dir(workspace)
        converted = workspace / "converted"
        archive = converted / "已入库"
        content_a = "# A\n\nFirst independent synthetic source body."
        content_b = "# B\n\nSecond different synthetic source body."
        for name, content in (("A.md", content_a), ("B.md", content_b)):
            (converted / name).write_text(content, encoding="utf-8")
            completed = self.run_live_cli(workspace, stubs)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

        (archive / "B.md").write_text(content_a, encoding="utf-8")
        completed = self.run_live_cli(workspace, stubs)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        manifest = json.loads(
            (workspace / "index" / "manifest.json").read_text(encoding="utf-8")
        )
        db_dir = workspace / "index" / "chroma_db"
        client = chromadb.PersistentClient(
            path=str(db_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        collection = client.get_collection("papers")
        rows = collection.get(include=["metadatas"])
        self.assertEqual({meta["source"] for meta in rows["metadatas"]}, {"A.md"})
        self.assertEqual(manifest["files"]["B.md"]["chunks"], 0)
        self.assertEqual(manifest["files"]["B.md"]["dup_of"], "A.md")
        self.assertEqual(collection.count(), manifest["files"]["A.md"]["chunks"])
        self.assertEqual(collection.count(), manifest["total_chunks"])
        self.assertTrue((archive / "A.md").is_file())
        self.assertTrue((archive / "B.md").is_file())
        del collection, client
        b3_fakes.release_chroma(db_dir)

    def test_normalized_identity_is_written_and_same_model_rerun_reuses_rows(self):
        workspace = self.workspace()
        body = "Synthetic normalized passage used only for model identity regression."
        target, selected, report, front = self.normalized_inputs(
            workspace, {"PMID_8401.md": body}
        )
        backends = b3_fakes.FakeBackends(SimpleNamespace(staging_db=target.staging_db))
        self.addCleanup(backends.close)
        first = self.run_normalized(target, selected, report, front, backends)
        expected_fingerprint = build_index.model_fingerprint(target.model_dir)
        manifest = json.loads(target.manifest.read_text(encoding="utf-8"))
        ready_path = build_index.ready_marker_path(target.manifest)
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["model_fingerprint"], expected_fingerprint)
        self.assertEqual(ready["model_fingerprint"], expected_fingerprint)
        self.assertEqual(first["encoded_sources"], 1)
        self.assertEqual(first["chunks"], backends.collection.count())
        before_calls = len(backends.encoder.calls)

        second = self.run_normalized(target, selected, report, front, backends)
        self.assertEqual(second["encoded_sources"], 0)
        self.assertEqual(second["reused_sources"], 1)
        self.assertEqual(len(backends.encoder.calls), before_calls)
        self.assertEqual(json.loads(ready_path.read_text(encoding="utf-8"))["model_fingerprint"], expected_fingerprint)
        self.assertEqual(backends.collection.count(), manifest["total_chunks"])

    def test_normalized_model_change_refuses_reuse_and_preserves_identity_and_vectors(self):
        workspace = self.workspace()
        body = "Synthetic normalized passage retained across a rejected model change."
        target, selected, report, front = self.normalized_inputs(
            workspace, {"PMID_8402.md": body}
        )
        backends = b3_fakes.FakeBackends(SimpleNamespace(staging_db=target.staging_db))
        self.addCleanup(backends.close)
        self.run_normalized(target, selected, report, front, backends)
        manifest_before = target.manifest.read_bytes()
        rows_before = self.normalized_rows(backends)
        calls_before = len(backends.encoder.calls)
        old_identity = json.loads(manifest_before)["model_fingerprint"]
        (target.model_dir / "config.json").write_text(
            '{"fixture":"changed synthetic model config"}\n', encoding="utf-8"
        )

        with self.assertRaisesRegex(
            ingest_normalized.NormalizedIndexError,
            "fingerprint is different.*new isolated target",
        ):
            self.run_normalized(target, selected, report, front, backends)

        self.assertFalse(build_index.ready_marker_path(target.manifest).exists())
        self.assertEqual(target.manifest.read_bytes(), manifest_before)
        self.assertEqual(json.loads(target.manifest.read_text(encoding="utf-8"))["model_fingerprint"], old_identity)
        self.assertEqual(self.normalized_rows(backends), rows_before)
        self.assertEqual(len(backends.encoder.calls), calls_before)

    def test_normalized_nonempty_ledger_without_fingerprint_refuses_silent_reuse(self):
        workspace = self.workspace()
        body = "Synthetic normalized passage for a missing identity ledger."
        target, selected, report, front = self.normalized_inputs(
            workspace, {"PMID_8403.md": body}
        )
        backends = b3_fakes.FakeBackends(SimpleNamespace(staging_db=target.staging_db))
        self.addCleanup(backends.close)
        self.run_normalized(target, selected, report, front, backends)
        manifest = json.loads(target.manifest.read_text(encoding="utf-8"))
        manifest.pop("model_fingerprint")
        index_common.write_json_atomic(target.manifest, manifest)
        manifest_before = target.manifest.read_bytes()
        rows_before = self.normalized_rows(backends)
        calls_before = len(backends.encoder.calls)

        with self.assertRaisesRegex(
            ingest_normalized.NormalizedIndexError,
            "fingerprint is missing.*new isolated target",
        ):
            self.run_normalized(target, selected, report, front, backends)

        self.assertFalse(build_index.ready_marker_path(target.manifest).exists())
        self.assertEqual(target.manifest.read_bytes(), manifest_before)
        self.assertEqual(self.normalized_rows(backends), rows_before)
        self.assertEqual(len(backends.encoder.calls), calls_before)

    def test_normalized_failed_upsert_keeps_partial_identity_and_removes_ready(self):
        workspace = self.workspace()
        body = "Synthetic normalized passage for interrupted identity build."
        target, selected, report, front = self.normalized_inputs(
            workspace, {"PMID_8404.md": body}
        )
        backends = b3_fakes.FakeBackends(SimpleNamespace(staging_db=target.staging_db))
        self.addCleanup(backends.close)
        collection = backends.collection
        backends.wrap_collection(
            b3_fakes.CrashingCollection(collection, fail_on_upsert=1)
        )

        with self.assertRaisesRegex(RuntimeError, "injected upsert failure"):
            self.run_normalized(target, selected, report, front, backends)

        manifest = json.loads(target.manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(
            manifest["model_fingerprint"],
            build_index.model_fingerprint(target.model_dir),
        )
        self.assertFalse(build_index.ready_marker_path(target.manifest).exists())
        self.assertEqual(collection.count(), 1)

    def test_live_cli_reports_startup_failure_with_nonzero_exit(self):
        workspace = self.workspace()
        stubs = self.cli_workspace(workspace)
        model_dir = self.model_dir(workspace)
        manifest = workspace / "index" / "manifest.json"
        _write_json(manifest, self.manifest_base(model_dir, {}, 0))
        before = manifest.read_bytes()
        db = workspace / "index" / "chroma_db"

        completed = self.run_live_cli(workspace, stubs)
        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Chroma DB directory is missing", completed.stdout + completed.stderr)
        self.assertFalse(db.exists(), "live preflight must run before PersistentClient")
        self.assertEqual(manifest.read_bytes(), before)


    def test_live_cli_refuses_existing_directory_without_persistent_database(self):
        for state in ("missing-directory", "empty-directory", "residual-file-only"):
            with self.subTest(state=state):
                workspace = self.workspace()
                content = "# Stored source\n\nSynthetic indexed content."
                pair = self.seed_pair(
                    workspace,
                    {"PMID_3.md": content},
                    {"PMID_3.md": 1},
                    {"PMID_3.md": 1},
                )
                stubs = self.cli_workspace(workspace)
                self.without_persistent_database(pair, state)
                before = pair["manifest"].read_bytes()

                completed = self.run_live_cli(workspace, stubs)

                output = completed.stdout + completed.stderr
                self.assertNotEqual(completed.returncode, 0, output)
                if state == "missing-directory":
                    self.assertIn("Chroma DB directory is missing", output)
                    self.assertFalse(pair["db"].exists())
                else:
                    self.assertIn("persistent state is missing or empty", output)
                    expected = (
                        {"chroma.sqlite3-wal": b"synthetic leftover"}
                        if state == "residual-file-only" else {}
                    )
                    self.assertEqual(
                        {path.name: path.read_bytes() for path in pair["db"].iterdir()},
                        expected,
                    )
                self.assertFalse(
                    (pair["db"] / "chroma.sqlite3").exists(),
                    "live preflight must not initialize a missing SQLite database",
                )
                self.assertEqual(pair["manifest"].read_bytes(), before)

    def test_reconcile_repairs_counts_and_preserves_v2_for_incremental_and_live(self):
        workspace = self.workspace()
        stubs = self.cli_workspace(workspace)
        content = "# Same paper\n\nSynthetic passage body for B1."
        pair = self.seed_pair(
            workspace,
            {"PMID_1.md": content, "alias.md": content},
            {"PMID_1.md": 2},
            {"PMID_1.md": 1, "alias.md": 0},
            aliases={"alias.md": "PMID_1.md"},
            total_override=3,
        )
        before = json.loads(pair["manifest"].read_text(encoding="utf-8"))

        with self.patched_reconciler(pair), contextlib.redirect_stdout(io.StringIO()):
            reconcile_manifest.main()
        b3_fakes.release_chroma(pair["db"])
        repaired = json.loads(pair["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(repaired["total_chunks"], 2)
        self.assertEqual(repaired["embedding_text_version"], before["embedding_text_version"])
        self.assertEqual(repaired["chunking"], before["chunking"])
        self.assertEqual(repaired["model_fingerprint"], before["model_fingerprint"])
        self.assertEqual(repaired["files"]["PMID_1.md"]["chunks"], 2)
        self.assertEqual(repaired["files"]["PMID_1.md"]["repr"], before["files"]["PMID_1.md"]["repr"])
        self.assertEqual(repaired["files"]["alias.md"], before["files"]["alias.md"])

        summary = rag_core.index_preflight(
            pair["db"], pair["model"], manifest_path=pair["manifest"]
        )
        self.assertEqual(summary["collection_count"], 2)
        b3_fakes.release_chroma(pair["db"])

        mapping_dir = self.write_synthetic_metadata(
            pair["root"], ["PMID_1.md", "alias.md"]
        )
        self.assertTrue((mapping_dir / "mapping.json").is_file())
        with mock.patch.multiple(
            build_index,
            ROOT=pair["root"],
            INDEX_DIR=pair["index"],
            CHROMA_DIR=pair["db"],
            MANIFEST_PATH=pair["manifest"],
            CONVERTED=pair["converted"],
            ARCHIVE=pair["archive"],
            LOCAL_MODEL_DIR=pair["model"],
        ), mock.patch.object(build_index, "SentenceTransformer", side_effect=AssertionError("must not encode")):
            build_index.main([])
        b3_fakes.release_chroma(pair["db"])

        live = self.run_live_cli(workspace, stubs)
        self.assertEqual(live.returncode, 0, live.stdout + live.stderr)
        self.assertIn("[INGEST] ready", live.stdout)
        self.assertFalse((workspace / "encode-called.txt").exists())
        after_live = json.loads(pair["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(after_live["embedding_text_version"], "passage-v2")
        self.assertEqual(after_live["files"]["PMID_1.md"]["repr"], repaired["files"]["PMID_1.md"]["repr"])

    def test_reconcile_refuses_changed_content_mixed_representation_and_unknown_model(self):
        cases = ("changed-content", "unknown-row-version", "mixed-row-version", "unknown-model")
        for case in cases:
            with self.subTest(case=case):
                workspace = self.workspace()
                text = "# Source\n\nSynthetic evidence body."
                row_count = 2 if case == "mixed-row-version" else 1
                row_versions = None
                if case == "unknown-row-version":
                    row_versions = {"PMID_1.md": "legacy-v1"}
                elif case == "mixed-row-version":
                    row_versions = {"PMID_1.md": ["passage-v2", "legacy-v1"]}
                pair = self.seed_pair(
                    workspace,
                    {"PMID_1.md": text},
                    {"PMID_1.md": row_count},
                    {"PMID_1.md": row_count},
                    row_versions=row_versions,
                )
                before = pair["manifest"].read_bytes()
                if case == "changed-content":
                    (pair["archive"] / "PMID_1.md").write_text(text + " changed", encoding="utf-8")
                    expected = "source content changed"
                elif case == "unknown-row-version":
                    expected = "unknown representation"
                elif case == "mixed-row-version":
                    expected = "mixed representation versions"
                else:
                    (pair["model"] / "config.json").write_text('{"fixture": false}\n', encoding="utf-8")
                    expected = "model fingerprint mismatch"
                with self.patched_reconciler(pair), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex((RuntimeError, rag_core.IndexIntegrityError), expected):
                        reconcile_manifest.main()
                self.assertEqual(pair["manifest"].read_bytes(), before)

    def test_reconcile_refuses_missing_db_and_nonempty_ledger_for_empty_collection(self):
        for state in ("missing-directory", "empty-directory", "residual-file-only"):
            with self.subTest(state=state):
                workspace = self.workspace()
                pair = self.seed_pair(
                    workspace,
                    {"PMID_1.md": "# A\n\nfixture"},
                    {"PMID_1.md": 1},
                    {"PMID_1.md": 1},
                )
                self.without_persistent_database(pair, state)
                before = pair["manifest"].read_bytes()
                with self.patched_reconciler(pair), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(rag_core.IndexIntegrityError) as raised:
                        reconcile_manifest.main()
                if state == "missing-directory":
                    self.assertIn("DB directory is missing", str(raised.exception))
                    self.assertFalse(pair["db"].exists())
                else:
                    self.assertIn("persistent state is missing or empty", str(raised.exception))
                    expected = (
                        {"chroma.sqlite3-wal": b"synthetic leftover"}
                        if state == "residual-file-only" else {}
                    )
                    self.assertEqual(
                        {path.name: path.read_bytes() for path in pair["db"].iterdir()},
                        expected,
                    )
                self.assertFalse((pair["db"] / "chroma.sqlite3").exists())
                self.assertEqual(pair["manifest"].read_bytes(), before)

        empty_workspace = self.workspace()
        empty_pair = self.seed_pair(
            empty_workspace,
            {"PMID_2.md": "# B\n\nfixture"},
            {},
            {"PMID_2.md": 1},
        )
        client = chromadb.PersistentClient(
            path=str(empty_pair["db"]), settings=ChromaSettings(anonymized_telemetry=False)
        )
        client.get_or_create_collection("papers", metadata={"hnsw:space": "cosine"})
        del client
        b3_fakes.release_chroma(empty_pair["db"])
        before = empty_pair["manifest"].read_bytes()
        with self.patched_reconciler(empty_pair), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "empty Chroma collection"):
                reconcile_manifest.main()
        b3_fakes.release_chroma(empty_pair["db"])
        self.assertEqual(empty_pair["manifest"].read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
