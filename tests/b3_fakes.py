"""Shared offline fakes for the B3 continuous-runner and writer-boundary tests.

Everything here is deterministic and local: a character-level tokenizer, a
two-dimensional fake encoder, and a real Chroma client in a throwaway
directory.  No model weights are loaded and no network access is used.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "index"))

import index_common  # noqa: E402
from file_lock import writer_lock_path  # noqa: E402
from rebuild_b3 import B3Paths  # noqa: E402


class FakeTokenizer:
    """One token per character: bounded fixtures behave like short documents."""

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        return [ord(character) for character in text]

    def decode(self, ids, skip_special_tokens=True,  # noqa: ARG002
               clean_up_tokenization_spaces=False):  # noqa: ARG002
        return "".join(chr(value) for value in ids)


class FakeEncoder:
    """Deterministic 2-D vectors; records what was actually passed to encode."""

    def __init__(self):
        self.calls = []
        self.on_encode = None

    def encode(self, texts):
        texts = list(texts)
        self.calls.append(texts)
        if self.on_encode is not None:
            self.on_encode(len(self.calls), texts)
        return [
            [float(len(text) % 89) + 1.0, float(len(text) % 17) + 1.0]
            for text in texts
        ]

    @property
    def unit_sizes(self):
        return [len(call) for call in self.calls]

    @property
    def texts_seen(self):
        return [text for call in self.calls for text in call]


class CrashingCollection:
    """Proxy that fails the Nth upsert to simulate a crash mid-source."""

    def __init__(self, inner, fail_on_upsert=None):
        self._inner = inner
        self._fail_on = fail_on_upsert
        self.upserts = 0

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def upsert(self, **kwargs):
        self.upserts += 1
        result = self._inner.upsert(**kwargs)
        if self._fail_on is not None and self.upserts >= self._fail_on:
            # Fail *after* the write reached Chroma: this models the harshest
            # interruption, where rows exist without a committed manifest entry.
            raise RuntimeError("injected upsert failure")
        return result


class FakeBackends:
    """Runner backends backed by a real Chroma store and a fake encoder.

    The Chroma client is created lazily so a test can prove that a code path
    never opened the database at all.
    """

    def __init__(self, paths, device="cpu"):
        self.paths = paths
        self.device = device
        self.encoder = FakeEncoder()
        self.tokenizer = FakeTokenizer()
        self.collection_opened = False
        self.closed = False
        self._client = None
        self._collection = None

    def open(self):
        if self._collection is not None:
            return self._collection
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        Path(self.paths.staging_db).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.paths.staging_db),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            "papers", metadata={"hnsw:space": "cosine"}
        )
        return self._collection

    def encode(self, texts):
        return self.encoder.encode(texts)

    @property
    def collection(self):
        self.collection_opened = True
        return self.open()

    def close(self):
        self.closed = True
        self._client = None
        self._collection = None
        release_chroma(self.paths.staging_db)

    def wrap_collection(self, proxy):
        self._collection = proxy
        self.collection_opened = True


def release_chroma(db_path):
    """Drop Chroma's process-wide client registry so temp dirs can be removed."""
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except Exception:  # pragma: no cover - chroma version without the registry
        return
    registry = getattr(SharedSystemClient, "_identifier_to_system", {})
    for identifier in list(registry):
        if str(db_path) in str(identifier):
            registry.pop(identifier, None)
    for _ in range(3):
        gc.collect()


def rename_directory(source, target, attempts=40, delay=0.25):
    """Rename a directory, waiting out a lingering Windows file handle.

    Chroma keeps SQLite handles open until its client is collected; on Windows
    the rename then fails with a sharing violation for a short while.  This is
    environment latency, not a logic error, so the retry is bounded and loud.
    """
    last = None
    for _ in range(attempts):
        try:
            Path(source).rename(target)
            return True
        except OSError as exc:  # pragma: no cover - depends on host timing
            last = exc
            time.sleep(delay)
    raise last


def make_paths(root):
    """A fully isolated path set below ``root``."""
    root = Path(root)
    index_dir = root / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    converted = root / "converted"
    archive = converted / "已入库"
    archive.mkdir(parents=True, exist_ok=True)
    staging_db = index_dir / "chroma_db.v2-staging"
    return B3Paths(
        index_dir=index_dir,
        sources=index_dir / "b3-sources.json",
        staging_db=staging_db,
        manifest=index_dir / "manifest.b3.partial.json",
        progress=index_dir / "b3-progress.json",
        stop_flag=index_dir / "b3-rebuild.stop",
        # Same derivation as production: one lock per target directory, shared
        # by the runner, the builders, the live ingest and the publisher.
        lock=writer_lock_path(staging_db),
        converted=converted,
        archive=archive,
        model_dir=ROOT / ".model-cache" / "BAAI__bge-m3",
    )


def write_corpus(archive, files):
    for name, text in files.items():
        (Path(archive) / name).write_text(text, encoding="utf-8")
    return files


def simple_meta_lookup(name):
    """Metadata for synthetic fixtures: title = file stem, no DOI."""
    return {
        "title": Path(name).stem,
        "year": "",
        "journal": "",
        "doi": "",
        "authors": "",
        "pub_type": "",
        "license": "unverified",
        "mac": "unknown",
        "folder": "",
        "evidence": "",
    }, False


def plan_sources(paths):
    """Build the source list for a synthetic corpus without loading a model."""
    import rebuild_b3

    tokenizer = FakeTokenizer()
    payload = rebuild_b3.build_sources_payload(paths, tokenizer,
                                               meta_lookup=simple_meta_lookup)
    index_common.write_json_atomic(paths.sources, payload)
    return payload


def patch_meta(mapping):
    """Force ``build_index.load_meta`` to return a fixed metadata mapping."""
    import build_index

    by_stem = {}
    for stem, meta in mapping.items():
        entry = dict(simple_meta_lookup(f"{stem}.md")[0])
        entry.update(meta)
        entry["filename"] = f"{stem}.md"
        by_stem[stem] = entry
    return mock.patch.object(build_index, "load_meta", return_value=(by_stem, {}, {}))


def paragraph(prefix, length=600):
    """A single paragraph of roughly ``length`` characters."""
    unit = f"{prefix} "
    return (unit * ((length // len(unit)) + 1))[:length]
