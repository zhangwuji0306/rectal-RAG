"""Continuous, stoppable B3 passage-representation-v2 rebuild runner.

One process handles sources one after another until every source is complete, a
stop request arrives at a short-operation boundary, or a concrete error makes
further progress impossible.  There are no work batches, no batch plans and no
per-batch checkpoints.

Guarantees:

* The committed boundary is the manifest: a source is complete only after all of
  its rows are written and the manifest entry is committed atomically.  An
  interrupted source is deleted on resume and redone; completed sources are
  never re-encoded.
* A stop request is honoured after the current short operation (one small
  encoding unit or one upsert), never after a whole paper or an accumulated
  256-chunk pending list.
* An existing stop request prevents any new encoding or writing, and an
  explicit ``--clear-stop`` only consumes the request that was already present
  when the process started; a newer request is preserved and ends the run.
* A single OS file lock owns the staging database, so two writers cannot share
  it, and a crashed writer never leaves the lock held.
* The runner never publishes.  Publishing is a separate process
  (``index/publish_index.py``).

Commands::

    python index/rebuild_b3.py --plan              # (re)build the source list
    python index/rebuild_b3.py --adopt-existing    # one-time migration of the old 4-batch staging
    python index/rebuild_b3.py --resume [--clear-stop] [--device cpu]
    python index/rebuild_b3.py --status            # no model, no Chroma
    python index/rebuild_b3.py --verify            # structural check of the staging pair
    python index/rebuild_b3.py --stop              # request a stop; no model, no Chroma
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

INDEX_DIR = Path(__file__).resolve().parent
ROOT = INDEX_DIR.parent
sys.path.insert(0, str(INDEX_DIR))

import index_common  # noqa: E402
from file_lock import (  # noqa: E402
    FileLock,
    LockUnavailable,
    lock_is_free,
    writer_lock_path,
)

COLLECTION_NAME = "papers"
SOURCES_SCHEMA_VERSION = 1
PROGRESS_SCHEMA_VERSION = 1
MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_TEXT_VERSION = "passage-v2"
UNIT_CHUNKS = 8
VALID_STATES = ("running", "stopping", "stopped", "failed", "completed")
# Derived run states reported by ``--status``.  ``interrupted`` means the
# persisted progress claims a run but no process holds the target writer lock.
DERIVED_RUNNING = "running"
DERIVED_INTERRUPTED = "interrupted"
DERIVED_IDLE = "idle"
STOP_REQUESTED = False


@dataclass(frozen=True)
class B3Paths:
    """Every path the runner may touch, so tests can isolate a run."""

    index_dir: Path
    sources: Path
    staging_db: Path
    manifest: Path
    progress: Path
    stop_flag: Path
    lock: Path
    converted: Path
    archive: Path
    model_dir: Path


STAGING_DB = INDEX_DIR / "chroma_db.v2-staging"

DEFAULT_PATHS = B3Paths(
    index_dir=INDEX_DIR,
    sources=INDEX_DIR / "b3-sources.json",
    staging_db=STAGING_DB,
    manifest=INDEX_DIR / "manifest.b3.partial.json",
    progress=INDEX_DIR / "b3-progress.json",
    stop_flag=INDEX_DIR / "b3-rebuild.stop",
    # Every writer on a target directory takes the same lock, so the runner,
    # a build, a live ingest, a reconcile write and a publication are mutually
    # exclusive on the staging target but independent of other targets.
    lock=writer_lock_path(STAGING_DB),
    converted=ROOT / "converted",
    archive=ROOT / "converted" / "已入库",
    model_dir=ROOT / ".model-cache" / "BAAI__bge-m3",
)

LEGACY = {
    "plan": INDEX_DIR / "b3-rebuild-4batch-plan.json",
    "manifest": INDEX_DIR / "manifest.b3-4batch.partial.json",
    "checkpoint": INDEX_DIR / "b3-rebuild-4batch-checkpoint.json",
    "staging": INDEX_DIR / "chroma_db.v2-staging-4batch",
}


# ---------------------------------------------------------------------------
# Lazy, testable collaborator access
# ---------------------------------------------------------------------------

def build_index_module():
    """Import build_index (and its heavy deps) only where it is needed."""
    import build_index  # noqa: PLC0415 - deliberately lazy

    return build_index


def corpus_items(paths):
    return index_common.markdown_files(paths.converted, paths.archive)


def corpus_inventory(paths):
    return index_common.hashed_corpus(corpus_items(paths))


def load_tokenizer(paths, device="cpu"):
    """Prefer a tokenizer-only load; fall back to the full model."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415

        model_dir = paths.model_dir
        if Path(model_dir).is_dir():
            return AutoTokenizer.from_pretrained(str(model_dir))
    except Exception:  # noqa: BLE001 - fall back to the encoder's tokenizer
        pass
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(str(paths.model_dir), device=device)
    return model[0].tokenizer


class RealBackends:
    """Model + Chroma access used by a real run."""

    def __init__(self, paths, device="cpu"):
        self.paths = paths
        self.device = device
        self._model = None
        self._client = None
        self._collection = None
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self._model is not None:
                self._tokenizer = self._model[0].tokenizer
            else:
                self._tokenizer = load_tokenizer(self.paths, self.device)
        return self._tokenizer

    def encode(self, texts):
        import torch  # noqa: PLC0415

        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            torch.set_num_threads(max(1, os.cpu_count() or 8))
            model_path = (
                str(self.paths.model_dir)
                if Path(self.paths.model_dir).is_dir()
                else MODEL_NAME
            )
            self._model = SentenceTransformer(model_path, device=self.device)
            if self.device == "cuda":
                self._model.half()
                torch.cuda.empty_cache()
        batch_size = 8 if self.device == "cuda" else 48
        return self._model.encode(
            list(texts), batch_size=batch_size, normalize_embeddings=True,
            show_progress_bar=False,
        )

    @property
    def collection(self):
        if self._collection is None:
            import chromadb  # noqa: PLC0415
            from chromadb.config import Settings as ChromaSettings  # noqa: PLC0415

            self.paths.staging_db.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.paths.staging_db),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def close(self):
        collection, client, model = self._collection, self._client, self._model
        self._collection = self._client = self._model = self._tokenizer = None
        del collection, client
        if model is not None:
            del model
        try:
            import gc  # noqa: PLC0415
            import torch  # noqa: PLC0415

            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        # Windows refuses to rename a directory that still has an open SQLite
        # handle, so the client must be released before the directory moves.
        try:
            from rag_core import release_client  # noqa: PLC0415

            release_client(self.paths.staging_db)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Stop requests
# ---------------------------------------------------------------------------

def _request_stop_signal(signum=None, _frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    label = signal.Signals(signum).name if signum else "signal"
    print(f"stop requested ({label}); finishing the current short operation", flush=True)


def install_signal_handlers():
    signal.signal(signal.SIGINT, _request_stop_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _request_stop_signal)


def stop_requested(paths):
    return STOP_REQUESTED or Path(paths.stop_flag).exists()


def stop_token(paths):
    """Identify the currently pending stop request (None when there is none)."""
    path = Path(paths.stop_flag)
    if not path.is_file():
        return None
    return index_common.sha256_file(path)


def request_stop(paths):
    """Create or refresh the stop request without touching model or Chroma."""
    token = uuid.uuid4().hex
    payload = {
        "request_id": token,
        "requested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested_by_pid": os.getpid(),
    }
    index_common.write_json_atomic(paths.stop_flag, payload)
    return payload


def consume_stop(paths, token):
    """Delete the identified stop request; keep a newer one.

    Returns True when the request was consumed and the run may start.
    """
    if token is None:
        return True
    path = Path(paths.stop_flag)
    if not path.is_file():
        return True
    if index_common.sha256_file(path) != token:
        # A different (newer) request arrived while this process was starting.
        return False
    path.unlink()
    return True


# ---------------------------------------------------------------------------
# Source list and manifest
# ---------------------------------------------------------------------------

def load_sources(paths):
    payload = index_common.read_json(paths.sources, default=None)
    if not payload:
        raise RuntimeError(
            f"source list is missing: {paths.sources}; run --plan first"
        )
    return payload


def source_by_name(sources_payload):
    return {item["source"]: item for item in sources_payload["sources"]}


def canonical_sources(sources_payload):
    return sorted(
        item["source"] for item in sources_payload["sources"] if item["is_canonical"]
    )


def aliases_by_canonical(sources_payload):
    mapping = {}
    for item in sources_payload["sources"]:
        if not item["is_canonical"]:
            mapping.setdefault(item["canonical_source"], []).append(item["source"])
    return mapping


def build_sources_payload(paths, tokenizer, meta_lookup=None):
    """Build the single source list from the current corpus (no batch fields)."""
    build_index = build_index_module()
    if meta_lookup is None:
        by_stem, by_folder, by_doi = build_index.load_meta()

        def meta_lookup(name):
            meta, is_alias = build_index.meta_for(name, by_stem, by_folder, by_doi)
            return meta, is_alias

    items = corpus_items(paths)
    hashed = index_common.hashed_corpus(items)
    groups = index_common.duplicate_groups(hashed)
    entries = []
    for name, path in items:
        digest = index_common.sha256_file(path)
        canonical = groups.get(digest, name)
        meta, is_alias = meta_lookup(name)
        if name == canonical:
            text = path.read_text(encoding="utf-8", errors="replace")
            estimated = len(build_index.chunk_markdown(text, tokenizer))
        else:
            estimated = 0
        entries.append({
            "source": name,
            "sha256": digest,
            "canonical_source": canonical,
            "is_canonical": name == canonical,
            "estimated_chunks": estimated,
            "paper_id": build_index.paper_id_for(name, meta),
            "title": meta.get("title", ""),
            "mac": meta.get("mac", "unknown"),
            "supp": "yes" if is_alias else "no",
        })
    payload = {
        "schema_version": SOURCES_SCHEMA_VERSION,
        "stage": "B3",
        "plan_type": "continuous-by-source-v1",
        "model": build_index.MODEL_NAME,
        "model_fingerprint": (
            build_index.model_fingerprint(paths.model_dir)
            if Path(paths.model_dir).is_dir() else None
        ),
        "embedding_text_version": build_index.EMBEDDING_TEXT_VERSION,
        "chunking": dict(build_index.CHUNKING_CONFIG),
        "source_order": "canonical source filename, UTF-8 code-point ascending",
        "duplicate_policy": (
            "sha-identical sources share one canonical content source "
            "(PMID_* preferred); aliases carry zero chunks and a dup_of entry"
        ),
        "sources": entries,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    payload["sources_fingerprint"] = index_common.corpus_fingerprint(
        [(item["source"], item["sha256"]) for item in entries]
    )
    payload["planned_chunks"] = sum(
        item["estimated_chunks"] for item in entries if item["is_canonical"]
    )
    payload["canonical_source_count"] = sum(1 for item in entries if item["is_canonical"])
    return payload


def assert_corpus_matches_sources(paths, sources_payload):
    """Refuse to run when the corpus changed after the source list was built."""
    current = dict(corpus_inventory(paths))
    declared = {item["source"]: item["sha256"] for item in sources_payload["sources"]}
    added = sorted(set(current) - set(declared))
    removed = sorted(set(declared) - set(current))
    changed = sorted(
        name for name in set(current) & set(declared)
        if current[name] != declared[name]
    )
    if added or removed or changed:
        raise RuntimeError(
            "corpus changed after the B3 source list was built "
            f"(added={len(added)} {added[:3]}, removed={len(removed)} {removed[:3]}, "
            f"changed={len(changed)} {changed[:3]}); re-run --plan to refresh it "
            "explicitly instead of silently reusing stale ids"
        )
    return True


def empty_manifest(sources_payload):
    return {
        "version": 2,
        "status": "partial",
        "stage": "B3",
        "plan_type": sources_payload.get("plan_type"),
        "model": sources_payload.get("model", MODEL_NAME),
        "model_fingerprint": sources_payload.get("model_fingerprint"),
        "embedding_text_version": sources_payload.get("embedding_text_version",
                                                     EMBEDDING_TEXT_VERSION),
        "chunking": sources_payload.get("chunking"),
        "sources_fingerprint": sources_payload.get("sources_fingerprint"),
        "planned_chunks": sources_payload.get("planned_chunks"),
        "files": {},
        "total_chunks": 0,
        "built_at": None,
    }


def load_manifest(paths, sources_payload):
    manifest = index_common.read_json(paths.manifest, default=None)
    if manifest is None:
        return empty_manifest(sources_payload)
    manifest.setdefault("files", {})
    return manifest


def manifest_config_matches(manifest, sources_payload):
    """Return a list of configuration mismatches that force a fresh rebuild."""
    problems = []
    checks = (
        ("model", sources_payload.get("model")),
        ("model_fingerprint", sources_payload.get("model_fingerprint")),
        ("embedding_text_version", sources_payload.get("embedding_text_version")),
        ("chunking", sources_payload.get("chunking")),
    )
    for field, expected in checks:
        recorded = manifest.get(field)
        if recorded is not None and expected is not None and recorded != expected:
            problems.append(f"{field}: manifest={recorded!r} sources={expected!r}")
    return problems


def save_manifest(paths, manifest):
    manifest["total_chunks"] = sum(
        entry.get("chunks", 0) for entry in manifest.get("files", {}).values()
        if isinstance(entry, dict)
    )
    manifest["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    index_common.write_json_atomic(paths.manifest, manifest)


def progress_state(paths):
    return index_common.read_json(paths.progress, default=None)


# Progress notes that a later stage of the same build contradicts.  They are
# reported as stale instead of being shown as current state.
SUPERSEDED_NOTES = ("no encoding started",)


def write_progress(paths, manifest, status, sources_payload, **fields):
    """Persist run state.  Completion counts are always derived from the manifest."""
    files = manifest.get("files", {})
    completed = set(files)
    payload = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "stage": "B3",
        "status": status,
        "pid": os.getpid(),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "manifest": str(paths.manifest),
        "staging_db": str(paths.staging_db),
        "sources": str(paths.sources),
        "completed_source_entries": len(completed),
        "completed_canonical_sources": sum(
            1 for item in sources_payload["sources"]
            if item["is_canonical"] and item["source"] in completed
        ),
        "total_canonical_sources": sum(
            1 for item in sources_payload["sources"] if item["is_canonical"]
        ),
        "chunks": manifest.get("total_chunks", 0),
    }
    existing = progress_state(paths) or {}
    payload["started_at"] = existing.get("started_at") or fields.get(
        "started_at"
    ) or time.strftime("%Y-%m-%dT%H:%M:%S%z")
    payload["errors"] = (existing.get("errors") or [])[-5:]
    if "in_progress" in fields:
        payload["in_progress"] = fields["in_progress"]
    elif existing.get("in_progress") and status in ("running", "stopping"):
        payload["in_progress"] = existing["in_progress"]
    else:
        payload["in_progress"] = None
    # A note describes the run that wrote it, so it is never inherited: a
    # surviving "no encoding started" next to a non-zero chunk count is exactly
    # the stale state this runner must not publish.
    for key in ("last_committed_source", "device", "unit_chunks", "note",
                "migrated_from", "stop_reason", "run_id"):
        if key in fields:
            payload[key] = fields[key]
        elif key in existing and key in ("last_committed_source", "device",
                                         "unit_chunks", "migrated_from", "run_id"):
            if existing[key] is not None:
                payload[key] = existing[key]
    if status == "running" and "run_id" not in payload:
        payload["run_id"] = uuid.uuid4().hex[:12]
    index_common.write_json_atomic(paths.progress, payload)
    return payload


def derived_run_state(progress, lock_free, complete):
    """Turn persisted progress plus the real lock state into a run state.

    A PID is never proof that a runner is alive, and the persisted status is
    never trusted on its own: a record that claims ``running`` while the target
    writer lock is free describes a run that has stopped (crash, kill, or a
    power loss) and must be reported as interrupted.
    """
    recorded = (progress or {}).get("status")
    if complete:
        return "completed" if lock_free else "running"
    if recorded in ("running", "stopping"):
        if lock_free:
            return DERIVED_INTERRUPTED
        return "stopping" if recorded == "stopping" else DERIVED_RUNNING
    if recorded:
        return recorded
    return DERIVED_IDLE


def status_payload(paths):
    """Derived status that never loads a model or opens Chroma."""
    manifest = index_common.read_json(paths.manifest, default=None)
    sources = index_common.read_json(paths.sources, default=None)
    progress = index_common.read_json(paths.progress, default=None)
    files = (manifest or {}).get("files") or {}
    completed = set(files)
    canonical = [
        item["source"] for item in (sources or {}).get("sources", [])
        if item["is_canonical"]
    ]
    remaining = [name for name in sorted(canonical) if name not in completed]
    lock_free = _lock_is_free(paths)
    complete = bool(canonical) and not remaining
    recorded_status = (progress or {}).get("status")
    state = derived_run_state(progress, lock_free, complete)
    note = (progress or {}).get("note")
    note_stale = bool(
        note and any(marker in note for marker in SUPERSEDED_NOTES)
        and (progress or {}).get("chunks")
    )
    return {
        "state": state,
        "status": state,
        "recorded_progress_status": recorded_status,
        "progress_stale": state == DERIVED_INTERRUPTED and recorded_status in
        ("running", "stopping"),
        "note": note,
        "note_stale": note_stale,
        "last_checkpoint_at": (progress or {}).get("updated_at"),
        "last_committed_source": (progress or {}).get("last_committed_source"),
        "in_progress": (progress or {}).get("in_progress"),
        "recovery_hint": (
            "no process holds the staging writer lock; resume with "
            "'rebuild_b3.py --resume' (add --clear-stop for a pending request)"
            if state == DERIVED_INTERRUPTED else None
        ),
        "sources_file": str(paths.sources),
        "sources_present": sources is not None,
        "staging_db": str(paths.staging_db),
        "staging_db_present": Path(paths.staging_db).is_dir(),
        "manifest": str(paths.manifest),
        "manifest_present": manifest is not None,
        "manifest_status": (manifest or {}).get("status"),
        "total_chunks": (manifest or {}).get("total_chunks"),
        "completed_source_entries": len(completed),
        "completed_canonical_sources": sum(1 for name in canonical if name in completed),
        "total_canonical_sources": len(canonical),
        "remaining_canonical_sources": len(remaining),
        "next_source": remaining[0] if remaining else None,
        "planned_chunks": (sources or {}).get("planned_chunks"),
        "progress": progress,
        "stop_requested": stop_requested(paths),
        "stop_token": stop_token(paths),
        "writer_lock": str(paths.lock),
        "writer_lock_free": lock_free,
        "complete": complete,
    }


def _lock_is_free(paths):
    """Probe this target's writer lock.  A free lock is never proof of a run."""
    try:
        with FileLock(paths.lock):
            return True
    except LockUnavailable:
        return False


def acquire_writer(paths, label="B3 writer"):
    """Take the target writer lock for the whole duration of an operation."""
    lock = FileLock(paths.lock)
    try:
        lock.acquire()
    except LockUnavailable as exc:
        raise RuntimeError(
            f"another {label} already owns {paths.staging_db} ({paths.lock}); "
            "wait for it to exit or stop it before changing the same target"
        ) from exc
    return lock


# ---------------------------------------------------------------------------
# Staging verification
# ---------------------------------------------------------------------------

def _source_rows(collection, source):
    rows = collection.get(where={"source": source}, include=["metadatas", "documents"])
    return (
        list(rows.get("ids") or []),
        list(rows.get("metadatas") or []),
        list(rows.get("documents") or []),
    )


def collection_sources_and_versions(collection):
    return collection_scan(collection)


def collection_scan(collection):
    """Paged metadata scan -> ({source: row count}, {embedding_text_version: count})."""
    total = collection.count()
    counts = {}
    versions = {}
    for offset in range(0, total, 10000):
        rows = collection.get(limit=10000, offset=offset, include=["metadatas"])
        for metadata in rows.get("metadatas") or []:
            metadata = metadata or {}
            source = metadata.get("source")
            if not source:
                raise RuntimeError(f"row without source metadata at offset {offset}")
            counts[source] = counts.get(source, 0) + 1
            version = metadata.get("embedding_text_version")
            versions[version] = versions.get(version, 0) + 1
    return counts, versions


def expected_ids(item, chunk_count):
    return [f"{item['sha256'][:12]}-{index:03d}" for index in range(chunk_count)]


def verify_source_rows(collection, item, entry, meta, rows=None):
    """Verify one committed source against its manifest entry and the source list."""
    build_index = build_index_module()
    expected_count = entry.get("chunks", 0)
    ids, metadatas, documents = rows if rows is not None else _source_rows(
        collection, item["source"]
    )
    problems = []
    if len(ids) != expected_count:
        problems.append(f"rows={len(ids)} entry={expected_count}")
    if expected_count:
        if sorted(ids) != sorted(expected_ids(item, expected_count)):
            problems.append("row ids do not match the deterministic ids")
        expected_paper_id = build_index.paper_id_for(item["source"], meta)
        seen_chunks = set()
        for row_id, metadata, document in zip(ids, metadatas, documents):
            metadata = metadata or {}
            if metadata.get("source") != item["source"]:
                problems.append(f"{row_id}: source metadata mismatch")
                break
            if metadata.get("embedding_text_version") != EMBEDDING_TEXT_VERSION:
                problems.append(f"{row_id}: embedding_text_version mismatch")
                break
            if metadata.get("paper_id") != expected_paper_id:
                problems.append(f"{row_id}: paper_id mismatch")
                break
            if (metadata.get("title") or "") != (meta.get("title") or ""):
                problems.append(f"{row_id}: title changed since the row was written")
                break
            if "section_type" not in metadata or "section" not in metadata:
                problems.append(f"{row_id}: section metadata missing")
                break
            chunk_index = metadata.get("chunk")
            if not isinstance(chunk_index, int) or chunk_index in seen_chunks:
                problems.append(f"{row_id}: invalid or duplicated chunk index")
                break
            seen_chunks.add(chunk_index)
            if not isinstance(document, str) or not document.strip():
                problems.append(f"{row_id}: empty stored passage")
                break
    return problems


def deep_verify_source(source_path, entry, meta, tokenizer, ids, metadatas, documents):
    """Re-chunk a committed source and compare it with the stored rows.

    Proves the stored passages and section paths are exactly what the current
    chunker and representation inputs produce; no embedding is recomputed.
    """
    build_index = build_index_module()
    if source_path is None:
        return ["source path unavailable for deep verification"]
    text = Path(source_path).read_text(encoding="utf-8", errors="replace")
    chunks = build_index.chunk_markdown(text, tokenizer)
    problems = []
    if len(chunks) != entry.get("chunks", 0):
        return [f"re-chunked count {len(chunks)} != entry {entry.get('chunks')}"]
    order = {}
    for row_id, metadata, document in zip(ids, metadatas, documents):
        index = (metadata or {}).get("chunk")
        if not isinstance(index, int) or index in order:
            return ["stored rows have no usable chunk index"]
        order[index] = (document, (metadata or {}).get("section"), metadata)
    for index, chunk in enumerate(chunks):
        stored = order.get(index)
        if stored is None:
            problems.append(f"chunk {index} missing from staging")
            continue
        document, section, metadata = stored
        if document != chunk["text"]:
            problems.append(f"chunk {index} passage differs from the current chunker")
        if section != chunk["section"]:
            problems.append(f"chunk {index} section path differs")
        expected_text = build_index.build_embedding_text(
            meta.get("title"), chunk["section"], chunk["text"]
        )
        if metadata.get("section_type") != build_index.section_type(chunk["section"]):
            problems.append(f"chunk {index} section_type differs")
        if not expected_text:
            problems.append(f"chunk {index} embedding text is empty")
    return problems


def reconcile_staging(collection, paths, sources_payload, manifest, deep=False,
                      tokenizer=None):
    """Bring the staging database back to the committed manifest boundary.

    * rows of a source that is not committed are removed (unfinished work),
    * manifest entries that no longer match the source list or the stored rows
      are dropped so the source is redone,
    * unknown rows are removed.
    """
    build_index = build_index_module()
    by_name = source_by_name(sources_payload)
    files = manifest["files"]
    report = {
        "dropped_stale_entries": [],
        "dropped_unverified_entries": [],
        "removed_unknown_sources": [],
        "verified_sources": 0,
    }

    for name in sorted(set(files) - set(by_name)):
        report["dropped_stale_entries"].append(name)
        files.pop(name, None)

    counts, versions = collection_scan(collection)
    by_stem, by_folder, by_doi = build_index.load_meta()
    source_paths = dict(corpus_items(paths))
    for name in sorted(files):
        item = by_name[name]
        entry = files[name]
        meta, _is_alias = build_index.meta_for(name, by_stem, by_folder, by_doi)
        expected_repr = build_index.representation_signature(meta)
        if (
            entry.get("sha256") != item["sha256"]
            or entry.get("repr") != expected_repr
            or entry.get("embedding_text_version") != EMBEDDING_TEXT_VERSION
            or entry.get("chunks") != item["estimated_chunks"]
        ):
            report["dropped_unverified_entries"].append(name)
            files.pop(name, None)
            if counts.get(name):
                collection.delete(where={"source": name})
            continue
        rows = _source_rows(collection, name) if counts.get(name) or entry.get("chunks") else (
            [], [], []
        )
        problems = verify_source_rows(collection, item, entry, meta, rows=rows)
        if problems and rows[0]:
            # Rows whose deterministic id does not belong to this content
            # version are leftovers of another attempt; they are removed and
            # the committed rows are re-checked before the source is redone.
            expected = set(expected_ids(item, entry.get("chunks", 0)))
            stray = [row_id for row_id in rows[0] if row_id not in expected]
            if stray:
                collection.delete(ids=stray)
                rows = _source_rows(collection, name)
                problems = verify_source_rows(collection, item, entry, meta, rows=rows)
        if not problems and deep and entry.get("chunks"):
            problems = deep_verify_source(
                source_paths.get(name), entry, meta, tokenizer, *rows
            )
        if problems:
            report["dropped_unverified_entries"].append(f"{name}: {problems[:2]}")
            files.pop(name, None)
            if counts.get(name):
                collection.delete(where={"source": name})
            continue
        report["verified_sources"] += 1

    committed = set(files)
    for source in sorted(set(counts) - committed):
        report["removed_unknown_sources"].append(source)
        collection.delete(where={"source": source})
    return report, versions


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _delete_source(collection, source):
    collection.delete(where={"source": source})


def _entry_for(build_index, item, meta, chunk_count, dup_of=None):
    return build_index.manifest_entry(
        item["source"], item["sha256"], chunk_count, meta, dup_of=dup_of
    )


def run_resume(paths=DEFAULT_PATHS, device="cpu", clear_stop=False, backends=None,
               unit_chunks=UNIT_CHUNKS, deep_verify=False, max_sources=None,
               pre_start_stop_token=None):
    """Process sources continuously until complete, stopped, or failed."""
    build_index = build_index_module()
    backends = backends or RealBackends(paths, device)
    lock = acquire_writer(paths, "B3 writer")

    try:
        pending_token = stop_token(paths)
        if pending_token is not None:
            if not clear_stop:
                return _finish(paths, None, None, "stopped",
                               "stop request already present; no model loaded, "
                               "no encoding or writing started")
            if pre_start_stop_token != pending_token:
                return _finish(paths, None, None, "stopped",
                               "a new stop request arrived while starting; kept")
            if not consume_stop(paths, pending_token):
                return _finish(paths, None, None, "stopped",
                               "a new stop request arrived while starting; kept")
            print("consumed the stop request that was pending at startup", flush=True)

        sources_payload = load_sources(paths)
        assert_corpus_matches_sources(paths, sources_payload)
        manifest = load_manifest(paths, sources_payload)
        problems = manifest_config_matches(manifest, sources_payload)
        if problems:
            raise RuntimeError(
                "staging manifest was produced with a different configuration: "
                + "; ".join(problems)
            )
        if manifest["files"] and not Path(paths.staging_db).is_dir():
            raise RuntimeError(
                f"manifest lists {len(manifest['files'])} completed sources but "
                f"{paths.staging_db} does not exist; refusing to start a new empty index"
            )

        if stop_requested(paths):
            return _finish(paths, manifest, sources_payload, "stopped",
                           "stop requested before any encoding")

        collection = backends.collection
        if collection.count():
            foreign = {
                version: count for version, count in collection_scan(collection)[1].items()
                if count and version != EMBEDDING_TEXT_VERSION
            }
            if foreign:
                raise RuntimeError(
                    f"staging contains rows with another representation: {foreign}"
                )

        by_stem, by_folder, by_doi = build_index.load_meta()
        report, _versions = reconcile_staging(
            collection, paths, sources_payload, manifest,
            deep=deep_verify, tokenizer=backends.tokenizer,
        )
        for key in ("dropped_stale_entries", "dropped_unverified_entries",
                    "removed_unknown_sources"):
            if report[key]:
                print(f"reconcile {key}: {len(report[key])} {report[key][:3]}", flush=True)
        print(f"reconcile verified {report['verified_sources']} completed sources", flush=True)

        by_name = source_by_name(sources_payload)
        aliases = aliases_by_canonical(sources_payload)
        order = canonical_sources(sources_payload)
        paths_by_name = dict(corpus_items(paths))
        pending_sources = [name for name in order if name not in manifest["files"]]

        if not pending_sources:
            return _complete(paths, manifest, sources_payload, collection, build_index)

        start = time.time()
        write_progress(paths, manifest, "running", sources_payload,
                       device=device, unit_chunks=unit_chunks,
                       in_progress=None, started_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       run_id=uuid.uuid4().hex[:12])
        total_canonical = len(order)
        done_this_run = 0
        for name in pending_sources:
            if max_sources is not None and done_this_run >= max_sources:
                write_progress(paths, manifest, "stopped", sources_payload,
                               note=f"max_sources={max_sources} reached (test cap)")
                print(f"stopping after {done_this_run} sources (test cap)", flush=True)
                return manifest
            if stop_requested(paths):
                write_progress(paths, manifest, "stopped", sources_payload,
                               stop_reason="stop request seen before the next source")
                print("stop requested; no new source started", flush=True)
                return manifest

            item = by_name[name]
            meta, _is_alias = build_index.meta_for(name, by_stem, by_folder, by_doi)
            if not item["estimated_chunks"]:
                manifest["files"][name] = _entry_for(build_index, item, meta, 0)
                _commit_aliases(build_index, manifest, by_name, aliases, name,
                                by_stem, by_folder, by_doi)
                save_manifest(paths, manifest)
                write_progress(paths, manifest, "running", sources_payload,
                               last_committed_source=name, in_progress=None,
                               device=device, unit_chunks=unit_chunks)
                done_this_run += 1
                print(f"[{_completed_canonical(manifest, order)}/{total_canonical}] "
                      f"{name} -> 0 chunks (no indexable text)", flush=True)
                continue

            write_progress(paths, manifest, "running", sources_payload,
                           in_progress={"source": name,
                                        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                        "rows_written": 0},
                           device=device, unit_chunks=unit_chunks)
            outcome = _process_source(
                paths, build_index, collection, backends, item, meta, manifest,
                sources_payload, unit_chunks, paths_by_name.get(name),
            )
            if outcome == "stopped":
                _delete_source(collection, name)
                write_progress(paths, manifest, "stopped", sources_payload,
                               stop_reason=f"stopped during {name}", in_progress=None)
                print(f"stopped during {name}; its unfinished rows were discarded", flush=True)
                return manifest
            if outcome == "failed":
                write_progress(paths, manifest, "failed", sources_payload,
                               in_progress=None)
                return manifest
            _commit_aliases(build_index, manifest, by_name, aliases, name,
                            by_stem, by_folder, by_doi)
            save_manifest(paths, manifest)
            write_progress(paths, manifest, "running", sources_payload,
                           last_committed_source=name, in_progress=None,
                           device=device, unit_chunks=unit_chunks)
            done_this_run += 1
            elapsed = time.time() - start
            print(
                f"[{_completed_canonical(manifest, order)}/{total_canonical}] {name} -> "
                f"{manifest['files'][name]['chunks']} chunks "
                f"(total {manifest['total_chunks']}, {elapsed:.0f}s this run)",
                flush=True,
            )

        if stop_requested(paths):
            write_progress(paths, manifest, "stopped", sources_payload,
                           stop_reason="stop request seen before final verification")
            return manifest
        return _complete(paths, manifest, sources_payload, collection, build_index)
    finally:
        try:
            backends.close()
        finally:
            lock.release()


def _completed_canonical(manifest, canonical_order):
    files = manifest["files"]
    return sum(1 for name in canonical_order if name in files)


def _commit_aliases(build_index, manifest, by_name, aliases, canonical,
                    by_stem, by_folder, by_doi):
    for alias in aliases.get(canonical, []):
        alias_item = by_name[alias]
        alias_meta, _ = build_index.meta_for(alias, by_stem, by_folder, by_doi)
        manifest["files"][alias] = _entry_for(
            build_index, alias_item, alias_meta, 0, dup_of=canonical
        )


def _process_source(paths, build_index, collection, backends, item, meta, manifest,
                    sources_payload, unit_chunks, source_path):
    """Chunk, encode and commit one canonical source. Returns ok/stopped/failed."""
    name = item["source"]
    if source_path is None:
        _record_error(paths, manifest, sources_payload, name, "source file missing")
        return "failed"
    digest = index_common.sha256_file(source_path)
    if digest != item["sha256"]:
        _record_error(
            paths, manifest, sources_payload, name,
            "content sha changed; re-run --plan before reusing old ids",
        )
        return "failed"
    text = Path(source_path).read_text(encoding="utf-8", errors="replace")
    chunks = build_index.chunk_markdown(text, backends.tokenizer)
    if len(chunks) != item["estimated_chunks"]:
        _record_error(
            paths, manifest, sources_payload, name,
            f"chunk count changed ({len(chunks)} != {item['estimated_chunks']})",
        )
        return "failed"

    documents = [chunk["text"] for chunk in chunks]
    sections = [chunk["section"] for chunk in chunks]
    metadatas = [
        build_index.chunk_metadata(meta, name, sections[index], index, False)
        for index in range(len(chunks))
    ]
    embedding_texts = [
        build_index.build_embedding_text(meta.get("title"), sections[index], documents[index])
        for index in range(len(chunks))
    ]
    ids = expected_ids(item, len(chunks))

    written = 0
    for start in range(0, len(chunks), unit_chunks):
        end = min(start + unit_chunks, len(chunks))
        if stop_requested(paths):
            return "stopped"
        embeddings = backends.encode(embedding_texts[start:end])
        if embeddings is None or len(embeddings) != end - start:
            _record_error(paths, manifest, sources_payload, name,
                          "encoder returned a different number of vectors")
            return "failed"
        if stop_requested(paths):
            # Nothing has been written for this unit yet; drop the whole source.
            return "stopped"
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            # Pass the encoder output through unchanged: Chroma accepts a numpy
            # array, while a list of numpy scalars is rejected.
            embeddings=embeddings,
        )
        written = end
        if stop_requested(paths):
            return "stopped"
    if written != len(chunks):
        _record_error(paths, manifest, sources_payload, name, "incomplete write")
        return "failed"

    stored_ids, stored_metadatas, stored_documents = _source_rows(collection, name)
    if len(stored_ids) != len(chunks):
        _record_error(paths, manifest, sources_payload, name,
                      f"staging holds {len(stored_ids)} rows for {len(chunks)} chunks")
        return "failed"
    entry = _entry_for(build_index, item, meta, len(chunks))
    problems = verify_source_rows(
        collection, item, entry, meta,
        rows=(stored_ids, stored_metadatas, stored_documents),
    )
    if problems:
        _record_error(paths, manifest, sources_payload, name, "; ".join(problems[:2]))
        return "failed"
    manifest["files"][name] = entry
    return "ok"


def _record_error(paths, manifest, sources_payload, source, message):
    payload = progress_state(paths) or {}
    errors = (payload.get("errors") or [])[-4:]
    errors.append({
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source,
        "error": message,
    })
    payload.setdefault("schema_version", PROGRESS_SCHEMA_VERSION)
    payload.setdefault("stage", "B3")
    payload["errors"] = errors
    payload["status"] = "failed"
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    payload["pid"] = os.getpid()
    payload["in_progress"] = None
    payload["manifest"] = str(paths.manifest)
    payload["staging_db"] = str(paths.staging_db)
    payload["sources"] = str(paths.sources)
    files = (manifest or {}).get("files", {})
    payload["completed_source_entries"] = len(files)
    payload["chunks"] = (manifest or {}).get("total_chunks", 0)
    index_common.write_json_atomic(paths.progress, payload)
    print(f"FAILED {source}: {message}", flush=True)


def _finish(paths, manifest, sources_payload, status, reason):
    if sources_payload is None:
        sources_payload = index_common.read_json(paths.sources, default=None)
    if sources_payload is None:
        print(f"no source list at {paths.sources}; nothing to run", flush=True)
        return manifest
    if manifest is None:
        manifest = load_manifest(paths, sources_payload)
    write_progress(paths, manifest, status, sources_payload,
                   stop_reason=reason, in_progress=None)
    print(reason, flush=True)
    return manifest


def _complete(paths, manifest, sources_payload, collection, build_index):
    """Verify the whole staging pair and mark the build complete (no publish)."""
    verification = verify_complete_staging(paths, sources_payload, manifest, collection)
    manifest["status"] = "complete"
    save_manifest(paths, manifest)
    marker = build_index.write_ready_marker(
        paths.manifest, manifest, paths.staging_db,
        corpus_inventory(paths), paths.model_dir,
    )
    write_progress(paths, manifest, "completed", sources_payload,
                   in_progress=None, note="build complete; publishing is a separate step")
    print(json.dumps(verification, ensure_ascii=False, indent=2), flush=True)
    print(f"B3 build COMPLETE (not published). Ready marker: "
          f"{build_index.ready_marker_path(paths.manifest)}", flush=True)
    print(
        "Switch with:\n"
        f'  .venv\\Scripts\\python.exe -B index\\publish_index.py '
        f'--staging-db "{paths.staging_db}" '
        f'--staging-manifest "{paths.manifest}"',
        flush=True,
    )
    return manifest


def structure_report(sources_payload, manifest, collection):
    """Structural summary of the committed part of a partial staging index."""
    by_name = source_by_name(sources_payload)
    files = manifest["files"]
    counts, versions = collection_scan(collection)
    expected_nonzero = {
        name for name in files
        if files[name].get("chunks") and by_name.get(name)
    }
    return {
        "manifest_entries": len(files),
        "committed_chunks": sum(entry.get("chunks", 0) for entry in files.values()),
        "rows_in_staging": collection.count(),
        "rows_match_manifest": collection.count() == sum(
            entry.get("chunks", 0) for entry in files.values()
        ),
        "committed_nonzero_sources": len(expected_nonzero),
        "unknown_row_sources": sorted(set(counts) - set(files)),
        "foreign_representations": {
            version: count for version, count in versions.items()
            if count and version != EMBEDDING_TEXT_VERSION
        },
    }


def verify_complete_staging(paths, sources_payload, manifest, collection):
    """Full structural verification against the whole source list."""
    by_name = source_by_name(sources_payload)
    files = manifest["files"]
    canonical = canonical_sources(sources_payload)
    missing = [name for name in canonical if name not in files]
    if missing:
        raise RuntimeError(f"{len(missing)} canonical sources are not committed: {missing[:3]}")
    alias_missing = [
        item["source"] for item in sources_payload["sources"]
        if not item["is_canonical"] and item["source"] not in files
    ]
    if alias_missing:
        raise RuntimeError(f"missing duplicate-alias entries: {alias_missing[:3]}")

    expected_chunks = sum(
        by_name[name]["estimated_chunks"] for name in canonical
    )
    if manifest["total_chunks"] != expected_chunks:
        raise RuntimeError(
            f"manifest chunks {manifest['total_chunks']} != planned {expected_chunks}"
        )
    counts, versions = collection_scan(collection)
    foreign = {v: c for v, c in versions.items() if c and v != EMBEDDING_TEXT_VERSION}
    if foreign:
        raise RuntimeError(f"staging holds foreign representations: {foreign}")
    expected_nonzero = {name for name in canonical if by_name[name]["estimated_chunks"]}
    if set(counts) != expected_nonzero:
        extra = sorted(set(counts) - expected_nonzero)
        absent = sorted(expected_nonzero - set(counts))
        raise RuntimeError(
            f"staging source set mismatch: extra={extra[:3]} absent={absent[:3]}"
        )
    total_rows = collection.count()
    if total_rows != manifest["total_chunks"]:
        raise RuntimeError(
            f"staging rows {total_rows} != manifest chunks {manifest['total_chunks']}"
        )
    zero_aliases = [
        item["source"] for item in sources_payload["sources"]
        if not item["is_canonical"]
    ]
    for name in zero_aliases:
        if files[name].get("chunks"):
            raise RuntimeError(f"duplicate alias {name} must carry zero chunks")
        if files[name].get("dup_of") != by_name[name]["canonical_source"]:
            raise RuntimeError(f"duplicate alias {name} points at the wrong canonical source")
    return {
        "canonical_sources": len(canonical),
        "alias_entries": len(zero_aliases),
        "chunks": total_rows,
        "nonzero_sources": len(expected_nonzero),
        "manifest_entries": len(files),
    }


def verify_only(paths=DEFAULT_PATHS, device="cpu", backends=None, lock=None,
                deep=False):
    own_backends = backends is None
    own_lock = lock is None and backends is None
    if own_lock:
        lock = acquire_writer(paths, "B3 writer (verify)")
    sources_payload = load_sources(paths)
    manifest = load_manifest(paths, sources_payload)
    backends = backends or RealBackends(paths, device)
    try:
        result = verify_complete_staging(
            paths, sources_payload, manifest, backends.collection
        )
        if deep:
            result["deep"] = verify_sources_deep(
                paths, sources_payload, manifest, backends.collection,
                backends.tokenizer,
            )
        return result
    finally:
        if own_backends:
            backends.close()
        if own_lock:
            lock.release()


def verify_sources_deep(paths, sources_payload, manifest, collection, tokenizer):
    """Re-check every committed source row-by-row against its content.

    Structural verification compares source sets and totals; this pass proves
    the stored passages, section paths, chunk ids, metadata and canonical/alias
    relations of each committed source, re-chunking the Markdown on disk.  It is
    strictly read-only: a mismatch is reported, never repaired.
    """
    build_index = build_index_module()
    by_name = source_by_name(sources_payload)
    by_stem, by_folder, by_doi = build_index.load_meta()
    source_paths = dict(corpus_items(paths))
    counts, _versions = collection_scan(collection)
    report = {"checked": 0, "problems": [], "empty_sources": 0}
    for name in sorted(manifest["files"]):
        item = by_name.get(name)
        entry = manifest["files"][name]
        if item is None:
            report["problems"].append({"source": name, "problems": ["not in the source list"]})
            continue
        meta, _is_alias = build_index.meta_for(name, by_stem, by_folder, by_doi)
        rows = _source_rows(collection, name) if counts.get(name) else ([], [], [])
        problems = verify_source_rows(collection, item, entry, meta, rows=rows)
        if not problems and entry.get("chunks"):
            problems = deep_verify_source(
                source_paths.get(name), entry, meta, tokenizer, *rows
            )
        elif not entry.get("chunks"):
            report["empty_sources"] += 1
            if item["is_canonical"] and item["estimated_chunks"]:
                problems = [
                    f"canonical source planned {item['estimated_chunks']} chunks but "
                    "the ledger commits zero"
                ]
            if not item["is_canonical"] and entry.get("dup_of") != item["canonical_source"]:
                problems = ["duplicate alias points at the wrong canonical source"]
        if problems:
            report["problems"].append({"source": name, "problems": problems[:3]})
        else:
            report["checked"] += 1
    report["committed_entries"] = len(manifest["files"])
    report["rows_in_staging"] = sum(counts.values())
    return report


# ---------------------------------------------------------------------------
# One-time migration of the former four-batch staging
# ---------------------------------------------------------------------------

def adopt_existing(paths=DEFAULT_PATHS, device="cpu", backends=None,
                   legacy=None, deep=True):
    """Migrate the four-batch artifacts into the continuous runner layout.

    The staging directory is renamed *before* any Chroma client is opened: a
    Windows SQLite handle would block the rename, and a crash before the new
    ledger is written only costs a re-verification, because the legacy plan and
    manifest are retained until the migration completes.
    """
    legacy = dict(legacy or LEGACY)
    legacy_staging = Path(legacy["staging"])
    lock = acquire_writer(paths, "B3 writer")

    try:
        new_exists = Path(paths.staging_db).is_dir()
        legacy_exists = legacy_staging.is_dir()
        if new_exists and legacy_exists:
            raise RuntimeError(
                f"both {paths.staging_db} and {legacy_staging} exist; resolve this "
                "manually before migrating"
            )
        if not new_exists and not legacy_exists:
            raise RuntimeError(
                f"neither {paths.staging_db} nor {legacy_staging} exists; nothing to adopt"
            )

        payload = _adopt_from_legacy_files(legacy)
        if not payload["sources"]["sources"]:
            raise RuntimeError("legacy plan contains no sources")

        migrated_now = False
        if not new_exists:
            legacy_staging.rename(paths.staging_db)
            migrated_now = True
            print(f"migrated staging {legacy_staging} -> {paths.staging_db}", flush=True)

        existing = index_common.read_json(paths.manifest, default=None)
        already_advanced = bool(existing and existing.get("files"))
        if already_advanced:
            # A ledger already exists (perhaps with work done after the
            # migration); verify it and never downgrade it to the legacy one.
            payload["manifest"] = existing
            payload["manifest"].setdefault("files", {})

        own_backends = backends is None
        active = backends or RealBackends(paths, device)
        active_collection = None
        try:
            active_collection = active.collection
            report = _adopt_verify_against(
                paths, active, active_collection, payload, deep=deep
            )
            structural = structure_report(
                payload["sources"], payload["manifest"], active_collection
            )
        finally:
            if own_backends:
                active.close()

        index_common.write_json_atomic(paths.sources, payload["sources"])
        save_manifest(paths, payload["manifest"])
        write_progress(
            paths, payload["manifest"], "stopped", payload["sources"],
            in_progress=None, migrated_from=str(legacy_staging),
            note="migrated from the four-batch staging; no encoding started",
        )
        verification = structural
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if migrated_now:
            _archive_legacy(legacy)
        result = dict(report)
        result["migrated_now"] = migrated_now
        result["already_advanced"] = already_advanced
        result["structural"] = verification
        return result
    finally:
        lock.release()


def _paths_with_staging(paths, staging_db):
    fields = dict(paths.__dict__)
    fields["staging_db"] = Path(staging_db)
    return B3Paths(**fields)


def _adopt_from_legacy_files(legacy):
    """Read the legacy plan/manifest into the continuous-runner schema."""
    plan_path = Path(legacy["plan"])
    manifest_path = Path(legacy["manifest"])
    plan = index_common.read_json(plan_path, default=None)
    if plan is None:
        raise RuntimeError(f"legacy plan is missing: {plan_path}")
    sources = []
    for item in plan["sources"]:
        sources.append({
            "source": item["source"],
            "sha256": item["sha256"],
            "canonical_source": item["canonical_source"],
            "is_canonical": item["is_canonical"],
            "estimated_chunks": item["estimated_chunks"],
            "paper_id": item.get("paper_id", ""),
            "title": item.get("title", ""),
            "mac": item.get("mac", "unknown"),
            "supp": item.get("supp", "no"),
        })
    sources_payload = {
        "schema_version": SOURCES_SCHEMA_VERSION,
        "stage": "B3",
        "plan_type": "continuous-by-source-v1",
        "model": plan.get("model", MODEL_NAME),
        "model_fingerprint": plan.get("model_fingerprint"),
        "embedding_text_version": plan.get("embedding_text_version",
                                           EMBEDDING_TEXT_VERSION),
        "chunking": plan.get("chunking"),
        "source_order": plan.get("source_order"),
        "duplicate_policy": (
            "sha-identical sources share one canonical content source "
            "(PMID_* preferred); aliases carry zero chunks and a dup_of entry"
        ),
        "sources": sources,
        "migrated_from": {
            "plan": str(plan_path),
            "plan_fingerprint": plan.get("plan_fingerprint"),
        },
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    sources_payload["sources_fingerprint"] = index_common.corpus_fingerprint(
        [(item["source"], item["sha256"]) for item in sources]
    )
    sources_payload["planned_chunks"] = sum(
        item["estimated_chunks"] for item in sources if item["is_canonical"]
    )
    sources_payload["canonical_source_count"] = sum(
        1 for item in sources if item["is_canonical"]
    )

    old_manifest = index_common.read_json(manifest_path, default=None) or {"files": {}}
    build_index = build_index_module()
    by_name = {item["source"]: item for item in sources}
    by_stem, by_folder, by_doi = build_index.load_meta()
    files = {}
    for name, entry in (old_manifest.get("files") or {}).items():
        item = by_name.get(name)
        if item is None:
            raise RuntimeError(f"legacy manifest contains an unknown source: {name}")
        meta, _is_alias = build_index.meta_for(name, by_stem, by_folder, by_doi)
        files[name] = build_index.manifest_entry(
            name, item["sha256"], entry.get("chunks", 0), meta,
            dup_of=entry.get("dup_of"),
        )
    manifest = empty_manifest(sources_payload)
    manifest["files"] = files
    manifest["migrated_from"] = {
        "manifest": str(manifest_path),
        "plan_fingerprint": plan.get("plan_fingerprint"),
    }
    return {"sources": sources_payload, "manifest": manifest}


def _adopt_verify_against(paths, backends, collection, payload, deep, destructive=False):
    """Verify every migrated entry against the staging database.

    Verification is non-destructive by default: an entry that cannot be proven
    is dropped from the in-memory ledger and reported, and left for the normal
    resume reconciliation to remove.  A failed or interrupted migration must
    never delete rows before the new ledger has been written.
    """
    manifest = payload["manifest"]
    build_index = build_index_module()
    by_name = source_by_name(payload["sources"])
    by_stem, by_folder, by_doi = build_index.load_meta()
    tokenizer = backends.tokenizer if deep else None
    source_paths = dict(corpus_items(paths))
    counts, versions = collection_scan(collection)
    report = {"verified": 0, "dropped": [], "foreign_versions": {}, "deep": bool(deep)}
    for version, count in versions.items():
        if count and version != EMBEDDING_TEXT_VERSION:
            report["foreign_versions"][version] = count
    for name in sorted(list(manifest["files"])):
        item = by_name[name]
        entry = manifest["files"][name]
        meta, _is_alias = build_index.meta_for(name, by_stem, by_folder, by_doi)
        rows = _source_rows(collection, name) if counts.get(name) else ([], [], [])
        problems = verify_source_rows(collection, item, entry, meta, rows=rows)
        if not problems and deep and entry.get("chunks"):
            problems = deep_verify_source(
                source_paths.get(name), entry, meta, tokenizer, *rows
            )
        if problems:
            report["dropped"].append({"source": name, "problems": problems[:3]})
            manifest["files"].pop(name, None)
            if destructive and counts.get(name):
                collection.delete(where={"source": name})
            continue
        report["verified"] += 1
    committed = set(manifest["files"])
    for source in sorted(set(counts) - committed):
        report["dropped"].append(
            {"source": source, "problems": ["rows without a committed entry"]}
        )
        if destructive:
            collection.delete(where={"source": source})
    report["committed_entries"] = len(committed)
    report["chunks"] = sum(
        entry.get("chunks", 0) for entry in manifest["files"].values()
    )
    report["rows_in_staging"] = sum(counts.values())
    return report


def _adopt_verify(paths, payload, backends, deep):
    collection = backends.collection
    report = _adopt_verify_against(paths, backends, collection, payload, deep=deep)
    index_common.write_json_atomic(paths.manifest, payload["manifest"])
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def _archive_legacy(legacy):
    """Keep the former batch artifacts as history, not as a second entry point."""
    target = ROOT / "archive" / "B3_4batch"
    target.mkdir(parents=True, exist_ok=True)
    for key in ("plan", "manifest", "checkpoint"):
        source = Path(legacy[key])
        if source.is_file():
            shutil.copy2(source, target / source.name)
    print(f"legacy batch artifacts copied to {target} for the record", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None, paths=DEFAULT_PATHS):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true", help="(re)build the source list")
    group.add_argument("--adopt-existing", action="store_true",
                       help="one-time migration of the four-batch staging")
    group.add_argument("--resume", action="store_true", help="run continuously")
    group.add_argument("--status", action="store_true", help="derived status (no model/Chroma)")
    group.add_argument("--verify", action="store_true", help="verify the staging pair")
    group.add_argument("--stop", action="store_true", help="request a stop")
    parser.add_argument("--clear-stop", action="store_true",
                        help="with --resume: consume the stop request pending at startup")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--unit-chunks", type=int, default=UNIT_CHUNKS,
                        help="chunks per encoding unit (internal computation unit)")
    parser.add_argument("--max-sources", type=int, default=None,
                        help="test/debug cap; never marks the build complete")
    parser.add_argument("--no-deep-verify", action="store_true",
                        help="with --adopt-existing: skip re-chunk verification")
    parser.add_argument("--deep-verify", action="store_true",
                        help="with --resume: re-chunk every reused source and compare "
                             "the stored passages before accepting it")
    parser.add_argument("--json", action="store_true", help="machine-readable status")
    args = parser.parse_args(argv)

    if args.stop:
        payload = request_stop(paths)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.status:
        payload = status_payload(paths)
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.plan:
        # --plan rewrites the source list the runner reads, so it takes the same
        # writer lock and refuses to run next to a live runner.
        lock = acquire_writer(paths, "B3 writer (plan)")
        try:
            tokenizer = load_tokenizer(paths, args.device)
            payload = build_sources_payload(paths, tokenizer)
            index_common.write_json_atomic(paths.sources, payload)
        finally:
            lock.release()
        print(json.dumps({
            "sources": len(payload["sources"]),
            "canonical": payload["canonical_source_count"],
            "planned_chunks": payload["planned_chunks"],
            "fingerprint": payload["sources_fingerprint"],
        }, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.adopt_existing:
        result = adopt_existing(paths, device=args.device, deep=not args.no_deep_verify)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.verify:
        result = verify_only(paths, args.device, deep=args.deep_verify)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0

    install_signal_handlers()
    pre_start_token = stop_token(paths)
    run_resume(
        paths, device=args.device, clear_stop=args.clear_stop,
        unit_chunks=args.unit_chunks, max_sources=args.max_sources,
        pre_start_stop_token=pre_start_token, deep_verify=args.deep_verify,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
