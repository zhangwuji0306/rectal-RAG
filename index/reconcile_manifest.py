"""Repair manifest counts from an already-existing Chroma ``papers`` DB.

The normal build/ingest paths update the manifest at the end of a run.  If two
writers overlap, the last writer can persist an older in-memory snapshot. This
utility uses Chroma rows to repair source and total chunk counts while retaining
the existing v2 representation identity and source hashes. It refuses to bind
changed Markdown to existing vectors.

Usage (from E:\\writing-rag):
    .venv\\Scripts\\python.exe index\\reconcile_manifest.py --dry-run
    .venv\\Scripts\\python.exe index\\reconcile_manifest.py

Sources with stored chunks are reconciled from Chroma. Existing zero-chunk
aliases are retained only when their content still matches a stored canonical
source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONVERTED = ROOT / "converted"
ARCHIVE = CONVERTED / "已入库"
INDEX_DIR = ROOT / "index"
CHROMA_DIR = INDEX_DIR / "chroma_db"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
BACKUP_DIR = ROOT / ".claude" / "tmp" / "manifest-backups"
DEFAULT_MODEL = "BAAI/bge-m3"
MODEL_DIR = ROOT / ".model-cache" / "BAAI__bge-m3"
BATCH_SIZE = 2000

sys.path.insert(0, str(INDEX_DIR))
from file_lock import LockUnavailable, target_write_exclusion  # noqa: E402
from rag_core import (  # noqa: E402
    IndexIntegrityError,
    PASSAGE_V2_VERSION,
    require_persistent_database,
    validate_v2_manifest_identity,
)

os.environ.setdefault("HF_HOME", str(ROOT / ".model-cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".model-cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file(source: str) -> Path | None:
    """Find a source in the two supported Markdown locations."""
    for base in (CONVERTED, ARCHIVE):
        candidate = (base / source).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None


def collect_sources(collection):
    """Return source -> {chunks, mac, embedding_text_version} from Chroma."""
    total = collection.count()
    sources = {}
    for offset in range(0, total, BATCH_SIZE):
        rows = collection.get(
            include=["metadatas"], limit=BATCH_SIZE, offset=offset
        )["metadatas"]
        for meta in rows:
            source = (meta or {}).get("source")
            if not source:
                raise RuntimeError(f"Chroma row at offset {offset} has no source")
            version = (meta or {}).get("embedding_text_version")
            entry = sources.setdefault(
                source,
                {
                    "chunks": 0,
                    "mac": (meta or {}).get("mac", "unknown"),
                    "embedding_text_version": version,
                },
            )
            entry["chunks"] += 1
            current_mac = (meta or {}).get("mac", "unknown")
            if entry["mac"] != current_mac:
                raise RuntimeError(
                    f"inconsistent mac metadata for source {source!r}: "
                    f"{entry['mac']!r} vs {current_mac!r}"
                )
            if entry["embedding_text_version"] != version:
                raise RuntimeError(
                    f"mixed representation versions for source {source!r}: "
                    f"{entry['embedding_text_version']!r} vs {version!r}"
                )
    versions = {entry["embedding_text_version"] for entry in sources.values()}
    if len(versions) > 1:
        detail = ", ".join(repr(version) for version in sorted(versions, key=str))
        raise RuntimeError(f"mixed representation versions in Chroma collection: {detail}")
    if versions and next(iter(versions)) != PASSAGE_V2_VERSION:
        version = next(iter(versions))
        raise RuntimeError(f"unknown representation in Chroma collection: {version!r}")
    return sources, total


def build_manifest(sources, collection_count, old_manifest):
    files = {}
    missing = []
    old_files = old_manifest["files"]
    for source in sorted(sources):
        old_entry = old_files.get(source)
        if not isinstance(old_entry, dict):
            raise RuntimeError(
                f"Chroma source {source!r} has no prior ledger representation; "
                "refusing to invent one"
            )
        path = source_file(source)
        if path is None:
            missing.append(source)
            continue
        current_sha = sha256(path)
        if current_sha != old_entry.get("sha256"):
            raise RuntimeError(
                f"source content changed after indexing: {source!r}; "
                "refusing to bind the new SHA-256 to existing vectors"
            )
        if old_entry.get("mac", "unknown") != sources[source]["mac"]:
            raise RuntimeError(
                f"source metadata changed after indexing: {source!r}; refusing reconcile"
            )
        files[source] = dict(old_entry)
        files[source]["chunks"] = sources[source]["chunks"]
    if missing:
        sample = ", ".join(missing[:10])
        raise RuntimeError(
            f"{len(missing)} Chroma sources have no Markdown file under "
            f"converted/ or converted/已入库 (sample: {sample})"
        )
    # Preserve proven zero-row entries, including duplicate aliases. If an
    # alias no longer points at a source that still has rows, omit it so the
    # next incremental build can classify it again without re-encoding the
    # unchanged canonical source.
    for source, entry in old_files.items():
        if source in files or not isinstance(entry, dict) or entry.get("chunks") != 0:
            continue
        path = source_file(source)
        if path is None:
            continue
        current_sha = sha256(path)
        if current_sha != entry.get("sha256"):
            continue
        duplicate_of = entry.get("dup_of")
        if duplicate_of is not None:
            canonical = files.get(duplicate_of)
            if canonical is None or canonical.get("sha256") != current_sha:
                continue
        files[source] = dict(entry)

    chunk_sum = sum(item["chunks"] for item in sources.values())
    if chunk_sum != collection_count:
        raise RuntimeError(
            f"chunk count changed during scan: metadata sum={chunk_sum}, "
            f"collection.count()={collection_count}"
        )
    manifest = dict(old_manifest)
    manifest.update({
        "files": files,
        "total_chunks": chunk_sum,
        "reconciled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Reconcile manifest.json from Chroma")
    parser.add_argument("--dry-run", action="store_true", help="validate and print counts only")
    parser.add_argument("--no-backup", action="store_true", help="do not save the old manifest")
    args = parser.parse_args()

    if args.dry_run:
        return reconcile_once(args)

    # The write mode rewrites the production manifest, so it must own the same
    # target writer lock as every other writer of that index directory.
    try:
        with target_write_exclusion([CHROMA_DIR], label="reconcile_manifest"):
            return reconcile_once(args)
    except LockUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(3) from exc


def reconcile_once(args):
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)
    if not CHROMA_DIR.is_dir():
        raise IndexIntegrityError(f"Chroma DB directory is missing: {CHROMA_DIR}")
    old_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    validate_v2_manifest_identity(
        old_manifest, model_dir=MODEL_DIR, expected_model=DEFAULT_MODEL
    )
    require_persistent_database(CHROMA_DIR)

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_collection("papers")
    sources, collection_count = collect_sources(collection)
    old_files = old_manifest.get("files", {})
    if collection_count == 0 and (
        old_manifest.get("total_chunks", 0) != 0
        or any(
            isinstance(entry, dict) and entry.get("chunks", 0) != 0
            for entry in old_files.values()
        )
    ):
        raise RuntimeError(
            "non-empty manifest is paired with an empty Chroma collection; "
            "refusing to reconcile lost vectors as an empty index"
        )
    manifest = build_manifest(sources, collection_count, old_manifest)

    print(
        f"Chroma: {len(sources)} sources / {collection_count} chunks\n"
        f"manifest: {len(old_files)} sources / "
        f"{old_manifest.get('total_chunks', 0)} chunks\n"
        f"reconciled: {len(manifest['files'])} sources / "
        f"{manifest['total_chunks']} chunks"
    )
    if args.dry_run:
        print("DRY-RUN: manifest not changed")
        return

    if not args.no_backup:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = BACKUP_DIR / f"manifest-{stamp}.json"
        shutil.copy2(MANIFEST_PATH, backup)
        print(f"backup: {backup}")

    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, MANIFEST_PATH)
    print(f"written: {MANIFEST_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - concise CLI failure
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
