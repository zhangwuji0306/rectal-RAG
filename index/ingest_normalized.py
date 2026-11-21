"""Build an isolated passage-v2 index pair from verified normalized documents.

``直肠癌文献爬取/scripts/normalized_ingest.py`` decides which normalized
documents may be ingested (identity agrees with the canonical task, the A6
verdict is bound to the current content version, and the normalized document was
produced from exactly that version) and copies them into a staging directory.
This module turns exactly those documents into an index pair.

It reuses the passage-v2 writer of :mod:`build_index` - the same chunker, the
same embedding text built by ``build_embedding_text``, the same chunk metadata
and the same content-sha duplicate rule - so a normalized passage is stored and
encoded exactly like a converted one.  It never writes ``converted/``, never
touches ``index/b3-sources.json``, and never writes to the B3 staging pair or to
production: the target pair is explicit and isolated.

The pair this module produces is an acceptance target for the normalized path,
not a production candidate: its ready marker describes the normalized corpus, so
the production publication entry point refuses it.

Usage::

    python index/ingest_normalized.py --plan
    python index/ingest_normalized.py --build --device cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

INDEX_DIR = Path(__file__).resolve().parent
ROOT = INDEX_DIR.parent
SCRIPTS_DIR = ROOT / "直肠癌文献爬取" / "scripts"
sys.path.insert(0, str(INDEX_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

import index_common  # noqa: E402
from file_lock import LockUnavailable, target_write_exclusion  # noqa: E402

COLLECTION_NAME = "papers"
DEFAULT_NORMALIZED_ROOT = ROOT / "normalized"
DEFAULT_STAGING_DB = INDEX_DIR / "chroma_db.normalized-staging"
DEFAULT_STAGING_MANIFEST = INDEX_DIR / "manifest.normalized-staging.json"
B3_STAGING_DB = INDEX_DIR / "chroma_db.v2-staging"


class NormalizedIndexError(RuntimeError):
    """Raised when the normalized inputs must not become an index."""


@dataclass(frozen=True)
class NormalizedTarget:
    """Every path this entry point may touch."""

    normalized_root: Path
    staging_db: Path
    manifest: Path
    converted_roots: tuple
    mapping_paths: tuple
    task_db: Path | None
    model_dir: Path


def default_target(**overrides) -> NormalizedTarget:
    values = {
        "normalized_root": DEFAULT_NORMALIZED_ROOT,
        "staging_db": DEFAULT_STAGING_DB,
        "manifest": DEFAULT_STAGING_MANIFEST,
        "converted_roots": (ROOT / "converted", ROOT / "converted" / "已入库"),
        "mapping_paths": (
            ROOT / ".claude" / "tmp" / "mapping.json",
            ROOT / ".claude" / "tmp" / "mapping_pmid.json",
        ),
        "task_db": ROOT / "直肠癌文献爬取" / "tasks.sqlite",
        "model_dir": ROOT / ".model-cache" / "BAAI__bge-m3",
    }
    values.update(overrides)
    return NormalizedTarget(**values)


def build_index_module():
    import build_index  # noqa: PLC0415 - deliberately lazy

    return build_index


def normalized_ingest_module():
    import normalized_ingest  # noqa: PLC0415 - deliberately lazy

    return normalized_ingest


def validate_target(target: NormalizedTarget):
    """Refuse a target that would overwrite production or the B3 staging pair."""
    protected = (
        ("production database", INDEX_DIR / "chroma_db"),
        ("production manifest", INDEX_DIR / "manifest.json"),
        ("B3 staging database", B3_STAGING_DB),
        ("B3 staging manifest", INDEX_DIR / "manifest.b3.partial.json"),
    )
    for label, path in protected:
        if index_common.same_path(target.staging_db, path) or index_common.same_path(
            target.manifest, path
        ):
            raise NormalizedIndexError(
                f"the normalized target must not be {label} ({path}); use "
                "index/publish_index.py for a production switch"
            )
    if index_common.is_inside(target.manifest, target.staging_db) or index_common.is_inside(
        target.staging_db, target.manifest
    ):
        raise NormalizedIndexError(
            "the normalized staging manifest and database must not contain each other"
        )
    return True


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

def strip_front_matter(text):
    """Return the document body without its leading ``---`` block."""
    if not text.startswith("---"):
        return text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for position in range(1, len(lines)):
        if lines[position].strip() == "---":
            return "\n".join(lines[position + 1:]).lstrip("\n")
    return text


def document_meta(front, name):
    """Index metadata for one normalized document, from its front matter."""
    return {
        "title": front.get("title") or Path(name).stem,
        "year": str(front.get("year") or ""),
        "journal": front.get("journal") or "",
        "doi": front.get("doi") or "",
        "authors": front.get("authors") or "",
        "pub_type": "",
        "license": front.get("license") or "unverified",
        "mac": front.get("mac") or "unknown",
        "folder": "",
        "evidence": "",
        # Provenance of this passage: which canonical source produced the
        # normalized document and which validated content version it holds.
        "normalized_source": front.get("source") or "",
        "normalized_pmcid": front.get("pmcid") or "",
        "validated_content_sha256": (front.get("content_sha256") or "").casefold(),
        "passage_origin": "normalized",
    }


def _identity_keys(entry):
    keys = []
    if entry.identity.get("pmid"):
        keys.append(("pmid", str(entry.identity["pmid"])))
    if entry.identity.get("doi"):
        keys.append(("doi", str(entry.identity["doi"]).casefold()))
    if not keys:
        keys.append(("file", Path(entry.normalized_path).name.casefold()))
    return keys


def _group_entries(entries):
    """Union entries that share a PMID or DOI, so one paper is one source."""
    parent = list(range(len(entries)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    seen = {}
    for index, entry in enumerate(entries):
        for key in _identity_keys(entry):
            if key in seen:
                union(seen[key], index)
            else:
                seen[key] = index
    groups = {}
    for index in range(len(entries)):
        groups.setdefault(find(index), []).append(entries[index])
    return [groups[key] for key in sorted(groups)]


def select_documents(target: NormalizedTarget, prefer=None):
    """Return the ingestable normalized documents, grouped per paper.

    Only entries the adapter proved ingestable are considered.  A paper that
    appears in more than one ingestable document with different content is a
    conflict and must be resolved explicitly with ``prefer``; identical content
    is deduplicated to one canonical document with zero-chunk aliases.
    """
    normalized_ingest = normalized_ingest_module()
    plan = normalized_ingest.plan_ingestion(
        target.normalized_root,
        converted_roots=target.converted_roots,
        mapping_paths=target.mapping_paths,
        out_root=target.staging_db.parent / f"{target.staging_db.name}.candidates",
        db_path=target.task_db,
    )
    ingestable = [
        entry for entry in plan["entries"]
        if entry.ingestable and entry.action == normalized_ingest.STAGE_ACTION
    ]
    selected = []
    conflicts = []
    for group in _group_entries(ingestable):
        digests = {}
        for entry in group:
            digests.setdefault(
                index_common.sha256_file(entry.normalized_path), []
            ).append(entry)
        if len(group) == 1 or len(digests) == 1:
            names = [Path(entry.normalized_path).name for entry in group]
            canonical_name = index_common.canonical_rel(names)
            for entry in group:
                is_canonical = Path(entry.normalized_path).name == canonical_name
                selected.append({
                    "source": Path(entry.normalized_path).name,
                    "path": Path(entry.normalized_path),
                    "entry": entry,
                    "is_canonical": is_canonical,
                    "dup_of": None if is_canonical else canonical_name,
                    "sha256": index_common.sha256_file(entry.normalized_path),
                })
            continue
        chosen = None
        if prefer:
            wanted = str(prefer)
            for entry in group:
                if Path(entry.normalized_path).name == wanted or str(
                    entry.normalized_path
                ) == wanted:
                    chosen = entry
                    break
        if chosen is None:
            conflicts.append({
                "identity": dict(group[0].identity),
                "documents": [
                    {
                        "path": str(entry.normalized_path),
                        "sha256": index_common.sha256_file(entry.normalized_path),
                        "declared_source": entry.identity.get("source"),
                    }
                    for entry in group
                ],
                "reason": (
                    "同一篇文献有多个可通过核验的 normalized 文档但内容不同；"
                    "必须用 --prefer 明确选择来源，不能静默丢弃正文"
                ),
            })
            continue
        selected.append({
            "source": Path(chosen.normalized_path).name,
            "path": Path(chosen.normalized_path),
            "entry": chosen,
            "is_canonical": True,
            "dup_of": None,
            "sha256": index_common.sha256_file(chosen.normalized_path),
        })
    report = {
        "normalized_root": str(target.normalized_root),
        "task_db": str(target.task_db) if target.task_db else None,
        "planned_documents": plan["summary"]["normalized_documents"],
        "ingestable_documents": len(ingestable),
        "selected_sources": len(selected),
        "canonical_sources": sum(1 for item in selected if item["is_canonical"]),
        "alias_sources": sum(1 for item in selected if not item["is_canonical"]),
        "conflicts": conflicts,
        "not_ingestable": [
            {
                "path": str(entry.normalized_path),
                "verification": entry.verification,
                "reason": entry.verification_reason,
                "action": entry.action,
            }
            for entry in plan["entries"]
            if not entry.ingestable
        ],
    }
    if conflicts:
        raise NormalizedIndexError(
            f"{len(conflicts)} paper(s) have conflicting ingestable normalized "
            "documents; resolve them with --prefer: "
            + "; ".join(conflict["documents"][0]["path"] for conflict in conflicts[:2])
        )
    return selected, report


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class ModelAdapter:
    """Give the shared writer the encoder interface it expects."""

    def __init__(self, backends, device):
        self._backends = backends
        self.device = device

    def encode(self, texts, batch_size=None, **_ignored):
        return self._backends.encode(texts)


class RealBackends:
    """Model and Chroma access for one isolated normalized target."""

    def __init__(self, target: NormalizedTarget, device="cpu"):
        self.target = target
        self.device = device
        self._model = None
        self._collection = None
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            if self._model is not None:
                self._tokenizer = self._model[0].tokenizer
            else:
                from transformers import AutoTokenizer  # noqa: PLC0415

                self._tokenizer = AutoTokenizer.from_pretrained(
                    str(self.target.model_dir)
                )
        return self._tokenizer

    def encode(self, texts):
        import torch  # noqa: PLC0415

        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            model_path = (
                str(self.target.model_dir)
                if Path(self.target.model_dir).is_dir()
                else "BAAI/bge-m3"
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

            Path(self.target.staging_db).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.target.staging_db),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = client.get_or_create_collection(
                COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def close(self):
        self._collection = None
        if self._model is not None:
            self._model = None
        try:
            from rag_core import release_client  # noqa: PLC0415

            release_client(self.target.staging_db)
        except Exception:  # noqa: BLE001 - release is best effort
            pass


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def empty_manifest(build_index, task_db):
    return {
        "version": 2,
        "status": "partial",
        "stage": "normalized-ingest",
        "model": build_index.MODEL_NAME,
        "embedding_text_version": build_index.EMBEDDING_TEXT_VERSION,
        "chunking": dict(build_index.CHUNKING_CONFIG),
        "task_db": str(task_db) if task_db else None,
        "files": {},
        "total_chunks": 0,
        "built_at": None,
    }


def verify_rows(collection, source, expected_count, expected_ids,
                expected_documents=None, expected_version=None):
    """Prove the stored rows of one source are exactly what was submitted."""
    rows = collection.get(where={"source": source}, include=["metadatas", "documents"])
    ids = list(rows.get("ids") or [])
    if len(ids) != expected_count:
        return [f"rows={len(ids)} expected={expected_count}"]
    if sorted(ids) != sorted(expected_ids):
        return ["stored row ids do not match the deterministic ids"]
    by_id = {
        row_id: (metadata or {}, document)
        for row_id, metadata, document in zip(
            rows.get("ids") or [], rows.get("metadatas") or [],
            rows.get("documents") or [],
        )
    }
    for index, row_id in enumerate(expected_ids):
        metadata, document = by_id[row_id]
        if metadata.get("source") != source:
            return ["stored row source metadata does not match the queried source"]
        if (metadata or {}).get("passage_origin") != "normalized":
            return ["stored row is not marked as a normalized passage"]
        if expected_version is not None and metadata.get(
            "embedding_text_version"
        ) != expected_version:
            return ["stored row uses a different embedding text version"]
        if not isinstance(document, str) or not document.strip():
            return ["stored passage is empty"]
        if expected_documents is not None and document != expected_documents[index]:
            return ["stored passage text does not match the current source"]
    return []


def _expected_ids(sha256_value, count):
    """Return the deterministic row ids used by the shared writer."""
    return [f"{sha256_value[:12]}-{index:03d}" for index in range(count)]


def verify_target_state(build_index, collection, selected, manifest,
                        prepared_by_source):
    """Verify selected sources, manifest entries and stored rows are one set."""
    selected_names = {item["source"] for item in selected}
    manifest_files = manifest.get("files") or {}
    if set(manifest_files) != selected_names:
        return [
            "manifest source set differs from selected source set: "
            f"selected={len(selected_names)} manifest={len(manifest_files)}"
        ]

    actual_sources, version_counts = build_index.collection_scan(collection)
    build_index.assert_representation_compatible(
        manifest, version_counts, "normalized staging index"
    )
    expected_row_sources = {
        name for name, entry in manifest_files.items()
        if isinstance(entry, dict) and entry.get("chunks", 0)
    }
    if actual_sources != expected_row_sources:
        return [
            "stored source set differs from manifest: "
            f"actual={len(actual_sources)} expected={len(expected_row_sources)}"
        ]

    expected_total = sum(
        entry.get("chunks", 0) for entry in manifest_files.values()
        if isinstance(entry, dict)
    )
    if collection.count() != expected_total:
        return [f"stored rows={collection.count()} expected={expected_total}"]

    problems = []
    for item in selected:
        name = item["source"]
        entry = manifest_files[name]
        prepared = prepared_by_source.get(name)
        expected_count = len(prepared["ids"]) if prepared is not None else 0
        if entry.get("chunks") != expected_count:
            problems.append(
                f"{name}: manifest chunks={entry.get('chunks')} "
                f"expected={expected_count}"
            )
            continue
        expected_documents = prepared["documents"] if prepared is not None else []
        row_problems = verify_rows(
            collection,
            name,
            expected_count,
            _expected_ids(item["sha256"], expected_count),
            expected_documents=expected_documents,
            expected_version=build_index.EMBEDDING_TEXT_VERSION,
        )
        problems.extend(f"{name}: {problem}" for problem in row_problems)
    return problems


def build(target: NormalizedTarget, *, device="cpu", prefer=None, backends=None,
          deep=False):
    """Encode the ingestable normalized documents into the isolated target."""
    build_index = build_index_module()
    validate_target(target)
    own_backends = backends is None
    active = backends
    manifest = None
    manifest_started = False
    ready_path = build_index.ready_marker_path(target.manifest)
    try:
        with target_write_exclusion([target.staging_db], label="normalized ingest"):
            try:
                manifest = index_common.read_json(target.manifest, default=None) or \
                    empty_manifest(build_index, target.task_db)
                manifest.setdefault("files", {})
            except BaseException:
                ready_path.unlink(missing_ok=True)
                raise
            try:
                model_fingerprint = build_index.model_fingerprint(target.model_dir)
                recorded_fingerprint = manifest.get("model_fingerprint")
                ledger_nonempty = bool(manifest.get("files")) or bool(
                    manifest.get("total_chunks")
                )
                if ledger_nonempty and recorded_fingerprint != model_fingerprint:
                    reason = (
                        "missing" if not recorded_fingerprint else "different"
                    )
                    raise NormalizedIndexError(
                        f"normalized target model fingerprint is {reason}: "
                        f"manifest={recorded_fingerprint!r}, current={model_fingerprint!r}; "
                        "refusing to reuse existing vectors. Choose a new isolated "
                        "target and rebuild."
                    )

                if active is None:
                    active = RealBackends(target, device)
                collection = active.collection
                if collection.count() and recorded_fingerprint != model_fingerprint:
                    reason = "missing" if not recorded_fingerprint else "different"
                    raise NormalizedIndexError(
                        f"normalized target model fingerprint is {reason} for a "
                        "non-empty Chroma collection; refusing to reuse existing "
                        "vectors. Choose a new isolated target and rebuild."
                    )

                # A previous ready marker can never describe an unfinished
                # selection.  The old identity is compared above before either
                # it or the vector set can be rewritten.
                ready_path.unlink(missing_ok=True)
                manifest["model_fingerprint"] = model_fingerprint
                manifest["status"] = "partial"
                index_common.write_json_atomic(target.manifest, manifest)
                manifest_started = True

                selected, report = select_documents(target, prefer=prefer)
                empty = empty_manifest(build_index, target.task_db)
                problems = [
                    f"{field}: manifest={manifest.get(field)!r} sources="
                    f"{empty[field]!r}"
                    for field in ("model", "embedding_text_version", "chunking")
                    if manifest.get(field) is not None
                    and manifest.get(field) != empty[field]
                ]
                if problems:
                    raise NormalizedIndexError(
                        "the normalized staging pair was built with a different "
                        "configuration: " + "; ".join(problems)
                    )
                actual_sources, version_counts = build_index.collection_scan(collection)
                build_index.assert_representation_compatible(
                    manifest, version_counts, "normalized staging index"
                )

                selected_names = {item["source"] for item in selected}
                stale_sources = (
                    set(manifest["files"]) | actual_sources
                ) - selected_names
                for source in sorted(stale_sources):
                    build_index.replace_source(
                        collection, source, [], [], [], None
                    )
                    manifest["files"].pop(source, None)

                encoded = 0
                skipped = 0
                prepared_by_source = {}
                for item in selected:
                    name = item["source"]
                    sha = item["sha256"]
                    text = item["path"].read_text(
                        encoding="utf-8", errors="replace"
                    )
                    front, body = normalized_front_and_body(text)
                    meta = document_meta(front, name)
                    signature = build_index.representation_signature(meta)
                    if not item["is_canonical"]:
                        build_index.replace_source(
                            collection, name, [], [], [], None
                        )
                        manifest["files"][name] = build_index.manifest_entry(
                            name, sha, 0, meta, dup_of=item["dup_of"]
                        )
                        prepared_by_source[name] = None
                        continue

                    prepared = build_index.prepare_source(
                        name, sha, body, active.tokenizer, meta, False
                    )
                    prepared_by_source[name] = prepared
                    expected_count = len(prepared["ids"]) if prepared else 0
                    current = build_index.entry_is_current(
                        manifest["files"].get(name, {}), sha, signature
                    ) and manifest["files"].get(name, {}).get(
                        "chunks"
                    ) == expected_count
                    if current:
                        row_problems = verify_rows(
                            collection,
                            name,
                            expected_count,
                            _expected_ids(sha, expected_count),
                            expected_documents=(
                                prepared["documents"] if prepared else []
                            ),
                            expected_version=build_index.EMBEDDING_TEXT_VERSION,
                        )
                        if not row_problems:
                            skipped += 1
                            continue

                    if prepared is None:
                        build_index.replace_source(
                            collection, name, [], [], [], None
                        )
                        manifest["files"][name] = build_index.manifest_entry(
                            name, sha, 0, meta
                        )
                        continue
                    batch = 8 if device == "cuda" else 48
                    entry = build_index.encode_and_store(
                        collection, ModelAdapter(active, device), prepared, batch
                    )
                    row_problems = verify_rows(
                        collection,
                        name,
                        len(prepared["ids"]),
                        prepared["ids"],
                        expected_documents=prepared["documents"],
                        expected_version=build_index.EMBEDDING_TEXT_VERSION,
                    )
                    if row_problems:
                        raise NormalizedIndexError(
                            f"{name}: stored rows failed verification: "
                            f"{row_problems[:2]}"
                        )
                    manifest["files"][name] = entry
                    encoded += 1
                    print(
                        f"[NORMALIZED] {name} -> {entry['chunks']} chunks",
                        flush=True,
                    )

                manifest["total_chunks"] = sum(
                    item.get("chunks", 0)
                    for item in manifest["files"].values()
                )
                final_problems = verify_target_state(
                    build_index,
                    collection,
                    selected,
                    manifest,
                    prepared_by_source,
                )
                if final_problems:
                    raise NormalizedIndexError(
                        "normalized target is not internally consistent: "
                        + "; ".join(final_problems[:4])
                    )
                manifest["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                manifest["status"] = "complete"
                index_common.write_json_atomic(target.manifest, manifest)
                inventory = sorted(
                    (item["source"], item["sha256"]) for item in selected
                )
                marker = build_index.write_ready_marker(
                    target.manifest,
                    manifest,
                    target.staging_db,
                    inventory,
                    target.model_dir,
                )
                report.update({
                    "encoded_sources": encoded,
                    "reused_sources": skipped,
                    "manifest_entries": len(manifest["files"]),
                    "chunks": manifest["total_chunks"],
                    "manifest": str(target.manifest),
                    "staging_db": str(target.staging_db),
                    "ready_marker": str(ready_path),
                    "ready_marker_sha256": marker["manifest_sha256"],
                })
                return report
            except BaseException:
                # A partial write may have reached Chroma before an exception.
                # The manifest and marker must advertise that this target is not
                # publishable; the next run repairs it from the current selection.
                ready_path.unlink(missing_ok=True)
                if manifest_started:
                    manifest["status"] = "partial"
                    try:
                        index_common.write_json_atomic(target.manifest, manifest)
                    except Exception:  # pragma: no cover - disk failure is original error
                        pass
                raise
    finally:
        if own_backends and active is not None:
            active.close()


def normalized_front_and_body(text):
    """Return the parsed front matter and the body of a normalized document."""
    normalized_ingest = normalized_ingest_module()
    return normalized_ingest.parse_front_matter(text), strip_front_matter(text)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="report the selection only")
    parser.add_argument("--build", action="store_true", help="encode the selected sources")
    parser.add_argument("--normalized-root", default=str(DEFAULT_NORMALIZED_ROOT))
    parser.add_argument("--task-db", default=str(
        ROOT / "直肠癌文献爬取" / "tasks.sqlite"
    ))
    parser.add_argument("--staging-db", default=str(DEFAULT_STAGING_DB))
    parser.add_argument("--manifest", default=str(DEFAULT_STAGING_MANIFEST))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--prefer", default=None,
                        help="resolve a same-paper source conflict by filename")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    target = default_target(
        normalized_root=Path(args.normalized_root),
        staging_db=Path(args.staging_db),
        manifest=Path(args.manifest),
        task_db=Path(args.task_db) if args.task_db else None,
    )
    try:
        if args.build:
            result = build(target, device=args.device, prefer=args.prefer)
        else:
            selected, result = select_documents(target, prefer=args.prefer)
            result["selection"] = [
                {"source": item["source"], "is_canonical": item["is_canonical"],
                 "dup_of": item["dup_of"], "path": str(item["path"])}
                for item in selected
            ]
    except (NormalizedIndexError, LockUnavailable) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
