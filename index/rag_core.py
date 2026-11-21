"""Reusable RAG retrieval core: query the built Chroma index from any project.

The index (chroma_db) and the local bge-m3 model are self-contained under
E:\\writing-rag, so other projects on this machine can query them directly
without copying the ~2.3 GB model or rebuilding the index.

Usage from another project:
    import sys
    sys.path.insert(0, r"E:\\writing-rag\\index")
    from rag_core import retrieve, dedupe_by_doi, summary_list

    hits = retrieve("mucin pool prognosis", k=8, mac_only=True)
    refs = dedupe_by_doi(hits)      # literature-level, deduped by DOI
    for r in summary_list(refs):
        print(r["title"], "|", r["journal"], "|", r["year"], "|", r["doi"])

All paths default to the writing-rag ones and can be overridden per call.
"""

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_DIR = ROOT / "index" / "chroma_db"
DEFAULT_MODEL_DIR = ROOT / ".model-cache" / "BAAI__bge-m3"
DEFAULT_MODEL_NAME = "BAAI/bge-m3"  # fallback hub name if local copy missing
PASSAGE_V2_VERSION = "passage-v2"

os.environ.setdefault("HF_HOME", str(ROOT / ".model-cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".model-cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from sentence_transformers import SentenceTransformer  # noqa: E402

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

# Module-level caches so repeated calls inside one process load each requested
# model/collection once.  The complete key prevents a DB or collection from
# being silently reused for another retrieval target.
_MODEL_CACHE = {}
_COLLECTION_CACHE = {}


class IndexIntegrityError(RuntimeError):
    """Raised when the persisted index cannot be proven internally consistent."""


def _resolved_path(value, default):
    path = Path(value) if value is not None else Path(default)
    return path.expanduser().resolve()


def _cache_key(model_dir, device, db_dir, collection_name):
    model_path = _resolved_path(model_dir, DEFAULT_MODEL_DIR)
    db_path = _resolved_path(db_dir, DEFAULT_DB_DIR)
    return (str(model_path), str(device), str(db_path), str(collection_name))


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(model_dir):
    """Return a deterministic, inexpensive fingerprint of a local model tree.

    Configuration/tokenizer files are content-hashed.  Large weight files are
    represented by their relative path and byte size so a normal preflight does
    not reread the 2+ GB local model; a changed model layout or weight size is
    still detected.  A manifest may additionally pin this value in its
    ``model_fingerprint`` field.
    """
    path = _resolved_path(model_dir, DEFAULT_MODEL_DIR)
    if not path.is_dir():
        raise IndexIntegrityError(f"model directory is missing: {path}")

    entries = []
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = file_path.relative_to(path).as_posix()
        size = file_path.stat().st_size
        entry = {"path": relative, "size": size}
        if size <= 1024 * 1024:
            entry["sha256"] = _sha256_file(file_path)
        entries.append(entry)
    if not entries:
        raise IndexIntegrityError(f"model directory contains no files: {path}")
    payload = json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def release_client(db_dir=None, collection_name="papers"):
    """Drop cached clients for one database directory.

    Chroma keeps SQLite handles in a process-wide registry.  Windows refuses to
    rename or remove a directory that still has an open handle, so a process
    that is about to move an index directory (publication, migration) must
    release them explicitly before the rename.
    """
    db_path = _resolved_path(db_dir, DEFAULT_DB_DIR)
    for key in [key for key in _COLLECTION_CACHE if key[2] == str(db_path)
                and key[3] == str(collection_name)]:
        _COLLECTION_CACHE.pop(key, None)
    try:
        from chromadb.api.shared_system_client import SharedSystemClient
    except Exception:  # pragma: no cover - chroma build without the registry
        return False
    registry = getattr(SharedSystemClient, "_identifier_to_system", {})
    wanted = os.path.normcase(str(db_path))
    released = False
    for identifier in list(registry):
        try:
            candidate = Path(identifier)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            matches = os.path.normcase(str(candidate.resolve())) == wanted
        except (OSError, ValueError):  # pragma: no cover - unusual identifier
            matches = False
        if matches:
            del registry[identifier]
            released = True
    import gc

    gc.collect()
    return released


def _manifest_path(db_dir, manifest_path):
    if manifest_path is not None:
        path = Path(manifest_path).expanduser()
        if not path.is_absolute():
            path = Path(db_dir).parent / path
        return path.resolve()
    return _resolved_path(db_dir, DEFAULT_DB_DIR).parent / "manifest.json"


def _is_nonnegative_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def require_persistent_database(db_dir):
    """Refuse to open a ledger-backed target without its existing SQLite DB."""
    db_path = _resolved_path(db_dir, DEFAULT_DB_DIR)
    if not db_path.is_dir():
        raise IndexIntegrityError(f"Chroma DB directory is missing: {db_path}")
    sqlite_path = db_path / "chroma.sqlite3"
    try:
        if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
            raise IndexIntegrityError(
                "Chroma DB persistent state is missing or empty: "
                f"{sqlite_path}"
            )
    except OSError as exc:
        raise IndexIntegrityError(
            f"cannot inspect Chroma DB persistent state: {sqlite_path}"
        ) from exc
    return sqlite_path


def validate_v2_manifest_identity(manifest, model_dir=None,
                                  expected_model=DEFAULT_MODEL_NAME):
    """Validate representation and model identity without trusting row counts.

    Reconciliation may repair count fields, but it must not invent the
    representation signature or model identity for vectors already on disk.
    """
    if not isinstance(manifest, dict):
        raise IndexIntegrityError("manifest root must be an object")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise IndexIntegrityError("manifest.files must be an object")
    model = manifest.get("model")
    if not isinstance(model, str) or not model:
        raise IndexIntegrityError("manifest.model must be a non-empty string")
    if expected_model is not None and model != expected_model:
        raise IndexIntegrityError(
            f"manifest model mismatch: {model!r} != {expected_model!r}"
        )
    if manifest.get("embedding_text_version") != PASSAGE_V2_VERSION:
        raise IndexIntegrityError(
            "manifest does not declare passage-v2; refusing to relabel unknown vectors"
        )
    if not isinstance(manifest.get("chunking"), dict):
        raise IndexIntegrityError("manifest.chunking must be an object")
    recorded_fingerprint = manifest.get("model_fingerprint")
    if not _is_sha256(recorded_fingerprint):
        raise IndexIntegrityError(
            "manifest.model_fingerprint is missing or invalid; model identity is unproven"
        )

    for source, entry in files.items():
        if not isinstance(source, str) or not source:
            raise IndexIntegrityError("manifest source names must be non-empty strings")
        if not isinstance(entry, dict):
            raise IndexIntegrityError(f"manifest entry is not an object: {source!r}")
        if not _is_sha256(entry.get("sha256")):
            raise IndexIntegrityError(f"invalid sha256 in manifest entry: {source!r}")
        if not _is_sha256(entry.get("repr")):
            raise IndexIntegrityError(
                f"representation signature is missing or invalid: {source!r}"
            )
        if entry.get("embedding_text_version") != PASSAGE_V2_VERSION:
            raise IndexIntegrityError(
                f"entry does not declare passage-v2: {source!r}"
            )

    model_path = _resolved_path(model_dir, DEFAULT_MODEL_DIR)
    fingerprint = model_fingerprint(model_path)
    if recorded_fingerprint != fingerprint:
        raise IndexIntegrityError(
            f"model fingerprint mismatch: manifest={recorded_fingerprint!r}, "
            f"actual={fingerprint!r}"
        )
    return fingerprint


def index_write_preflight(db_dir=None, model_dir=None, device="cpu",
                          collection_name="papers", manifest_path=None,
                          expected_model=DEFAULT_MODEL_NAME):
    """Validate an existing pair or prove a target is genuinely empty."""
    db_path = _resolved_path(db_dir, DEFAULT_DB_DIR)
    manifest_file = _manifest_path(db_path, manifest_path)
    if manifest_file.exists():
        return index_preflight(
            db_dir=db_path,
            model_dir=model_dir,
            device=device,
            collection_name=collection_name,
            manifest_path=manifest_file,
            expected_model=expected_model,
        )
    if db_path.exists():
        if not db_path.is_dir():
            raise IndexIntegrityError(f"Chroma DB path is not a directory: {db_path}")
        try:
            populated = next(db_path.iterdir(), None) is not None
        except OSError as exc:
            raise IndexIntegrityError(f"cannot inspect Chroma DB directory: {db_path}") from exc
        if populated:
            raise IndexIntegrityError(
                f"Chroma DB exists without a manifest; refusing to initialize: {db_path}"
            )
    return None


def _collection_profile(collection):
    """Return per-source row counts and representation counts in bounded pages."""
    total = collection.count()
    source_counts = {}
    version_counts = {}
    for offset in range(0, total, 10000):
        rows = collection.get(limit=10000, offset=offset, include=["metadatas"])
        for metadata in rows.get("metadatas") or []:
            source = (metadata or {}).get("source")
            if not source:
                raise IndexIntegrityError(
                    f"collection row at offset {offset} has no source metadata"
                )
            source_counts[source] = source_counts.get(source, 0) + 1
            version = (metadata or {}).get("embedding_text_version")
            version_counts[version] = version_counts.get(version, 0) + 1
    return source_counts, version_counts


def index_preflight(db_dir=None, model_dir=None, device="cpu",
                    collection_name="papers", manifest_path=None,
                    expected_model=DEFAULT_MODEL_NAME):
    """Validate the persisted index before model or collection use.

    The check is deliberately read-only.  It never creates a missing Chroma
    directory or collection.  A returned summary is useful to reports/tests;
    all integrity failures use :class:`IndexIntegrityError`.
    """
    db_path = _resolved_path(db_dir, DEFAULT_DB_DIR)
    if not db_path.is_dir():
        raise IndexIntegrityError(f"Chroma DB directory is missing: {db_path}")

    manifest_file = _manifest_path(db_path, manifest_path)
    if not manifest_file.is_file():
        raise IndexIntegrityError(f"manifest is missing: {manifest_file}")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexIntegrityError(f"manifest cannot be read: {manifest_file}") from exc
    model_path = _resolved_path(model_dir, DEFAULT_MODEL_DIR)
    fingerprint = validate_v2_manifest_identity(
        manifest, model_dir=model_dir, expected_model=expected_model
    )
    files = manifest.get("files")
    total_chunks = manifest.get("total_chunks")
    if not _is_nonnegative_int(total_chunks):
        raise IndexIntegrityError("manifest.total_chunks must be a non-negative integer")

    nonzero_sources = {}
    manifest_chunk_sum = 0
    for source, entry in files.items():
        chunks = entry.get("chunks")
        if not _is_nonnegative_int(chunks):
            raise IndexIntegrityError(f"invalid chunks in manifest entry: {source!r}")
        manifest_chunk_sum += chunks
        if chunks:
            nonzero_sources[source] = chunks
    if manifest_chunk_sum != total_chunks:
        raise IndexIntegrityError(
            f"manifest chunk sum mismatch: entries={manifest_chunk_sum}, "
            f"total_chunks={total_chunks}"
        )

    require_persistent_database(db_path)
    client = chromadb.PersistentClient(
        path=str(db_path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:  # Chroma uses version-specific exception classes.
        raise IndexIntegrityError(
            f"expected Chroma collection is missing: {collection_name!r}"
        ) from exc
    collection_count = collection.count()
    if collection_count != total_chunks:
        raise IndexIntegrityError(
            f"collection count mismatch: Chroma={collection_count}, "
            f"manifest={total_chunks}"
        )
    actual_source_counts, version_counts = _collection_profile(collection)
    actual_sources = set(actual_source_counts)
    if actual_source_counts != nonzero_sources:
        raise IndexIntegrityError(
            "per-source chunk mismatch: "
            f"Chroma={actual_source_counts!r}, manifest={nonzero_sources!r}"
        )
    if collection_count and version_counts != {PASSAGE_V2_VERSION: collection_count}:
        raise IndexIntegrityError(
            f"collection representation mismatch: {version_counts!r}; "
            f"expected only {PASSAGE_V2_VERSION!r}"
        )

    return {
        "db_dir": str(db_path),
        "manifest_path": str(manifest_file),
        "collection": collection_name,
        "collection_count": collection_count,
        "source_count": len(actual_sources),
        "manifest_file_count": len(files),
        "manifest_source_count": len(nonzero_sources),
        "manifest_total_chunks": total_chunks,
        "model_path": str(model_path),
        "model_fingerprint": fingerprint,
    }


def _get_model(model_dir=None, device="cpu", db_dir=None, collection_name="papers"):
    key = _cache_key(model_dir, device, db_dir, collection_name)
    if key not in _MODEL_CACHE:
        path = Path(key[0])
        model_path = str(path) if path.exists() else DEFAULT_MODEL_NAME
        _MODEL_CACHE[key] = SentenceTransformer(model_path, device=device)
    return _MODEL_CACHE[key]


def _get_collection(db_dir=None, collection_name="papers", model_dir=None, device="cpu"):
    key = _cache_key(model_dir, device, db_dir, collection_name)
    if key not in _COLLECTION_CACHE:
        path = Path(key[2])
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        try:
            _COLLECTION_CACHE[key] = client.get_collection(collection_name)
        except Exception as exc:  # Chroma uses version-specific exception classes.
            raise IndexIntegrityError(
                f"expected Chroma collection is missing: {collection_name!r}"
            ) from exc
    return _COLLECTION_CACHE[key]


def retrieve(query, k=8, mac_only=False, db_dir=None, model_dir=None, device="cpu",
             candidate_k=None, collection_name="papers", manifest_path=None,
             ranking_trace=None):
    """Semantic search over the built index.

    Returns a list of dicts (rank, score, source, paper_id, mac, title, year,
    journal, doi, authors, section, section_type, snippet) sorted by score
    descending.
    mac_only=True restricts to MAC-classified chunks (metadata filter).
    candidate_k can be larger than k when callers need enough chunks for
    literature-level DOI deduplication.
    """
    index_preflight(
        db_dir=db_dir,
        model_dir=model_dir,
        device=device,
        collection_name=collection_name,
        manifest_path=manifest_path,
    )
    model = _get_model(model_dir, device, db_dir, collection_name)
    collection = _get_collection(db_dir, collection_name, model_dir, device)
    count = collection.count()
    if not count:
        return []
    n_results = min(max(1, candidate_k or k), count)
    q = model.encode([query], normalize_embeddings=True)
    where = {"mac": "yes"} if mac_only else None
    res = collection.query(query_embeddings=q, n_results=n_results, where=where)

    out = []
    for i, (doc, meta, dist) in enumerate(
        zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ):
        out.append(
            {
                "rank": i + 1,
                "score": round(1 - dist, 4),  # cosine similarity (normalized)
                "source": meta.get("source", ""),
                "paper_id": meta.get("paper_id", ""),
                "mac": meta.get("mac", "unknown"),
                "title": meta.get("title", ""),
                "year": meta.get("year", ""),
                "journal": meta.get("journal", ""),
                "doi": meta.get("doi", ""),
                "authors": meta.get("authors", ""),
                "pub_type": meta.get("pub_type", ""),
                "license": meta.get("license", "unverified"),
                "section": meta.get("section", ""),
                "section_type": meta.get("section_type", "other"),
                "embedding_text_version": meta.get("embedding_text_version", ""),
                "chunk": meta.get("chunk", 0),
                "snippet": doc[:400],
            }
        )
    if ranking_trace is not None:
        ranking_trace['candidates'] = [
            {**{key: hit[key] for key in ('rank', 'score', 'source', 'chunk', 'doi', 'section')},
             'id': res['ids'][0][i], 'distance': res['distances'][0][i]}
            for i, hit in enumerate(out)]
    return out


def retrieve_dense(query, candidate_k, ranking_trace=None, **kwargs):
    trace = {}
    hits = retrieve(query, k=candidate_k, candidate_k=candidate_k,
                    ranking_trace=trace, **kwargs)
    for hit, record in zip(hits, trace.get('candidates', [])):
        hit['id'] = record['id']
        hit['distance'] = record['distance']
    if ranking_trace is not None:
        ranking_trace.update(trace)
    return hits


def retrieve_sparse(query, candidate_k, sparse_path, ranking_trace=None,
                    db_dir=None, manifest_path=None):
    from sparse_index import SparseIndex
    index = SparseIndex(sparse_path, db_dir=db_dir, manifest_path=manifest_path)
    try:
        hits = index.search(query, candidate_k)
    finally:
        index.close()
    if ranking_trace is not None:
        ranking_trace['candidates'] = [_passage_record(hit) for hit in hits]
    return hits


def _passage_record(hit):
    return {key: hit[key] for key in ('id', 'rank', 'score', 'source', 'chunk',
            'doi', 'section', 'distance', 'contributions') if key in hit}


def fuse_rrf(dense_hits, sparse_hits, rrf_k=60):
    if rrf_k < 1:
        raise ValueError('rrf_k must be positive')
    merged = {}
    for branch, hits in (('dense', dense_hits), ('sparse', sparse_hits)):
        seen = set()
        for rank, hit in enumerate(hits, 1):
            passage_id = hit['id']
            if passage_id in seen:
                raise ValueError('duplicate passage ID within branch')
            seen.add(passage_id)
            if passage_id not in merged:
                merged[passage_id] = {**hit, 'score': 0.0, 'contributions': {}}
            item = merged[passage_id]
            if any(item.get(k) != hit.get(k) for k in ('source', 'chunk', 'doi')):
                raise ValueError('branch passage identity mismatch')
            contribution = 1 / (rrf_k + rank)
            item['contributions'][branch] = {'rank': rank, 'score': contribution}
            item['score'] += contribution
    ordered = sorted(merged.values(), key=lambda hit: (-hit['score'], hit['id']))
    for rank, hit in enumerate(ordered, 1):
        hit['rank'] = rank
    return ordered


def retrieve_literature_mode(query, k=50, mode='dense-only', candidate_k=250,
                             sparse_path=None, dense_k=125, sparse_k=125,
                             rrf_k=60, ranking_trace=None, **kwargs):
    """Explicit total-slot budget; no filling after overlap or branch shortage."""
    if mode not in ('dense-only', 'sparse-only', 'hybrid'):
        raise ValueError('unknown retrieval mode')
    if not 1 <= k <= 50 or candidate_k < 1:
        raise ValueError('invalid literature/candidate budget')
    if mode == 'hybrid' and (dense_k < 1 or sparse_k < 1 or dense_k+sparse_k > candidate_k):
        raise ValueError('hybrid branches exceed total passage budget')
    dense, sparse = [], []
    if mode != 'dense-only':
        if sparse_path is None:
            raise ValueError('explicit sparse path required')
        if kwargs.get('mac_only', False):
            raise ValueError('sparse/hybrid mac_only is unsupported')
        sparse = retrieve_sparse(query, sparse_k if mode == 'hybrid' else candidate_k,
                                 sparse_path, db_dir=kwargs.get('db_dir'),
                                 manifest_path=kwargs.get('manifest_path'))
    if mode != 'sparse-only':
        dense = retrieve_dense(query, dense_k if mode == 'hybrid' else candidate_k, **kwargs)
    hits = fuse_rrf(dense, sparse, rrf_k) if mode == 'hybrid' else dense or sparse
    ranked = dedupe_by_doi(hits)
    selected = ranked[:k]
    if ranking_trace is not None:
        groups = {}
        for hit in hits:
            key = (hit.get('doi') or '').strip().lower() or hit['source']
            groups.setdefault(key, []).append(hit)
        aggregates = []
        for rank, entry in enumerate(ranked, 1):
            key = (entry.get('doi') or '').strip().lower() or entry['source']
            members = groups[key]
            representative = max(members, key=lambda h: h['score'])
            aggregates.append({'rank': rank, 'key': key, 'source': entry['source'],
                               'doi': entry.get('doi', ''), 'score': entry['score'],
                               'representative_id': representative['id'],
                               'member_ids': [h['id'] for h in members],
                               'n_hits': len(members), 'sections': entry['sections']})
        ranking_trace.update({'mode': mode, 'candidate_budget': candidate_k,
            'requested': {'dense': dense_k if mode == 'hybrid' else candidate_k if mode == 'dense-only' else 0,
                          'sparse': sparse_k if mode == 'hybrid' else candidate_k if mode == 'sparse-only' else 0},
            'branches': {'dense': [_passage_record(h) for h in dense],
                         'sparse': [_passage_record(h) for h in sparse]},
            'intersection_ids': sorted({h['id'] for h in dense} & {h['id'] for h in sparse}),
            'passages': [_passage_record(h) for h in hits], 'aggregates': aggregates,
            'passage_candidates_returned': len(hits),
            'literature_candidates_before_limit': len(ranked), 'returned_count': len(selected),
            'truncation_position': k if len(ranked) > k else None,
            'truncated_literature_count': max(0, len(ranked)-k),
            'ordered_hits': [{'rank': i, 'doi': h.get('doi', ''), 'source': h['source'],
                              'score': h['score']} for i,h in enumerate(selected, 1)]})
    return summary_list(selected)


def retrieve_literature(query, k=8, mac_only=False, db_dir=None, model_dir=None,
                        device="cpu", candidate_k=None, collection_name="papers",
                        manifest_path=None, ranking_trace=None):
    """Retrieve up to ``k`` DOI/source-level candidates.

    Dense search first collects a larger candidate pool, then deduplicates by
    DOI. This avoids returning eight chunks from one paper while hiding other
    relevant papers from the caller.
    When supplied, ``ranking_trace`` receives same-call count, budget, cutoff,
    and DOI/source/score metadata without passage content.
    """
    pool = candidate_k or max(40, k * 5)
    hits = retrieve(query, k=pool, mac_only=mac_only, db_dir=db_dir,
                    model_dir=model_dir, device=device, candidate_k=pool,
                    collection_name=collection_name, manifest_path=manifest_path)
    ranked = dedupe_by_doi(hits)
    selected = ranked[:k]
    if ranking_trace is not None:
        ordered_hits = []
        for rank, hit in enumerate(selected, 1):
            item = {
                "rank": rank,
                "doi": hit.get("doi", ""),
                "source": hit.get("source", ""),
            }
            score = hit.get("score")
            if score is not None:
                item["score"] = score
            ordered_hits.append(item)
        ranking_trace.update({
            "candidate_budget": pool,
            "passage_candidates_returned": len(hits),
            "literature_candidates_before_limit": len(ranked),
            "returned_count": len(selected),
            "truncation_position": k if len(ranked) > k else None,
            "truncated_literature_count": max(0, len(ranked) - k),
            "ordered_hits": ordered_hits,
        })
    return summary_list(selected)


def dedupe_by_doi(results):
    """Merge hits of the same paper into one literature-level entry.

    Keyed by lowercased DOI; entries without DOI fall back to the source
    filename. Keeps the highest-score hit's fields, accumulates section
    locations and counts hits (n_hits). Returns entries sorted by score.
    """
    merged = {}
    for r in results:
        key = (r.get("doi") or "").strip().lower() or r["source"]
        if key not in merged:
            entry = dict(r)
            entry["sections"] = [r["section"]] if r.get("section") else []
            entry["n_hits"] = 1
            merged[key] = entry
        else:
            m = merged[key]
            if r["score"] > m["score"]:
                for f in ("title", "year", "journal", "doi", "authors", "pub_type",
                          "license", "source", "paper_id", "mac"):
                    if r.get(f):
                        m[f] = r[f]
                m["score"] = r["score"]
            if r.get("section") and r["section"] not in m["sections"]:
                m["sections"].append(r["section"])
            m["n_hits"] += 1
    return sorted(merged.values(), key=lambda x: -x["score"])


def summary_list(entries):
    """Structured citation-ready list, without snippet noise.

    Each entry: title, year, journal, doi, authors, mac, source, sections,
    n_hits. `entries` may be raw retrieve() output or dedupe_by_doi() output.
    """
    return [
        {
            "title": e["title"],
            "year": e["year"],
            "journal": e["journal"],
            "doi": e["doi"],
            "authors": e["authors"],
            "mac": e["mac"],
            "pub_type": e.get("pub_type", ""),
            "license": e.get("license", "unverified"),
            "source": e["source"],
            "paper_id": e.get("paper_id", ""),
            "sections": e.get("sections", [e["section"]] if e.get("section") else []),
            "n_hits": e.get("n_hits", 1),
        }
        for e in entries
    ]
