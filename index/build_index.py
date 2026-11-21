"""Build a local Chroma vector index over converted Markdown.

Embeddings : BAAI/bge-m3 (local, CPU), markdown-heading-aware chunking.
Storage    : Chroma persistent DB at index/chroma_db, collection "papers".
Metadata   : merged from .claude/tmp/mapping.json (Zotero/CSV) + alias table
             for known supplementary / split files.

Idempotent: files whose sha256 is unchanged are skipped (incremental).
Both converted/*.md and converted/已入库/*.md are inputs; the archive move does
not cause a second conversion or embedding pass.  Rebuild all with --force.

Duplicate content policy: md files that are sha-identical (e.g. one paper
converted under both a PMID_* name and a legacy Zotero/Word name, or the same
article downloaded under several PMIDs) hold identical deterministic chunk
ids, so Chroma can physically store them only once.  Each duplicate group is
indexed under one canonical name (PMID_* names preferred, else the smallest
name); the other files of the group get a chunks=0 ledger entry (``dup_of``)
and are never upserted.  Manifest totals therefore equal real Chroma rows even
on a --force rebuild.

Usage:
    python index/build_index.py            # incremental
    python index/build_index.py --force    # full rebuild
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONVERTED = ROOT / "converted"
ARCHIVE = CONVERTED / "已入库"
INDEX_DIR = ROOT / "index"
CHROMA_DIR = INDEX_DIR / "chroma_db"
MANIFEST_PATH = INDEX_DIR / "manifest.json"
MODEL_NAME = "BAAI/bge-m3"
LOCAL_MODEL_DIR = ROOT / ".model-cache" / "BAAI__bge-m3"  # pre-downloaded local copy
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 80
MIN_CHUNK_TOKENS = 16  # drop tiny noise fragments (title-page art, etc.)
EMBEDDING_TEXT_VERSION = "passage-v2"
EMBEDDING_BATCH_CHUNKS = 256
EMBEDDING_BATCH_SIZE_CUDA = 8
EMBEDDING_BATCH_SIZE_CPU = 48
SECTION_TYPES = (
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
    "supplement",
    "other",
)
# Chunking parameters participate in the embedding input (through the section
# path) and in the stored passage, so they are part of the representation
# signature recorded per source.
CHUNKING_CONFIG = {
    "max_tokens": CHUNK_MAX_TOKENS,
    "overlap_tokens": CHUNK_OVERLAP_TOKENS,
    "min_tokens": MIN_CHUNK_TOKENS,
}


class RepresentationMismatch(RuntimeError):
    """Raised when a writer would mix incompatible passage representations."""

# Keep the ~2.3 GB model cache inside the workspace (self-contained).
os.environ.setdefault("HF_HOME", str(ROOT / ".model-cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".model-cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from sentence_transformers import SentenceTransformer  # noqa: E402

import torch  # noqa: E402

import chromadb  # noqa: E402
from chromadb.config import Settings as ChromaSettings  # noqa: E402

try:
    from rag_core import index_preflight, model_fingerprint
except ImportError:  # pragma: no cover - supports package-style imports in tests
    from .rag_core import index_preflight, model_fingerprint

try:
    import index_common
except ImportError:  # pragma: no cover - package-style imports
    from . import index_common

try:
    from file_lock import LockUnavailable, target_write_exclusion
except ImportError:  # pragma: no cover - package-style imports
    from .file_lock import LockUnavailable, target_write_exclusion

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Metadata lookup: converted md stem -> mapping.json entry
# ---------------------------------------------------------------------------

ALIASES = {
    # converted stem -> (lookup kind, value)
    "ESGAR202601 PARTI": ("folder", "7TE8XYFI"),
    "ESGAR202601 PARTII": ("folder", "FZQ7B3VD"),
    "zrac039_supplementary_data": ("folder", "LMF8QQ6J"),            # Enblad 2022 supp
    "cancers-1559433-supplementary": ("doi", "10.3390/cancers14051297"),  # Bong 2022 supp
    "NIHMS2115008-supplement-Supplement": ("doi", "10.1007/s00330-025-11967-6"),  # Javed-Tayyab 2025 supp
}


def load_meta():
    by_stem = {}
    by_folder = {}
    by_doi = {}
    for mapping_path in (
        ROOT / ".claude" / "tmp" / "mapping.json",        # MAC corpus (first: wins DOI conflicts)
        ROOT / ".claude" / "tmp" / "mapping_pmid.json",   # PMID batch (generated from 索引信息.csv)
    ):
        if not mapping_path.exists():
            continue
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        for f in mapping["files"]:
            fn = f.get("filename") or ""
            if fn:
                by_stem[Path(fn).stem] = f
            if f.get("folder"):
                by_folder[f["folder"]] = f
            doi = (f.get("doi") or "").strip().lower()
            if doi:
                by_doi.setdefault(doi, f)
    return by_stem, by_folder, by_doi


def meta_for(rel_name, by_stem, by_folder, by_doi):
    """Return (meta_dict, is_supp_or_alias) for a converted file."""
    stem = Path(rel_name).stem
    if stem in ALIASES:
        kind, value = ALIASES[stem]
        f = by_folder.get(value) if kind == "folder" else by_doi.get(value.lower())
        if f:
            return _entry_meta(f), True
    f = by_stem.get(stem)
    if f:
        return _entry_meta(f), False
    return {
        "title": stem,
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


def _entry_meta(f):
    return {
        "title": f.get("title") or Path(f.get("filename") or "?").stem,
        "year": str(f.get("year") or ""),
        "journal": f.get("journal") or "",
        "doi": f.get("doi") or "",
        "authors": f.get("author") or "",
        "pub_type": f.get("pub_type") or f.get("pubtype") or "",
        "license": f.get("license") or "unverified",
        "mac": "yes" if f.get("mac") else "no",
        "folder": f.get("folder") or "",
        "evidence": f.get("evidence") or "",
    }


def section_type(section):
    """Map a heading path to the stable section type vocabulary."""
    parts = [part.strip().casefold() for part in (section or "").split(" > ")]
    for part in reversed(parts):
        if re.search(r"\b(supplement(?:ary)?|appendix|附录|补充)\b", part):
            return "supplement"
        if re.search(r"\b(reference|references|bibliography|参考文献)\b", part):
            return "references"
        if re.search(r"\babstract\b|摘要", part):
            return "abstract"
        if re.search(
            r"\b(methods?|materials?|methodology|experimental|statistical analysis)\b"
            r"|方法|材料",
            part,
        ):
            return "methods"
        if re.search(r"\b(results?|findings|outcomes?)\b|结果|发现", part):
            return "results"
        if re.search(r"\b(discussion|interpretation)\b|讨论", part):
            return "discussion"
        if re.search(r"\b(conclusions?)\b|结论", part):
            return "conclusion"
        if re.search(r"\b(introduction|background|rationale)\b|引言|背景", part):
            return "introduction"
    return "other"


def paper_id_for(source, meta):
    """Return a stable paper identity shared by DOI-linked source files."""
    doi = (meta.get("doi") or "").strip().casefold()
    if doi:
        return f"doi:{doi}"
    return f"source:{Path(source).stem}"


def build_embedding_text(title, section, text):
    """Build the v2 representation while leaving the stored passage untouched."""
    title = (title or "").strip() or "Unknown title"
    section = (section or "").strip() or "other"
    return f"Title: {title}\nSection: {section}\nContent:\n{text}"


def chunk_metadata(meta, source, section, chunk_index, is_alias):
    """Return metadata shared by all chunks of one source passage."""
    result = dict(meta)
    result.update({
        "source": source,
        "paper_id": paper_id_for(source, meta),
        "section": section,
        "section_type": section_type(section),
        "chunk": chunk_index,
        "supp": "yes" if is_alias else "no",
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
    })
    return result


def representation_signature(meta):
    """Fingerprint every input that changes the embedding text.

    The passage text itself is covered by the content sha256 of the Markdown
    source.  The remaining embedding inputs are the representation version, the
    title drawn from the metadata mapping and the chunking configuration that
    decides the section path.  A source whose content sha is unchanged is only
    reusable when this signature also matches.
    """
    payload = json.dumps(
        {
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "title": (meta or {}).get("title") or "",
            "chunking": CHUNKING_CONFIG,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def entry_is_current(entry, sha256_value, signature):
    """Return True when a manifest entry already describes the wanted work."""
    if not isinstance(entry, dict):
        return False
    return (
        entry.get("sha256") == sha256_value
        and entry.get("repr") == signature
        and entry.get("embedding_text_version") == EMBEDDING_TEXT_VERSION
    )


def manifest_entry(source, sha256_value, chunk_count, meta, is_alias=False, dup_of=None):
    """Build one manifest entry including its representation signature."""
    entry = {
        "sha256": sha256_value,
        "chunks": chunk_count,
        "mac": (meta or {}).get("mac", "unknown"),
        "repr": representation_signature(meta),
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
    }
    if dup_of is not None:
        entry["dup_of"] = dup_of
    return entry


# ---------------------------------------------------------------------------
# Chunking (heading-aware, ~800 tokens, paragraph granularity)
# ---------------------------------------------------------------------------

def chunk_markdown(text, tokenizer, max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP_TOKENS):
    def count_tokens(s):
        return len(tokenizer.encode(s, add_special_tokens=False))

    def split_token_windows(value):
        """Split an overlong unit by token windows without dropping its tail."""
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return [value]
        step = max_tokens - overlap
        if step <= 0:
            raise ValueError("overlap must be smaller than max_tokens")
        out = []
        start = 0
        while start < len(token_ids):
            end = min(start + max_tokens, len(token_ids))
            piece = tokenizer.decode(
                token_ids[start:end],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if piece:
                out.append(piece)
            if end == len(token_ids):
                break
            start += step
        return out

    def split_sentences(para):
        units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n", para) if u.strip()]
        out, buf, bt = [], [], 0
        for u in units:
            ut = count_tokens(u)
            if ut > max_tokens:
                if buf:
                    out.append(" ".join(buf))
                    buf, bt = [], 0
                out.extend(split_token_windows(u))
                continue
            if buf and bt + ut > max_tokens:
                out.append(" ".join(buf))
                buf, bt = [], 0
            buf.append(u)
            bt += ut
        if buf:
            out.append(" ".join(buf))
        return out

    lines = text.splitlines()
    chunks = []
    heading_stack = []  # (level, title)
    buf = []

    def flush():
        nonlocal buf
        if not buf:
            return
        body = "\n".join(buf).strip()
        buf = []
        if not body:
            return
        section = " > ".join(t for _, t in heading_stack)
        if count_tokens(body) < MIN_CHUNK_TOKENS:  # drop noise fragments
            return
        if count_tokens(body) <= max_tokens:
            chunks.append({"text": body, "section": section})
            return
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        acc, acc_toks = [], 0
        for p in paras:
            pt = count_tokens(p)
            if pt > max_tokens:
                if acc:
                    chunks.append({"text": "\n\n".join(acc), "section": section})
                    acc, acc_toks = [], 0
                for sub in split_sentences(p):
                    chunks.append({"text": sub, "section": section})
                continue
            if acc and acc_toks + pt > max_tokens:
                chunks.append({"text": "\n\n".join(acc), "section": section})
                carry, ct = [], 0  # keep tail paragraphs as overlap
                for pp in reversed(acc):
                    t = count_tokens(pp)
                    if ct + t > overlap:
                        break
                    carry.insert(0, pp)
                    ct += t
                acc, acc_toks = carry, ct
            acc.append(p)
            acc_toks += pt
        if acc:
            chunks.append({"text": "\n\n".join(acc), "section": section})

    in_fence = False
    for ln in lines:
        stripped = ln.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or stripped.startswith("![") or stripped.startswith("<!--"):
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            flush()
            lvl, title = len(m.group(1)), m.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= lvl:
                heading_stack.pop()
            heading_stack.append((lvl, title))
        else:
            buf.append(ln)
    flush()
    return chunks


def markdown_files():
    """Return one path per Markdown filename; top-level wins on collision."""
    return index_common.markdown_files(CONVERTED, ARCHIVE)


def _canonical_key(name):
    """Duplicate-group ordering: PMID_* names win, then plain lexicographic."""
    return index_common.canonical_key(name)


def canonical_rel(group):
    """Pick the canonical filename of a sha-identical duplicate group."""
    return index_common.canonical_rel(group)


def duplicate_aliases(file_hash):
    """Map each duplicate source to its canonical source using the shared rule."""
    groups = {}
    for source, digest in file_hash.items():
        groups.setdefault(digest, []).append(source)
    aliases = {}
    for sources in groups.values():
        if len(sources) < 2:
            continue
        canonical = canonical_rel(sources)
        aliases.update({source: canonical for source in sources if source != canonical})
    return aliases


def entry_role_is_current(entry, dup_of=None):
    """Return whether a ledger entry has the role selected for this corpus."""
    if not isinstance(entry, dict):
        return False
    if dup_of is not None:
        return entry.get("chunks", 0) == 0 and entry.get("dup_of") == dup_of
    return "dup_of" not in entry


def _source_ids(collection, source):
    rows = collection.get(where={"source": source}, include=["metadatas"])
    return list(rows.get("ids") or [])


def replace_source(collection, source, ids, documents, metadatas, embeddings):
    """Write new chunks before removing stale ones for this source.

    If embedding or the write fails, the previous chunks remain available.  A
    deterministic id plus ``upsert`` also makes an interrupted retry repair
    the harmless duplicate-free state.
    """
    old_ids = _source_ids(collection, source)
    if documents:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        stale_ids = [item for item in old_ids if item not in set(ids)]
    else:
        stale_ids = old_ids
    if stale_ids:
        collection.delete(ids=stale_ids)


def prepare_source(rel, sha256_value, text, tokenizer, meta, is_alias):
    """Turn one Markdown source into the exact rows a writer must store.

    Shared by the offline builder and the live ingestor so both writers embed
    the same text, store the same passages and record the same metadata.
    Returns ``None`` when the source yields no indexable chunk.
    """
    chunks = chunk_markdown(text, tokenizer)
    if not chunks:
        return None
    documents = [chunk["text"] for chunk in chunks]
    sections = [chunk["section"] for chunk in chunks]
    ids = [f"{sha256_value[:12]}-{index:03d}" for index in range(len(chunks))]
    metadatas = [
        chunk_metadata(meta, rel, sections[index], index, is_alias)
        for index in range(len(chunks))
    ]
    embedding_docs = [
        build_embedding_text(meta.get("title"), sections[index], documents[index])
        for index in range(len(chunks))
    ]
    return {
        "source": rel,
        "sha256": sha256_value,
        "meta": meta,
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "embedding_docs": embedding_docs,
    }


def encode_and_store(collection, model, prepared, batch_size):
    """Embed and store one prepared source, then return its manifest entry."""
    embeddings = model.encode(
        prepared["embedding_docs"],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    replace_source(
        collection, prepared["source"], prepared["ids"],
        prepared["documents"], prepared["metadatas"], embeddings,
    )
    return manifest_entry(
        prepared["source"], prepared["sha256"], len(prepared["ids"]), prepared["meta"]
    )


def replace_sources(collection, source_items, embeddings):
    """Upsert a bounded group of sources in one Chroma operation."""
    old_ids = {
        item["source"]: _source_ids(collection, item["source"])
        for item in source_items
    }
    ids = [chunk_id for item in source_items for chunk_id in item["ids"]]
    new_ids = set(ids)
    documents = [doc for item in source_items for doc in item["documents"]]
    metadatas = [meta for item in source_items for meta in item["metadatas"]]
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    stale_ids = [
        old_id
        for item in source_items
        for old_id in old_ids[item["source"]]
        if old_id not in new_ids
    ]
    if stale_ids:
        collection.delete(ids=stale_ids)


def collection_scan(collection):
    """Return (source names, embedding_text_version -> row count).

    One paged metadata pass.  A single ``get(limit=count)`` exceeds SQLite's
    variable limit on large collections (56k+ chunks), so page by offset like
    ``reconcile_manifest`` does.
    """
    count = collection.count()
    sources = set()
    versions = {}
    if not count:
        return sources, versions
    batch = 10000
    for offset in range(0, count, batch):
        rows = collection.get(limit=batch, offset=offset, include=["metadatas"])
        for meta in (rows.get("metadatas") or []):
            meta = meta or {}
            source = meta.get("source")
            if source:
                sources.add(source)
            version = meta.get("embedding_text_version")
            versions[version] = versions.get(version, 0) + 1
    return sources, versions


def collection_sources(collection):
    """Return all source names currently present in Chroma."""
    return collection_scan(collection)[0]


def assert_representation_compatible(manifest, version_counts, label):
    """Refuse to write when the target index uses another representation.

    Both the manifest ledger and the rows already stored in Chroma are checked:
    relabelling a manifest is not enough to make a legacy index a v2 index.
    """
    recorded = manifest.get("embedding_text_version")
    nonzero_entries = sum(
        1 for entry in (manifest.get("files") or {}).values()
        if isinstance(entry, dict) and entry.get("chunks")
    )
    if recorded != EMBEDDING_TEXT_VERSION and nonzero_entries:
        raise RepresentationMismatch(
            f"{label}: manifest declares embedding_text_version="
            f"{recorded!r} with {nonzero_entries} chunked sources; refusing to "
            f"mix it with {EMBEDDING_TEXT_VERSION!r}"
        )
    foreign = {
        version: count for version, count in version_counts.items()
        if count and version != EMBEDDING_TEXT_VERSION
    }
    if foreign:
        detail = ", ".join(
            f"{version!r}={count}" for version, count in sorted(
                foreign.items(), key=lambda item: str(item[0])
            )
        )
        raise RepresentationMismatch(
            f"{label}: existing rows do not use {EMBEDDING_TEXT_VERSION!r} ({detail}); "
            "refusing to write a mixed-representation index"
        )
    return True


def clear_collection(collection):
    """Delete every row in a collection in bounded batches."""
    while True:
        rows = collection.get(limit=10000, include=["metadatas"])
        ids = rows.get("ids") or []
        if not ids:
            return
        collection.delete(ids=ids)


def write_manifest_atomic(manifest, manifest_path=MANIFEST_PATH):
    """Atomically publish the manifest after all Chroma writes succeed."""
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, manifest_path)


def ready_marker_path(manifest_path):
    """Path of the ``ready`` marker that accompanies a staged index pair."""
    manifest_path = Path(manifest_path)
    return manifest_path.with_name(manifest_path.name + ".ready.json")


def verify_complete_corpus(manifest, current_names):
    """Refuse to declare a rebuild complete unless it covers every source.

    A locally self-consistent manifest is not enough: a capped or interrupted
    run also produces a manifest that matches its own collection.
    """
    declared = set((manifest.get("files") or {}).keys())
    missing = sorted(set(current_names) - declared)
    unknown = sorted(declared - set(current_names))
    if missing or unknown:
        raise RepresentationMismatch(
            "staged rebuild does not cover the whole corpus: "
            f"missing={len(missing)} {missing[:3]}, unknown={len(unknown)} {unknown[:3]}"
        )
    return True


def write_ready_marker(manifest_path, manifest, run_db, corpus_entries, model_path):
    """Record the validated staging pair; the publish step consumes this."""
    marker = {
        "stage": "index-staging",
        "status": "ready",
        "db": str(Path(run_db)),
        "manifest": str(Path(manifest_path)),
        "manifest_sha256": index_common.sha256_file(manifest_path),
        "model": manifest.get("model"),
        "model_fingerprint": manifest.get("model_fingerprint"),
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "chunking": CHUNKING_CONFIG,
        "total_chunks": manifest.get("total_chunks"),
        "source_entries": len(manifest.get("files") or {}),
        "chunked_sources": sum(
            1 for entry in (manifest.get("files") or {}).values()
            if isinstance(entry, dict) and entry.get("chunks")
        ),
        "corpus_fingerprint": index_common.corpus_fingerprint(corpus_entries),
        "corpus_source_count": len(corpus_entries),
        "model_path": str(model_path),
        "built_at": manifest.get("built_at"),
    }
    index_common.write_json_atomic(ready_marker_path(manifest_path), marker)
    return marker


def resolve_run_paths(args, parser):
    """Resolve the write target and enforce the isolation boundary.

    Custom ``--db``/``--manifest`` must be supplied as a pair, must not alias
    the production paths, and a ``--force`` rebuild never targets production:
    it stages the new index and requires a separate publish step.
    """
    production_db = Path(CHROMA_DIR)
    production_manifest = Path(MANIFEST_PATH)
    if (args.db is None) != (args.manifest is None):
        parser.error(
            "--db and --manifest must be given together: a half-specified pair "
            "would leave the other path pointing at production"
        )
    staged = False
    if args.db is not None:
        run_db = Path(args.db).expanduser()
        run_manifest = Path(args.manifest).expanduser()
        for label, value, production in (
            ("--db", run_db, production_db),
            ("--manifest", run_manifest, production_manifest),
        ):
            if index_common.same_path(value, production) or index_common.is_inside(
                production, value
            ):
                parser.error(
                    f"{label}={value} resolves to the production path "
                    f"{production}; use index/publish_index.py for the switch"
                )
        if index_common.same_path(run_db, run_manifest):
            parser.error("--db and --manifest must not resolve to the same path")
    elif args.force:
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        run_db = INDEX_DIR / f"chroma_db.staging-{stamp}"
        run_manifest = INDEX_DIR / f"manifest.staging-{stamp}.json"
        staged = True
    else:
        run_db = production_db
        run_manifest = production_manifest
    return run_db, run_manifest, staged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="Build the local passage-v2 vector index."
    )
    ap.add_argument("--force", action="store_true",
                    help="rebuild everything into a staging pair (never publishes)")
    ap.add_argument("--limit", type=int, default=0, help="only process first N files (debug)")
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto (default)")
    ap.add_argument("--db", default=None, help="alternate Chroma directory (testing/isolation)")
    ap.add_argument("--manifest", default=None, help="alternate manifest path (testing/isolation)")
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    INDEX_DIR.mkdir(exist_ok=True)
    if args.force and args.limit:
        ap.error(
            "--limit cannot be combined with --force: a capped run is a partial index "
            "and must never be reported as a complete rebuild"
        )
    run_db, run_manifest, staged = resolve_run_paths(args, ap)
    # Hold the target writer lock for the whole build: probing and releasing
    # would let a live ingest, the B3 runner or a publication start in the
    # middle of this run.  Writers on other targets stay independent.
    try:
        with target_write_exclusion([run_db], label="index build"):
            return _build(args, run_db, run_manifest, staged)
    except LockUnavailable as exc:
        raise SystemExit(f"ERROR: {exc}")


def _build(args, run_db, run_manifest, staged):
    if staged:
        print(f"Full rebuild staging DB: {run_db}", flush=True)
        print(f"Full rebuild staging manifest: {run_manifest}", flush=True)

    manifest = {
        "version": 1,
        "model": MODEL_NAME,
        "embedding_text_version": EMBEDDING_TEXT_VERSION,
        "chunking": CHUNKING_CONFIG,
        "files": {},
    }
    # An explicit full rebuild owns the reset below, including recovery from
    # an interrupted prior rebuild whose manifest and collection differ.
    if run_manifest.exists() and not args.force:
        manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    manifest.setdefault("files", {})

    model_path = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_NAME
    if run_manifest.exists() and not args.force:
        # Never let a present manifest make a missing DB look like a new empty
        # index.  The preflight is read-only and runs before any Chroma access
        # that could create a directory or collection.
        index_preflight(
            db_dir=run_db,
            model_dir=model_path,
            device=args.device,
            collection_name="papers",
            manifest_path=run_manifest,
            expected_model=MODEL_NAME,
        )

    torch.set_num_threads(max(1, os.cpu_count() or 8))
    print(f"torch threads: {torch.get_num_threads()}")

    by_stem, by_folder, by_doi = load_meta()
    file_items = markdown_files()
    current_names = {name for name, _ in file_items}

    # One pass over the corpus: group files by content sha so duplicate
    # content is never upserted twice (identical content -> identical chunk
    # ids -> the second write would silently re-own the first file's rows).
    file_hash = {}
    for rel, md in file_items:
        h = hashlib.sha256(md.read_bytes()).hexdigest()
        file_hash[rel] = h
    alias_of = duplicate_aliases(file_hash)
    dup_skips = set(alias_of)

    todo = []
    skipped = 0
    for rel, md in file_items:
        h = file_hash[rel]
        entry = manifest["files"].get(rel, {})
        meta, _ = meta_for(rel, by_stem, by_folder, by_doi)
        signature = representation_signature(meta)
        if not args.force and entry_is_current(entry, h, signature) and \
                entry_role_is_current(entry, alias_of.get(rel)):
            skipped += 1
            continue
        todo.append((md, h, meta, signature))
    if args.limit:
        todo = todo[: args.limit]
    stale_manifest = set(manifest["files"]) - current_names
    if not todo and not stale_manifest and not args.force:
        print(f"No new files ({skipped} already indexed). Use --force to rebuild.")
        return

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"device: {device}")
    print(f"Loading model {MODEL_NAME} (local dir: {LOCAL_MODEL_DIR}) ...")
    model = SentenceTransformer(model_path, device=device)
    if device == "cuda":
        model.half()  # fp16 to fit the 2 GB laptop GPU
        torch.cuda.empty_cache()
    tokenizer = model[0].tokenizer

    client = chromadb.PersistentClient(
        path=str(run_db), settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        "papers", metadata={"hnsw:space": "cosine"}
    )

    if args.force:
        clear_collection(collection)
        manifest = {
            "version": 1,
            "model": MODEL_NAME,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "chunking": CHUNKING_CONFIG,
            "files": {},
        }
        todo = []
        for rel, md in file_items:
            meta, _alias = meta_for(rel, by_stem, by_folder, by_doi)
            todo.append((md, file_hash[rel], meta, representation_signature(meta)))
        if args.limit:
            todo = todo[: args.limit]
    else:
        _, version_counts = collection_scan(collection)
        assert_representation_compatible(
            manifest, version_counts, f"incremental build target {run_manifest}"
        )
        # Remove manifest entries and Chroma rows whose Markdown source was
        # removed.  Archive moves retain the filename and therefore remain
        # the same logical source.
        for source in sorted(stale_manifest | (collection_sources(collection) - current_names)):
            replace_source(collection, source, [], [], [], None)
            manifest["files"].pop(source, None)

    t0 = time.time()
    total_chunks = 0
    done_chunks = 0
    pending = []
    pending_chunks = 0
    aliases_to_finalize = []
    batch = EMBEDDING_BATCH_SIZE_CUDA if device == "cuda" else EMBEDDING_BATCH_SIZE_CPU

    def flush_pending():
        nonlocal pending, pending_chunks, total_chunks, done_chunks
        if not pending:
            return
        embedding_docs = [
            text
            for item in pending
            for text in item["embedding_docs"]
        ]
        embeddings = model.encode(
            embedding_docs,
            batch_size=batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        replace_sources(collection, pending, embeddings)
        for item in pending:
            chunk_count = len(item["documents"])
            manifest["files"][item["source"]] = manifest_entry(
                item["source"], item["sha256"], chunk_count, item["meta"]
            )
            total_chunks += chunk_count
            done_chunks += chunk_count
            tag = (
                f"[{item['meta']['mac']}]"
                if item["meta"]["mac"] != "unknown"
                else "[?]"
            )
            print(
                f"  {tag} {item['source']}  -> {chunk_count} chunks "
                f"(cum {done_chunks})",
                flush=True,
            )
        manifest["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["total_chunks"] = sum(
            item.get("chunks", 0) for item in manifest["files"].values()
        )
        if Path(model_path).is_dir():
            manifest["model_fingerprint"] = model_fingerprint(model_path)
        write_manifest_atomic(manifest, run_manifest)
        pending = []
        pending_chunks = 0

    for md, h, meta, signature in todo:
        rel = md.name
        if pending and pending_chunks >= EMBEDDING_BATCH_CHUNKS:
            flush_pending()
        _, is_alias = meta_for(rel, by_stem, by_folder, by_doi)
        if rel in dup_skips:
            aliases_to_finalize.append((rel, h, meta, alias_of[rel]))
            print(f"  DUP (indexed as {alias_of[rel]}): {rel}", flush=True)
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        prepared = prepare_source(rel, h, text, tokenizer, meta, is_alias)
        if prepared is None:
            replace_source(collection, rel, [], [], [], None)
            print(f"  SKIP (no text): {rel}")
            manifest["files"][rel] = manifest_entry(rel, h, 0, meta)
            continue
        pending.append(prepared)
        pending_chunks += len(prepared["ids"])

    flush_pending()

    # Retire rows left by a source that became a duplicate only after the
    # canonical's rows are safely stored.  Shared deterministic IDs now belong
    # to the canonical source, so source-scoped deletion cannot remove them.
    for rel, h, meta, canonical in aliases_to_finalize:
        replace_source(collection, rel, [], [], [], None)
        manifest["files"][rel] = manifest_entry(
            rel, h, 0, meta, dup_of=canonical
        )

    manifest["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if Path(model_path).is_dir():
        manifest["model_fingerprint"] = model_fingerprint(model_path)
    manifest["total_chunks"] = sum(
        item.get("chunks", 0) for item in manifest["files"].values()
    )
    write_manifest_atomic(manifest, run_manifest)
    corpus_entries = sorted(file_hash.items())
    if staged:
        verify_complete_corpus(manifest, current_names)
        # Validate the complete staged pair.  This is read-only and catches
        # count/model mismatches; publishing is a separate operation executed
        # by index/publish_index.py from a short-lived process.
        staged_summary = index_preflight(
            db_dir=run_db,
            model_dir=model_path,
            device=args.device,
            collection_name="papers",
            manifest_path=run_manifest,
            expected_model=MODEL_NAME,
        )
        marker = write_ready_marker(
            run_manifest, manifest, run_db, corpus_entries, model_path
        )
        print(
            f"Staging preflight PASS: sources={staged_summary['source_count']} "
            f"chunks={staged_summary['collection_count']}",
            flush=True,
        )
        print(f"Ready marker: {ready_marker_path(run_manifest)}", flush=True)
        print(
            "NOT published. Switch with a separate process:\n"
            f'  .venv\\Scripts\\python.exe index\\publish_index.py '
            f'--staging-db "{run_db}" --staging-manifest "{run_manifest}"',
            flush=True,
        )
    print(f"\nDone: {len(todo)} files, {total_chunks} chunks, {skipped} skipped, "
          f"{time.time() - t0:.0f}s. Index at {run_db}")
    return manifest


if __name__ == "__main__":
    main()
