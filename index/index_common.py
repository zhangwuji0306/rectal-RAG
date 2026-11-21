"""Lightweight helpers shared by the index builders.

This module deliberately avoids importing ``sentence_transformers``,
``torch`` or ``chromadb``: the stop/status/publish entry points must be usable
without loading a model or opening a Chroma client.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CHUNK_STATUS_COMPLETE = "complete"
CHUNK_STATUS_PARTIAL = "partial"


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_key(name):
    """Duplicate-group ordering: PMID_* names win, then plain lexicographic."""
    return (0 if name.startswith("PMID_") else 1, name)


def canonical_rel(group):
    """Pick the canonical filename of a sha-identical duplicate group."""
    return sorted(group, key=canonical_key)[0]


def markdown_files(converted, archive):
    """Return one (name, path) per Markdown filename; top-level wins."""
    by_name = {}
    for base in (converted, archive):
        base = Path(base)
        if not base.exists():
            continue
        for path in sorted(base.glob("*.md")):
            by_name.setdefault(path.name, path)
    return [(name, by_name[name]) for name in sorted(by_name)]


def hashed_corpus(file_items):
    """Return the sorted ``(name, sha256)`` inventory of a Markdown corpus."""
    return sorted((name, sha256_file(path)) for name, path in file_items)


def corpus_fingerprint(hashed_entries):
    """Fingerprint a corpus source set including every source's content hash."""
    payload = json.dumps(
        sorted([str(name), str(digest)] for name, digest in hashed_entries),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def duplicate_groups(hashed_entries):
    """Map content sha256 -> canonical name for every duplicated group."""
    groups = {}
    for name, digest in hashed_entries:
        groups.setdefault(digest, []).append(name)
    return {
        digest: canonical_rel(names)
        for digest, names in groups.items()
        if len(names) > 1
    }


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path, value, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def normalised(path):
    """Return a comparable absolute path (case-insensitive on Windows)."""
    resolved = Path(path).expanduser().resolve()
    return os.path.normcase(str(resolved))


def same_path(left, right):
    return normalised(left) == normalised(right)


def is_inside(child, parent):
    """True when ``child`` is ``parent`` itself or lives below it."""
    child_norm = normalised(child)
    parent_norm = normalised(parent)
    return child_norm == parent_norm or child_norm.startswith(parent_norm.rstrip("\\/") + os.sep)
