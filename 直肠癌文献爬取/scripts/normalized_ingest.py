# -*- coding: utf-8 -*-
"""A-R3 normalized corpus ingestion adapter.

``normalized/PMID_<pmid>.md`` documents come from A7 and carry the validated
content version in their front matter. This adapter makes that corpus usable by
an index build without becoming part of one:

* it reads the front matter (PMID/DOI/source/license/content_sha256);
* it deduplicates against the existing ``converted/`` Markdown by paper
  identity (PMID/DOI/filename, including the legacy mapping files);
* it decides the preferred source for one paper and reports the decision;
* it never overwrites existing Markdown and never writes into ``converted/``.

Planning is the default; ``apply`` copies the chosen normalized documents into a
separate staging directory. ``index/`` is intentionally untouched: the already
agreed ``index/b3-sources.json`` list belongs to the fixed-corpus B3 rebuild,
while adding normalized documents to the index is a separate corpus-expansion
task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a1_schema import _connect  # noqa: E402
from content_validation import content_binding  # noqa: E402


PROJECT = Path(__file__).resolve().parents[2]
NORMALIZED_ROOT = PROJECT / "normalized"
CONVERTED_ROOT = PROJECT / "converted"
STAGING_ROOT = PROJECT / "normalized_ingest"
FRONT_MATTER_FIELDS = ("pmid", "doi", "pmcid", "title", "source", "license", "content_sha256")
IDENTITY_FIELDS = ("pmid", "doi")
CONVERTED_SUFFIXES = (".md", ".markdown")


class IngestionPlanError(ValueError):
    """Raised when a staging request cannot be honoured safely."""


@dataclass(frozen=True)
class NormalizedDocument:
    """One normalized Markdown document and the identity it declares."""

    path: Path
    front_matter: Mapping[str, Any]
    pmid: str | None
    doi: str | None
    source: str | None
    license: str | None
    content_sha256: str | None
    missing_front_matter: tuple[str, ...] = ()
    binding: str | None = None

    @property
    def identity(self) -> tuple[tuple[str, str], ...]:
        keys: list[tuple[str, str]] = []
        if self.pmid:
            keys.append(("pmid", self.pmid))
        if self.doi:
            keys.append(("doi", self.doi))
        return tuple(keys)


@dataclass
class IngestionEntry:
    """The ingestion decision for one normalized document."""

    pmid: str | None
    normalized_path: str
    identity: Mapping[str, Any] = field(default_factory=dict)
    missing_front_matter: tuple[str, ...] = ()
    existing_converted: list[str] = field(default_factory=list)
    preferred_source: str | None = None
    preferred_path: str | None = None
    reason: str = ""
    action: str = "skip"
    target: str | None = None
    verification: str = "not_checked"
    verification_reason: str = ""
    ingestable: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "pmid": self.pmid,
            "normalized_path": self.normalized_path,
            "identity": dict(self.identity),
            "missing_front_matter": list(self.missing_front_matter),
            "existing_converted": list(self.existing_converted),
            "preferred_source": self.preferred_source,
            "preferred_path": self.preferred_path,
            "reason": self.reason,
            "action": self.action,
            "target": self.target,
            "verification": self.verification,
            "verification_reason": self.verification_reason,
            "ingestable": self.ingestable,
        }


# ``stage`` is reserved for a document that may become a production index
# source.  ``stage_candidate`` copies an unverified document into the staging
# directory for inspection only: it is never ingestable, whatever it looks like.
STAGE_ACTION = "stage"
CANDIDATE_ACTION = "stage_candidate"
COPY_ACTIONS = (STAGE_ACTION, CANDIDATE_ACTION)


# ---------------------------------------------------------------------------
# front matter and identity
# ---------------------------------------------------------------------------


def _scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value.startswith('"'):
        try:
            return json.loads(value)
        except ValueError:
            return value.strip('"')
    if value.isdigit():
        return int(value)
    return value


def parse_front_matter(text: str) -> dict[str, Any]:
    """Parse the leading ``---`` block of a normalized document body."""

    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    result: dict[str, Any] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = _scalar(value)
    return result


def read_front_matter(path: str | Path) -> dict[str, Any]:
    """Read the leading ``---`` block of a normalized Markdown document."""

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_front_matter(text)


def _identity_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_doi(value: Any) -> str | None:
    text = _identity_value(value)
    if not text:
        return None
    return text.casefold()


def _pmid_from_name(path: Path) -> str | None:
    match = re.search(r"PMID[_-]?(\d+)", path.name, flags=re.IGNORECASE)
    return match.group(1) if match else None


def load_normalized_document(
    path: str | Path, *, db_path: str | Path | None = None
) -> NormalizedDocument:
    """Load one normalized document, optionally annotating the A6 binding."""

    document_path = Path(path)
    front = read_front_matter(document_path)
    pmid = _identity_value(front.get("pmid")) or _pmid_from_name(document_path)
    doi = _normalized_doi(front.get("doi"))
    missing = tuple(
        name
        for name in ("pmid", "doi", "source", "license")
        if not _identity_value(front.get(name))
    )
    binding: str | None = None
    if db_path is not None and pmid:
        row = read_task(db_path, pmid)
        binding = "task_missing" if row is None else (
            "bound" if content_binding(row).bound else "unbound"
        )
    return NormalizedDocument(
        path=document_path,
        front_matter=front,
        pmid=pmid,
        doi=doi,
        source=_identity_value(front.get("source")),
        license=_identity_value(front.get("license")),
        content_sha256=_identity_value(front.get("content_sha256")),
        missing_front_matter=missing,
        binding=binding,
    )


def read_task(db_path: str | Path, pmid: str | int) -> dict[str, Any] | None:
    """Read one canonical task row without ever creating or writing the DB."""

    path = Path(db_path)
    if not path.is_file():
        return None
    uri = f"file:{path.as_posix()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        conn = _connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM tasks WHERE pmid=?", (int(pmid),)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def verify_document(
    document: NormalizedDocument,
    task: Mapping[str, Any] | None,
    *,
    db_path: str | Path | None,
) -> tuple[str, str]:
    """Decide whether a normalized document may become an index source.

    Ingestable means: the canonical task exists, the document's declared
    identity agrees with it, its A6 verdict is still bound to the current
    content version, and the content version the normalized document was
    produced from is exactly that current version.  Anything else is reported
    with a reason and stays a candidate.
    """

    if db_path is None:
        return (
            "no_database",
            "未提供 canonical tasks.sqlite，无法核对身份与当前验证版本；"
            "该条目只是候选，不具备可入库语义",
        )
    name_pmid = _pmid_from_name(document.path)
    front_pmid = _identity_value(document.front_matter.get("pmid"))
    if name_pmid and front_pmid and str(name_pmid) != str(front_pmid):
        return (
            "identity_conflict",
            f"文件名 PMID {name_pmid} 与 front matter PMID {front_pmid} 不一致",
        )
    if not document.identity:
        return "missing_identity", "没有 PMID/DOI 身份，无法与 canonical 任务或语料去重"
    if task is None:
        return "task_missing", f"canonical tasks 中不存在 PMID {document.pmid}"
    task_doi = _normalized_doi(task.get("doi"))
    if document.doi and task_doi and document.doi != task_doi:
        return (
            "identity_conflict",
            f"front matter DOI {document.doi} 与 canonical DOI {task_doi} 不一致",
        )
    binding = content_binding(task)
    if not binding.bound:
        return "unbound", f"A6 判定未绑定当前内容版本: {binding.reason}"
    declared = (_identity_value(document.content_sha256) or "").casefold()
    if not declared:
        return "content_version_missing", "front matter 未记录 content_sha256"
    expected = (binding.expected_sha256 or "").casefold()
    if declared != expected:
        return (
            "content_version_mismatch",
            f"normalized content_sha256={declared} 不是当前验证版本 {expected}",
        )
    return "verified", "normalized 内容版本与 canonical 当前验证版本一致"


# ---------------------------------------------------------------------------
# existing converted corpus
# ---------------------------------------------------------------------------


def _mapping_identities(paths: Iterable[str | Path]) -> dict[str, dict[str, str]]:
    """Map a converted filename to the identity recorded by a legacy mapping."""

    identities: dict[str, dict[str, str]] = {}
    for raw_path in paths:
        mapping_path = Path(raw_path)
        if not mapping_path.is_file():
            continue
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        records = payload.get("files") if isinstance(payload, Mapping) else payload
        for record in records or []:
            if not isinstance(record, Mapping):
                continue
            filename = _identity_value(record.get("filename"))
            if not filename:
                continue
            identity: dict[str, str] = {}
            pmid = _identity_value(record.get("pmid"))
            if pmid:
                identity["pmid"] = str(int(str(pmid))) if str(pmid).isdigit() else pmid
            doi = _normalized_doi(record.get("doi"))
            if doi:
                identity["doi"] = doi
            identities[Path(filename).name] = identity
            identities[Path(filename).stem] = identity
    return identities


def converted_identity_keys(
    path: str | Path, *, mapping: Mapping[str, Mapping[str, str]] | None = None
) -> set[tuple[str, str]]:
    """Return the identity keys a converted Markdown file can be matched on."""

    converted = Path(path)
    keys: set[tuple[str, str]] = set()
    front = read_front_matter(converted)
    pmid = _identity_value(front.get("pmid")) or _pmid_from_name(converted)
    if pmid:
        keys.add(("pmid", pmid))
    doi = _normalized_doi(front.get("doi"))
    if doi:
        keys.add(("doi", doi))
    recorded = (mapping or {}).get(converted.name) or (mapping or {}).get(converted.stem)
    if recorded:
        if recorded.get("pmid"):
            keys.add(("pmid", recorded["pmid"]))
        if recorded.get("doi"):
            keys.add(("doi", recorded["doi"]))
    return keys


def converted_index(
    roots: Iterable[str | Path],
    *,
    mapping_paths: Iterable[str | Path] = (),
) -> dict[tuple[str, str], list[Path]]:
    """Index existing converted Markdown by paper identity."""

    mapping = _mapping_identities(mapping_paths)
    index: dict[tuple[str, str], list[Path]] = {}
    for raw_root in roots:
        root = Path(raw_root)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.casefold() not in CONVERTED_SUFFIXES:
                continue
            for key in converted_identity_keys(path, mapping=mapping):
                index.setdefault(key, []).append(path)
    return index


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def preferred_converted(paths: Sequence[Path]) -> Path | None:
    """Pick the preferred converted document for one paper.

    A ``PMID_*`` name inside the archive wins over the top level, which in turn
    wins over a legacy filename: the archive is the ingested corpus and a
    PMID-named file is the canonical name used by the index duplicate rule.
    """

    if not paths:
        return None

    def rank(path: Path) -> tuple[int, int, str]:
        archived = 0 if path.parent.name == "已入库" else 1
        named = 0 if _pmid_from_name(path) else 1
        return (archived, named, path.name.casefold())

    return sorted(paths, key=rank)[0]


def plan_ingestion(
    normalized_root: str | Path = NORMALIZED_ROOT,
    *,
    converted_roots: Sequence[str | Path] = (CONVERTED_ROOT,),
    mapping_paths: Sequence[str | Path] = (),
    out_root: str | Path = STAGING_ROOT,
    db_path: str | Path | None = None,
    pmids: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    """Plan the ingestion of normalized documents without writing anything.

    ``db_path`` is the canonical task database.  Without it nothing can be
    checked against a canonical task, so every document stays a candidate.
    """

    root = Path(normalized_root)
    documents = sorted(root.glob("*.md")) if root.is_dir() else []
    if pmids is not None:
        wanted = {str(pmid) for pmid in pmids}
        documents = [
            path
            for path in documents
            if (_pmid_from_name(path) or "") in wanted
        ]
    index = converted_index(converted_roots, mapping_paths=mapping_paths)
    staging = Path(out_root)
    entries: list[IngestionEntry] = []
    for path in documents:
        document = load_normalized_document(path, db_path=db_path)
        task = read_task(db_path, document.pmid) if (db_path and document.pmid) else None
        verification, verification_reason = verify_document(
            document, task, db_path=db_path
        )
        entry = IngestionEntry(
            pmid=document.pmid,
            normalized_path=str(path),
            identity={
                "pmid": document.pmid,
                "doi": document.doi,
                "source": document.source,
                "license": document.license,
                "content_sha256": document.content_sha256,
                "binding": document.binding,
                "verification": verification,
            },
            missing_front_matter=document.missing_front_matter,
            verification=verification,
            verification_reason=verification_reason,
            ingestable=verification == "verified",
        )
        converted: list[Path] = []
        for key in document.identity:
            for candidate in index.get(key, []):
                if candidate not in converted:
                    converted.append(candidate)
        entry.existing_converted = [str(item) for item in converted]
        if not document.identity:
            entry.action = "missing_identity"
            entry.ingestable = False
            entry.reason = "front matter 与文件名都没有 PMID/DOI，无法与 converted 去重"
            entries.append(entry)
            continue
        existing = preferred_converted(converted)
        if existing is not None:
            entry.preferred_source = "converted"
            entry.preferred_path = str(existing)
            entry.action = "skip_existing_converted"
            # The existing converted body stays the source of record; the
            # normalized copy is never used to replace it.
            entry.ingestable = False
            entry.reason = (
                "该文献已有 converted 正文，按身份去重后优先保留既有来源；"
                "normalized 文档不用于覆盖（corpus 扩展为独立任务）"
            )
            entries.append(entry)
            continue
        entry.preferred_source = "normalized"
        entry.preferred_path = str(path)
        entry.target = str(staging / path.name)
        if entry.ingestable:
            entry.action = STAGE_ACTION
            entry.reason = (
                "converted 中不存在同身份正文，且 normalized 内容版本与 canonical "
                "当前验证版本一致，可作为该文献的入库来源"
            )
        else:
            entry.action = CANDIDATE_ACTION
            entry.reason = (
                "converted 中不存在同身份正文，但该 normalized 文档未通过可入库核验："
                f"{verification} - {verification_reason}；只作为候选复制到 staging 供核对"
            )
        entries.append(entry)
    summary = {
        "normalized_documents": len(documents),
        "converted_candidates": len(index),
        "verified": sum(1 for entry in entries if entry.verification == "verified"),
        "stage": sum(1 for entry in entries if entry.action == STAGE_ACTION),
        "stage_candidates": sum(
            1 for entry in entries if entry.action == CANDIDATE_ACTION
        ),
        "skip_existing_converted": sum(
            1 for entry in entries if entry.action == "skip_existing_converted"
        ),
        "missing_identity": sum(
            1 for entry in entries if entry.action == "missing_identity"
        ),
        "not_ingestable": sum(1 for entry in entries if not entry.ingestable),
        "staged": 0,
        "skipped_unchanged": 0,
        "conflicts": 0,
        "refused_unverified": 0,
    }
    return {
        "normalized_root": str(root),
        "converted_roots": [str(Path(item)) for item in converted_roots],
        "out_root": str(staging),
        "task_db": str(db_path) if db_path is not None else None,
        "entries": entries,
        "summary": summary,
    }


def apply_plan(plan: Mapping[str, Any], *, require_verified: bool = False) -> dict[str, Any]:
    """Copy the planned documents into the staging root, never overwriting.

    ``require_verified`` is the production mode: only a document that passed the
    ingestable check is copied, and an unverified candidate is reported as
    refused instead of being staged as if it were usable.
    """

    staging = Path(str(plan["out_root"]))
    summary = dict(plan["summary"])
    summary["staged"] = 0
    summary["skipped_unchanged"] = 0
    summary["conflicts"] = 0
    summary["refused_unverified"] = 0
    applied: list[dict[str, Any]] = []
    for entry in plan["entries"]:
        record = {
            "pmid": entry.pmid,
            "target": entry.target,
            "action": entry.action,
            "verification": entry.verification,
            "ingestable": entry.ingestable,
        }
        if entry.action not in COPY_ACTIONS or not entry.target:
            applied.append(record)
            continue
        if require_verified and not entry.ingestable:
            summary["refused_unverified"] += 1
            record["action"] = "refused_unverified"
            record["reason"] = entry.verification_reason
            applied.append(record)
            continue
        source = Path(str(entry.normalized_path))
        target = Path(str(entry.target))
        if target.exists():
            if target.read_bytes() == source.read_bytes():
                summary["skipped_unchanged"] += 1
                record["action"] = "skip_unchanged"
                applied.append(record)
                continue
            summary["conflicts"] += 1
            record["action"] = "conflict"
            record["reason"] = "目标已存在且内容不同，不覆盖既有 Markdown"
            applied.append(record)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.staging")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
        summary["staged"] += 1
        record["action"] = "staged"
        applied.append(record)
    return {"out_root": str(staging), "summary": summary, "applied": applied}


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A-R3 normalized 入库适配器（计划默认只读，不写入 converted/）"
    )
    parser.add_argument("command", choices=("plan", "apply"), nargs="?", default="plan")
    parser.add_argument("--normalized-root", default=str(NORMALIZED_ROOT))
    parser.add_argument(
        "--converted-root",
        action="append",
        default=None,
        help="可重复；默认 converted/ 与 converted/已入库/",
    )
    parser.add_argument("--mapping", action="append", default=[])
    parser.add_argument("--out", default=str(STAGING_ROOT))
    parser.add_argument("--db", default=None)
    parser.add_argument("--pmid", action="append", default=None)
    args = parser.parse_args(argv)

    converted_roots = args.converted_root or [
        str(CONVERTED_ROOT),
        str(CONVERTED_ROOT / "已入库"),
    ]
    plan = plan_ingestion(
        args.normalized_root,
        converted_roots=converted_roots,
        mapping_paths=args.mapping,
        out_root=args.out,
        db_path=args.db,
        pmids=args.pmid,
    )
    result = apply_plan(plan, require_verified=True) if args.command == "apply" else {
        "out_root": plan["out_root"],
        "task_db": plan["task_db"],
        "summary": plan["summary"],
        "entries": [entry.as_dict() for entry in plan["entries"]],
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
