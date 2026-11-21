# -*- coding: utf-8 -*-
"""A9 read-only unified corpus and acquisition report."""
from __future__ import annotations
import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
CRAWLER_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = CRAWLER_ROOT.parent
DEFAULT_RAW_ROOT = PROJECT_ROOT / "corpus_raw"
DEFAULT_NORMALIZED_ROOT = PROJECT_ROOT / "normalized"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from common import load_config, validate_pdf
from content_validation import content_binding

LEGACY_STATUS_ALIASES = {"done": "metadata_only", "failed": "retryable_error", "not_found": "metadata_only", "downloading": "fetching"}
REQUIRED_ERROR_BUCKETS = ("timeout", "rate_limit", "server_error", "invalid_content", "metadata_mismatch")
SOURCE_BUCKETS = ("PMC AWS", "Europe PMC", "Unpaywall", "Other", "Unknown")
FORMAT_BUCKETS = ("xml", "pdf", "txt", "html", "unknown")
PDF_NAME_RE = re.compile(r"^PMID_(\d+)\.pdf$", re.IGNORECASE)
PMID_MD_RE = re.compile(r"^PMID_(\d+)\.md$", re.IGNORECASE)
RAW_ARTIFACT_NAMES = frozenset({"article.xml", "article.txt", "article.html", "article.pdf"})
# Gate results are computed, never assumed. "NOT CHECKED" means this run had no
# evidence for the gate; it must not be read as a pass.
GATE_RESULTS = ("PASS", "FAIL", "NOT SATISFIED", "NOT CHECKED")
PMC_AWS_INVENTORY_KEYS = ("pmc_aws_inventory", "pmc_aws_inventory_path", "pmc_aws_inventory_file")
VERIFICATION_SCOPES = ("offline_unit_tests", "single_article_real_run", "full_corpus_rollout")

def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _read_only_connection(path):
    resolved = Path(path).resolve().as_posix()
    # A WAL database without its side files cannot always be opened with plain
    # mode=ro (the production snapshot behaved this way, see the A review); the
    # immutable snapshot URI keeps the report read-only in that case.
    try:
        conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.Error:
        try:
            conn.close()
        except Exception:
            pass
        conn = sqlite3.connect(f"file:{resolved}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def _table_exists(conn, name):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None

def _canonical_status(value):
    raw = str(value or "pending").strip()
    return LEGACY_STATUS_ALIASES.get(raw, raw)

def _source_bucket(value):
    normalized = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized in {"pmc_aws", "pmcaws", "pmc_aws_resolver"}:
        return "PMC AWS"
    if normalized in {"europepmc", "europe_pmc", "europe_pmc_xml"}:
        return "Europe PMC"
    if normalized in {"unpaywall", "unpaywall_api"}:
        return "Unpaywall"
    return "Other" if normalized else "Unknown"

def _is_success_attempt(row):
    outcome = str(row.get("outcome") or "").strip().casefold()
    if row.get("error_class") or row.get("retryable") in (1, True, "1", "true", "TRUE"):
        return False
    return outcome in {"success", "succeeded", "fulltext_ready"}

def _error_bucket(value):
    value = str(value or "").strip().casefold()
    if value in {"timeout", "rate_limit", "server_error"}:
        return value
    if value in {"invalid_pdf", "invalid_xml", "content_too_small", "invalid_content"}:
        return "invalid_content"
    if value in {"metadata_mismatch", "identifier_mismatch"}:
        return "metadata_mismatch"
    return None

def _legacy_error_bucket(value):
    text = str(value or "").casefold()
    if not text:
        return None
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "429" in text or "rate limit" in text or "rate_limit" in text:
        return "rate_limit"
    if any(code in text for code in ("500", "502", "503", "504", "server error")):
        return "server_error"
    if any(token in text for token in ("invalid pdf", "truncated", "invalid content")):
        return "invalid_content"
    if "metadata mismatch" in text or "identity mismatch" in text:
        return "metadata_mismatch"
    return None

def _coverage(rows, field, denominator):
    count = sum(1 for row in rows if row.get(field) not in (None, ""))
    return {"count": count, "denominator": denominator, "percent": round(count / denominator * 100, 2) if denominator else 0.0}

def _load_manifest(path):
    if not Path(path).is_file():
        return {"exists": False, "file_count": 0, "total_chunks": 0, "pmid_names": []}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    files = payload.get("files", {}) if isinstance(payload, dict) else {}
    keys = list(files.keys()) if isinstance(files, dict) else []
    values = list(files.values() if isinstance(files, dict) else files or [])
    computed = sum(int(item.get("chunks", 0) or 0) for item in values if isinstance(item, dict))
    return {"exists": True, "file_count": len(values), "total_chunks": int(payload.get("total_chunks", computed) or 0), "computed_chunks": computed, "built_at": payload.get("built_at"), "reconciled_at": payload.get("reconciled_at"), "pmid_names": keys}

def _load_tracker(path):
    if not Path(path).is_file():
        return {"exists": False, "converted": 0, "failed": 0}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {"exists": True, "converted": len(payload.get("converted", {}) or {}), "failed": len(payload.get("failed", {}) or {})}

def _count_markdown(path):
    path = Path(path)
    return sum(1 for item in path.rglob("*.md") if item.is_file()) if path.is_dir() else 0

def _count_raw_fulltext(path):
    path = Path(path)
    if path.is_file():
        return int(path.name.casefold() in RAW_ARTIFACT_NAMES)
    if not path.is_dir():
        return 0
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and item.name.casefold() in RAW_ARTIFACT_NAMES
    )

def _pdf_inventory(pdf_dir, min_size):
    pdf_dir = Path(pdf_dir)
    files = _pdf_files(pdf_dir)
    valid = 0
    for item in files:
        try:
            valid += int(validate_pdf(item.read_bytes(), min_size))
        except OSError:
            pass
    pmids = {int(PDF_NAME_RE.match(item.name).group(1)) for item in files}
    return {"files": len(files), "valid": valid, "invalid": len(files) - valid, "pmids": pmids, "scanned": True}


def _pdf_files(pdf_dir):
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.is_dir():
        return []
    return [item for item in pdf_dir.iterdir() if item.is_file() and PDF_NAME_RE.match(item.name)]


def _pdf_inventory_names(pdf_dir):
    """List the legacy PDF corpus without parsing its content.

    Re-parsing every PDF on each report run was the expensive part of the old
    report; the file-level cross-check stays available from the names alone.
    """

    files = _pdf_files(pdf_dir)
    return {
        "files": len(files),
        "valid": None,
        "invalid": None,
        "pmids": {int(PDF_NAME_RE.match(item.name).group(1)) for item in files},
        "scanned": False,
    }


def _front_matter(path, limit=25):
    """Read only the leading front-matter block of a normalized document."""

    try:
        with Path(path).open(encoding="utf-8", errors="replace") as handle:
            lines = []
            for index, line in enumerate(handle):
                if index >= limit:
                    break
                lines.append(line.rstrip("\n"))
    except OSError:
        return {}
    if not lines or lines[0].strip() != "---":
        return {}
    result = {}
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip('"') or None
    return result


def _normalized_documents(root):
    """Map normalized PMID-named documents to the PMID they declare."""

    root = Path(root)
    if not root.is_dir():
        return {}
    documents = {}
    for path in sorted(root.glob("*.md")):
        match = PMID_MD_RE.match(path.name)
        front = _front_matter(path)
        raw_pmid = front.get("pmid") or (match.group(1) if match else None)
        if not raw_pmid:
            continue
        digits = "".join(char for char in str(raw_pmid) if char.isdigit())
        if digits:
            documents[digits] = path
    return documents


def _task_format(row):
    value = str(row.get("content_format") or "").strip().casefold()
    if value in FORMAT_BUCKETS:
        return value
    path = str(row.get("content_path") or row.get("validated_content_path") or "")
    suffix = Path(path).suffix.casefold().lstrip(".")
    return suffix if suffix in FORMAT_BUCKETS else "unknown"


def _readiness(tasks, normalized_root, manifest):
    """Separate scheduling status from content readiness (A-R4).

    Readiness is decided by the A6 verdict bound to the current content version
    (``content_binding``), not by ``tasks.status``. This does not re-read corpus
    bytes: it checks that the recorded validated identity still matches the
    task's current content pointer.
    """

    bound, unbound_verdicts = [], []
    for row in tasks:
        if content_binding(row).bound:
            bound.append(row)
        elif row.get("file_valid") == 1 and row.get("bibliographic_match") == 1:
            # A verdict exists but no longer describes the incumbent content.
            unbound_verdicts.append(row)
    metadata_only = [row for row in tasks if _canonical_status(row.get("status")) == "metadata_only"]
    with_body = [row for row in metadata_only if content_binding(row).bound]
    normalized = _normalized_documents(normalized_root)
    indexed_pmids = {
        str(int(PMID_MD_RE.match(name).group(1)))
        for name in (manifest.get("pmid_names") or [])
        if PMID_MD_RE.match(name)
    }
    bound_pmids = {str(int(row["pmid"])) for row in bound}
    return {
        "dimension_note": (
            "scheduling status（tasks.status，调度维度）与 content readiness"
            "（绑定当前内容版本的双验证，正文维度）分开统计，两者不相加。"
        ),
        "scheduling_status_counts": dict(Counter(_canonical_status(row.get("status")) for row in tasks)),
        "content_readiness": {
            "validated_body_bound_to_current_content": len(bound),
            "verdict_without_content_binding": len(unbound_verdicts),
            "verdict_without_content_binding_pmids": [int(row["pmid"]) for row in unbound_verdicts],
        },
        "metadata_only": {
            "total": len(metadata_only),
            "with_validated_body": len(with_body),
            "without_validated_body": len(metadata_only) - len(with_body),
            "note": "metadata_only 仅为调度状态；是否有可用已验证正文以上两列为准。",
        },
        "normalized_readiness": {
            "documents": len(normalized),
            "pmids": sorted(normalized, key=int),
            "indexed_normalized": "NOT CHECKED",
            "note": "normalized/ 不是 build_index 的输入目录；接入方案见 normalized_ingest.py（A-R3）。",
        },
        "indexed_readiness": {
            "manifest_pmid_named_entries": len(indexed_pmids),
            "bound_pmids_present_in_manifest": len(bound_pmids & indexed_pmids),
            "note": "manifest 以 converted 文件名记账，normalized 文档的入库状态未检查。",
        },
    }


def _format_pipeline(tasks, normalized_root, manifest):
    """Per-format fetched/validated/normalized/indexed counts (A-R4)."""

    normalized = _normalized_documents(normalized_root)
    manifest_pmids = {
        str(int(PMID_MD_RE.match(name).group(1)))
        for name in (manifest.get("pmid_names") or [])
        if PMID_MD_RE.match(name)
    }
    rows = {}
    for fmt in FORMAT_BUCKETS:
        format_tasks = [row for row in tasks if _task_format(row) == fmt]
        fetched = [row for row in format_tasks if row.get("content_path")]
        validated = [row for row in format_tasks if content_binding(row).bound]
        validated_pmids = {str(int(row["pmid"])) for row in validated}
        normalized_pmids = {
            pmid for pmid, _path in normalized.items()
            if pmid in {str(int(row["pmid"])) for row in format_tasks}
        }
        rows[fmt] = {
            "fetched": len(fetched),
            "validated": len(validated),
            "normalized": len(normalized_pmids),
            "indexed": len(validated_pmids & manifest_pmids),
        }
    return rows


def _provider_configuration(config_path=None):
    """Report provider configuration facts without contacting any provider."""

    config = {}
    try:
        config = load_config() if config_path is None else json.loads(
            Path(config_path).read_text(encoding="utf-8")
        )
    except Exception:
        config = {}
    inventory = next(
        (
            str(config.get(key)).strip()
            for key in PMC_AWS_INVENTORY_KEYS
            if str(config.get(key) or "").strip()
        ),
        "",
    )
    inventory_path = Path(inventory) if inventory else None
    return {
        "pmc_aws_inventory_path": inventory or None,
        "pmc_aws_inventory_configured": bool(inventory_path and inventory_path.is_file()),
        "api_key_present": bool(os.environ.get("NCBI_API_KEY")),
        "unpaywall_email_present": bool(os.environ.get("UNPAYWALL_EMAIL")),
        "crossref_mailto_present": bool(os.environ.get("CROSSREF_MAILTO")),
        "note": "只报告是否配置/是否存在，不写出任何参数值。",
    }


def _source_queries(attempts, candidates, configuration):
    """Distinguish "not configured", "call failed" and "queried, no candidate"."""

    result = {}
    for bucket in SOURCE_BUCKETS:
        bucket_attempts = [row for row in attempts if _source_bucket(row.get("source")) == bucket]
        bucket_candidates = [
            row for row in candidates if _source_bucket(row.get("source")) == bucket
        ]
        failed = [
            row
            for row in bucket_attempts
            if str(row.get("error_class") or "").strip() or row.get("retryable") in (1, True)
        ]
        if bucket_candidates:
            outcome = "已产生候选"
        elif failed:
            outcome = "调用失败"
        elif bucket == "PMC AWS" and not configuration.get("pmc_aws_inventory_configured"):
            outcome = "配置缺失"
        elif bucket_attempts:
            outcome = "已查询无候选"
        else:
            outcome = "未查询"
        result[bucket] = {
            "attempts": len(bucket_attempts),
            "failed_attempts": len(failed),
            "candidates": len(bucket_candidates),
            "outcome": outcome,
        }
    return result

def _chroma_count(path):
    path = Path(path)
    if not path.is_dir():
        return {"available": False, "count": None, "error": "directory_missing"}
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(path))
        return {"available": True, "count": client.get_collection("papers").count(), "error": None}
    except Exception as exc:
        return {"available": False, "count": None, "error": type(exc).__name__}

def build_report(*, db_path, pdf_dir, raw_root, normalized_root, converted_root, tracker_path, manifest_path, chroma_dir, snapshot_at=None, read_chroma=True, read_pdf_inventory=False):
    paths = {name: Path(value) for name, value in {"db": db_path, "pdf": pdf_dir, "raw": raw_root, "normalized": normalized_root, "converted": converted_root, "tracker": tracker_path, "manifest": manifest_path, "chroma": chroma_dir}.items()}
    conn = _read_only_connection(paths["db"])
    try:
        tasks = [dict(row) for row in conn.execute("SELECT * FROM tasks ORDER BY pmid")]
        attempts = [dict(row) for row in conn.execute("SELECT * FROM fetch_attempts")] if _table_exists(conn, "fetch_attempts") else []
        candidates = [dict(row) for row in conn.execute("SELECT * FROM source_candidates")] if _table_exists(conn, "source_candidates") else []
        schema_version = None
        if _table_exists(conn, "schema_version"):
            row = conn.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
            schema_version = row["version"] if row else None
    finally:
        conn.close()
    raw_status = Counter(str(row.get("status") or "pending") for row in tasks)
    canonical_status = Counter(_canonical_status(row.get("status")) for row in tasks)
    target_rows = [row for row in tasks if _canonical_status(row.get("status")) != "archived"]
    canonical_source_attempts = Counter()
    canonical_source_pmids = defaultdict(set)
    for row in attempts:
        if _is_success_attempt(row):
            bucket = _source_bucket(row.get("source"))
            canonical_source_attempts[bucket] += 1
            canonical_source_pmids[bucket].add(int(row["pmid"]))
    legacy_source_tasks = Counter()
    legacy_source_pmids = defaultdict(set)
    for row in tasks:
        if str(row.get("status") or "") != "done":
            continue
        route = str(row.get("route") or "").strip().casefold()
        if route.startswith("europepmc"):
            bucket = "Europe PMC"
        elif "pmc_aws" in route or "pmcaws" in route:
            bucket = "PMC AWS"
        elif "unpaywall" in route:
            bucket = "Unpaywall"
        elif route:
            bucket = "Other"
        else:
            bucket = "Unknown"
        legacy_source_tasks[bucket] += 1
        legacy_source_pmids[bucket].add(int(row["pmid"]))
    canonical_errors = Counter(_error_bucket(row.get("error_class")) for row in attempts)
    canonical_errors.pop(None, None)
    legacy_errors = Counter(_legacy_error_bucket(row.get("last_error")) for row in tasks)
    legacy_errors.pop(None, None)
    min_size = int(load_config().get("min_pdf_size", 30720))
    # A-R4: the legacy PDF corpus is not re-parsed on every run; the content
    # check is opt-in and the file-level cross-check uses names only.
    pdfs = (
        _pdf_inventory(paths["pdf"], min_size)
        if read_pdf_inventory
        else _pdf_inventory_names(paths["pdf"])
    )
    manifest = _load_manifest(paths["manifest"])
    tracker = _load_tracker(paths["tracker"])
    chroma = _chroma_count(paths["chroma"]) if read_chroma else {"available": False, "count": None, "error": "skipped"}
    done_pmids = {int(row["pmid"]) for row in tasks if str(row.get("status") or "") == "done"}
    configuration = _provider_configuration()
    metadata = {
        "title": _coverage(target_rows, "title", len(target_rows)),
        "doi": _coverage(target_rows, "doi", len(target_rows)),
        "pmcid": _coverage([dict(row, pmcid=row.get("pmcid") or row.get("pmc")) for row in target_rows], "pmcid", len(target_rows)),
        "metadata_updated_at": _coverage(target_rows, "metadata_updated_at", len(target_rows)),
    }
    source_successes = {}
    for bucket in SOURCE_BUCKETS:
        source_successes[bucket] = {
            "canonical_attempts": canonical_source_attempts[bucket],
            "canonical_unique_pmids": len(canonical_source_pmids[bucket]),
            "legacy_done_tasks": legacy_source_tasks[bucket],
            "legacy_unique_pmids": len(legacy_source_pmids[bucket]),
        }
    converted_archive = _count_markdown(paths["converted"] / "已入库")
    project_root = CRAWLER_ROOT.parent
    readiness = _readiness(tasks, paths["normalized"], manifest)
    format_pipeline = _format_pipeline(tasks, paths["normalized"], manifest)
    source_queries = _source_queries(attempts, candidates, configuration)
    gates, gates_detail = _evaluate_gates(
        tasks=tasks,
        attempts=attempts,
        candidates=candidates,
        readiness=readiness,
        manifest=manifest,
        chroma=chroma,
        paths=paths,
        configuration=configuration,
        status_sum_matches=sum(raw_status.values()) == len(tasks),
        metadata_coverage=metadata,
    )
    return {
        "snapshot_at": snapshot_at or _utc_now(),
        "data_sources": {key: str(value) for key, value in paths.items()},
        "sqlite": {"schema_version": schema_version, "task_count": len(tasks), "raw_status": dict(sorted(raw_status.items())), "canonical_status": dict(sorted(canonical_status.items())), "fetch_attempts": len(attempts), "source_candidates": len(candidates), "candidate_sources": dict(sorted(Counter(_source_bucket(row.get("source")) for row in candidates).items()))},
        "corpus": {"corpus_target": len(target_rows), "metadata_ready": canonical_status["metadata_ready"], "fulltext_ready": canonical_status["fulltext_ready"], "metadata_only": canonical_status["metadata_only"], "retryable_error": canonical_status["retryable_error"], "archived": canonical_status["archived"], "excluded": canonical_status["excluded"], "pending": canonical_status["pending"], "other_canonical_states": {key: value for key, value in sorted(canonical_status.items()) if key not in {"metadata_ready", "fulltext_ready", "metadata_only", "retryable_error", "archived", "excluded", "pending"}}},
        "readiness": readiness,
        "format_pipeline": format_pipeline,
        "a2_refresh": {
            "metadata_updated_at": metadata["metadata_updated_at"],
            "pmcid": metadata["pmcid"],
            "note": "A2 refresh 功能已实现；此处只报告 metadata_updated_at 的实际覆盖，不代表全库已刷新。",
        },
        "provider_configuration": configuration,
        "source_queries": source_queries,
        "verification_scope": _verification_scope(target_rows, readiness, paths),
        "metadata_coverage": metadata,
        "source_successes": source_successes,
        "errors": {"canonical_fetch_attempts": {name: canonical_errors[name] for name in REQUIRED_ERROR_BUCKETS}, "legacy_last_error_heuristic": {name: legacy_errors[name] for name in REQUIRED_ERROR_BUCKETS}},
        "files": {"raw_fulltext": _count_raw_fulltext(paths["raw"]), "normalized_documents": _count_markdown(paths["normalized"]), "converted_markdown_archive": converted_archive, "converted_markdown_top_level": sum(1 for item in paths["converted"].glob("*.md") if item.is_file()) if paths["converted"].is_dir() else 0, "conversion_failed": tracker["failed"], "conversion_tracker_converted": tracker["converted"], "flash_failed_pdfs": sum(1 for item in (project_root / "flash-failed").rglob("*.pdf")), "legacy_pdf_corpus": {key: value for key, value in pdfs.items() if key != "pmids"}},
        "index": {"manifest": manifest, "chroma": chroma},
        "cross_checks": {"sqlite_status_sum_matches_tasks": sum(raw_status.values()) == len(tasks), "target_plus_archived_equals_tasks": len(target_rows) + canonical_status["archived"] == len(tasks), "legacy_done_vs_pdf": {"done_tasks": len(done_pmids), "legacy_pdf_files": pdfs["files"], "valid_legacy_pdfs": pdfs["valid"], "done_without_pdf": len(done_pmids - pdfs["pmids"]), "pdf_without_done": len(pdfs["pmids"] - done_pmids), "content_scanned": pdfs["scanned"]}, "manifest_files_vs_converted_archive": {"manifest_files": manifest["file_count"], "converted_markdown_archive": converted_archive, "equal": manifest["file_count"] == converted_archive}, "manifest_chunks_vs_chroma": {"manifest_chunks": manifest.get("total_chunks", 0), "chroma_count": chroma.get("count"), "equal": chroma.get("count") == manifest.get("total_chunks") if chroma.get("count") is not None else None}},
        "gates": gates,
        "gates_detail": gates_detail,
    }


def _gate(result, *, scope, evidence, checked):
    if result not in GATE_RESULTS:
        raise ValueError(f"non-canonical gate result: {result!r}")
    return result, {"result": result, "scope": scope, "evidence": evidence, "checked": checked}


def _evaluate_gates(
    *,
    tasks,
    attempts,
    candidates,
    readiness,
    manifest,
    chroma,
    paths,
    configuration,
    status_sum_matches,
    metadata_coverage,
):
    """Compute every Acquisition Gate from this run's evidence.

    No gate is a constant: a gate without evidence this run reports
    NOT CHECKED instead of PASS.
    """

    gates = {}
    details = {}

    def record(name, result, *, scope, evidence, checked):
        value, detail = _gate(result, scope=scope, evidence=evidence, checked=checked)
        gates[name] = value
        details[name] = detail

    record(
        "sqlite_authority",
        "PASS" if status_sum_matches and tasks is not None else "FAIL",
        scope="本次 snapshot",
        evidence=f"{paths['db']}（tasks 状态分布求和与任务数一致={status_sum_matches}）",
        checked="读取 tasks 并核对状态分布求和",
    )
    provenance_used = bool(attempts) and bool(candidates)
    record(
        "provenance_tables_actually_used",
        "PASS" if provenance_used else "NOT SATISFIED",
        scope="本次 snapshot（生产数据）",
        evidence=(
            f"fetch_attempts={len(attempts)}、source_candidates={len(candidates)}"
        ),
        checked="两表是否有真实记录",
    )
    http_ok, http_evidence = _check_http_error_semantics()
    record(
        "http_error_semantics",
        http_ok,
        scope="代码契约（state_machine 分类函数），与语料状态无关",
        evidence=http_evidence,
        checked="503 不进 source_not_found；429 可重试",
    )
    pmcid_count = metadata_coverage["pmcid"]["count"]
    record(
        "pmcid_metadata_refresh",
        "PASS" if pmcid_count else "NOT SATISFIED",
        scope="本次 snapshot 覆盖 + A2 模块可用性",
        evidence=(
            f"target PMCID 覆盖 {pmcid_count}/"
            f"{metadata_coverage['pmcid']['denominator']}；"
            f"refresh_pubmed_metadata.py / pubmed_metadata.py"
        ),
        checked="PMCID 覆盖是否非零（全库刷新未由本报表验证）",
    )
    modules_result, modules_evidence = _check_official_modules()
    record(
        "official_resolvers_and_validation",
        modules_result,
        scope="模块可用性 + 本次 canonical 计数",
        evidence=(
            f"{modules_evidence}；raw={readiness['content_readiness']['validated_body_bound_to_current_content']} "
            f"条绑定验证正文，normalized={readiness['normalized_readiness']['documents']} 篇"
        ),
        checked="oa_resolver/normalization/content_validation 可导入且本次语料计数已读取",
    )
    fixture_root = PROJECT_ROOT / "tests" / "fixtures"
    fixtures = len(list(fixture_root.glob("*"))) if fixture_root.is_dir() else 0
    record(
        "fixture_isolation",
        "PASS" if fixtures else "NOT CHECKED",
        scope="测试 fixture 与生产语料隔离",
        evidence=f"{fixture_root}（{fixtures} 个 fixture 文件）；本报告只读打开 SQLite",
        checked="fixture 目录存在且报表未写入生产语料",
    )
    record(
        "content_readiness_evidence",
        (
            "PASS"
            if readiness["content_readiness"]["validated_body_bound_to_current_content"]
            else "NOT SATISFIED"
        ),
        scope="本次 snapshot（记录的身份绑定，未重新读取语料字节）",
        evidence=(
            "绑定当前内容版本的双验证任务 "
            f"{readiness['content_readiness']['validated_body_bound_to_current_content']} 条；"
            "存在判定但未绑定的 "
            f"{readiness['content_readiness']['verdict_without_content_binding']} 条"
        ),
        checked="content_binding：验证记录的内容路径/sha256 与当前内容指针一致",
    )
    record(
        "index_consistency",
        _index_gate(manifest, chroma),
        scope="本次 manifest/Chroma 读取",
        evidence=(
            f"manifest files={manifest['file_count']}、chunks={manifest.get('total_chunks', 0)}、"
            f"chroma={chroma.get('count')}（{chroma.get('error') or 'ok'}）"
        ),
        checked="manifest 与 Chroma 计数是否一致",
    )
    record(
        "pmc_aws_inventory",
        "NOT SATISFIED" if not configuration["pmc_aws_inventory_configured"] else "PASS",
        scope="本机配置",
        evidence=(
            "未配置本地 PMC AWS inventory"
            if not configuration["pmc_aws_inventory_configured"]
            else f"inventory={configuration['pmc_aws_inventory_path']}"
        ),
        checked="是否存在本地 inventory 文件",
    )
    return gates, details


def _check_http_error_semantics():
    try:
        from state_machine import classify_http_status

        not_found = classify_http_status(404)
        server = classify_http_status(503)
        rate = classify_http_status(429)
        ok = (
            not_found.source_level
            and not server.source_level
            and server.error_class == "server_error"
            and rate.retryable
        )
        evidence = (
            f"404->{not_found.error_class}(source_level={not_found.source_level})、"
            f"503->{server.error_class}、429->{rate.error_class}(retryable={rate.retryable})"
        )
        return ("PASS" if ok else "FAIL"), evidence
    except Exception as exc:  # pragma: no cover - defensive
        return "NOT CHECKED", f"无法导入 state_machine: {type(exc).__name__}"


def _check_official_modules():
    names = ("oa_resolver", "normalization", "content_validation")
    missing = []
    for name in names:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        return "FAIL", "无法导入: " + ", ".join(missing)
    return "PASS", "、".join(f"{name}.py" for name in names)


def _index_gate(manifest, chroma):
    if chroma.get("count") is None:
        return "NOT CHECKED"
    return "PASS" if chroma.get("count") == manifest.get("total_chunks") else "FAIL"


def _verification_scope(target_rows, readiness, paths):
    """Report the three run scopes separately (A-R4).

    The rollout coverage uses the corpus target as denominator, the same pool
    every other coverage figure in this report uses; archived tasks are outside
    the corpus target by definition and must not enlarge this denominator.
    """

    bound = readiness["content_readiness"]["validated_body_bound_to_current_content"]
    coverage = _coverage(target_rows, "metadata_updated_at", len(target_rows))
    return {
        "offline_unit_tests": {
            "result": "NOT CHECKED",
            "evidence": "tests/test_a*.py（需单独运行，本报表不执行测试）",
            "note": "离线单测通过只说明离线覆盖，不代表生产链路已跑通。",
        },
        "single_article_real_run": {
            "result": "OBSERVED" if bound else "NOT OBSERVED",
            "observed": bound,
            "evidence": f"{paths['db']} 中绑定验证正文的 canonical 任务数={bound}",
            "note": "单篇真实成功不等于全语料 rollout。",
        },
        "full_corpus_rollout": {
            "result": "NOT STARTED" if coverage["count"] == 0 else "PARTIAL",
            "metadata_updated_at": coverage,
            "evidence": (
                f"metadata_updated_at 覆盖 {_fmt_percent(coverage)}"
                "（分母=corpus target，与 A2 覆盖口径一致）"
            ),
            "note": "全量 rollout 以 metadata/content 覆盖为准，未完成前不得解释为全库完成。",
        },
    }

def _fmt_percent(value):
    return f"{value['count']}/{value['denominator']} ({value['percent']:.2f}%)"

def render_markdown(report):
    sqlite = report["sqlite"]; corpus = report["corpus"]; files = report["files"]; cross = report["cross_checks"]; metadata = report["metadata_coverage"]; source = report["source_successes"]; errors = report["errors"]; gates = report["gates"]; manifest = report["index"]["manifest"]; chroma = report["index"]["chroma"]
    data_sources = report.get("data_sources") or {}
    readiness = report.get("readiness") or {}
    content_readiness = readiness.get("content_readiness") or {}
    metadata_only = readiness.get("metadata_only") or {}
    normalized_readiness = readiness.get("normalized_readiness") or {}
    indexed_readiness = readiness.get("indexed_readiness") or {}
    format_pipeline = report.get("format_pipeline") or {}
    a2_refresh = report.get("a2_refresh") or {}
    provider = report.get("provider_configuration") or {}
    source_queries = report.get("source_queries") or {}
    scopes = report.get("verification_scope") or {}
    gate_detail = report.get("gates_detail") or {}
    legacy_pdfs = files['legacy_pdf_corpus']
    legacy_pdf_note = (
        f"{legacy_pdfs['files']}（有效 {legacy_pdfs['valid']}）"
        if legacy_pdfs.get('valid') is not None
        else f"{legacy_pdfs['files']}（未解析内容）"
    )
    lines = [
        "# A9 统一报告：Reporting 与 Acquisition 总验收", "",
        f"> Snapshot: {report['snapshot_at']}",
        "> 本报告由 a9_reporting.py 只读生成；SQLite、文件系统、manifest 和 Chroma 均未被修改。", "",
        "## 数据源与统计口径", "",
        "### 数据源",
        "",
        f"- SQLite: {data_sources.get('db', 'not provided')}",
        f"- Legacy PDF corpus: {data_sources.get('pdf', 'not provided')}",
        f"- Canonical raw fulltext: {data_sources.get('raw', 'not provided')}",
        f"- Canonical normalized documents: {data_sources.get('normalized', 'not provided')}",
        f"- RAG Markdown: {data_sources.get('converted', 'not provided')}",
        f"- Conversion tracker: {data_sources.get('tracker', 'not provided')}",
        f"- Manifest: {data_sources.get('manifest', 'not provided')}",
        f"- Chroma: {data_sources.get('chroma', 'not provided')}",
        "",
        "- SQLite tasks 是任务状态的唯一权威源；fetch_attempts/source_candidates 单独作为 A1–A5 provenance 证据，不以旧 CSV 反推 canonical attempt。",
        "- Corpus target 定义为 canonical 状态不是 archived 的任务；当前与 2000 年及以后目标池一致。",
        "- Metadata ready、Fulltext ready、Metadata only、Retryable error、Archived、Excluded 按 canonical 状态计数；旧状态 done→metadata_only、failed→retryable_error、not_found→metadata_only。",
        "- “Fulltext ready”在本报告分两处出现：调度状态计数（tasks.status）与 content readiness（绑定当前内容版本的双验证）。两者是不同维度，不相加；详见“准备度维度”。",
        "- 来源成功分别报告 A5 canonical fetch_attempts 和旧下载器 tasks.status=done + route；两者不可混算。",
        "- raw_fulltext 统计 A5 canonical raw 目录中的 article.xml/article.txt/article.html/article.pdf artifact 文件；normalized_documents 统计 A7 canonical Markdown，旧 pdfs_merged 和 RAG converted/已入库单列。",
        "- legacy PDF 目录默认只按文件名核对，不解析 PDF 内容（如需内容校验用 --pdf-inventory 显式开启）。", "",
        "## Corpus 与 RAG 准备度", "", "| 指标 | 数量 |", "|---|---:|",
        f"| Corpus target | {corpus['corpus_target']} |", f"| Metadata ready | {corpus['metadata_ready']} |", f"| Fulltext ready（调度状态 tasks.status） | {corpus['fulltext_ready']} |", f"| Metadata only | {corpus['metadata_only']} |", f"| Retryable error | {corpus['retryable_error']} |", f"| Archived | {corpus['archived']} |", f"| Excluded | {corpus['excluded']} |", f"| Pending | {corpus['pending']} |", "",
        "### 准备度维度（调度状态 vs 正文就绪）", "",
        readiness.get("dimension_note", "调度状态与正文就绪分开统计。"), "",
        "| 维度 | 指标 | 数量 |", "|---|---|---:|",
        f"| 调度状态 | Fulltext ready（tasks.status） | {corpus['fulltext_ready']} |",
        f"| 正文就绪 | 绑定当前内容版本的双验证正文 | {content_readiness.get('validated_body_bound_to_current_content', 'NOT CHECKED')} |",
        f"| 正文就绪 | 有判定但未绑定当前内容版本 | {content_readiness.get('verdict_without_content_binding', 'NOT CHECKED')} |",
        f"| Metadata only | 合计（调度状态） | {metadata_only.get('total', 'NOT CHECKED')} |",
        f"| Metadata only | 其中有可用已验证正文 | {metadata_only.get('with_validated_body', 'NOT CHECKED')} |",
        f"| Metadata only | 其中没有可用已验证正文 | {metadata_only.get('without_validated_body', 'NOT CHECKED')} |",
        f"| Normalized | normalized 文档 | {normalized_readiness.get('documents', 'NOT CHECKED')} |",
        f"| Indexed | normalized 是否入库 | {normalized_readiness.get('indexed_normalized', 'NOT CHECKED')} |",
        f"| Indexed | manifest 中 PMID 命名条目 | {indexed_readiness.get('manifest_pmid_named_entries', 'NOT CHECKED')} |",
        "",
        "### 格式维度统计（已获取 / 已验证 / 已标准化 / 已索引）", "",
        "| 格式 | 已获取 | 已验证 | 已标准化 | 已索引（manifest PMID 条目） |", "|---|---:|---:|---:|---:|",
    ]
    for fmt in FORMAT_BUCKETS:
        row = format_pipeline.get(fmt) or {}
        lines.append(
            f"| {fmt} | {row.get('fetched', 'NOT CHECKED')} | {row.get('validated', 'NOT CHECKED')} | "
            f"{row.get('normalized', 'NOT CHECKED')} | {row.get('indexed', 'NOT CHECKED')} |"
        )
    lines += [
        "",
        "> 已索引列仅统计 manifest 中以 PMID 命名的条目；converted 中的旧 Zotero/Word 名字由 index/ 自身的映射处理，“normalized 已入库”为 NOT CHECKED（normalized/ 尚不是 build_index 输入）。", "",
        "### A2 metadata refresh 实际覆盖", "",
        f"- metadata_updated_at 覆盖：{_fmt_percent(a2_refresh.get('metadata_updated_at') or metadata['metadata_updated_at'])}",
        f"- PMCID 覆盖：{_fmt_percent(a2_refresh.get('pmcid') or metadata['pmcid'])}",
        f"- {a2_refresh.get('note', '覆盖不代表全库已刷新。')}", "",
        "### 来源查询状态（配置缺失 / 调用失败 / 已查询无候选）", "",
        "| 来源 | 尝试 | 失败尝试 | 候选 | 结论 |", "|---|---:|---:|---:|---|",
    ]
    for bucket in SOURCE_BUCKETS:
        row = source_queries.get(bucket) or {}
        lines.append(
            f"| {bucket} | {row.get('attempts', 'NOT CHECKED')} | {row.get('failed_attempts', 'NOT CHECKED')} | "
            f"{row.get('candidates', 'NOT CHECKED')} | {row.get('outcome', 'NOT CHECKED')} |"
        )
    lines += [
        "",
        f"- PMC AWS inventory：{'已配置 ' + str(provider.get('pmc_aws_inventory_path')) if provider.get('pmc_aws_inventory_configured') else '未配置本地 inventory（无法区分“未查询”与“无候选”）'}",
        f"- 凭据存在性（只报有无，不写值）：NCBI_API_KEY={provider.get('api_key_present')}、UNPAYWALL_EMAIL={provider.get('unpaywall_email_present')}、CROSSREF_MAILTO={provider.get('crossref_mailto_present')}", "",
        "### 验证范围（三件不同的事）", "",
        "| 范围 | 结论 | 证据 |", "|---|---|---|",
    ]
    for scope in VERIFICATION_SCOPES:
        row = scopes.get(scope) or {}
        lines.append(
            f"| {scope} | {row.get('result', 'NOT CHECKED')} | {row.get('evidence', 'not provided')} |"
        )
    lines += [
        "",
        "### 元数据覆盖（Corpus target 分母）", "", "| 字段 | 覆盖 |", "|---|---:|",
        f"| DOI | {_fmt_percent(metadata['doi'])} |", f"| PMCID（pmcid 或 legacy pmc） | {_fmt_percent(metadata['pmcid'])} |", f"| Title | {_fmt_percent(metadata['title'])} |", f"| metadata_updated_at | {_fmt_percent(metadata['metadata_updated_at'])} |", "",
        "### 文件与索引", "", "| 指标 | 数量 |", "|---|---:|",
        f"| raw_fulltext artifacts（canonical corpus_raw/） | {files['raw_fulltext']} |", f"| normalized_documents（canonical normalized/） | {files['normalized_documents']} |", f"| converted Markdown archive | {files['converted_markdown_archive']} |", f"| conversion_failed（tracker） | {files['conversion_failed']} |", f"| flash-failed PDFs | {files['flash_failed_pdfs']} |", f"| legacy pdfs_merged PDFs | {legacy_pdf_note} |", f"| manifest files / chunks | {manifest['file_count']} / {manifest['total_chunks']} |", f"| Chroma papers chunks | {chroma['count'] if chroma['count'] is not None else '不可用'} |", "",
        "## 来源成功数", "", "| 来源 | canonical attempts | canonical unique PMIDs | legacy done tasks |", "|---|---:|---:|---:|",
    ]
    for bucket in SOURCE_BUCKETS:
        row = source[bucket]
        lines.append(f"| {bucket} | {row['canonical_attempts']} | {row['canonical_unique_pmids']} | {row['legacy_done_tasks']} |")
    lines += ["", "## 错误分类", "", "| 类别 | canonical fetch_attempts | legacy last_error 文本辅助统计 |", "|---|---:|---:|"]
    for name in REQUIRED_ERROR_BUCKETS:
        lines.append(f"| {name} | {errors['canonical_fetch_attempts'][name]} | {errors['legacy_last_error_heuristic'][name]} |")
    provenance_used = gates.get("provenance_tables_actually_used") == "PASS"
    provenance_evidence = (
        f"生产 snapshot 为 fetch_attempts={sqlite['fetch_attempts']}、source_candidates={sqlite['source_candidates']}，两表均有真实记录。"
        if provenance_used
        else f"生产 snapshot 为 fetch_attempts={sqlite['fetch_attempts']}、source_candidates={sqlite['source_candidates']}，不能宣称生产已使用。"
    )
    provenance_conclusion = (
        "当前代码与离线 fixture 覆盖了 Acquisition 设计所需的状态、来源解析、重试、JATS normalization 和 bibliographic validation；生产 SQLite 已产生真实 A1–A5 provenance rows，provenance-use Gate 已满足。"
        if provenance_used
        else "当前代码与离线 fixture 覆盖了 Acquisition 设计所需的状态、来源解析、重试、JATS normalization 和 bibliographic validation；但生产 SQLite 尚未产生完整的 A1–A5 provenance rows，provenance-use Gate 未满足。该缺口应通过真实 Acquisition rollout 补齐，不应通过回填或改写历史数据伪造证据。"
    )
    legacy_pdf_cross = cross['legacy_done_vs_pdf']
    legacy_pdf_cross_text = (
        f"- 旧下载器 done 与 PDF：done {legacy_pdf_cross['done_tasks']}，PDF {legacy_pdf_cross['legacy_pdf_files']}，"
        f"有效 PDF {legacy_pdf_cross['valid_legacy_pdfs']}，done 无 PDF {legacy_pdf_cross['done_without_pdf']}，"
        f"PDF 无 done {legacy_pdf_cross['pdf_without_done']}。"
        if legacy_pdf_cross.get('content_scanned', True)
        else f"- 旧下载器 done 与 PDF：done {legacy_pdf_cross['done_tasks']}，PDF {legacy_pdf_cross['legacy_pdf_files']}，"
             f"done 无 PDF {legacy_pdf_cross['done_without_pdf']}，PDF 无 done {legacy_pdf_cross['pdf_without_done']}；"
             "PDF 内容有效性本次未解析（--pdf-inventory 可显式开启）。"
    )
    lines += ["", "> legacy 文本统计仅用于解释历史失败，不作为 Acquisition Gate 的结构化证据。", "", "## Acquisition Gate", "", "| Gate | 结论 | 适用范围 | 证据与检查方式 |", "|---|---|---|---|",
        f"| SQLite 是唯一权威源 | {gates.get('sqlite_authority', 'NOT CHECKED')} | {_gate_row(gate_detail, 'sqlite_authority')[0]} | a1_schema.py、state_machine.py、tasks.sqlite schema v{sqlite['schema_version']}；{sqlite['task_count']} 条任务；{_gate_row(gate_detail, 'sqlite_authority')[1]} |",
        f"| fetch_attempts 与 source_candidates 已实际使用 | {gates.get('provenance_tables_actually_used', 'NOT CHECKED')} | {_gate_row(gate_detail, 'provenance_tables_actually_used')[0]} | oa_resolver.py 写入 source_candidates，fulltext_client.py/state_machine.py 写入 fetch_attempts；{provenance_evidence} |",
        f"| 503 不进 not_found，429 可重试 | {gates.get('http_error_semantics', 'NOT CHECKED')} | {_gate_row(gate_detail, 'http_error_semantics')[0]} | {_gate_row(gate_detail, 'http_error_semantics')[1]} |",
        f"| PMCID 可由 metadata refresh 补充 | {gates.get('pmcid_metadata_refresh', 'NOT CHECKED')} | {_gate_row(gate_detail, 'pmcid_metadata_refresh')[0]} | pubmed_metadata.py、refresh_pubmed_metadata.py；test_a2_metadata.py；{_gate_row(gate_detail, 'pmcid_metadata_refresh')[1]} |",
        f"| PMC AWS、Europe PMC XML、JATS normalization、bibliographic validation 可用 | {gates.get('official_resolvers_and_validation', 'NOT CHECKED')} | {_gate_row(gate_detail, 'official_resolvers_and_validation')[0]} | {_gate_row(gate_detail, 'official_resolvers_and_validation')[1]} |",
        f"| 生产 fixture 与真实语料隔离 | {gates.get('fixture_isolation', 'NOT CHECKED')} | {_gate_row(gate_detail, 'fixture_isolation')[0]} | {_gate_row(gate_detail, 'fixture_isolation')[1]} |",
        f"| 正文就绪有绑定当前内容版本的验证证据 | {gates.get('content_readiness_evidence', 'NOT CHECKED')} | {_gate_row(gate_detail, 'content_readiness_evidence')[0]} | {_gate_row(gate_detail, 'content_readiness_evidence')[1]} |",
        f"| 索引账实一致（manifest vs Chroma） | {gates.get('index_consistency', 'NOT CHECKED')} | {_gate_row(gate_detail, 'index_consistency')[0]} | {_gate_row(gate_detail, 'index_consistency')[1]} |",
        f"| PMC AWS 本地 inventory 已配置 | {gates.get('pmc_aws_inventory', 'NOT CHECKED')} | {_gate_row(gate_detail, 'pmc_aws_inventory')[0]} | {_gate_row(gate_detail, 'pmc_aws_inventory')[1]} |", "",
        "## 交叉核对", "",
        f"- SQLite 状态分布求和：{'PASS' if cross['sqlite_status_sum_matches_tasks'] else 'FAIL'}；target + archived = tasks：{'PASS' if cross['target_plus_archived_equals_tasks'] else 'FAIL'}。",
        legacy_pdf_cross_text,
        f"- manifest 文件数与 converted/已入库：{cross['manifest_files_vs_converted_archive']['manifest_files']} vs {cross['manifest_files_vs_converted_archive']['converted_markdown_archive']}，{'一致' if cross['manifest_files_vs_converted_archive']['equal'] else '不一致'}。",
        f"- manifest chunks 与 Chroma：{cross['manifest_chunks_vs_chroma']['manifest_chunks']} vs {cross['manifest_chunks_vs_chroma']['chroma_count']}，{'一致' if cross['manifest_chunks_vs_chroma']['equal'] else '不一致/不可用'}。", "",
        "## Gate 结论", "",
        provenance_conclusion, "",
        "本报告的 Gate 由本次读取到的数据与代码检查结果产生；无法在离线只读报表中判定的项目记为 NOT CHECKED，而不是 PASS。", "",
    ]
    return "\n".join(lines)


def _gate_row(detail, name):
    """Return (scope, evidence) for one gate, tolerating hand-built reports."""

    entry = (detail or {}).get(name) or {}
    return (
        entry.get("scope", "未提供（旧报表结构）"),
        entry.get("evidence", "未提供（旧报表结构）"),
    )

def _build_parser():
    parser = argparse.ArgumentParser(description="Generate the read-only A9 unified report")
    parser.add_argument("--db", default=str(CRAWLER_ROOT / "tasks.sqlite"))
    parser.add_argument("--pdf-dir", default=str(CRAWLER_ROOT / "pdfs_merged"))
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    parser.add_argument("--normalized-root", default=str(DEFAULT_NORMALIZED_ROOT))
    parser.add_argument("--converted-root", default=str(PROJECT_ROOT / "converted"))
    parser.add_argument("--tracker", default=str(PROJECT_ROOT / ".claude" / "conversion-tracker.json"))
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "index" / "manifest.json"))
    parser.add_argument("--chroma-dir", default=str(PROJECT_ROOT / "index" / "chroma_db"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "整改手册" / "docs" / "reviews" / "stage-A9-unified-report.md"))
    parser.add_argument(
        "--json-out",
        default=None,
        help="同时写出机器可读的 JSON 报表（内容与 Markdown 同一份快照）",
    )
    parser.add_argument("--snapshot-at", default=None)
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument(
        "--pdf-inventory",
        action="store_true",
        help="显式解析旧 pdfs_merged PDF 内容（默认只按文件名核对）",
    )
    return parser

def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    report = build_report(db_path=args.db, pdf_dir=args.pdf_dir, raw_root=args.raw_root, normalized_root=args.normalized_root, converted_root=args.converted_root, tracker_path=args.tracker, manifest_path=args.manifest, chroma_dir=args.chroma_dir, snapshot_at=args.snapshot_at, read_chroma=not args.skip_chroma, read_pdf_inventory=args.pdf_inventory)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(report), encoding="utf-8")
    json_output = None
    if args.json_out:
        json_output = Path(args.json_out)
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"output": str(output), "json_output": str(json_output) if json_output else None, "snapshot_at": report["snapshot_at"], "gates": report["gates"]}, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
