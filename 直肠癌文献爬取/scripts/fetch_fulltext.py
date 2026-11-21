# -*- coding: utf-8 -*-
"""A5 single-PMID full-text acquisition into the raw corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a1_schema import DEFAULT_DB_PATH, _connect, migrate_to_v2, utc_now
from content_validation import (
    INVALIDATED_ON_CONTENT_REPLACEMENT,
    content_binding,
    ensure_validation_schema,
    validate_task_content,
)
from fulltext_client import (
    FetchResult,
    HttpResponse,
    UnifiedHttpClient,
    load_project_config,
    redact_text,
    redact_url,
)
from ingest_pmids import normalize_pmids
from lease_batch import (
    ACQUISITION_STATUSES,
    DEFAULT_LEASE_SECONDS,
    LeaseConflictError,
    assert_write_owner_in_connection,
    claim_task,
    heartbeat_task,
    release_task,
)
from oa_resolver import (
    INSTITUTIONAL_REPOSITORY,
    PUBLISHER,
    SOURCE_PRIORITY,
    UNPAYWALL,
    _candidate_sort_key,
)


RAW_ROOT = Path(__file__).resolve().parents[2] / "corpus_raw"
FETCHABLE_FORMATS = frozenset({"xml", "txt", "html", "pdf"})
# A-R3: only XML/PDF can pass A6 validation, so only they may be fetched as
# ingestable candidates. TXT/HTML stay fetchable in principle but are skipped
# as candidates and the loop falls back to the next source.
INGESTABLE_FORMATS = frozenset({"xml", "pdf"})
UNVERIFIED_FORMATS = frozenset({"txt", "html"})
FORMAT_FILENAMES = {
    "xml": "article.xml",
    "txt": "article.txt",
    "html": "article.html",
    "pdf": "article.pdf",
}
LEGAL_OA_SOURCES = frozenset({UNPAYWALL, PUBLISHER, INSTITUTIONAL_REPOSITORY})
TERMINAL_SKIP_STATUSES = frozenset({"archived", "excluded", "fulltext_ready"})
# A-R2: the lease is renewed to cover the client's own bounded retry budget for
# one candidate plus this margin, so a single long HTTP/backoff/validation pass
# runs on a lease that was extended for the duration of that operation.
LEASE_BUDGET_MARGIN_SECONDS = 30.0


def _text(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _pmcid(value: Any) -> str | None:
    value = _text(value)
    if value and not value.upper().startswith("PMC"):
        value = "PMC" + value
    return value.upper() if value else None


def _format(candidate: Mapping[str, Any]) -> str | None:
    value = _text(candidate.get("format"))
    if value:
        value = value.casefold()
        if value in {"text", "plain", "plaintext"}:
            value = "txt"
        elif value in {"xhtml", "htm"}:
            value = "html"
        return value if value in FETCHABLE_FORMATS else None
    url = _text(candidate.get("url")) or ""
    suffix = Path(url.split("?", 1)[0].split("#", 1)[0]).suffix.casefold().lstrip(".")
    return suffix if suffix in FETCHABLE_FORMATS else "html"


def _fetch_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Apply A5's format/source sequence with A4 as the tie-break owner."""
    source = candidate.get("source")
    fmt = _format(candidate)
    if source == "PMC_AWS" and fmt == "xml":
        rank = 0
    elif source == "EuropePMC" and fmt == "xml":
        rank = 1
    elif source == "PMC_AWS" and fmt == "txt":
        rank = 2
    elif fmt in {"xml", "html"} and source in LEGAL_OA_SOURCES:
        rank = 3
    elif fmt == "pdf":
        rank = 4
    else:
        rank = 5
    return (rank, *_candidate_sort_key(candidate))


def order_candidates(candidates: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return only fetchable A4 candidates in deterministic A5 order."""
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = dict(candidate)
        row["format"] = _format(row)
        if row.get("source") not in SOURCE_PRIORITY:
            continue
        if not row.get("url") or row["format"] not in FETCHABLE_FORMATS:
            continue
        if row["source"] in LEGAL_OA_SOURCES and row["format"] not in {"xml", "html", "pdf"}:
            continue
        rows.append(row)
    return sorted(rows, key=_fetch_rank)


@dataclass(frozen=True)
class RawArtifact:
    path: Path
    source_json: Path
    sha256: str
    format: str
    retrieved_at: str
    backup_path: Path | None = None


class RawCorpusWriter:
    """Publish article bytes and source.json as one raw directory."""

    def __init__(self, raw_root: str | Path = RAW_ROOT) -> None:
        self.raw_root = Path(raw_root)

    @staticmethod
    def _atomic_write_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_write_text(path: Path, value: str) -> None:
        RawCorpusWriter._atomic_write_bytes(path, value.encode("utf-8"))

    @staticmethod
    def _remove_directory(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)

    @classmethod
    def _publish_directory(cls, staging: Path, target: Path) -> Path | None:
        backup: Path | None = None
        target_moved = False
        staging_installed = False
        try:
            if target.exists():
                backup = Path(
                    tempfile.mkdtemp(
                        prefix=f".{target.name}.backup-", dir=target.parent
                    )
                )
                backup.rmdir()
                try:
                    os.replace(target, backup)
                except Exception:
                    # Detect a completed rename even if the call raised after
                    # the filesystem action. The original target still being
                    # present is not evidence that it belongs to this publish.
                    target_moved = backup.exists() and not target.exists()
                    raise
                target_moved = True
            try:
                os.replace(staging, target)
            except Exception:
                # staging disappearing proves this invocation installed it;
                # target.exists() alone could still refer to the old directory.
                staging_installed = not staging.exists() and target.exists()
                raise
            staging_installed = True
            return backup
        except Exception as publish_error:
            recovery_error: Exception | None = None
            if staging_installed:
                try:
                    cls._remove_directory(target)
                except Exception as exc:
                    recovery_error = exc

            if recovery_error is None and target_moved:
                try:
                    if backup is None or not backup.exists():
                        raise FileNotFoundError(
                            f"original raw backup is missing for {target}"
                        )
                    os.replace(backup, target)
                    target_moved = False
                except Exception as exc:
                    recovery_error = exc
            elif not target_moved and backup is not None and backup.exists():
                # This can only be the empty placeholder created above.
                try:
                    backup.rmdir()
                except Exception as exc:
                    recovery_error = exc

            try:
                cls._remove_directory(staging)
            except Exception as exc:
                if recovery_error is None:
                    recovery_error = exc

            if recovery_error is not None:
                retained = ""
                if target_moved and backup is not None and backup.exists():
                    retained = (
                        f"; recovery copy remains at {backup}; original raw directory retained at backup"
                    )
                raise OSError(
                    f"raw directory publish failed for {target}; rollback/recovery "
                    f"failed: {recovery_error!r}; original publish error: "
                    f"{publish_error!r}{retained}"
                ) from recovery_error
            raise

    def commit(self, artifact: RawArtifact) -> None:
        """Finalize a published artifact after its SQLite metadata is durable."""
        if artifact.backup_path is not None:
            self._remove_directory(artifact.backup_path)

    def rollback(self, artifact: RawArtifact) -> None:
        """Remove the published artifact and restore the previous PMID directory."""

        target = artifact.path.parent
        backup = artifact.backup_path
        if backup is None:
            self._remove_directory(target)
            return
        if not backup.exists():
            raise OSError(
                f"previous raw directory backup is missing at {backup}; "
                f"new raw directory remains at {target}"
            )
        try:
            self._remove_directory(target)
        except Exception as cleanup_error:
            raise OSError(
                f"new raw directory could not be removed; previous directory "
                f"remains at {backup}"
            ) from cleanup_error
        try:
            os.replace(backup, target)
        except Exception as restore_error:
            raise OSError(
                f"previous raw directory restore failed; recovery copy remains at {backup}"
            ) from restore_error

    def write(
        self,
        *,
        pmid: int,
        task: Mapping[str, Any],
        candidate: Mapping[str, Any],
        response: HttpResponse,
        defer_commit: bool = False,
    ) -> RawArtifact:
        fmt = _format(candidate)
        if fmt not in FORMAT_FILENAMES:
            raise ValueError(f"unsupported raw format: {fmt!r}")
        retrieved_at = utc_now()
        content = bytes(response.body)
        digest = hashlib.sha256(content).hexdigest()
        directory = self.raw_root / f"PMID_{int(pmid)}"
        article_path = directory / FORMAT_FILENAMES[fmt]
        source_json_path = directory / "source.json"
        record = {
            "PMID": int(pmid),
            "DOI": _text(task.get("doi")),
            "PMCID": _pmcid(candidate.get("pmcid") or task.get("pmcid") or task.get("pmc")),
            "source": _text(candidate.get("source")),
            "URL": redact_url(str(candidate.get("url"))),
            "license": _text(candidate.get("license") or task.get("license") or "unverified"),
            "retrieved_at": retrieved_at,
            "sha256": digest,
            "format": fmt,
            "version": _text(candidate.get("version") or candidate.get("article_version")) or "unknown",
        }
        record = {
            key: redact_text(value) if isinstance(value, str) else value
            for key, value in record.items()
        }
        self.raw_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=self.raw_root))
        staged_article = staging / FORMAT_FILENAMES[fmt]
        staged_source_json = staging / "source.json"
        try:
            self._atomic_write_bytes(staged_article, content)
            self._atomic_write_text(
                staged_source_json,
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            backup_path = self._publish_directory(staging, directory)
        except Exception as write_error:
            try:
                self._remove_directory(staging)
            except Exception as cleanup_error:
                raise OSError(
                    f"{write_error}; staging cleanup failed at {staging}: {cleanup_error}"
                ) from write_error
            raise
        artifact = RawArtifact(
            path=article_path,
            source_json=source_json_path,
            sha256=digest,
            format=fmt,
            retrieved_at=retrieved_at,
            backup_path=backup_path,
        )
        if not defer_commit:
            self.commit(artifact)
            artifact = RawArtifact(
                path=artifact.path,
                source_json=artifact.source_json,
                sha256=artifact.sha256,
                format=artifact.format,
                retrieved_at=artifact.retrieved_at,
            )
        return artifact


def _task_and_candidates(
    pmid: int,
    db_path: str | Path,
    candidates: Sequence[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    migrate_to_v2(db_path)
    ensure_validation_schema(db_path)
    conn = _connect(db_path)
    try:
        task_row = conn.execute("SELECT * FROM tasks WHERE pmid=?", (pmid,)).fetchone()
        if task_row is None:
            raise ValueError(f"task PMID {pmid} does not exist")
        if candidates is None:
            candidate_rows = conn.execute(
                "SELECT * FROM source_candidates WHERE pmid=?", (pmid,)
            ).fetchall()
            candidates = [dict(row) for row in candidate_rows]
        return dict(task_row), order_candidates(candidates)
    finally:
        conn.close()


def _verdict_reset_clause(conn) -> str:
    """Assignments that invalidate a previous A6 verdict on content replace."""

    existing = {row[1] for row in conn.execute('PRAGMA table_info("tasks")')}
    return "".join(
        f"{name}=NULL, "
        for name in INVALIDATED_ON_CONTENT_REPLACEMENT
        if name in existing
    )


def _update_task_content(
    pmid: int,
    candidate: Mapping[str, Any],
    artifact: RawArtifact,
    *,
    db_path: str | Path,
    worker_id: str | None = None,
) -> None:
    """Commit the new content pointer, invalidating the previous verdict.

    A-R1/A-R2: replacing the artifact clears the A6 verified-content binding so
    a stale ``file_valid=1`` can never sit next to new bytes, and the commit
    obeys the incumbent task lease.
    """

    timestamp = artifact.retrieved_at
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        assert_write_owner_in_connection(
            conn,
            pmid,
            worker_id,
            now=timestamp,
            operation="A5 content pointer update",
        )
        conn.execute(
            f"""UPDATE tasks SET content_status=?, content_source=?, content_format=?,
                content_path=?, source_url=?, content_sha256=?, source_updated_at=?,
                retrieved_at=?, {_verdict_reset_clause(conn)}updated_at=? WHERE pmid=?""",
            (
                "fetched",
                candidate.get("source"),
                artifact.format,
                str(artifact.path),
                redact_url(str(candidate.get("url"))),
                artifact.sha256,
                timestamp,
                timestamp,
                timestamp,
                pmid,
            ),
        )
        if conn.total_changes != 1:
            raise ValueError(f"task PMID {pmid} disappeared during content update")
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _task_row(pmid: int, db_path: str | Path) -> dict[str, Any]:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM tasks WHERE pmid=?", (pmid,)).fetchone()
        if row is None:
            raise ValueError(f"task PMID {pmid} does not exist")
        return dict(row)
    finally:
        conn.close()


def _current_status(pmid: int, db_path: str | Path) -> str:
    return str(_task_row(pmid, db_path)["status"])


def _held_by_another(row: Mapping[str, Any], now: str | None = None) -> bool:
    """Return whether a live lease is held by somebody other than this caller."""

    owner = row.get("worker_id")
    lease_until = row.get("lease_until")
    if not owner or not lease_until:
        return False
    current = now or utc_now()
    return bool(str(lease_until) >= current)


def _lease_live(row: Mapping[str, Any], now: str | None = None) -> bool:
    lease_until = row.get("lease_until")
    if not lease_until:
        return False
    return bool(str(lease_until) >= (now or utc_now()))


def _request_budget_seconds(client: UnifiedHttpClient) -> float:
    """Upper bound of one candidate request, read from the client's policy."""

    settings = getattr(client, "settings", None)
    if settings is None:
        return 0.0
    attempts = max(1, int(getattr(settings, "max_attempts", 1) or 1))
    per_attempt = max(
        0.0, float(getattr(settings, "connect_timeout", 0.0) or 0.0)
    ) + max(0.0, float(getattr(settings, "read_timeout", 0.0) or 0.0))
    backoff = max(0.0, float(getattr(settings, "backoff_base_seconds", 0.0) or 0.0))
    backoff_max = max(
        0.0, float(getattr(settings, "backoff_max_seconds", 0.0) or 0.0)
    )
    jitter = max(0.0, float(getattr(settings, "jitter_seconds", 0.0) or 0.0))
    sleeps = sum(
        min(backoff_max, backoff * (2 ** (index - 1))) + jitter
        for index in range(2, attempts + 1)
    )
    return per_attempt * attempts + sleeps


def _operation_lease_seconds(client: UnifiedHttpClient, lease_seconds: int) -> int:
    """Lease length that covers one bounded fetch/validate pass (A-R2)."""

    budget = _request_budget_seconds(client) + LEASE_BUDGET_MARGIN_SECONDS
    return max(int(lease_seconds), int(budget) + 1)


def _lease_lost_result(
    pmid: int,
    db_path: str | Path,
    attempts: int,
    failures: list[dict[str, Any]],
    detail: str,
) -> dict[str, Any]:
    """Report a lease loss after the refusing commit already rolled back."""

    return {
        "pmid": pmid,
        "status": _current_status(pmid, db_path),
        "outcome": "lease_lost",
        "attempts": attempts,
        "failures": failures,
        "detail": detail,
    }


def acquire_one(
    pmid: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    raw_root: str | Path = RAW_ROOT,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    client: UnifiedHttpClient | None = None,
    writer: RawCorpusWriter | None = None,
    worker_id: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    refresh: bool = False,
    now: str | None = None,
) -> dict[str, Any]:
    """Acquire one PMID from A4 candidates; no bulk mode is provided.

    The single-article path leases the task through the existing A8 API with a
    temporary worker identity, validates each fetched candidate with A6, and
    falls back to the next candidate while the real failure evidence is kept.
    A task that already has a validated body is skipped unless ``refresh`` is
    requested; an HTTP 200 alone is never an ingestable success.
    """

    pmid = normalize_pmids([pmid])[0]
    task, candidate_rows = _task_and_candidates(pmid, db_path, candidates)
    existing_status = _current_status(pmid, db_path)
    if existing_status in TERMINAL_SKIP_STATUSES:
        return {
            "pmid": pmid,
            "status": existing_status,
            "outcome": "skipped_terminal_status",
            "attempts": 0,
            "failures": [],
        }
    binding = content_binding(task)
    if binding.bound and not refresh:
        # A-R4: status=metadata_only alone must not trigger a re-fetch over a
        # body that is already validated; refresh is an explicit operation.
        return {
            "pmid": pmid,
            "status": existing_status,
            "outcome": "skipped_validated_content",
            "attempts": 0,
            "failures": [],
            "validated_content_sha256": binding.expected_sha256,
        }
    if not candidate_rows:
        return {
            "pmid": pmid,
            "status": existing_status,
            "outcome": "no_fetchable_candidate",
            "attempts": 0,
            "failures": [],
        }

    worker_id = worker_id or f"a5-{uuid.uuid4().hex[:12]}"
    claimed = claim_task(
        worker_id,
        db_path=db_path,
        lease_seconds=lease_seconds,
        pmid=pmid,
        claimable_statuses=ACQUISITION_STATUSES,
        now=now,
    )
    if claimed is None:
        row = _task_row(pmid, db_path)
        if str(row.get("worker_id") or "") == worker_id and _lease_live(row, now):
            # The same worker already owns this task; extend and continue
            # instead of reporting a lease conflict against itself.
            heartbeat_task(
                pmid, worker_id, db_path=db_path, lease_seconds=lease_seconds, now=now
            )
            claimed = row
        else:
            return {
                "pmid": pmid,
                "status": str(row["status"]),
                "outcome": (
                    "skipped_leased" if _held_by_another(row, now) else "not_claimable"
                ),
                "worker_id": row.get("worker_id"),
                "attempts": 0,
                "failures": [],
            }
    task = claimed

    failures: list[dict[str, Any]] = [
        {
            "source": candidate["source"],
            "url": redact_url(str(candidate["url"])),
            "format": candidate["format"],
            "stage": "format_gate",
            "error_class": None,
            "error_detail": (
                f"A6 has no verifier for {candidate['format']}; skipped as a "
                "non-ingestable candidate"
            ),
            "attempts": 0,
        }
        for candidate in candidate_rows
        if candidate["format"] not in INGESTABLE_FORMATS
    ]
    ingestable = [
        candidate
        for candidate in candidate_rows
        if candidate["format"] in INGESTABLE_FORMATS
    ]
    if not ingestable:
        release_task(pmid, worker_id, db_path=db_path, now=now)
        return {
            "pmid": pmid,
            "status": _current_status(pmid, db_path),
            "outcome": "no_ingestable_candidate",
            "attempts": 0,
            "failures": failures,
        }

    client = client or UnifiedHttpClient.from_config(
        load_project_config(), db_path=db_path
    )
    # A-R2: bind *this* attempt's identity. A shared client must not keep the
    # worker identity of whatever task it served before.
    client.bind_worker(worker_id)
    writer = writer or RawCorpusWriter(raw_root)
    operation_lease_seconds = _operation_lease_seconds(client, lease_seconds)

    def renew_lease() -> bool:
        """Renew the lease for the bounded duration of the next operation."""

        return heartbeat_task(
            pmid,
            worker_id,
            db_path=db_path,
            lease_seconds=operation_lease_seconds,
            now=now,
        )

    attempts = 0
    fetched: dict[str, Any] | None = None
    for candidate in ingestable:
        def persist_content(response: HttpResponse, c: Mapping[str, Any] = candidate) -> RawArtifact:
            artifact = writer.write(
                pmid=pmid, task=task, candidate=c, response=response, defer_commit=True
            )
            try:
                _update_task_content(
                    pmid, c, artifact, db_path=db_path, worker_id=worker_id
                )
            except Exception:
                writer.rollback(artifact)
                raise
            try:
                writer.commit(artifact)
            except Exception:
                # SQLite metadata is already durable; keep the new raw artifact.
                pass
            return artifact

        # The request below may take the whole retry budget; renew first so the
        # operation starts from a lease that covers it (A-R2).
        if not renew_lease():
            return _lease_lost_result(
                pmid,
                db_path,
                attempts,
                failures,
                f"PMID {pmid} is no longer leased to {worker_id!r}; "
                "the next candidate request was not started",
            )
        result: FetchResult = client.get(
            pmid=pmid,
            source=str(candidate["source"]),
            url=str(candidate["url"]),
            route=f"fulltext:{candidate['format']}",
            identifier=_pmcid(candidate.get("pmcid") or task.get("pmcid")) or _text(task.get("doi")) or str(pmid),
            success_handler=persist_content,
        )
        attempts += result.attempts
        # The request may have outlived the lease (a long Retry-After, a slow
        # read). Ownership is proved again before validation or the next
        # candidate, so nothing is committed on a lease that has gone.
        if not renew_lease():
            return _lease_lost_result(
                pmid,
                db_path,
                attempts,
                failures,
                f"PMID {pmid} lost its lease to {worker_id!r} during the "
                "candidate request; the content commit was refused",
            )
        if not result.ok:
            failures.append(
                {
                    "source": candidate["source"],
                    "url": redact_url(str(candidate["url"])),
                    "format": candidate["format"],
                    "stage": "http",
                    "error_class": result.error_class,
                    "error_detail": result.error_detail,
                    "attempts": result.attempts,
                }
            )
            continue
        # A-R3: a fetched artifact is validated before it can count as success.
        try:
            validation = validate_task_content(
                pmid, db_path=db_path, raw_root=raw_root, worker_id=worker_id
            )
        except LeaseConflictError as exc:
            return _lease_lost_result(pmid, db_path, attempts, failures, str(exc))
        if validation["file_valid"] == 1 and validation["bibliographic_match"] == 1:
            release_task(pmid, worker_id, db_path=db_path, now=now)
            return {
                "pmid": pmid,
                "status": _current_status(pmid, db_path),
                "outcome": "raw_fetched",
                "source": candidate["source"],
                "format": candidate["format"],
                "path": str(result.artifact.path),
                "source_json": str(result.artifact.source_json),
                "attempts": attempts,
                "failures": failures,
                "content_status": validation["content_status"],
                "validation_method": validation["validation_method"],
                "validated_content_sha256": validation["content_sha256"],
            }
        failures.append(
            {
                "source": candidate["source"],
                "url": redact_url(str(candidate["url"])),
                "format": candidate["format"],
                "stage": "content_validation",
                "error_class": validation["validation_error"],
                "error_detail": validation["validation_detail"],
                "validation_outcome": validation["outcome"],
                "file_valid": validation["file_valid"],
                "bibliographic_match": validation["bibliographic_match"],
                "attempts": result.attempts,
                "artifact_path": str(result.artifact.path),
            }
        )
        fetched = {
            "source": candidate["source"],
            "format": candidate["format"],
            "path": str(result.artifact.path),
            "source_json": str(result.artifact.source_json),
        }
    release_task(pmid, worker_id, db_path=db_path, now=now)
    return {
        "pmid": pmid,
        "status": _current_status(pmid, db_path),
        # A-R3: a raw artifact that A6 rejected is not an ingestable success.
        "outcome": "fetched_unvalidated" if fetched else "failed",
        "attempts": attempts,
        "failures": failures,
        "raw_written": fetched is not None,
        **(fetched or {}),
    }


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A5 fetch one PMID into corpus_raw")
    parser.add_argument("pmid", type=int)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--raw-root", default=str(RAW_ROOT))
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-acquire even when a validated body already exists",
    )
    args = parser.parse_args(argv)
    result = acquire_one(
        args.pmid,
        db_path=args.db,
        raw_root=args.raw_root,
        refresh=args.refresh,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return (
        0
        if result["outcome"]
        in {
            "raw_fetched",
            "no_fetchable_candidate",
            "skipped_terminal_status",
            "skipped_validated_content",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(run_cli())
