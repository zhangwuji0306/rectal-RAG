# -*- coding: utf-8 -*-
"""A8 task leases and canonical batch membership.

The lease operations use SQLite's write transaction as the claim boundary.
Only an unexpired lease owner may heartbeat or finish a task, and reclaiming
never falls back to updated_at.  batches/batch_members provide a query-based
view over the historical per-run SQLite files while retaining the legacy
files for compatibility.

A-R2 extends the same ownership rule to every canonical task write: a task
status transition, a content-pointer commit, an A6 verdict commit and a legacy
artifact submission are all refused while another worker holds a live lease,
and a caller that presents a worker identity also needs that identity to be the
stored owner of a task whose lease is still unexpired.  Only a caller that
presents no identity at all (import, migration, fixture) may write an unowned
task.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from a1_schema import DEFAULT_DB_PATH, _connect, migrate_to_v2, utc_now
from state_machine import (
    CANONICAL_STATES,
    StateTransitionError,
    can_transition,
    canonical_status,
)


DEFAULT_LEASE_SECONDS = 15 * 60
CLAIMABLE_STATUS = "pending"
RECLAIMABLE_STATUSES = ("fetching", "downloading")
# A5 canonical acquisition starts from any state that is not terminal and not
# already satisfied by a validated body.
ACQUISITION_STATUSES = (
    "pending",
    "metadata_ready",
    "oa_resolved",
    "fetching",
    "metadata_only",
    "retryable_error",
)
# Legacy downloader spellings remain accepted as completion targets (A-R2.3).
LEGACY_COMPLETION_STATUSES = ("done", "failed", "not_found", "downloading")
COMPLETION_STATUSES = frozenset(CANONICAL_STATES) | frozenset(
    LEGACY_COMPLETION_STATUSES
)


class LeaseConflictError(RuntimeError):
    """Raised when an operation would overwrite an active worker lease."""


def _required_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _timestamp(value: str | datetime | None) -> str:
    if value is None:
        return utc_now()
    if isinstance(value, datetime):
        current = value
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    if not isinstance(value, str) or not value.strip():
        raise ValueError("now must be an ISO timestamp, datetime, or None")
    text = value.strip()
    try:
        current = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO timestamp: {value!r}") from exc
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _lease_until(now: str, lease_seconds: int) -> str:
    if isinstance(lease_seconds, bool) or not isinstance(lease_seconds, int):
        raise ValueError("lease_seconds must be a positive integer")
    if lease_seconds <= 0:
        raise ValueError("lease_seconds must be a positive integer")
    current = datetime.fromisoformat(now)
    return (current + timedelta(seconds=lease_seconds)).isoformat()


def lease_write_allowed(
    row: Mapping[str, Any],
    worker_id: str | None,
    *,
    now: str | None = None,
) -> bool:
    """Return whether ``worker_id`` may write to this task right now.

    Two cases are deliberately separated:

    * **No caller identity** (``worker_id is None``). This is the controlled
      import/migration/fixture path. It may write while the task has no
      incumbent owner, and is refused as soon as somebody holds the task.
    * **A caller identity is supplied.** This is an acquisition write and it
      must be the *currently stored* owner of an unexpired lease. A supplied
      caller with an empty stored owner is refused: reclaiming a task clears
      ``worker_id``, and a late worker from the previous ownership must not be
      able to write through the hole that clearing leaves.
    """

    incumbent = row.get("worker_id")
    has_incumbent = incumbent is not None and str(incumbent).strip() != ""
    caller = (worker_id or "").strip()
    if not caller:
        # Import/migration: only an unowned task is writable without identity.
        return not has_incumbent
    if not has_incumbent:
        return False
    if str(incumbent).strip() != caller:
        return False
    lease_until = row.get("lease_until")
    if lease_until is None or not str(lease_until).strip():
        return False
    current = now or utc_now()
    return str(lease_until) >= current


def assert_write_owner_in_connection(
    conn: sqlite3.Connection,
    pmid: int,
    worker_id: str | None,
    *,
    now: str | datetime | None = None,
    operation: str = "task write",
) -> None:
    """Refuse a task write that the caller does not currently own.

    A caller without any identity may only write an unowned task; a caller that
    presents a worker identity must be the stored owner of an unexpired lease.
    """

    row = conn.execute(
        "SELECT pmid, status, worker_id, lease_until FROM tasks WHERE pmid=?",
        (int(pmid),),
    ).fetchone()
    if row is None:
        raise ValueError(f"task PMID {pmid} does not exist")
    current = _timestamp(now)
    if lease_write_allowed(dict(row), worker_id, now=current):
        return
    raise LeaseConflictError(
        f"{operation} for PMID {pmid} refused: stored owner is "
        f"{row['worker_id']!r} with lease_until {row['lease_until']!r}, "
        f"caller={worker_id!r}"
    )


def assert_write_owner(
    pmid: int,
    worker_id: str | None,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    now: str | datetime | None = None,
    operation: str = "task write",
) -> None:
    """Ownership check for callers that write outside a lease transaction."""

    conn = _connect(db_path)
    try:
        assert_write_owner_in_connection(
            conn, pmid, worker_id, now=now, operation=operation
        )
    finally:
        conn.close()


def _ensure_batch_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            query_text TEXT,
            query_hash TEXT,
            searched_at TEXT,
            imported_at TEXT,
            input_sha256 TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            note TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS batch_members (
            batch_id TEXT NOT NULL REFERENCES batches(batch_id),
            pmid INTEGER NOT NULL REFERENCES tasks(pmid),
            original_status TEXT,
            added_at TEXT NOT NULL,
            PRIMARY KEY (batch_id, pmid)
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_batch_members_pmid "
        "ON batch_members(pmid)"
    )


def ensure_lease_schema(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Ensure A1 plus the additive A8 batch tables exist."""

    migrate_to_v2(db_path)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_batch_tables(conn)
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def claim_task(
    worker_id: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    pmid: int | None = None,
    batch_id: str | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    now: str | datetime | None = None,
    claimable_statuses: Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """Atomically claim one task and assign its lease to a worker.

    ``claimable_statuses`` defaults to the queue's ``pending`` state; A5 passes
    the acquisition states so that a single-article run leases the task through
    this same API instead of writing its status directly (A-R2). A task whose
    ``next_retry_at`` is still in the future is never handed out: retry
    scheduling is read from that column rather than reset by hand.
    """

    worker_id = _required_text(worker_id, "worker_id")
    if pmid is not None and (
        isinstance(pmid, bool) or not isinstance(pmid, int) or pmid <= 0
    ):
        raise ValueError("pmid must be a positive integer or None")
    if batch_id is not None:
        batch_id = _required_text(batch_id, "batch_id")
    statuses = tuple(
        dict.fromkeys(
            str(status).strip()
            for status in (
                claimable_statuses if claimable_statuses is not None else (CLAIMABLE_STATUS,)
            )
        )
    )
    if not statuses or any(not status for status in statuses):
        raise ValueError("claimable_statuses must contain non-empty statuses")
    ensure_lease_schema(db_path)
    claimed_at = _timestamp(now)
    lease_until = _lease_until(claimed_at, lease_seconds)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        marks = ",".join("?" for _ in statuses)
        sql = (
            "SELECT t.* FROM tasks AS t "
            f"WHERE t.status IN ({marks}) "
            "AND (t.lease_until IS NULL OR t.lease_until<?) "
            "AND (t.next_retry_at IS NULL OR t.next_retry_at<=?)"
        )
        params: list[Any] = [*statuses, claimed_at, claimed_at]
        if batch_id is not None:
            sql += (
                " AND EXISTS (SELECT 1 FROM batch_members bm "
                "WHERE bm.batch_id=? AND bm.pmid=t.pmid)"
            )
            params.append(batch_id)
        if pmid is not None:
            sql += " AND t.pmid=?"
            params.append(pmid)
        if min_year is not None:
            sql += " AND t.year>=?"
            params.append(min_year)
        if max_year is not None:
            sql += " AND t.year<=?"
            params.append(max_year)
        sql += " ORDER BY t.pmid LIMIT 1"
        row = conn.execute(sql, params).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None

        cursor = conn.execute(
            f"""UPDATE tasks SET status='fetching', worker_id=?, lease_until=?,
                heartbeat_at=?, updated_at=?, attempts=COALESCE(attempts,0)+1
                WHERE pmid=? AND status IN ({marks})
                  AND (lease_until IS NULL OR lease_until<?)
                  AND (next_retry_at IS NULL OR next_retry_at<=?)""",
            (
                worker_id,
                lease_until,
                claimed_at,
                claimed_at,
                row["pmid"],
                *statuses,
                claimed_at,
                claimed_at,
            ),
        )
        if cursor.rowcount != 1:
            conn.execute("ROLLBACK")
            return None
        conn.execute("COMMIT")
        result = dict(row)
        result.update(
            {
                "status": "fetching",
                "worker_id": worker_id,
                "lease_until": lease_until,
                "heartbeat_at": claimed_at,
                "updated_at": claimed_at,
                "attempts": int(row["attempts"] or 0) + 1,
            }
        )
        return result
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def heartbeat_task(
    pmid: int,
    worker_id: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    now: str | datetime | None = None,
) -> bool:
    """Extend a still-valid lease owned by worker_id."""

    worker_id = _required_text(worker_id, "worker_id")
    ensure_lease_schema(db_path)
    heartbeat_at = _timestamp(now)
    lease_until = _lease_until(heartbeat_at, lease_seconds)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """UPDATE tasks SET heartbeat_at=?, lease_until=?, updated_at=?
               WHERE pmid=? AND worker_id=?
                 AND lease_until IS NOT NULL AND lease_until>=?""",
            (heartbeat_at, lease_until, heartbeat_at, pmid, worker_id, heartbeat_at),
        )
        conn.execute("COMMIT")
        return cursor.rowcount == 1
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def release_task(
    pmid: int,
    worker_id: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    now: str | datetime | None = None,
) -> bool:
    """Drop an owned lease without changing the task's derived status.

    A5 releases the lease after its candidate loop; the document status stays
    as A3 derived it from the recorded attempts.
    """

    worker_id = _required_text(worker_id, "worker_id")
    ensure_lease_schema(db_path)
    released_at = _timestamp(now)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """UPDATE tasks SET worker_id=NULL, lease_until=NULL, heartbeat_at=NULL,
                updated_at=?
                WHERE pmid=? AND worker_id=?
                  AND lease_until IS NOT NULL AND lease_until>=?""",
            (released_at, pmid, worker_id, released_at),
        )
        conn.execute("COMMIT")
        return cursor.rowcount == 1
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def reclaim_expired(
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    now: str | datetime | None = None,
    batch_id: str | None = None,
) -> list[int]:
    """Return expired fetching tasks to pending using lease expiry only."""

    if batch_id is not None:
        batch_id = _required_text(batch_id, "batch_id")
    ensure_lease_schema(db_path)
    current = _timestamp(now)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        sql = (
            "SELECT t.pmid FROM tasks AS t "
            "WHERE t.status IN (?, ?) AND t.worker_id IS NOT NULL "
            "AND t.lease_until IS NOT NULL AND t.lease_until<?"
        )
        params: list[Any] = [*RECLAIMABLE_STATUSES, current]
        if batch_id is not None:
            sql += (
                " AND EXISTS (SELECT 1 FROM batch_members bm "
                "WHERE bm.batch_id=? AND bm.pmid=t.pmid)"
            )
            params.append(batch_id)
        pmids = [int(row[0]) for row in conn.execute(sql, params).fetchall()]
        if pmids:
            marks = ",".join("?" for _ in pmids)
            conn.execute(
                f"""UPDATE tasks SET status='pending', worker_id=NULL,
                    lease_until=NULL, heartbeat_at=NULL, updated_at=?
                    WHERE pmid IN ({marks}) AND status IN (?, ?)
                      AND worker_id IS NOT NULL AND lease_until IS NOT NULL
                      AND lease_until<?""",
                [current, *pmids, *RECLAIMABLE_STATUSES, current],
            )
        conn.execute("COMMIT")
        return pmids
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def complete_task(
    pmid: int,
    worker_id: str,
    status: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    attempts: int | None = None,
    route: str | None = None,
    last_error: str | None = None,
    pdf_path: str | None = None,
    license: str | None = None,
    now: str | datetime | None = None,
) -> bool:
    """Finish a task only while the caller still owns an unexpired lease.

    A-R2: the target must be a canonical state or a kept legacy spelling, and
    the transition must be legal for the task's current state. An arbitrary
    string is rejected before any write happens.
    """

    worker_id = _required_text(worker_id, "worker_id")
    if not isinstance(status, str) or status.strip() not in COMPLETION_STATUSES:
        raise ValueError(f"non-canonical completion status: {status!r}")
    status = status.strip()
    ensure_lease_schema(db_path)
    completed_at = _timestamp(now)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        task = conn.execute(
            "SELECT status, worker_id, lease_until FROM tasks WHERE pmid=?", (pmid,)
        ).fetchone()
        if task is None:
            raise ValueError(f"task PMID {pmid} does not exist")
        # Ownership failure stays a quiet False (the caller reports it); only a
        # non-canonical status or an illegal transition is an error.
        if not lease_write_allowed(dict(task), worker_id, now=completed_at):
            conn.execute("ROLLBACK")
            return False
        current = canonical_status(task["status"])
        target = canonical_status(status)
        if not can_transition(current, target):
            raise StateTransitionError(
                f"invalid completion transition: {task['status']} -> {status}"
            )
        assignments = [
            "status=?", "route=?", "last_error=?", "pdf_path=?",
            "updated_at=?", "worker_id=NULL", "lease_until=NULL", "heartbeat_at=NULL",
        ]
        values: list[Any] = [status, route, last_error, pdf_path, completed_at]
        if attempts is not None:
            assignments.insert(1, "attempts=?")
            values.insert(1, attempts)
        if license is not None:
            assignments.insert(2 if attempts is not None else 1, "license=?")
            values.insert(2 if attempts is not None else 1, license)
        values.extend([pmid, worker_id, completed_at])
        cursor = conn.execute(
            f"""UPDATE tasks SET {', '.join(assignments)}
                WHERE pmid=? AND worker_id=?
                  AND lease_until IS NOT NULL AND lease_until>=?""",
            values,
        )
        conn.execute("COMMIT")
        return cursor.rowcount == 1
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def register_batch(
    batch_id: str,
    pmids: Iterable[int],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    source: str = "unknown",
    query_text: str | None = None,
    note: str | None = None,
    original_status: Mapping[int, str] | None = None,
    created_at: str | datetime | None = None,
) -> dict[str, int | str]:
    """Register immutable batch membership without deleting history."""

    batch_id = _required_text(batch_id, "batch_id")
    source = _required_text(source, "source")
    values = list(dict.fromkeys(pmids))
    if not values:
        raise ValueError("pmids must not be empty")
    if any(
        isinstance(pmid, bool) or not isinstance(pmid, int) or pmid <= 0
        for pmid in values
    ):
        raise ValueError("pmids must contain positive integers")
    ensure_lease_schema(db_path)
    created = _timestamp(created_at)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            "SELECT source, query_text, note FROM batches WHERE batch_id=?",
            (batch_id,),
        ).fetchone()
        if existing is None:
            conn.execute(
                """INSERT INTO batches
                    (batch_id, source, query_text, created_at, note)
                    VALUES (?, ?, ?, ?, ?)""",
                (batch_id, source, query_text, created, note),
            )
        elif (existing["source"], existing["query_text"], existing["note"]) != (
            source, query_text, note
        ):
            raise ValueError(f"batch_id already exists with different provenance: {batch_id}")

        existing_members = {
            int(row[0])
            for row in conn.execute(
                "SELECT pmid FROM batch_members WHERE batch_id=?", (batch_id,)
            ).fetchall()
        }
        if existing_members and existing_members != set(values):
            raise ValueError(
                f"batch_id already exists with different membership: {batch_id}"
            )

        added = 0
        for pmid in values:
            task = conn.execute(
                "SELECT status FROM tasks WHERE pmid=?", (pmid,)
            ).fetchone()
            if task is None:
                raise ValueError(f"task PMID {pmid} does not exist")
            cursor = conn.execute(
                """INSERT OR IGNORE INTO batch_members
                    (batch_id, pmid, original_status, added_at)
                    VALUES (?, ?, ?, ?)""",
                (
                    batch_id,
                    pmid,
                    (original_status or {}).get(pmid, task["status"]),
                    created,
                ),
            )
            added += cursor.rowcount
        conn.execute("COMMIT")
        return {"batch_id": batch_id, "members": len(values), "added": added}
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def query_batch(
    batch_id: str,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return current task rows for one canonical batch membership."""

    batch_id = _required_text(batch_id, "batch_id")
    ensure_lease_schema(db_path)
    conn = _connect(db_path)
    try:
        if conn.execute(
            "SELECT 1 FROM batches WHERE batch_id=?", (batch_id,)
        ).fetchone() is None:
            raise ValueError(f"batch not found: {batch_id}")
        rows = conn.execute(
            """SELECT bm.batch_id, bm.pmid, bm.original_status, bm.added_at,
                       t.*
                FROM batch_members AS bm
                JOIN tasks AS t ON t.pmid=bm.pmid
                WHERE bm.batch_id=?
                ORDER BY bm.pmid""",
            (batch_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def sync_batch_rows(
    batch_id: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    source: str = "legacy-batch",
    now: str | datetime | None = None,
) -> dict[str, int | str]:
    """Apply one legacy batch snapshot to canonical tasks and close the batch."""

    batch_id = _required_text(batch_id, "batch_id")
    source = _required_text(source, "source")
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("batch rows must not be empty")
    pmids = [int(row["pmid"]) for row in materialized]
    ensure_lease_schema(db_path)
    current = _timestamp(now)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute(
            "SELECT 1 FROM batches WHERE batch_id=?", (batch_id,)
        ).fetchone() is None:
            conn.execute(
                "INSERT INTO batches(batch_id,source,created_at,note) VALUES(?,?,?,?)",
                (batch_id, source, current, "registered during legacy batch sync"),
            )
            for pmid in pmids:
                task = conn.execute(
                    "SELECT status FROM tasks WHERE pmid=?", (pmid,)
                ).fetchone()
                if task is None:
                    raise ValueError(f"task PMID {pmid} does not exist")
                conn.execute(
                    """INSERT OR IGNORE INTO batch_members
                        (batch_id,pmid,original_status,added_at)
                        VALUES(?,?,?,?)""",
                    (batch_id, pmid, task["status"], current),
                )
        expected = {
            int(row[0])
            for row in conn.execute(
                "SELECT pmid FROM batch_members WHERE batch_id=?", (batch_id,)
            ).fetchall()
        }
        actual = set(pmids)
        if expected != actual:
            raise ValueError(
                f"batch membership mismatch for {batch_id}: "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )
        updated = 0
        for row in materialized:
            task_values = (
                row.get("status"),
                row.get("attempts"),
                row.get("route"),
                row.get("last_error"),
                row.get("pdf_path"),
                row.get("license") or "unverified",
                row.get("updated_at") or current,
                row["pmid"],
            )
            cursor = conn.execute(
                """UPDATE tasks SET status=?, attempts=?, route=?, last_error=?,
                    pdf_path=?, license=?, updated_at=?, worker_id=NULL,
                    lease_until=NULL, heartbeat_at=NULL
                    WHERE pmid=?
                      AND (lease_until IS NULL OR lease_until<?)""",
                (*task_values, current),
            )
            if cursor.rowcount != 1:
                raise LeaseConflictError(
                    f"cannot sync PMID {row['pmid']} while its lease is active"
                )
            updated += cursor.rowcount
        conn.execute(
            "UPDATE batches SET completed_at=? WHERE batch_id=?",
            (current, batch_id),
        )
        conn.execute("COMMIT")
        return {"batch_id": batch_id, "rows": len(materialized), "updated": updated}
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()

