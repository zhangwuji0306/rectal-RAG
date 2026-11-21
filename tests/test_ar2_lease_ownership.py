# -*- coding: utf-8 -*-
"""Offline regression tests for A-R2: lease ownership over every task write.

Covers the review's scenarios: two workers competing, the old worker finishing
late, an expired lease being re-claimed, bypass attempts through the generic
status/pointer interfaces, illegal completion states and an unowned
single-article CLI run, plus the legacy downloader's artifact write.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "直肠癌文献爬取" / "scripts"
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(SCRIPTS))

import a1_schema as schema  # noqa: E402
import content_validation as validation  # noqa: E402
import fetch_fulltext as fetch  # noqa: E402
import fulltext_client as client_module  # noqa: E402
import ingest_pmids as ingest  # noqa: E402
import oa_resolver as oa  # noqa: E402
import state_machine as state  # noqa: E402
from lease_batch import (  # noqa: E402
    ACQUISITION_STATUSES,
    LeaseConflictError,
    assert_write_owner,
    claim_task,
    complete_task,
    ensure_lease_schema,
    lease_write_allowed,
    reclaim_expired,
    register_batch,
    release_task,
)

OK_XML = (FIXTURES / "a5_article_ok.xml").read_bytes()
WRONG_XML = (FIXTURES / "a6_article_wrong_identity.xml").read_bytes()
PDF_BODY = b"%PDF-1.4\n" + b"x" * 64 + b"\n%%EOF\n"


def stamp(seconds: int = 0) -> str:
    """Return a real, timezone-aware timestamp shifted by ``seconds``."""

    return (
        datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=seconds)
    ).isoformat()


def candidate(*, source=oa.PMC_AWS, fmt="xml", url=None):
    return {
        "source": source,
        "url": url or f"https://aws.example/PMC5001.{fmt}",
        "format": fmt,
        "pmcid": "PMC5001",
        "license": "CC BY 4.0",
        "version": "publishedVersion",
    }


class AR2LeaseOwnershipTests(unittest.TestCase):
    pmid = 5001

    # ---------------------------------------------------------------- helpers

    def make_db(self, path: Path, *, status: str = "oa_resolved", pmids=(5001,)) -> None:
        schema.migrate_to_v2(path)
        validation.ensure_validation_schema(path)
        ensure_lease_schema(path)
        ingest.ingest_pmids(
            list(pmids), source="ar2-test", batch_id="ar2-test-batch", db_path=path
        )
        conn = sqlite3.connect(path)
        conn.execute(
            """UPDATE tasks SET pmcid='PMC5001', doi='10.1000/a5',
               title='A5 full-text test article', year=2021, authors='Walker, Anna K',
               first_author_family='walker', status=? WHERE pmid=?""",
            (status, self.pmid),
        )
        conn.commit()
        conn.close()

    def task(self, db: Path, pmid: int | None = None) -> dict:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            return dict(
                conn.execute(
                    "SELECT * FROM tasks WHERE pmid=?", (pmid or self.pmid,)
                ).fetchone()
            )
        finally:
            conn.close()

    def attempts(self, db: Path) -> list[dict]:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute("SELECT * FROM fetch_attempts")]
        finally:
            conn.close()

    def attempts_in_order(self, db: Path) -> list[dict]:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            return [
                dict(row)
                for row in conn.execute("SELECT * FROM fetch_attempts ORDER BY rowid")
            ]
        finally:
            conn.close()

    def task_write_state(self, db: Path, pmid: int | None = None) -> dict:
        """The task fields a refused writer must not change."""

        row = self.task(db, pmid)
        return {
            "status": row["status"],
            "content_status": row["content_status"],
            "content_path": row["content_path"],
            "content_sha256": row["content_sha256"],
            "validated_content_path": row["validated_content_path"],
            "validated_content_sha256": row["validated_content_sha256"],
        }

    @staticmethod
    def response(status: int, body: bytes = OK_XML):
        return client_module.HttpResponse(
            status_code=status,
            headers={"Content-Type": "application/xml"},
            body=body,
            url="https://mock.example/response",
        )

    @staticmethod
    def sequence_transport(items):
        pending = list(items)
        calls: list[str] = []

        def transport(url, **_kwargs):
            calls.append(url)
            return pending.pop(0)

        return transport, calls

    def make_client(self, db: Path, transport):
        return client_module.UnifiedHttpClient(
            settings=client_module.RetrySettings(max_attempts=1),
            transport=transport,
            sleeper=lambda _seconds: None,
            random_value=lambda: 0.0,
            db_path=db,
        )

    def acquire(self, db: Path, raw_root: Path, client, **kwargs):
        return fetch.acquire_one(
            self.pmid,
            db_path=db,
            raw_root=raw_root,
            candidates=[candidate()],
            client=client,
            **kwargs,
        )

    # ------------------------------------------------------ competing workers

    def test_lease_gate_separates_import_writes_from_caller_identified_writes(self):
        """A caller identity must never be admitted by an empty stored owner."""

        unowned = {
            "pmid": self.pmid,
            "status": "pending",
            "worker_id": None,
            "lease_until": None,
        }
        # The review's reproduction: a reclaimed task (owner cleared) let
        # caller="expired-worker" through. Only the no-identity import path may
        # write an unowned task.
        self.assertFalse(
            lease_write_allowed(unowned, "expired-worker", now=stamp())
        )
        self.assertFalse(lease_write_allowed(unowned, "worker-a", now=stamp()))
        self.assertTrue(lease_write_allowed(unowned, None, now=stamp()))
        self.assertTrue(lease_write_allowed(unowned, "", now=stamp()))

        owned = {
            "pmid": self.pmid,
            "status": "fetching",
            "worker_id": "worker-a",
            "lease_until": stamp(600),
        }
        self.assertTrue(lease_write_allowed(owned, "worker-a", now=stamp()))
        # An identity-less caller never overrides a stored owner, and another
        # worker never does either.
        self.assertFalse(lease_write_allowed(owned, None, now=stamp()))
        self.assertFalse(lease_write_allowed(owned, "worker-b", now=stamp()))
        # The owner itself is refused once its lease is gone.
        self.assertFalse(
            lease_write_allowed(
                {**owned, "lease_until": stamp(-600)}, "worker-a", now=stamp()
            )
        )
        self.assertFalse(
            lease_write_allowed(
                {**owned, "lease_until": None}, "worker-a", now=stamp()
            )
        )

    def test_caller_without_a_stored_owner_is_refused_by_every_write_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            before = self.task_write_state(db)

            with self.assertRaises(LeaseConflictError):
                state.transition_task_status(
                    self.pmid, "metadata_only", db_path=db, worker_id="expired-worker"
                )
            artifact = fetch.RawCorpusWriter(raw_root).write(
                pmid=self.pmid,
                task={},
                candidate=candidate(),
                response=self.response(200),
            )
            with self.assertRaises(LeaseConflictError):
                fetch._update_task_content(
                    self.pmid,
                    candidate(),
                    artifact,
                    db_path=db,
                    worker_id="expired-worker",
                )
            with self.assertRaises(LeaseConflictError):
                validation.validate_task_content(
                    self.pmid,
                    db_path=db,
                    raw_root=raw_root,
                    worker_id="expired-worker",
                )
            self.assertFalse(
                complete_task(self.pmid, "expired-worker", "done", db_path=db)
            )
            # The attempt row stays factual, but the task state is untouched.
            recorded = state.record_fetch_attempt(
                self.pmid,
                oa.PMC_AWS,
                db_path=db,
                http_status=200,
                outcome="success",
                worker_id="expired-worker",
            )
            self.assertFalse(recorded["status_updated"])
            self.assertEqual(len(self.attempts(db)), 1)
            self.assertEqual(self.task_write_state(db), before)

    def test_no_identity_import_path_still_writes_an_unowned_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)

            # Controlled import/migration: no worker identity at all.
            state.transition_task_status(self.pmid, "metadata_only", db_path=db)
            self.assertEqual(self.task(db)["status"], "metadata_only")

            artifact = fetch.RawCorpusWriter(raw_root).write(
                pmid=self.pmid,
                task={},
                candidate=candidate(),
                response=self.response(200),
            )
            fetch._update_task_content(
                self.pmid, candidate(), artifact, db_path=db, worker_id=None
            )
            row = self.task(db)
            self.assertEqual(row["content_path"], str(artifact.path))
            self.assertEqual(row["content_sha256"], artifact.sha256)

            verdict = validation.validate_task_content(
                self.pmid, db_path=db, raw_root=raw_root
            )
            self.assertEqual(verdict["file_valid"], 1)
            row = self.task(db)
            self.assertEqual(row["file_valid"], 1)
            self.assertEqual(row["bibliographic_match"], 1)
            self.assertEqual(row["content_status"], "fulltext_ready")
            self.assertIsNone(row["worker_id"])

    def test_two_workers_competing_leave_one_owner_and_no_second_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            claimed = claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=900,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
            )
            self.assertEqual(claimed["worker_id"], "worker-a")
            transport, calls = self.sequence_transport([self.response(200)])
            result = self.acquire(
                db, raw_root, self.make_client(db, transport), worker_id="worker-b"
            )
            self.assertEqual(result["outcome"], "skipped_leased")
            self.assertEqual(result["worker_id"], "worker-a")
            self.assertEqual(calls, [])
            self.assertEqual(self.attempts(db), [])
            self.assertFalse((raw_root / f"PMID_{self.pmid}").exists())

    def test_owning_worker_may_continue_its_own_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=900,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
            )
            transport, calls = self.sequence_transport([self.response(200)])
            result = self.acquire(
                db, raw_root, self.make_client(db, transport), worker_id="worker-a"
            )
            self.assertEqual(result["outcome"], "raw_fetched")
            self.assertEqual(len(calls), 1)
            self.assertEqual(self.task(db)["content_status"], "fulltext_ready")

    def test_old_worker_cannot_write_after_the_lease_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=10,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
                now=stamp(),
            )
            self.assertEqual(reclaim_expired(db_path=db, now=stamp(11)), [self.pmid])
            claim_task(
                "worker-b",
                db_path=db,
                lease_seconds=900,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
                now=stamp(11),
            )
            before = self.task(db)

            with self.assertRaises(LeaseConflictError):
                state.transition_task_status(
                    self.pmid, "metadata_only", db_path=db, worker_id="worker-a"
                )
            self.assertFalse(
                complete_task(self.pmid, "worker-a", "done", db_path=db)
            )
            # The attempt row stays factual, but the task status is untouched.
            recorded = state.record_fetch_attempt(
                self.pmid,
                oa.PMC_AWS,
                db_path=db,
                http_status=200,
                outcome="success",
                worker_id="worker-a",
            )
            self.assertFalse(recorded["status_updated"])
            self.assertEqual(len(self.attempts(db)), 1)
            with self.assertRaises(LeaseConflictError):
                validation.validate_task_content(
                    self.pmid, db_path=db, raw_root=raw_root, worker_id="worker-a"
                )
            with self.assertRaises(LeaseConflictError):
                fetch._update_task_content(
                    self.pmid,
                    candidate(),
                    fetch.RawCorpusWriter(raw_root).write(
                        pmid=self.pmid,
                        task={},
                        candidate=candidate(),
                        response=self.response(200),
                    ),
                    db_path=db,
                    worker_id="worker-a",
                )
            after = self.task(db)
            self.assertEqual(after["worker_id"], "worker-b")
            self.assertEqual(after["status"], before["status"])
            self.assertEqual(after["content_sha256"], before["content_sha256"])
            self.assertEqual(before["worker_id"], "worker-b")

    def test_expired_lease_reclaimed_to_an_empty_owner_blocks_the_old_worker(self):
        """Expired → reclaimed → owner cleared → the old worker stays out."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=10,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
                now=stamp(),
            )
            self.assertEqual(reclaim_expired(db_path=db, now=stamp(11)), [self.pmid])
            reclaimed = self.task(db)
            self.assertIsNone(reclaimed["worker_id"])
            self.assertIsNone(reclaimed["lease_until"])
            before = self.task_write_state(db)

            with self.assertRaises(LeaseConflictError):
                state.transition_task_status(
                    self.pmid, "metadata_only", db_path=db, worker_id="worker-a"
                )
            with self.assertRaises(LeaseConflictError):
                assert_write_owner(self.pmid, "worker-a", db_path=db)
            artifact = fetch.RawCorpusWriter(raw_root).write(
                pmid=self.pmid,
                task={},
                candidate=candidate(),
                response=self.response(200),
            )
            with self.assertRaises(LeaseConflictError):
                fetch._update_task_content(
                    self.pmid,
                    candidate(),
                    artifact,
                    db_path=db,
                    worker_id="worker-a",
                )
            with self.assertRaises(LeaseConflictError):
                validation.validate_task_content(
                    self.pmid, db_path=db, raw_root=raw_root, worker_id="worker-a"
                )
            self.assertFalse(complete_task(self.pmid, "worker-a", "done", db_path=db))
            self.assertFalse(release_task(self.pmid, "worker-a", db_path=db))

            after = self.task_write_state(db)
            self.assertEqual(after, before)
            self.assertIsNone(after["content_path"])
            self.assertIsNone(after["content_sha256"])
            self.assertIsNone(after["validated_content_sha256"])
            # The identity-less import path is still available on the reclaimed
            # task; only the caller-identified acquisition write is refused.
            assert_write_owner(self.pmid, None, db_path=db)
            state.transition_task_status(self.pmid, "metadata_only", db_path=db)
            self.assertEqual(self.task(db)["status"], "metadata_only")

    def test_old_worker_returns_after_a_new_worker_completed_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=10,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
                now=stamp(),
            )
            self.assertEqual(reclaim_expired(db_path=db, now=stamp(11)), [self.pmid])
            claim_task(
                "worker-b",
                db_path=db,
                lease_seconds=900,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
                now=stamp(11),
            )
            self.assertTrue(
                complete_task(
                    self.pmid, "worker-b", "done", db_path=db, now=stamp(12)
                )
            )
            finished = self.task(db)
            self.assertEqual(finished["status"], "done")
            self.assertIsNone(finished["worker_id"])
            self.assertIsNone(finished["lease_until"])
            before = self.task_write_state(db)

            with self.assertRaises(LeaseConflictError):
                state.transition_task_status(
                    self.pmid, "retryable_error", db_path=db, worker_id="worker-a"
                )
            artifact = fetch.RawCorpusWriter(raw_root).write(
                pmid=self.pmid,
                task={},
                candidate=candidate(),
                response=self.response(200),
            )
            with self.assertRaises(LeaseConflictError):
                fetch._update_task_content(
                    self.pmid,
                    candidate(),
                    artifact,
                    db_path=db,
                    worker_id="worker-a",
                )
            self.assertFalse(complete_task(self.pmid, "worker-a", "done", db_path=db))

            after = self.task_write_state(db)
            self.assertEqual(after, before)
            self.assertEqual(after["status"], "done")
            self.assertIsNone(after["content_path"])
            self.assertIsNone(after["content_sha256"])

    def test_expired_lease_is_reclaimed_and_reacquired_by_a5(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=10,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
                now=stamp(),
            )
            transport, calls = self.sequence_transport([self.response(200)])
            result = self.acquire(
                db, raw_root, self.make_client(db, transport), worker_id="worker-b", now=stamp(11)
            )
            self.assertEqual(result["outcome"], "raw_fetched")
            self.assertEqual(len(calls), 1)
            row = self.task(db)
            self.assertIsNone(row["worker_id"])
            self.assertIsNone(row["lease_until"])
            self.assertEqual(row["file_valid"], 1)
            self.assertEqual(row["bibliographic_match"], 1)
            self.assertEqual(row["content_status"], "fulltext_ready")

    def test_generic_interfaces_cannot_bypass_an_active_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=900,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
            )
            with self.assertRaises(LeaseConflictError):
                state.transition_task_status(self.pmid, "metadata_only", db_path=db)
            with self.assertRaises(LeaseConflictError):
                state.transition_task_status(
                    self.pmid, "metadata_only", db_path=db, worker_id="worker-b"
                )
            with self.assertRaises(LeaseConflictError):
                assert_write_owner(self.pmid, None, db_path=db)
            assert_write_owner(self.pmid, "worker-a", db_path=db)
            artifact = fetch.RawCorpusWriter(raw_root).write(
                pmid=self.pmid,
                task={},
                candidate=candidate(),
                response=self.response(200),
            )
            with self.assertRaises(LeaseConflictError):
                fetch._update_task_content(
                    self.pmid, candidate(), artifact, db_path=db, worker_id=None
                )
            self.assertIsNone(self.task(db)["content_sha256"])
            # Without an incumbent the same interface still works.
            self.assertTrue(release_task(self.pmid, "worker-a", db_path=db))
            state.transition_task_status(self.pmid, "metadata_only", db_path=db)
            self.assertEqual(self.task(db)["status"], "metadata_only")

    def test_complete_task_rejects_unknown_states_and_illegal_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            self.make_db(db)
            claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=900,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
            )
            with self.assertRaises(ValueError):
                complete_task(self.pmid, "worker-a", "not_a_state", db_path=db)
            with self.assertRaises(state.StateTransitionError):
                # fulltext_ready needs a persisted dual-validated attempt.
                complete_task(self.pmid, "worker-a", "fulltext_ready", db_path=db)
            self.assertTrue(complete_task(self.pmid, "worker-a", "done", db_path=db))
            self.assertEqual(self.task(db)["status"], "done")

            # An archived task is terminal: an incumbent owner still cannot
            # complete it into a non-archived state.
            conn = sqlite3.connect(db)
            conn.execute(
                """UPDATE tasks SET status='archived', worker_id='worker-b',
                   lease_until=?, heartbeat_at=? WHERE pmid=?""",
                (stamp(900), stamp(), self.pmid),
            )
            conn.commit()
            conn.close()
            with self.assertRaises(state.StateTransitionError):
                complete_task(self.pmid, "worker-b", "not_found", db_path=db)

    # --------------------------------------------------------- long operations

    def test_a_long_request_runs_on_a_lease_renewed_for_the_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            short_lease = 5
            claimed = claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=short_lease,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
            )
            claimed_until = datetime.fromisoformat(claimed["lease_until"])
            self.assertLessEqual(
                (claimed_until - datetime.now(timezone.utc)).total_seconds(),
                short_lease + 1,
            )
            # The operation this request stands for lasts far longer than the
            # lease the claim handed out.
            operation_end = datetime.now(timezone.utc) + timedelta(seconds=120)
            observed: dict = {"calls": 0}

            def transport(url, **_kwargs):
                observed["calls"] += 1
                row = self.task(db)
                observed["worker_id"] = row["worker_id"]
                observed["read_at"] = datetime.now(timezone.utc)
                until = datetime.fromisoformat(row["lease_until"])
                observed["lease_until"] = until
                observed["covered"] = until >= operation_end
                return self.response(200)

            client = self.make_client(db, transport)
            result = self.acquire(
                db,
                raw_root,
                client,
                worker_id="worker-a",
                lease_seconds=short_lease,
            )
            self.assertEqual(observed["calls"], 1)
            self.assertEqual(observed["worker_id"], "worker-a")
            # No real wait: the assertion is that the lease observed at request
            # time already covers the whole operation, and that it was sized
            # from the client's bounded request budget instead of the 5 s.
            self.assertTrue(observed["covered"])
            budget_seconds = (
                fetch._request_budget_seconds(client) + fetch.LEASE_BUDGET_MARGIN_SECONDS
            )
            self.assertGreaterEqual(
                (observed["lease_until"] - observed["read_at"]).total_seconds(),
                budget_seconds - 2.0,
            )
            self.assertEqual(result["outcome"], "raw_fetched")
            row = self.task(db)
            self.assertEqual(row["content_status"], "fulltext_ready")
            self.assertEqual(row["file_valid"], 1)
            self.assertEqual(row["bibliographic_match"], 1)
            self.assertIsNone(row["worker_id"])

    def test_a_request_that_outlives_its_lease_never_commits_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=900,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
            )
            before = self.task_write_state(db)

            def transport(url, **_kwargs):
                # The request lasted longer than its own lease: expire it
                # without waiting for real time to pass.
                conn = sqlite3.connect(db)
                conn.execute(
                    "UPDATE tasks SET lease_until=? WHERE pmid=?",
                    (stamp(-1), self.pmid),
                )
                conn.commit()
                conn.close()
                return self.response(200)

            result = self.acquire(
                db, raw_root, self.make_client(db, transport), worker_id="worker-a"
            )
            self.assertEqual(result["outcome"], "lease_lost")
            after = self.task_write_state(db)
            self.assertEqual(after, before)
            self.assertIsNone(after["content_path"])
            self.assertIsNone(after["content_sha256"])
            self.assertFalse((raw_root / f"PMID_{self.pmid}").exists())
            self.assertEqual(
                sorted(item.name for item in raw_root.iterdir()), []
            )
            attempt = self.attempts_in_order(db)[-1]
            self.assertEqual(attempt["error_class"], "storage_error")
            self.assertEqual(attempt["worker_id"], "worker-a")

    def test_a_reclaim_during_a_request_never_commits_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=900,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
            )

            def transport(url, **_kwargs):
                # Another scheduler reclaimed the task mid-request and cleared
                # the owner, exactly as reclaim_expired does.
                conn = sqlite3.connect(db)
                conn.execute(
                    """UPDATE tasks SET status='pending', worker_id=NULL,
                       lease_until=NULL, heartbeat_at=NULL WHERE pmid=?""",
                    (self.pmid,),
                )
                conn.commit()
                conn.close()
                return self.response(200)

            result = self.acquire(
                db, raw_root, self.make_client(db, transport), worker_id="worker-a"
            )
            self.assertEqual(result["outcome"], "lease_lost")
            row = self.task(db)
            self.assertEqual(row["status"], "pending")
            self.assertIsNone(row["content_status"])
            self.assertIsNone(row["content_path"])
            self.assertIsNone(row["content_sha256"])
            self.assertIsNone(row["validated_content_sha256"])
            self.assertFalse((raw_root / f"PMID_{self.pmid}").exists())

    # ------------------------------------------------------ duplicate fetching

    def test_validated_body_is_not_refetched_unless_refresh_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            transport, calls = self.sequence_transport(
                [self.response(200), self.response(200)]
            )
            client = self.make_client(db, transport)
            first = self.acquire(db, raw_root, client)
            self.assertEqual(first["outcome"], "raw_fetched")
            row = self.task(db)
            self.assertEqual(row["status"], "metadata_only")
            self.assertEqual(row["content_status"], "fulltext_ready")
            before = (row["content_sha256"], row["validated_content_sha256"])

            second = self.acquire(db, raw_root, client)
            self.assertEqual(second["outcome"], "skipped_validated_content")
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                (self.task(db)["content_sha256"], self.task(db)["validated_content_sha256"]),
                before,
            )

            third = self.acquire(db, raw_root, client, refresh=True)
            self.assertEqual(third["outcome"], "raw_fetched", msg=third)
            self.assertEqual(len(calls), 2)

    def test_a_shared_client_rebinds_the_worker_identity_per_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            transport, calls = self.sequence_transport(
                [self.response(200), self.response(200)]
            )
            shared = self.make_client(db, transport)
            first = self.acquire(db, raw_root, shared, worker_id="worker-a")
            self.assertEqual(first["outcome"], "raw_fetched")
            self.assertEqual(shared.worker_id, "worker-a")

            second = self.acquire(
                db, raw_root, shared, worker_id="worker-b", refresh=True
            )
            self.assertEqual(second["outcome"], "raw_fetched")
            self.assertEqual(len(calls), 2)
            self.assertEqual(shared.worker_id, "worker-b")
            # Each attempt is recorded under the identity that really held the
            # lease; the previous worker's identity is not carried over.
            self.assertEqual(
                [row["worker_id"] for row in self.attempts_in_order(db)],
                ["worker-a", "worker-b"],
            )

    def test_retry_scheduling_reads_next_retry_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db, status="pending")
            conn = sqlite3.connect(db)
            conn.execute(
                "UPDATE tasks SET next_retry_at=? WHERE pmid=?", (stamp(600), self.pmid)
            )
            conn.commit()
            conn.close()
            self.assertIsNone(claim_task("worker-a", db_path=db, pmid=self.pmid))
            transport, calls = self.sequence_transport([self.response(200)])
            result = self.acquire(db, raw_root, self.make_client(db, transport))
            self.assertEqual(result["outcome"], "not_claimable")
            self.assertEqual(calls, [])

            conn = sqlite3.connect(db)
            conn.execute(
                "UPDATE tasks SET next_retry_at=? WHERE pmid=?", (stamp(-60), self.pmid)
            )
            conn.commit()
            conn.close()
            claimed = claim_task("worker-a", db_path=db, pmid=self.pmid)
            self.assertIsNotNone(claimed)

    # ------------------------------------------------------------- CLI surface

    def test_unowned_single_article_cli_does_not_bypass_a_live_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, raw_root = root / "tasks.sqlite", root / "corpus_raw"
            self.make_db(db)
            conn = sqlite3.connect(db)
            conn.execute(
                "INSERT INTO source_candidates (pmid, source, url, format, version, "
                "license, priority, resolved_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    self.pmid, oa.PMC_AWS, "https://aws.example/PMC5001.xml", "xml",
                    "publishedVersion", "CC BY 4.0", 1, "2026-09-16T00:00:00+00:00",
                ),
            )
            conn.commit()
            conn.close()
            claim_task(
                "worker-a",
                db_path=db,
                lease_seconds=900,
                pmid=self.pmid,
                claimable_statuses=ACQUISITION_STATUSES,
            )
            built: list[str] = []

            def forbidden_client(*_args, **_kwargs):
                built.append("called")
                raise AssertionError("the CLI must not start a request")

            buffer = io.StringIO()
            with mock.patch.object(
                fetch.UnifiedHttpClient, "from_config", side_effect=forbidden_client
            ), contextlib.redirect_stdout(buffer):
                exit_code = fetch.run_cli(
                    [str(self.pmid), "--db", str(db), "--raw-root", str(raw_root)]
                )
            payload = json.loads(buffer.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertEqual(payload["outcome"], "skipped_leased")
            self.assertEqual(payload["worker_id"], "worker-a")
            self.assertEqual(built, [])
            self.assertFalse((raw_root / f"PMID_{self.pmid}").exists())
            self.assertEqual(self.task(db)["worker_id"], "worker-a")

    # ------------------------------------------------------- legacy downloader

    def test_legacy_downloader_refuses_artifact_after_losing_the_lease(self):
        spec = importlib.util.spec_from_file_location(
            "ar2_downloader", SCRIPTS / "03_downloader.py"
        )
        downloader = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(downloader)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, pdf_dir = root / "tasks.sqlite", root / "pdfs"
            self.make_db(db, status="pending")
            claim_task("worker-other", db_path=db, lease_seconds=900, pmid=self.pmid)
            config = {
                "user_agent": "ar2-agent",
                "min_pdf_size": 1,
                "allow_scihub": False,
                "retry": {"max_attempts": 1},
            }
            transport, calls = self.sequence_transport(
                [self.response(200, PDF_BODY)]
            )
            original_from_config = downloader.UnifiedHttpClient.from_config

            def build_client(received_config, **kwargs):
                return original_from_config(
                    received_config,
                    transport=transport,
                    sleeper=lambda _seconds: None,
                    random_value=lambda: 0.0,
                    **kwargs,
                )

            with mock.patch.object(
                downloader, "setup_logger", return_value=_silent_logger()
            ), mock.patch.object(
                downloader.UnifiedHttpClient, "from_config", side_effect=build_client
            ):
                crawler = downloader.Crawler(config, str(db), str(pdf_dir), workers=1)
            try:
                with self.assertRaises(downloader.DownloadError):
                    crawler.try_europepmc({"pmid": self.pmid, "pmc": "PMC5001"})
            finally:
                crawler.conn.close()
            self.assertEqual(len(calls), 1)
            self.assertFalse((pdf_dir / f"PMID_{self.pmid}.pdf").exists())
            self.assertEqual(self.task(db)["worker_id"], "worker-other")

    def test_legacy_downloader_keeps_working_without_an_incumbent_lease(self):
        spec = importlib.util.spec_from_file_location(
            "ar2_downloader_free", SCRIPTS / "03_downloader.py"
        )
        downloader = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(downloader)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, pdf_dir = root / "tasks.sqlite", root / "pdfs"
            self.make_db(db, status="pending")
            config = {
                "user_agent": "ar2-agent",
                "min_pdf_size": 1,
                "allow_scihub": False,
                "retry": {"max_attempts": 1},
            }
            transport, _calls = self.sequence_transport([self.response(200, PDF_BODY)])
            original_from_config = downloader.UnifiedHttpClient.from_config

            def build_client(received_config, **kwargs):
                return original_from_config(
                    received_config,
                    transport=transport,
                    sleeper=lambda _seconds: None,
                    random_value=lambda: 0.0,
                    **kwargs,
                )

            with mock.patch.object(
                downloader, "setup_logger", return_value=_silent_logger()
            ), mock.patch.object(
                downloader.UnifiedHttpClient, "from_config", side_effect=build_client
            ):
                crawler = downloader.Crawler(config, str(db), str(pdf_dir), workers=1)
            try:
                path = crawler.try_europepmc({"pmid": self.pmid, "pmc": "PMC5001"})
            finally:
                crawler.conn.close()
            self.assertTrue(Path(path).exists())


    def test_legacy_downloader_adopts_an_unowned_task_for_the_artifact_write(self):
        spec = importlib.util.spec_from_file_location(
            "ar2_downloader_adopt", SCRIPTS / "03_downloader.py"
        )
        downloader = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(downloader)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, pdf_dir = root / "tasks.sqlite", root / "pdfs"
            self.make_db(db, status="pending")
            config = {
                "user_agent": "ar2-agent",
                "min_pdf_size": 1,
                "allow_scihub": False,
                "retry": {"max_attempts": 1},
            }
            transport, _calls = self.sequence_transport([self.response(200, PDF_BODY)])
            original_from_config = downloader.UnifiedHttpClient.from_config

            def build_client(received_config, **kwargs):
                return original_from_config(
                    received_config,
                    transport=transport,
                    sleeper=lambda _seconds: None,
                    random_value=lambda: 0.0,
                    **kwargs,
                )

            with mock.patch.object(
                downloader, "setup_logger", return_value=_silent_logger()
            ), mock.patch.object(
                downloader.UnifiedHttpClient, "from_config", side_effect=build_client
            ):
                crawler = downloader.Crawler(config, str(db), str(pdf_dir), workers=1)
            try:
                path = crawler.try_europepmc({"pmid": self.pmid, "pmc": "PMC5001"})
            finally:
                crawler.conn.close()
            self.assertTrue(Path(path).exists())
            # The ownerless task was leased by this crawler before the write, so
            # the artifact gate was not loosened to keep the legacy path alive.
            row = self.task(db)
            self.assertEqual(row["worker_id"], crawler.worker_id)
            self.assertIsNotNone(row["lease_until"])


def _silent_logger():
    import logging

    logger = logging.getLogger("ar2-downloader-test")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    return logger


if __name__ == "__main__":
    unittest.main()
