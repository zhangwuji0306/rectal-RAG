"""Offline regression tests for A8 leases and canonical batch queries."""

import concurrent.futures
import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "直肠癌文献爬取" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import a1_schema as schema
import ingest_pmids as ingest
import state_machine as state
from lease_batch import (
    claim_task,
    complete_task,
    heartbeat_task,
    query_batch,
    reclaim_expired,
    register_batch,
    sync_batch_rows,
)


class A8LeaseBatchTests(unittest.TestCase):
    def make_db(self, path, pmids=(8001,)):
        schema.migrate_to_v2(path)
        ingest.ingest_pmids(
            list(pmids), source="a8-test", batch_id=f"discovery-{pmids[0]}",
            db_path=path,
        )

    def read_task(self, path, pmid):
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            return dict(conn.execute(
                "SELECT * FROM tasks WHERE pmid=?", (pmid,)
            ).fetchone())
        finally:
            conn.close()

    def test_concurrent_claim_has_one_valid_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            self.make_db(db)
            args = dict(db_path=db, lease_seconds=30, now="2026-09-19T00:00:00+00:00")

            def claim(worker):
                return claim_task(worker, **args)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(claim, ("worker-a", "worker-b")))

            claims = [result for result in results if result is not None]
            self.assertEqual(len(claims), 1)
            row = self.read_task(db, 8001)
            self.assertEqual(row["status"], "fetching")
            self.assertEqual(row["worker_id"], claims[0]["worker_id"])
            self.assertEqual(row["lease_until"], "2026-09-19T00:00:30+00:00")
            self.assertEqual(row["heartbeat_at"], "2026-09-19T00:00:00+00:00")

    def test_unexpired_lease_cannot_be_claimed_or_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            self.make_db(db)
            claimed = claim_task(
                "worker-a", db_path=db, lease_seconds=30,
                now="2026-09-19T00:00:00+00:00",
            )
            self.assertIsNotNone(claimed)
            self.assertTrue(heartbeat_task(
                8001, "worker-a", db_path=db, lease_seconds=30,
                now="2026-09-19T00:00:10+00:00",
            ))
            self.assertEqual(
                reclaim_expired(db_path=db, now="2026-09-19T00:00:35+00:00"),
                [],
            )
            self.assertIsNone(claim_task(
                "worker-b", db_path=db, lease_seconds=30,
                now="2026-09-19T00:00:35+00:00",
            ))
            row = self.read_task(db, 8001)
            self.assertEqual(row["worker_id"], "worker-a")
            self.assertEqual(row["lease_until"], "2026-09-19T00:00:40+00:00")

    def test_expired_lease_can_be_reclaimed_and_stale_worker_cannot_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            self.make_db(db)
            claim_task(
                "worker-a", db_path=db, lease_seconds=10,
                now="2026-09-19T00:00:00+00:00",
            )
            self.assertEqual(
                reclaim_expired(db_path=db, now="2026-09-19T00:00:11+00:00"),
                [8001],
            )
            replacement = claim_task(
                "worker-b", db_path=db, lease_seconds=30,
                now="2026-09-19T00:00:11+00:00",
            )
            self.assertEqual(replacement["worker_id"], "worker-b")
            self.assertFalse(complete_task(
                8001, "worker-a", "done", db_path=db,
                now="2026-09-19T00:00:11+00:00",
            ))
            self.assertTrue(complete_task(
                8001, "worker-b", "done", db_path=db,
                attempts=replacement["attempts"], now="2026-09-19T00:00:12+00:00",
            ))
            row = self.read_task(db, 8001)
            self.assertEqual(row["status"], "done")
            self.assertIsNone(row["worker_id"])
            self.assertIsNone(row["lease_until"])
            self.assertIsNone(row["heartbeat_at"])

    def test_reclaim_does_not_use_old_updated_at_without_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            self.make_db(db)
            conn = sqlite3.connect(db)
            conn.execute(
                """UPDATE tasks SET status='fetching', worker_id='worker-a',
                   lease_until=NULL, heartbeat_at=NULL, updated_at=?
                   WHERE pmid=8001""",
                ("2020-01-01T00:00:00+00:00",),
            )
            conn.commit()
            conn.close()
            self.assertEqual(
                reclaim_expired(db_path=db, now="2026-09-19T00:00:00+00:00"),
                [],
            )
            self.assertEqual(self.read_task(db, 8001)["status"], "fetching")

    def test_batch_query_preserves_membership_and_current_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            self.make_db(db, pmids=(8101, 8102))
            result = register_batch(
                "pmc_b8", [8101, 8102], db_path=db, source="pmc",
                note="A8 fixture",
                original_status={8101: "pending", 8102: "metadata_ready"},
                created_at="2026-09-19T00:00:00+00:00",
            )
            self.assertEqual(result["added"], 2)
            self.assertEqual(
                register_batch(
                    "pmc_b8", [8101, 8102], db_path=db, source="pmc",
                    note="A8 fixture",
                    original_status={8101: "pending", 8102: "metadata_ready"},
                    created_at="2026-09-19T00:00:00+00:00",
                )["added"],
                0,
            )
            rows = query_batch("pmc_b8", db_path=db)
            self.assertEqual([row["pmid"] for row in rows], [8101, 8102])
            self.assertEqual(
                [row["original_status"] for row in rows],
                ["pending", "metadata_ready"],
            )
            synced = sync_batch_rows(
                "pmc_b8",
                [
                    {"pmid": 8101, "status": "done", "attempts": 1,
                     "route": "test", "last_error": None, "pdf_path": "a.pdf",
                     "license": "CC-BY", "updated_at": "2026-09-19T00:01:00+00:00"},
                    {"pmid": 8102, "status": "failed", "attempts": 2,
                     "route": "test", "last_error": "offline",
                     "pdf_path": None, "license": "unverified",
                     "updated_at": "2026-09-19T00:01:00+00:00"},
                ],
                db_path=db,
                now="2026-09-19T00:01:00+00:00",
            )
            self.assertEqual(synced["updated"], 2)
            rows = query_batch("pmc_b8", db_path=db)
            self.assertEqual([row["status"] for row in rows], ["done", "failed"])
            conn = sqlite3.connect(db)
            try:
                self.assertEqual(
                    conn.execute("SELECT completed_at FROM batches WHERE batch_id='pmc_b8'")
                    .fetchone()[0],
                    "2026-09-19T00:01:00+00:00",
                )
            finally:
                conn.close()

    def test_a3_status_update_keeps_active_lease_from_second_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            self.make_db(db)
            claim_task(
                "worker-a", db_path=db, lease_seconds=60,
                now="2026-09-19T00:00:00+00:00",
            )
            # The incumbent owner may write its own task state (A-R2); a caller
            # without that identity is refused instead.
            state.transition_task_status(
                8001, "pending", db_path=db, worker_id="worker-a",
                updated_at="2026-09-19T00:00:05+00:00",
            )
            self.assertIsNone(claim_task(
                "worker-b", db_path=db, lease_seconds=60,
                now="2026-09-19T00:00:05+00:00",
            ))
            row = self.read_task(db, 8001)
            self.assertEqual(row["status"], "pending")
            self.assertEqual(row["worker_id"], "worker-a")
            self.assertEqual(row["lease_until"], "2026-09-19T00:01:00+00:00")

    def test_reclaim_does_not_reset_non_active_history_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            self.make_db(db)
            claim_task(
                "worker-a", db_path=db, lease_seconds=10,
                now="2026-09-19T00:00:00+00:00",
            )
            state.transition_task_status(
                8001, "metadata_only", db_path=db, worker_id="worker-a",
                updated_at="2026-09-19T00:00:01+00:00",
            )
            self.assertEqual(reclaim_expired(
                db_path=db, now="2026-09-19T00:00:11+00:00"
            ), [])
            row = self.read_task(db, 8001)
            self.assertEqual(row["status"], "metadata_only")
            self.assertEqual(row["worker_id"], "worker-a")

    def test_batch_sync_rejects_active_lease_without_clearing_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            self.make_db(db)
            register_batch("pmc_active", [8001], db_path=db, source="pmc")
            claim_task(
                "worker-a", db_path=db, lease_seconds=60,
                now="2026-09-19T00:00:00+00:00",
            )
            with self.assertRaises(RuntimeError):
                sync_batch_rows(
                    "pmc_active",
                    [{"pmid": 8001, "status": "done", "attempts": 1}],
                    db_path=db, now="2026-09-19T00:00:05+00:00",
                )
            row = self.read_task(db, 8001)
            self.assertEqual(row["worker_id"], "worker-a")
            self.assertEqual(row["status"], "fetching")

if __name__ == "__main__":
    unittest.main()

