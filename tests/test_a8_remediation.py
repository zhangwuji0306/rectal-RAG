"""Regression tests for the A8 remediation boundaries."""

import importlib.util
import os
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
from lease_batch import ensure_lease_schema, register_batch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class A8RemediationTests(unittest.TestCase):
    def make_canonical(self, path, pmids=(9011, 9012)):
        schema.migrate_to_v2(path)
        ingest.ingest_pmids(
            list(pmids), source="a8-remediation", batch_id="discovery-remediation",
            db_path=path,
        )
        conn = sqlite3.connect(path)
        conn.executemany(
            "UPDATE tasks SET pmc=?, year=2024, title=?, title_norm=?, status='pending' WHERE pmid=?",
            [(f"PMC{pmid}", f"Article {pmid}", f"article {pmid}", pmid) for pmid in pmids],
        )
        conn.commit()
        conn.close()

    def test_additive_a8_migration_preserves_tasks_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tasks.sqlite"
            self.make_canonical(db, (9011,))
            conn = sqlite3.connect(db)
            conn.execute(
                "INSERT INTO fetch_attempts (attempt_id, pmid, source, started_at, outcome) "
                "VALUES (?,?,?,?,?)", ("attempt-9011", 9011, "fixture", "t0", "success")
            )
            conn.execute(
                "CREATE TABLE run_history (run TEXT, pmid INTEGER, status TEXT)"
            )
            conn.execute(
                "INSERT INTO run_history VALUES (?,?,?)", ("fixture", 9011, "done")
            )
            conn.commit()
            before = {
                "tasks": conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
                "attempts": conn.execute("SELECT COUNT(*) FROM fetch_attempts").fetchone()[0],
                "history": conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0],
            }
            conn.close()

            ensure_lease_schema(db)

            conn = sqlite3.connect(db)
            try:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], before["tasks"])
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM fetch_attempts").fetchone()[0], before["attempts"])
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0], before["history"])
                self.assertIsNotNone(conn.execute(
                    "SELECT name FROM sqlite_master WHERE name='batches'"
                ).fetchone())
                self.assertIsNotNone(conn.execute(
                    "SELECT name FROM sqlite_master WHERE name='batch_members'"
                ).fetchone())
            finally:
                conn.close()

    def test_merge_preserves_a8_and_a1_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            self.make_canonical(db, (9011,))
            register_batch(
                "fixture_batch", [9011], db_path=db, source="fixture",
                note="fixture batch", original_status={9011: "pending"},
            )
            conn = sqlite3.connect(db)
            conn.execute(
                "INSERT INTO fetch_attempts (attempt_id, pmid, source, started_at, outcome) "
                "VALUES (?,?,?,?,?)", ("attempt-merge", 9011, "fixture", "t0", "success")
            )
            conn.execute(
                "CREATE TABLE run_history (run TEXT, pmid INTEGER, doi TEXT, pmc TEXT, year INTEGER, "
                "title TEXT, title_norm TEXT, status TEXT, attempts INTEGER, route TEXT, "
                "last_error TEXT, pdf_path TEXT, license TEXT, updated_at TEXT)"
            )
            conn.execute(
                "INSERT INTO run_history VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("fixture", 9011, None, "PMC9011", 2024, "Article 9011", "article 9011",
                 "done", 1, "fixture", None, None, "CC-BY", "2026-09-20T00:00:00+00:00"),
            )
            conn.commit()
            conn.close()
            (root / "reports").mkdir()
            (root / "pdfs_merged").mkdir()

            merge = load_module("merge_a8_remediation", SCRIPTS / "11_merge_sqlite.py")
            merge.ROOT = str(root)
            merge.MERGED_DIR = str(root / "pdfs_merged")
            merge.BATCHES = []
            merge.main()

            conn = sqlite3.connect(db)
            try:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT status FROM tasks WHERE pmid=9011").fetchone()[0], "done")
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM run_history").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM fetch_attempts").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0], 1)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM batch_members").fetchone()[0], 1)
                self.assertEqual(
                    conn.execute("SELECT batch_id, pmid, original_status FROM batch_members").fetchone(),
                    ("fixture_batch", 9011, "pending"),
                )
            finally:
                conn.close()
            self.assertTrue(list(root.glob("tasks.sqlite.premerge-*")))

    def test_batch_builder_failure_keeps_old_snapshot_and_manifest(self):
        batch = load_module("batch_builder_a8_remediation", SCRIPTS / "12_build_pmc_batches.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            self.make_canonical(db, (9011, 9012))
            batch.build_batches(str(db), str(root), batch_size=1, max_batches=1)
            old_db = root / "tasks_pmc_b1.sqlite"
            manifest = root / "reports" / "pmc_batches.csv"
            old_db_bytes = old_db.read_bytes()
            manifest_bytes = manifest.read_bytes()

            conn = sqlite3.connect(db)
            conn.execute("DELETE FROM batch_members WHERE batch_id='pmc_b1'")
            conn.commit()
            conn.close()

            with self.assertRaises(ValueError):
                batch.build_batches(str(db), str(root), batch_size=1, max_batches=1)
            self.assertEqual(old_db.read_bytes(), old_db_bytes)
            self.assertEqual(manifest.read_bytes(), manifest_bytes)

    def test_batch_builder_success_registers_after_staging(self):
        batch = load_module("batch_builder_a8_success", SCRIPTS / "12_build_pmc_batches.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = root / "tasks.sqlite"
            self.make_canonical(db, (9011, 9012))
            result = batch.build_batches(str(db), str(root), batch_size=2)
            self.assertEqual(result["batches"], 1)
            self.assertTrue((root / "tasks_pmc_b1.sqlite").exists())
            conn = sqlite3.connect(db)
            try:
                self.assertEqual(conn.execute(
                    "SELECT COUNT(*) FROM batch_members WHERE batch_id='pmc_b1'"
                ).fetchone()[0], 2)
            finally:
                conn.close()


if __name__ == "__main__":
    unittest.main()
