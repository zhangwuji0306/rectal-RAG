"""Fault-injection tests for the publication boundary and the run-state report.

These tests reproduce the real exit windows the audit found (the production
directory moved, the staging directory promoted, the manifest replaced, all
with the step journal not yet written) by raising a ``BaseException`` from
inside the running publication, exactly as a killed process would.  Recovery is
then judged by the files and their identities, never by the recorded steps or a
returned status string.

Everything runs in temporary directories; the production index is only read.
"""

import argparse
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "index"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b3_fakes  # noqa: E402
import index_common  # noqa: E402
import publish_index  # noqa: E402
import rag_core  # noqa: E402
import rebuild_b3  # noqa: E402
from file_lock import FileLock, writer_lock_path  # noqa: E402
from test_b3_writer_boundary import make_manifest, write_marker  # noqa: E402


class ProcessExit(BaseException):
    """Stands in for the process dying: deliberately not an Exception."""


class PublishFixture:
    """A staging pair and a production pair inside one throwaway root."""

    def __init__(self, root, tag=""):
        self.root = Path(root)
        self.tag = tag
        prefix = f"{tag}-" if tag else ""
        self.staging_db, self.staging_manifest, staging = make_manifest(
            self.root / f"{prefix}staging", {"new.md": 3}
        )
        b3_fakes.release_chroma(self.staging_db)
        write_marker(self.staging_manifest, staging, self.staging_db)
        self.production_db, self.production_manifest, _ = make_manifest(
            self.root / f"{prefix}production", {"old.md": 2}
        )
        b3_fakes.release_chroma(self.production_db)
        self.journal = self.root / f"{prefix}journal.json"
        self.backups = self.root / f"{prefix}backups"

    def publish(self, **kwargs):
        result = publish_index.publish(
            self.staging_db, self.staging_manifest,
            self.production_db, self.production_manifest,
            journal_path=self.journal, manifest_backup_dir=self.backups,
            **kwargs,
        )
        return result


class PublishRecoveryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        b3_fakes.release_chroma(self.root)
        try:
            self._tmp.cleanup()
        except PermissionError:  # pragma: no cover - Windows handle latency
            pass

    @staticmethod
    def manifest_files(path):
        return set(json.loads(Path(path).read_text(encoding="utf-8"))["files"])

    def assert_old_pair(self, ctx):
        """The complete previous production pair is in place and queryable."""
        self.assertTrue(ctx.production_db.is_dir())
        self.assertEqual(self.manifest_files(ctx.production_manifest), {"old.md"})
        summary = rag_core.index_preflight(
            db_dir=ctx.production_db, manifest_path=ctx.production_manifest
        )
        self.assertEqual(summary["collection_count"], 2)
        b3_fakes.release_chroma(ctx.production_db)

    def assert_new_pair(self, ctx):
        self.assertTrue(ctx.production_db.is_dir())
        self.assertEqual(self.manifest_files(ctx.production_manifest), {"new.md"})
        summary = rag_core.index_preflight(
            db_dir=ctx.production_db, manifest_path=ctx.production_manifest
        )
        self.assertEqual(summary["collection_count"], 3)
        b3_fakes.release_chroma(ctx.production_db)

    @staticmethod
    def _die_after(action, match):
        fired = {"done": False}

        def wrapper(*args, **kwargs):
            result = action(*args, **kwargs)
            if not fired["done"] and match(args):
                fired["done"] = True
                raise ProcessExit()
            return result

        return wrapper, fired

    @staticmethod
    def _die_before(action, match):
        fired = {"done": False}

        def wrapper(*args, **kwargs):
            if not fired["done"] and match(args):
                fired["done"] = True
                raise ProcessExit()
            return action(*args, **kwargs)

        return wrapper, fired

    def interrupt(self, ctx, point):
        """Run a publication that dies at one named cross-file step."""
        real_rename = Path.rename
        moved = lambda a: Path(a[1]).name.startswith(  # noqa: E731
            ctx.production_db.name + ".previous"
        )
        promoted = lambda a: Path(a[1]) == ctx.production_db  # noqa: E731
        replaced = lambda a: Path(a[1]) == ctx.production_manifest  # noqa: E731
        backed_up = lambda a: Path(a[1]).parent == ctx.backups  # noqa: E731

        if point == "after-production-db-moved":
            wrapper, _ = self._die_after(real_rename, moved)
            patches = [mock.patch.object(Path, "rename", wrapper)]
        elif point == "before-production-db-moved":
            wrapper, _ = self._die_before(real_rename, moved)
            patches = [mock.patch.object(Path, "rename", wrapper)]
        elif point == "after-staging-db-promoted":
            wrapper, _ = self._die_after(real_rename, promoted)
            patches = [mock.patch.object(Path, "rename", wrapper)]
        elif point == "after-manifest-replaced":
            wrapper, _ = self._die_after(publish_index.os.replace, replaced)
            patches = [mock.patch.object(publish_index.os, "replace", wrapper)]
        elif point == "during-manifest-backup":
            wrapper, _ = self._die_after(publish_index.shutil.copy2, backed_up)
            patches = [mock.patch.object(publish_index.shutil, "copy2", wrapper)]
        else:  # pragma: no cover - guard
            raise AssertionError(f"unknown interruption point {point}")

        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            with self.assertRaises(ProcessExit):
                ctx.publish()

    # -- the reproduced windows -------------------------------------------

    def test_exit_after_the_production_directory_moved_restores_the_old_pair(self):
        ctx = PublishFixture(self.root)
        self.interrupt(ctx, "after-production-db-moved")

        journal = publish_index._journal_state(ctx.journal)
        self.assertEqual(journal["state"], "in-progress",
                         "a killed process must leave a recoverable journal")
        self.assertNotIn("production-db-moved", journal["steps"],
                         "the step log misses the real move - exactly the audit window")

        recovered = publish_index.recover(ctx.journal)

        self.assertEqual(recovered["state"], "rolled-back")
        self.assertEqual(recovered["result_pair"], "old")
        self.assert_old_pair(ctx)
        self.assertTrue(ctx.staging_db.is_dir())

    def test_exit_after_the_manifest_replace_yields_the_complete_new_pair(self):
        ctx = PublishFixture(self.root)
        self.interrupt(ctx, "after-manifest-replaced")

        journal = publish_index._journal_state(ctx.journal)
        self.assertNotIn("manifest-replaced", journal["steps"],
                         "the pair is already new; only the bookkeeping is missing")

        recovered = publish_index.recover(ctx.journal)

        self.assertEqual(recovered["state"], "completed")
        self.assertEqual(recovered["result_pair"], "new")
        self.assert_new_pair(ctx)

    def test_exit_before_the_move_rolls_back_without_touching_production(self):
        ctx = PublishFixture(self.root)
        self.interrupt(ctx, "before-production-db-moved")
        recovered = publish_index.recover(ctx.journal)
        self.assertEqual(recovered["state"], "rolled-back")
        self.assert_old_pair(ctx)

    def test_exit_after_the_promotion_returns_staging_and_production(self):
        ctx = PublishFixture(self.root)
        self.interrupt(ctx, "after-staging-db-promoted")
        self.assertFalse(ctx.staging_db.is_dir(), "the directory really moved")

        recovered = publish_index.recover(ctx.journal)

        self.assertEqual(recovered["state"], "rolled-back")
        self.assertEqual(recovered["result_pair"], "old")
        self.assert_old_pair(ctx)
        self.assertTrue(ctx.staging_db.is_dir(), "staging directory restored")

    def test_exit_during_the_backup_copy_leaves_the_old_pair(self):
        ctx = PublishFixture(self.root)
        self.interrupt(ctx, "during-manifest-backup")
        recovered = publish_index.recover(ctx.journal)
        self.assertEqual(recovered["state"], "rolled-back")
        self.assert_old_pair(ctx)

    # -- idempotence and re-interruption ----------------------------------

    def test_repeated_recovery_is_idempotent(self):
        for index, point in enumerate((
            "after-production-db-moved", "after-staging-db-promoted",
            "after-manifest-replaced",
        )):
            with self.subTest(point=point):
                ctx = PublishFixture(self.root, tag=f"idem{index}")
                self.interrupt(ctx, point)

                first = publish_index.recover(ctx.journal)
                second = publish_index.recover(ctx.journal)
                third = publish_index.recover(ctx.journal)

                self.assertEqual(first["state"], second["state"])
                self.assertEqual(second["state"], third["state"])
                self.assertEqual(first["result_pair"], second["result_pair"])
                if first["state"] == "completed":
                    self.assert_new_pair(ctx)
                else:
                    self.assert_old_pair(ctx)
                    self.assertTrue(ctx.staging_db.is_dir())

    def test_recovery_interrupted_twice_still_ends_paired(self):
        ctx = PublishFixture(self.root)
        self.interrupt(ctx, "after-production-db-moved")

        real_rename = Path.rename
        wrapper, fired = self._die_after(
            real_rename,
            lambda a: Path(a[0]).name.startswith(ctx.production_db.name + ".previous"),
        )
        with mock.patch.object(Path, "rename", wrapper):
            with self.assertRaises(ProcessExit):
                publish_index.recover(ctx.journal)
        self.assertTrue(fired["done"], "the recovery rename must really have happened")

        journal = publish_index._journal_state(ctx.journal)
        self.assertEqual(journal["state"], "in-progress",
                         "an interrupted recovery must stay retryable")

        recovered = publish_index.recover(ctx.journal)

        self.assertEqual(recovered["state"], "rolled-back")
        self.assert_old_pair(ctx)
        self.assertEqual(
            sorted(path.name for path in ctx.production_db.parent.glob(
                f"{ctx.production_db.name}.previous-*")),
            [],
            "no half-moved rollback directory may be left behind",
        )

    def test_intent_without_action_rolls_back_cleanly(self):
        """A journal that only records intents must not be trusted as done."""
        ctx = PublishFixture(self.root)
        index_common.write_json_atomic(ctx.journal, {
            "schema_version": 1,
            "state": "in-progress",
            "staging_db": str(ctx.staging_db),
            "staging_manifest": str(ctx.staging_manifest),
            "production_db": str(ctx.production_db),
            "production_manifest": str(ctx.production_manifest),
            "previous_db": str(ctx.production_db.parent /
                               f"{ctx.production_db.name}.previous-x"),
            "previous_manifest": str(ctx.backups / "manifest-pre-publish-x.json"),
            "steps": [],
            "intents": ["production-db-moved", "staging-db-promoted",
                        "manifest-replaced"],
        })
        recovered = publish_index.recover(ctx.journal)
        self.assertEqual(recovered["state"], "rolled-back")
        self.assert_old_pair(ctx)

    def test_unpairable_state_is_reported_instead_of_claimed(self):
        """A journal whose two files cannot be paired must not read as done."""
        ctx = PublishFixture(self.root)
        ctx.publish()
        index_common.write_json_atomic(ctx.journal, {
            "schema_version": 1,
            "state": "in-progress",
            "pair_id": "not-the-published-one",
            "staging_db": str(ctx.staging_db),
            "staging_manifest": str(ctx.staging_manifest),
            "production_db": str(ctx.production_db),
            "production_manifest": str(ctx.production_manifest),
            "previous_db": str(ctx.production_db.parent /
                               f"{ctx.production_db.name}.previous-y"),
            "previous_manifest": str(ctx.backups / "missing.json"),
            "had_production_db": False,
            "had_production_manifest": False,
            "staging_manifest_sha256": "0" * 64,
            "previous_manifest_sha256": None,
            "steps": [],
        })
        recovered = publish_index.recover(ctx.journal)
        self.assertEqual(recovered["state"], "in-progress")
        self.assertIn("recovery_incomplete", recovered)
        self.assertEqual(recovered["recovery_incomplete"]["state"], "mixed")

    def test_a_vanished_production_directory_is_never_reported_as_rolled_back(self):
        """The audit window: a rollback claim must require the database to exist."""
        ctx = PublishFixture(self.root)
        shutil.rmtree(ctx.production_db)
        ctx.production_manifest.unlink()
        index_common.write_json_atomic(ctx.journal, {
            "schema_version": 1,
            "state": "in-progress",
            "pair_id": "gone",
            "staging_db": str(ctx.staging_db),
            "staging_manifest": str(ctx.staging_manifest),
            "production_db": str(ctx.production_db),
            "production_manifest": str(ctx.production_manifest),
            "previous_db": str(ctx.production_db.parent /
                               f"{ctx.production_db.name}.previous-gone"),
            "previous_manifest": str(ctx.backups / "manifest-gone.json"),
            "had_production_db": True,
            "had_production_manifest": False,
            "staging_manifest_sha256": "a" * 64,
            "previous_manifest_sha256": None,
            "steps": [],
        })
        recovered = publish_index.recover(ctx.journal)
        self.assertEqual(recovered["state"], "in-progress",
                         "a missing production directory is not a restored pair")
        self.assertEqual(recovered["recovery_incomplete"]["state"], "mixed")
        self.assertFalse(ctx.production_db.is_dir())

    def test_a_missing_rollback_manifest_copy_refuses_instead_of_guessing(self):
        """Without the pre-publish manifest a rollback cannot be proven."""
        ctx = PublishFixture(self.root)
        ctx.publish()  # production now holds the new database and manifest
        index_common.write_json_atomic(ctx.journal, {
            "schema_version": 1,
            "state": "in-progress",
            "pair_id": "unknown",
            "staging_db": str(ctx.staging_db),
            "staging_manifest": str(ctx.staging_manifest),
            "production_db": str(ctx.production_db),
            "production_manifest": str(ctx.production_manifest),
            "previous_db": str(ctx.production_db.parent /
                               f"{ctx.production_db.name}.previous-lost"),
            "previous_manifest": str(ctx.backups / "manifest-lost.json"),
            "had_production_db": True,
            "had_production_manifest": True,
            "staging_manifest_sha256": index_common.sha256_file(ctx.staging_manifest),
            "previous_manifest_sha256": "c" * 64,
            "steps": [],
        })
        with self.assertRaises(publish_index.PublishRefused) as caught:
            publish_index.recover(ctx.journal)
        self.assertIn("pre-publish copy", str(caught.exception))
        self.assertEqual(publish_index._journal_state(ctx.journal)["state"],
                         "in-progress")


class PublishBoundaryTests(unittest.TestCase):
    """Locking, dry-run purity and layout validation."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        b3_fakes.release_chroma(self.root)
        try:
            self._tmp.cleanup()
        except PermissionError:  # pragma: no cover
            pass

    def test_dry_run_changes_nothing_and_creates_no_backup_directory(self):
        ctx = PublishFixture(self.root)
        result = ctx.publish(dry_run=True)

        self.assertEqual(result["state"], "dry-run")
        self.assertFalse(ctx.backups.exists(),
                         "dry-run must not create the backup directory")
        self.assertFalse(ctx.journal.exists(), "dry-run must not write the journal")
        self.assertTrue(ctx.staging_db.is_dir())
        self.assertFalse(list(ctx.root.glob("*-production.previous-*")))
        self.assertEqual(self.manifest_files(ctx.production_manifest), {"old.md"})

    def test_a_manifest_inside_the_moved_database_is_rejected_before_any_write(self):
        ctx = PublishFixture(self.root)
        nested_manifest = ctx.production_db / "manifest.json"
        nested_manifest.write_text(
            ctx.production_manifest.read_text(encoding="utf-8"), encoding="utf-8"
        )
        before = sorted(path.name for path in ctx.production_db.iterdir())

        with self.assertRaises(publish_index.PublishRefused) as caught:
            publish_index.publish(
                ctx.staging_db, ctx.staging_manifest,
                ctx.production_db, nested_manifest,
                journal_path=ctx.journal, manifest_backup_dir=ctx.backups,
            )

        self.assertIn("inside the production database", str(caught.exception))
        self.assertFalse(ctx.journal.exists(), "validation must precede every write")
        self.assertFalse(ctx.backups.exists())
        self.assertEqual(sorted(path.name for path in ctx.production_db.iterdir()),
                         before)
        self.assertTrue(ctx.staging_db.is_dir())
        self.assertIsNone(publish_index.read_pair_identity(ctx.staging_db))

    def test_recovery_must_not_run_without_the_write_locks(self):
        ctx = PublishFixture(self.root)
        index_common.write_json_atomic(ctx.journal, {
            "state": "in-progress",
            "staging_db": str(ctx.staging_db),
            "staging_manifest": str(ctx.staging_manifest),
            "production_db": str(ctx.production_db),
            "production_manifest": str(ctx.production_manifest),
            "previous_db": str(ctx.production_db.parent /
                               f"{ctx.production_db.name}.previous-z"),
            "previous_manifest": str(ctx.backups / "x.json"),
        })
        holder = FileLock(writer_lock_path(ctx.production_db)).acquire()
        try:
            with mock.patch.object(publish_index, "PUBLISH_LOCK",
                                   self.root / "publish.lock"), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = publish_index.main(["--recover", "--journal", str(ctx.journal)])
        finally:
            holder.release()

        self.assertEqual(code, 3, "recover must refuse without the writer lock")
        self.assertEqual(publish_index._journal_state(ctx.journal)["state"],
                         "in-progress")
        self.assert_old_pair_files(ctx)

    def test_publish_cli_refuses_while_a_writer_holds_the_target(self):
        ctx = PublishFixture(self.root)
        holder = FileLock(writer_lock_path(ctx.staging_db)).acquire()
        try:
            with mock.patch.object(publish_index, "PUBLISH_LOCK",
                                   self.root / "publish.lock"), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = publish_index.main([
                    "--staging-db", str(ctx.staging_db),
                    "--staging-manifest", str(ctx.staging_manifest),
                    "--production-db", str(ctx.production_db),
                    "--production-manifest", str(ctx.production_manifest),
                    "--journal", str(ctx.journal),
                    "--skip-corpus-check",
                ])
        finally:
            holder.release()

        self.assertEqual(code, 3)
        self.assertTrue(ctx.staging_db.is_dir())
        self.assert_old_pair_files(ctx)

    def test_publish_releases_the_locks_after_a_failure(self):
        ctx = PublishFixture(self.root)
        real_replace = publish_index.os.replace

        def failing_replace(source, target):
            if str(target) == str(ctx.production_manifest):
                raise OSError("injected failure")
            return real_replace(source, target)

        with mock.patch.object(publish_index, "PUBLISH_LOCK",
                               self.root / "publish.lock"), \
                mock.patch.object(publish_index.os, "replace", failing_replace), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(OSError):
                publish_index.main([
                    "--staging-db", str(ctx.staging_db),
                    "--staging-manifest", str(ctx.staging_manifest),
                    "--production-db", str(ctx.production_db),
                    "--production-manifest", str(ctx.production_manifest),
                    "--journal", str(ctx.journal),
                    "--skip-corpus-check",
                ])

        # A failed switch must leave both target locks available again.
        for label, target in (("staging", ctx.staging_db),
                              ("production", ctx.production_db)):
            lock = FileLock(writer_lock_path(target))
            try:
                lock.acquire()
            finally:
                lock.release()
            self.assertTrue(lock.held is False, f"{label} lock not released")
        self.assert_old_pair_files(ctx)

    def assert_old_pair_files(self, ctx):
        self.assertEqual(self.manifest_files(ctx.production_manifest), {"old.md"})
        self.assertTrue(ctx.production_db.is_dir())

    @staticmethod
    def manifest_files(path):
        return set(json.loads(Path(path).read_text(encoding="utf-8"))["files"])


class CrossProcessLockTests(unittest.TestCase):
    """A lock held by another process must really block this one."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._holder_serial = 0
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        b3_fakes.release_chroma(self.root)
        try:
            self._tmp.cleanup()
        except PermissionError:  # pragma: no cover
            pass

    def _hold_in_child(self, target):
        self._holder_serial += 1
        token = self._holder_serial
        ready = self.root / f"ready-{Path(target).name}-{token}.txt"
        release = self.root / f"release-{Path(target).name}-{token}.txt"
        released = self.root / f"released-{Path(target).name}-{token}.txt"
        script = textwrap.dedent(f"""
            import sys, time
            from pathlib import Path
            sys.path.insert(0, r"{ROOT / 'index'}")
            from file_lock import FileLock, writer_lock_path
            lock = FileLock(writer_lock_path(r"{target}"))
            lock.acquire()
            try:
                with open(r"{ready}", "w", encoding="ascii") as handle:
                    handle.write("LOCKED")
                    handle.flush()
                while not Path(r"{release}").is_file():
                    time.sleep(0.05)
            finally:
                lock.release()
                Path(r"{released}").write_text("RELEASED", encoding="ascii")
        """)
        process = subprocess.Popen([sys.executable, "-c", script],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        deadline = time.time() + 30
        while time.time() < deadline and not ready.is_file():
            if process.poll() is not None:
                raise AssertionError("the lock-holding child exited early")
            time.sleep(0.1)
        self.assertTrue(ready.is_file(), "the child never reported holding the lock")
        self.assertEqual(ready.read_text(encoding="ascii"), "LOCKED")
        self.assertIsNone(process.poll(), "the child exited after acquiring the lock")
        process._lock_ready = ready
        process._lock_release = release
        process._lock_released = released
        return process

    def _release_child(self, process):
        """Release the child lock explicitly, then clean up that handshake."""
        release = Path(process._lock_release)
        released = Path(process._lock_released)
        release.write_text("RELEASE", encoding="ascii")
        deadline = time.time() + 10
        while time.time() < deadline and not released.is_file():
            if process.poll() is not None:
                break
            time.sleep(0.05)
        if not released.is_file():
            process.kill()
        process.wait(timeout=10)
        self.assertTrue(released.is_file(), "the child never reported releasing the lock")
        for path in (Path(process._lock_ready), release, released):
            path.unlink(missing_ok=True)

    def _probe_in_child(self, target):
        """Exit 7 when the target lock is refused, 0 when it was acquired."""
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, r"{ROOT / 'index'}")
            from file_lock import FileLock, LockUnavailable, writer_lock_path
            lock = FileLock(writer_lock_path(r"{target}"))
            try:
                lock.acquire()
            except LockUnavailable:
                raise SystemExit(7)
            raise SystemExit(0)
        """)
        return subprocess.run([sys.executable, "-c", script],
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode

    def test_a_second_process_cannot_take_the_same_target_lock(self):
        target = self.root / "chroma_db.v2-staging"
        target.mkdir()
        process = self._hold_in_child(target)
        try:
            self.assertEqual(self._probe_in_child(target), 7)
        finally:
            self._release_child(process)

    def test_another_target_stays_independent(self):
        staging = self.root / "chroma_db.v2-staging"
        other = self.root / "chroma_db.other"
        staging.mkdir()
        other.mkdir()
        process = self._hold_in_child(staging)
        try:
            self.assertEqual(self._probe_in_child(other), 0,
                             "a different target must remain writable")
        finally:
            self._release_child(process)

    def test_resume_refuses_while_another_process_owns_the_target(self):
        paths = b3_fakes.make_paths(self.root)
        b3_fakes.write_corpus(paths.archive, {
            "alpha.md": "# Results\n\n" + b3_fakes.paragraph("a", 600) + "\n",
        })
        b3_fakes.plan_sources(paths)
        process = self._hold_in_child(paths.staging_db)
        backends = b3_fakes.FakeBackends(paths)
        try:
            with self.assertRaises(RuntimeError) as caught:
                rebuild_b3.run_resume(paths, backends=backends)
        finally:
            backends.close()
            self._release_child(process)
        self.assertIn("another B3 writer", str(caught.exception))
        self.assertEqual(backends.encoder.calls, [])

    def test_publish_and_recover_refuse_while_another_process_owns_a_target(self):
        staging_db, staging_manifest, staging = make_manifest(
            self.root / "staging", {"new.md": 2}
        )
        b3_fakes.release_chroma(staging_db)
        write_marker(staging_manifest, staging, staging_db)
        production_db, production_manifest, _ = make_manifest(
            self.root / "production", {"old.md": 1}
        )
        b3_fakes.release_chroma(production_db)
        journal = self.root / "journal.json"
        index_common.write_json_atomic(journal, {
            "state": "in-progress",
            "staging_db": str(staging_db),
            "staging_manifest": str(staging_manifest),
            "production_db": str(production_db),
            "production_manifest": str(production_manifest),
            "previous_db": str(production_db.parent / "chroma_db.previous-p"),
            "previous_manifest": str(self.root / "backups" / "p.json"),
        })

        for label, argv in (
            ("publish", [
                "--staging-db", str(staging_db),
                "--staging-manifest", str(staging_manifest),
                "--production-db", str(production_db),
                "--production-manifest", str(production_manifest),
                "--journal", str(journal), "--skip-corpus-check",
            ]),
            ("recover", ["--recover", "--journal", str(journal)]),
        ):
            with self.subTest(mode=label):
                index_common.write_json_atomic(
                    journal, {**index_common.read_json(journal), "state": "in-progress"}
                )
                process = self._hold_in_child(staging_db)
                try:
                    with mock.patch.object(publish_index, "PUBLISH_LOCK",
                                           self.root / "publish.lock"), \
                            contextlib.redirect_stderr(io.StringIO()):
                        code = publish_index.main(argv)
                finally:
                    self._release_child(process)
                self.assertEqual(code, 3,
                                 f"{label} must refuse while another process owns the target")
                self.assertTrue(staging_db.is_dir())
                self.assertEqual(
                    set(json.loads(
                        production_manifest.read_text(encoding="utf-8")
                    )["files"]),
                    {"old.md"},
                )


class RunStateTests(unittest.TestCase):
    """The status command must not report a dead run as running."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.paths = b3_fakes.make_paths(self.root)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        b3_fakes.release_chroma(self.paths.staging_db)
        try:
            self._tmp.cleanup()
        except PermissionError:  # pragma: no cover
            pass

    def write_corpus(self):
        b3_fakes.write_corpus(self.paths.archive, {
            "alpha.md": "# Results\n\n" + b3_fakes.paragraph("a", 600) + "\n",
            "beta.md": "# Results\n\n" + b3_fakes.paragraph("b", 600) + "\n",
        })
        b3_fakes.plan_sources(self.paths)

    def test_running_progress_with_a_free_lock_is_reported_as_interrupted(self):
        self.write_corpus()
        index_common.write_json_atomic(self.paths.progress, {
            "schema_version": 1,
            "status": "running",
            "pid": 7824,
            "updated_at": "2026-09-22T20:41:28+0800",
            "chunks": 13663,
            "note": "migrated from the four-batch staging; no encoding started",
            "in_progress": {"source": "alpha.md", "rows_written": 0},
        })

        status = rebuild_b3.status_payload(self.paths)

        self.assertEqual(status["state"], "interrupted")
        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(status["recorded_progress_status"], "running")
        self.assertTrue(status["progress_stale"])
        self.assertEqual(status["last_checkpoint_at"], "2026-09-22T20:41:28+0800")
        self.assertTrue(status["note_stale"], "the note contradicts the chunk count")
        self.assertIn("resume", status["recovery_hint"])
        self.assertEqual(status["progress"]["pid"], 7824,
                         "the recorded pid and history stay available")

    def test_a_live_writer_lock_is_reported_as_running(self):
        self.write_corpus()
        index_common.write_json_atomic(self.paths.progress,
                                       {"status": "running", "updated_at": "now"})
        holder = FileLock(self.paths.lock).acquire()
        try:
            status = rebuild_b3.status_payload(self.paths)
        finally:
            holder.release()
        self.assertEqual(status["state"], "running")
        self.assertFalse(status["writer_lock_free"])

    def test_plan_refuses_to_rewrite_the_list_while_a_writer_holds_the_target(self):
        self.write_corpus()
        before = self.paths.sources.read_bytes()
        holder = FileLock(self.paths.lock).acquire()
        try:
            with self.assertRaises(RuntimeError) as caught:
                rebuild_b3.main(["--plan"], paths=self.paths)
        finally:
            holder.release()
        self.assertIn("another B3 writer", str(caught.exception))
        self.assertEqual(self.paths.sources.read_bytes(), before)

    def test_resume_clears_a_superseded_note(self):
        self.write_corpus()
        index_common.write_json_atomic(self.paths.progress, {
            "status": "running", "chunks": 13663, "pid": 1,
            "note": "migrated from the four-batch staging; no encoding started",
        })
        backends = b3_fakes.FakeBackends(self.paths)
        self.addCleanup(backends.close)
        rebuild_b3.run_resume(self.paths, backends=backends)

        note = str(rebuild_b3.progress_state(self.paths).get("note") or "")
        self.assertNotIn("no encoding started", note)
        self.assertFalse(rebuild_b3.status_payload(self.paths)["note_stale"])


class CliHandoffTests(unittest.TestCase):
    def test_the_completion_hint_is_accepted_by_the_publisher(self):
        captured = io.StringIO()
        paths = mock.Mock()
        paths.staging_db = Path("index/chroma_db.v2-staging")
        paths.manifest = Path("index/manifest.b3.partial.json")
        manifest = {"files": {}, "total_chunks": 0, "status": "partial"}
        sources = {"sources": []}
        with mock.patch.object(rebuild_b3, "verify_complete_staging",
                               return_value={"chunks": 0}), \
                mock.patch.object(rebuild_b3, "save_manifest"), \
                mock.patch.object(rebuild_b3, "write_progress"), \
                mock.patch.object(rebuild_b3, "corpus_inventory", return_value=[]), \
                mock.patch.object(rebuild_b3, "build_index_module") as module, \
                contextlib.redirect_stdout(captured):
            rebuild_b3._complete(paths, manifest, sources, None, module.return_value)

        text = captured.getvalue()
        self.assertIn("--staging-manifest", text)
        line = next(line for line in text.splitlines() if "publish_index.py" in line)
        argv = line.split("publish_index.py", 1)[1].replace('"', "").split()

        parser = argparse.ArgumentParser()
        parser.add_argument("--staging-db", default=None)
        parser.add_argument("--staging-manifest", default=None)
        parser.add_argument("--production-db", default=None)
        parser.add_argument("--production-manifest", default=None)
        parser.add_argument("--journal", default=None)
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--recover", action="store_true")
        parser.add_argument("--skip-corpus-check", action="store_true")
        parsed = parser.parse_args(argv)

        self.assertEqual(Path(parsed.staging_db), paths.staging_db)
        self.assertEqual(Path(parsed.staging_manifest), paths.manifest)


if __name__ == "__main__":
    unittest.main()
