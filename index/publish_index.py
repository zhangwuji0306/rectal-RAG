"""Publish a validated staging index pair to the production paths.

This is deliberately a separate, short-lived process: it performs no model
loading and writes no vectors.  It exists so that "build the index" and
"switch production" stay two distinct, individually reviewable operations.

Guarantees:

* A staging pair is only publishable when its ``ready`` marker matches the
  staging manifest byte-for-byte, the manifest covers the whole corpus and the
  read-only preflight passes.  A capped or interrupted build cannot publish.
* The swap is journaled.  Each non-atomic cross-file step records its intent
  before it runs and its completion after it lands, and recovery decides what
  actually happened from the files and their identities rather than from the
  recorded steps.  An interruption therefore ends in either the complete old
  pair or the complete new pair, repeated recovery is safe, and a recovery that
  is itself interrupted can be resumed.
* The staging and production directories are identified by a publication
  identity written into the staging directory, so a moved directory is still
  recognisable after a crash.
* The whole publication holds the writer lock of every target directory for its
  entire duration, and ``--recover`` takes the same locks; no writer can start
  in the middle of a switch and no switch can start next to a writer.
* ``--dry-run`` is strictly read-only.

Usage::

    python index/publish_index.py --staging-db <dir> --staging-manifest <file> [--dry-run]
    python index/publish_index.py --recover
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
DEFAULT_PRODUCTION_DB = INDEX_DIR / "chroma_db"
DEFAULT_PRODUCTION_MANIFEST = INDEX_DIR / "manifest.json"
DEFAULT_JOURNAL = INDEX_DIR / "publish.journal.json"
PUBLISH_LOCK = INDEX_DIR / "publish.lock"
MANIFEST_BACKUP_DIR = ROOT / ".claude" / "tmp" / "manifest-backups"
PAIR_IDENTITY_NAME = ".publish-identity.json"
STEP_NAMES = (
    "manifest-backup",
    "production-db-moved",
    "staging-db-promoted",
    "manifest-replaced",
)

sys.path.insert(0, str(INDEX_DIR))
import index_common  # noqa: E402
from file_lock import FileLock, LockUnavailable, target_write_exclusion  # noqa: E402
from rag_core import IndexIntegrityError, index_preflight, release_client  # noqa: E402


class PublishRefused(RuntimeError):
    """Raised when a staging pair must not be published."""


def _preflight(staging_db, staging_manifest):
    return index_preflight(
        db_dir=staging_db,
        model_dir=ROOT / ".model-cache" / "BAAI__bge-m3",
        device="cpu",
        collection_name="papers",
        manifest_path=staging_manifest,
        expected_model="BAAI/bge-m3",
    )


def verify_staging_pair(staging_db, staging_manifest, corpus_names=None,
                        corpus_fingerprint=None):
    """Refuse publication unless the staging pair proves it is complete."""
    staging_db = Path(staging_db)
    staging_manifest = Path(staging_manifest)
    if not staging_db.is_dir():
        raise PublishRefused(f"staging database is missing: {staging_db}")
    if not staging_manifest.is_file():
        raise PublishRefused(f"staging manifest is missing: {staging_manifest}")

    marker_path = staging_manifest.with_name(staging_manifest.name + ".ready.json")
    if not marker_path.is_file():
        raise PublishRefused(
            f"staging pair has no ready marker ({marker_path}); publish only a "
            "verified staged build"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "ready":
        raise PublishRefused(f"ready marker status is {marker.get('status')!r}")
    actual_sha = index_common.sha256_file(staging_manifest)
    if marker.get("manifest_sha256") != actual_sha:
        raise PublishRefused(
            "staging manifest changed after it was marked ready "
            f"(marker={marker.get('manifest_sha256')}, actual={actual_sha})"
        )

    manifest = json.loads(staging_manifest.read_text(encoding="utf-8"))
    declared = set((manifest.get("files") or {}).keys())
    if corpus_names is not None:
        missing = sorted(set(corpus_names) - declared)
        unknown = sorted(declared - set(corpus_names))
        if missing or unknown:
            raise PublishRefused(
                "staging manifest does not cover the corpus: "
                f"missing={len(missing)} {missing[:3]}, unknown={len(unknown)} {unknown[:3]}"
            )
        if corpus_fingerprint is None:
            corpus_fingerprint = index_common.corpus_fingerprint(
                index_common.hashed_corpus(
                    index_common.markdown_files(
                        ROOT / "converted", ROOT / "converted" / "已入库"
                    )
                )
            )
        if marker.get("corpus_fingerprint") != corpus_fingerprint:
            raise PublishRefused(
                "the Markdown corpus changed after the staged build; rebuild "
                "before publishing"
            )
    summary = _preflight(staging_db, staging_manifest)
    if summary["collection_count"] != marker.get("total_chunks"):
        raise PublishRefused(
            f"preflight chunks {summary['collection_count']} != ready marker "
            f"{marker.get('total_chunks')}"
        )
    return {"marker": marker, "manifest": manifest, "preflight": summary}


# ---------------------------------------------------------------------------
# Journal and publication identity
# ---------------------------------------------------------------------------

def _write_journal(path, journal):
    index_common.write_json_atomic(path, journal)


def _journal_state(path):
    return index_common.read_json(path, default=None)


def _record(journal, path, key, step):
    journal.setdefault(key, []).append(step)
    _write_journal(path, journal)


def pair_identity_path(db_dir):
    return Path(db_dir) / PAIR_IDENTITY_NAME


def read_pair_identity(db_dir):
    return index_common.read_json(pair_identity_path(db_dir), default=None)


def write_pair_identity(db_dir, pair_id, manifest_sha256):
    """Mark a staging directory so a moved copy stays identifiable."""
    index_common.write_json_atomic(
        pair_identity_path(db_dir),
        {
            "schema_version": 1,
            "pair_id": pair_id,
            "manifest_sha256": manifest_sha256,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def _file_sha(path):
    path = Path(path)
    return index_common.sha256_file(path) if path.is_file() else None


def pair_state(journal):
    """Classify the production paths against the journal's two pair identities.

    Returns ``"new"`` when production already holds the complete staged pair,
    ``"old"`` when it holds the complete previous pair, and ``"mixed"`` when the
    two files disagree (an interrupted swap).
    """
    production_db = Path(journal["production_db"])
    production_manifest = Path(journal["production_manifest"])
    previous_db = Path(journal["previous_db"]) if journal.get("previous_db") else None

    identity = read_pair_identity(production_db) if production_db.is_dir() else None
    new_db = bool(identity and identity.get("pair_id") == journal.get("pair_id"))
    previous_db_present = bool(previous_db and previous_db.is_dir())
    had_db = journal.get("had_production_db")
    if had_db is None:
        had_db = production_db.is_dir() or previous_db_present
    if had_db:
        # The previous state had a production database, so it is only restored
        # once that directory is really back.
        old_db = production_db.is_dir() and not new_db
    else:
        # The previous state had no production database at all.
        old_db = (not production_db.is_dir()) and not previous_db_present

    manifest_sha = _file_sha(production_manifest)
    staging_sha = journal.get("staging_manifest_sha256")
    previous_sha = journal.get("previous_manifest_sha256")
    new_manifest = (
        manifest_sha is not None and staging_sha is not None
        and manifest_sha == staging_sha
    )
    # A swap only ever installs the staged manifest, so any manifest that is not
    # the staged one is the previous one.
    old_manifest = (not new_manifest) and (
        previous_sha is None or manifest_sha == previous_sha
    )

    if new_db and new_manifest:
        return "new"
    if old_db and old_manifest:
        return "old"
    return "mixed"


def pair_detail(journal):
    """Human-readable evidence behind :func:`pair_state`."""
    production_db = Path(journal["production_db"])
    previous_db = Path(journal["previous_db"]) if journal.get("previous_db") else None
    identity = read_pair_identity(production_db) if production_db.is_dir() else None
    return {
        "state": pair_state(journal),
        "production_db_present": production_db.is_dir(),
        "production_db_identity": (identity or {}).get("pair_id"),
        "expected_pair_id": journal.get("pair_id"),
        "previous_db_present": bool(previous_db and previous_db.is_dir()),
        "production_manifest_sha256": _file_sha(journal["production_manifest"]),
        "staging_manifest_sha256": journal.get("staging_manifest_sha256"),
        "previous_manifest_sha256": journal.get("previous_manifest_sha256"),
    }


def upgrade_journal(journal):
    """Fill in the identity fields a journal written by an older build lacks.

    The pair decision must not depend on the recorded steps, so a journal
    without them is completed from the files that are still on disk.
    """
    if not journal.get("pair_id"):
        staging_db = Path(journal.get("staging_db", ""))
        identity = read_pair_identity(staging_db) if staging_db.is_dir() else None
        journal["pair_id"] = (identity or {}).get("pair_id")
    if not journal.get("staging_manifest_sha256"):
        journal["staging_manifest_sha256"] = _file_sha(
            journal.get("staging_manifest", "")
        )
    if journal.get("previous_manifest_sha256") is None:
        journal["previous_manifest_sha256"] = _file_sha(
            journal.get("previous_manifest", "")
        )
    previous_db = Path(journal["previous_db"]) if journal.get("previous_db") else None
    production_db = Path(journal["production_db"])
    if "had_production_db" not in journal:
        journal["had_production_db"] = bool(
            production_db.is_dir() or (previous_db is not None and previous_db.is_dir())
        )
    if "had_production_manifest" not in journal:
        previous_manifest = (
            Path(journal["previous_manifest"]) if journal.get("previous_manifest") else None
        )
        journal["had_production_manifest"] = bool(
            previous_manifest is not None and previous_manifest.is_file()
        )
    return journal


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------

def _validate_targets(staging_db, staging_manifest, production_db, production_manifest,
                      journal_path=None, manifest_backup_dir=None):
    """Reject layouts that a swap cannot undo, before anything is written."""
    staging_db = Path(staging_db)
    staging_manifest = Path(staging_manifest)
    production_db = Path(production_db)
    production_manifest = Path(production_manifest)

    if index_common.same_path(staging_db, production_db):
        raise PublishRefused("staging database and production database are the same path")
    if index_common.is_inside(production_db, staging_db) or index_common.is_inside(
        staging_db, production_db
    ):
        raise PublishRefused(
            "staging and production database directories must not contain each other"
        )
    if index_common.same_path(staging_manifest, production_manifest):
        raise PublishRefused("staging manifest and production manifest are the same path")
    for label, value in (("staging manifest", staging_manifest),
                         ("production manifest", production_manifest),
                         ("journal", journal_path),
                         ("manifest backup directory", manifest_backup_dir)):
        if value is None:
            continue
        for name, directory in (("staging database", staging_db),
                                ("production database", production_db)):
            if index_common.is_inside(value, directory):
                raise PublishRefused(
                    f"the {label} ({value}) lives inside the {name} directory "
                    f"({directory}); a swap would move or leave it behind"
                )
    if index_common.same_path(staging_manifest, staging_db) or index_common.same_path(
        production_manifest, production_db
    ):
        raise PublishRefused("a manifest must not be a database directory path")


# ---------------------------------------------------------------------------
# Publish / recover
# ---------------------------------------------------------------------------

def publish(staging_db, staging_manifest, production_db, production_manifest,
            journal_path=DEFAULT_JOURNAL, dry_run=False, corpus_names=None,
            corpus_fingerprint=None, manifest_backup_dir=None):
    """Swap a verified staging pair into production with a journaled rollback."""
    staging_db = Path(staging_db)
    staging_manifest = Path(staging_manifest)
    production_db = Path(production_db)
    production_manifest = Path(production_manifest)
    journal_path = Path(journal_path)
    backup_dir = Path(manifest_backup_dir or MANIFEST_BACKUP_DIR)
    _validate_targets(staging_db, staging_manifest, production_db,
                      production_manifest, journal_path, backup_dir)

    existing = _journal_state(journal_path)
    if existing and existing.get("state") == "in-progress":
        raise PublishRefused(
            f"an unfinished publish journal exists at {journal_path}; run --recover first"
        )

    verification = verify_staging_pair(
        staging_db, staging_manifest, corpus_names, corpus_fingerprint
    )
    # The preflight opens the staging database; release the handles before any
    # directory is moved (Windows keeps the directory locked otherwise).
    release_client(staging_db)
    release_client(production_db)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    previous_db = production_db.parent / f"{production_db.name}.previous-{stamp}"
    previous_manifest = backup_dir / f"manifest-pre-publish-{stamp}.json"
    pair_id = uuid.uuid4().hex

    if dry_run:
        return {
            "state": "dry-run",
            "dry_run": True,
            "staging_db": str(staging_db),
            "staging_manifest": str(staging_manifest),
            "production_db": str(production_db),
            "production_manifest": str(production_manifest),
            "total_chunks": verification["marker"]["total_chunks"],
            "sources": verification["preflight"]["source_count"],
            "had_production_db": production_db.exists(),
            "had_production_manifest": production_manifest.exists(),
        }

    if previous_db.exists():
        raise PublishRefused(
            f"the rollback directory {previous_db} already exists; resolve it first"
        )
    backup_dir.mkdir(parents=True, exist_ok=True)
    if previous_manifest.exists():
        raise PublishRefused(
            f"the rollback manifest {previous_manifest} already exists; resolve it first"
        )

    journal = {
        "schema_version": 1,
        "state": "in-progress",
        "pair_id": pair_id,
        "stamp": stamp,
        "staging_db": str(staging_db),
        "staging_manifest": str(staging_manifest),
        "production_db": str(production_db),
        "production_manifest": str(production_manifest),
        "previous_db": str(previous_db),
        "previous_manifest": str(previous_manifest),
        "journal": str(journal_path),
        "had_production_db": production_db.exists(),
        "had_production_manifest": production_manifest.exists(),
        "staging_manifest_sha256": verification["marker"]["manifest_sha256"],
        "previous_manifest_sha256": _file_sha(production_manifest),
        "total_chunks": verification["marker"]["total_chunks"],
        "steps": [],
        "intents": [],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_journal(journal_path, journal)
    write_pair_identity(staging_db, pair_id, journal["staging_manifest_sha256"])

    try:
        _record(journal, journal_path, "intents", "manifest-backup")
        if journal["had_production_manifest"]:
            shutil.copy2(production_manifest, previous_manifest)
        _record(journal, journal_path, "steps", "manifest-backup")

        _record(journal, journal_path, "intents", "production-db-moved")
        if journal["had_production_db"]:
            release_client(production_db)
            production_db.rename(previous_db)
        _record(journal, journal_path, "steps", "production-db-moved")

        _record(journal, journal_path, "intents", "staging-db-promoted")
        release_client(staging_db)
        staging_db.rename(production_db)
        _record(journal, journal_path, "steps", "staging-db-promoted")

        _record(journal, journal_path, "intents", "manifest-replaced")
        temporary = production_manifest.with_name(production_manifest.name + ".publish.tmp")
        shutil.copy2(staging_manifest, temporary)
        os.replace(temporary, production_manifest)
        _record(journal, journal_path, "steps", "manifest-replaced")
    except Exception as exc:  # noqa: BLE001 - roll back to the paired state
        journal["error"] = f"{type(exc).__name__}: {exc}"
        journal = _rollback(journal, journal_path)
        raise

    state = pair_state(journal)
    journal["result_pair"] = state
    if state == "new":
        journal["state"] = "completed"
        journal["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    else:  # pragma: no cover - defensive: a landed swap must classify as new
        journal["state"] = "in-progress"
        journal["note"] = f"swap finished but the production pair classified as {state}"
    _write_journal(journal_path, journal)
    return journal


def _rollback(journal, journal_path):
    """Idempotently restore the complete previous pair.

    Every action is decided from the current files and identities, so this can
    run again after being interrupted part way through.
    """
    staging_db = Path(journal["staging_db"])
    production_db = Path(journal["production_db"])
    production_manifest = Path(journal["production_manifest"])
    previous_db = Path(journal["previous_db"]) if journal.get("previous_db") else None
    previous_manifest = (
        Path(journal["previous_manifest"]) if journal.get("previous_manifest") else None
    )
    restored = []

    # Drive the pair to the previous database first: the new database (if it
    # was promoted) goes back to staging, otherwise the moved directory returns.
    release_client(production_db)
    release_client(staging_db)
    identity = read_pair_identity(production_db) if production_db.is_dir() else None
    if identity and identity.get("pair_id") == journal.get("pair_id"):
        if staging_db.exists():
            raise PublishRefused(
                f"cannot roll back: both {staging_db} and the promoted copy in "
                f"{production_db} exist"
            )
        production_db.rename(staging_db)
        restored.append("staging-db")
    if previous_db is not None and previous_db.is_dir() and not production_db.exists():
        previous_db.rename(production_db)
        restored.append("production-db")

    # Pair the manifest with that database.
    current_sha = _file_sha(production_manifest)
    staging_sha = journal.get("staging_manifest_sha256")
    if current_sha is None or (staging_sha is not None and current_sha == staging_sha):
        if previous_manifest is not None and previous_manifest.is_file():
            temporary = production_manifest.with_name(
                production_manifest.name + ".publish.tmp"
            )
            shutil.copy2(previous_manifest, temporary)
            os.replace(temporary, production_manifest)
            restored.append("manifest")
        elif not journal.get("had_production_manifest"):
            if current_sha is not None:
                production_manifest.unlink(missing_ok=True)
                restored.append("manifest-removed")
        else:
            raise PublishRefused(
                "cannot roll back the manifest: the pre-publish copy "
                f"{previous_manifest} is missing"
            )
    leftover = production_manifest.with_name(production_manifest.name + ".publish.tmp")
    if leftover.exists():
        leftover.unlink()
        restored.append("temp-removed")

    state = pair_state(journal)
    journal["recovered_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    journal["rolled_back"] = restored
    journal["result_pair"] = state
    if state == "old":
        journal["state"] = "rolled-back"
        journal["steps"] = []
    else:
        # Keep the journal retryable instead of claiming a state we cannot prove.
        journal["state"] = "in-progress"
        journal["recovery_incomplete"] = pair_detail(journal)
    _write_journal(journal_path, journal)
    return journal


def recover(journal_path=DEFAULT_JOURNAL, production_db=None, production_manifest=None):
    """Finish or roll back an interrupted publication from its journal."""
    journal_path = Path(journal_path)
    journal = _journal_state(journal_path)
    if not journal:
        raise PublishRefused(f"no publish journal at {journal_path}")
    if journal.get("state") != "in-progress":
        return journal
    if production_db is not None:
        journal["production_db"] = str(production_db)
    if production_manifest is not None:
        journal["production_manifest"] = str(production_manifest)
    upgrade_journal(journal)

    if pair_state(journal) == "new":
        production_db_path = Path(journal["production_db"])
        try:
            summary = _preflight(production_db_path, journal["production_manifest"])
        except IndexIntegrityError as exc:
            journal["preflight_error"] = str(exc)
        else:
            journal["state"] = "completed"
            journal["recovered_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            journal["recovered_chunks"] = summary["collection_count"]
            journal["result_pair"] = "new"
        finally:
            release_client(production_db_path)
        _write_journal(journal_path, journal)
        if journal["state"] == "completed":
            return journal
    return _rollback(journal, journal_path)


def write_exclusion(staging_db, production_db):
    """Exclusive right to switch production, held for the whole operation.

    Covers both the staging target (moved by the swap) and the production
    target, so no build, ingest, reconcile or runner can touch either directory
    while the switch runs.  ``main`` additionally serialises publishers with the
    global publish lock, acquired first.
    """
    return target_write_exclusion(
        [staging_db, production_db], label="publish/recover"
    )


def _production_targets(journal):
    return journal.get("staging_db"), journal.get("production_db")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-db", default=None)
    parser.add_argument("--staging-manifest", default=None)
    parser.add_argument("--production-db", default=str(DEFAULT_PRODUCTION_DB))
    parser.add_argument("--production-manifest", default=str(DEFAULT_PRODUCTION_MANIFEST))
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL))
    parser.add_argument("--dry-run", action="store_true", help="verify only; change nothing")
    parser.add_argument("--recover", action="store_true", help="resolve an interrupted publish")
    parser.add_argument(
        "--skip-corpus-check", action="store_true",
        help="publish without comparing the staged manifest to the current corpus",
    )
    args = parser.parse_args(argv)

    journal_path = Path(args.journal)
    try:
        with FileLock(PUBLISH_LOCK):
            if args.recover:
                existing = _journal_state(journal_path)
                if not existing:
                    raise PublishRefused(f"no publish journal at {journal_path}")
                staging_db, production_db = _production_targets(existing)
                with write_exclusion(staging_db or args.production_db,
                                     production_db or args.production_db):
                    journal = recover(journal_path)
            else:
                if not args.staging_db or not args.staging_manifest:
                    parser.error(
                        "--staging-db and --staging-manifest are required (or use --recover)"
                    )
                corpus_names = None
                corpus_fingerprint = None
                if not args.skip_corpus_check:
                    inventory = index_common.hashed_corpus(
                        index_common.markdown_files(
                            ROOT / "converted", ROOT / "converted" / "已入库"
                        )
                    )
                    corpus_names = {name for name, _ in inventory}
                    corpus_fingerprint = index_common.corpus_fingerprint(inventory)
                with write_exclusion(args.staging_db, args.production_db):
                    journal = publish(
                        Path(args.staging_db), Path(args.staging_manifest),
                        Path(args.production_db), Path(args.production_manifest),
                        journal_path=journal_path, dry_run=args.dry_run,
                        corpus_names=corpus_names,
                        corpus_fingerprint=corpus_fingerprint,
                    )
    except LockUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except (PublishRefused, IndexIntegrityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({
        "state": journal["state"],
        "production_db": journal["production_db"],
        "total_chunks": journal.get("total_chunks"),
        "previous_db": journal.get("previous_db"),
        "previous_manifest": journal.get("previous_manifest"),
        "result_pair": journal.get("result_pair"),
        "recovery_incomplete": journal.get("recovery_incomplete"),
    }, ensure_ascii=False, indent=2), flush=True)
    return 4 if journal["state"] == "in-progress" else 0


if __name__ == "__main__":
    sys.exit(main())
