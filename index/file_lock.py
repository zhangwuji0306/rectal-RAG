"""Cross-process writer exclusion for index directories.

A standard-library operating-system file lock.  The lock is owned by the open
file handle, so it is released when the process exits for any reason - a
crashed writer never leaves the index permanently locked and a stale PID file
is never treated as proof of occupancy.

Usage::

    lock = FileLock(writer_lock_path(staging_db))
    with lock.acquire() as held:
        ...  # only one process at a time

``acquire()`` raises :class:`LockUnavailable` when another live process holds
the lock.

The lock is derived from the *target directory*: every writer that modifies one
index directory takes the same lock file, held for the whole modification
interval, so a build, a live ingest, a manifest reconcile, the B3 runner, a
publication and a publication recovery cannot overlap on the same target.
Writers on different target directories stay independent.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

try:  # Windows
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None


class LockUnavailable(RuntimeError):
    """Raised when another process currently holds the writer lock."""


class FileLock:
    """An advisory exclusive lock backed by the operating system."""

    def __init__(self, path):
        self.path = Path(path)
        self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self):
        if self._handle is not None:
            raise RuntimeError(f"lock already held by this object: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if msvcrt is not None:
                os.lseek(handle, 0, os.SEEK_SET)
                msvcrt.locking(handle, msvcrt.LK_NBLCK, 1)
            elif fcntl is not None:  # pragma: no cover - POSIX
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:  # pragma: no cover - no locking primitive available
                raise RuntimeError("no OS file-locking primitive is available")
        except OSError as exc:
            os.close(handle)
            raise LockUnavailable(
                f"another writer holds {self.path}; refusing a concurrent write"
            ) from exc
        # Record the owner for human diagnostics only; ownership is decided by
        # the OS lock, never by the content of this file.
        try:
            os.ftruncate(handle, 0)
            os.write(handle, f"pid={os.getpid()}\n".encode("ascii"))
            os.lseek(handle, 0, os.SEEK_SET)
        except OSError:
            pass
        self._handle = handle
        return self

    def release(self):
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if msvcrt is not None:
                os.lseek(handle, 0, os.SEEK_SET)
                msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:  # pragma: no cover - POSIX
                fcntl.flock(handle, fcntl.LOCK_UN)
        except OSError:
            pass
        finally:
            os.close(handle)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_exc):
        self.release()
        return False

    def __del__(self):  # pragma: no cover - defensive
        try:
            self.release()
        except Exception:
            pass


def writer_lock_path(target) -> Path:
    """Lock file guarding writes to one index directory.

    The lock lives next to the target, never inside it: publication renames the
    directory, and a lock inside it would travel with the rename.
    """
    target = Path(target)
    return target.parent / (target.name + ".writer.lock")


def lock_is_free(target) -> bool:
    """Probe a writer lock without holding it.  Never proof of a live process."""
    try:
        with FileLock(writer_lock_path(target)):
            return True
    except LockUnavailable:
        return False


@contextmanager
def target_write_exclusion(targets, label="index writer"):
    """Hold the writer lock of every listed target for the whole block.

    Acquisition order is a stable sort of the lock paths, so two processes that
    need several targets cannot deadlock against each other.
    """
    paths = sorted(
        {writer_lock_path(target) for target in targets},
        key=lambda path: str(path).lower(),
    )
    held = []
    try:
        for path in paths:
            try:
                held.append(FileLock(path).acquire())
            except LockUnavailable as exc:
                raise LockUnavailable(
                    f"{label}: another writer holds {path}; refusing to modify "
                    "the same target concurrently"
                ) from exc
        yield held
    finally:
        for lock in reversed(held):
            lock.release()
