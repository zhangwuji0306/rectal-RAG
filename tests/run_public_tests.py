"""Public entry point for the test suite (D-R5).

Runs exactly the public set from ``tests/suite_scope.py``: synthetic fixtures
only, no production vector index, no real model, no production database and no
local corpus.  Exits non-zero on any failure, and reports a module that could
not be imported as ``not-run`` instead of counting it as a pass.

    .venv/Scripts/python tests/run_public_tests.py
    .venv/Scripts/python tests/run_public_tests.py -v

Optional access guard, used by ``tests/test_entry_points.py`` to prove that the
public set never reads local-only data: every file access below a
``--guard-forbid`` path fails the run unless it is also below a ``--guard-allow``
path.

    .venv/Scripts/python tests/run_public_tests.py --guard-forbid E:\\writing-rag --guard-allow E:\\writing-rag\\.venv

Exit codes: 0 = all selected tests passed or explicitly skipped, 1 = failure or error,
2 = some modules could not be imported in this environment (reported, never
counted as a pass).
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import suite_scope  # noqa: E402

MODULES: tuple[str, ...] = suite_scope.PUBLIC_MODULES
ACTIVE_ENV = "RAG_PUBLIC_TESTS_ACTIVE"
AUDITED_EVENTS = (
    "open",
    "os.listdir",
    "os.scandir",
    "os.rename",
    "os.replace",
    "shutil.copyfile",
    "shutil.copytree",
)


def _normalise(path: str) -> str:
    return os.path.normcase(os.path.abspath(path)).rstrip("\\/")


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def install_access_guard(forbid: list[str], allow: list[str]) -> str:
    """Refuse every file access below a local-only root."""
    forbidden = [_normalise(item) for item in forbid]
    if not forbidden:
        return "off"
    allowed = [_normalise(item) for item in allow]

    def hook(event: str, args: tuple) -> None:
        if event not in AUDITED_EVENTS:
            return
        for raw in args:
            if isinstance(raw, int) or raw is None:
                continue
            try:
                text = os.fsdecode(raw)
            except (TypeError, ValueError):
                continue
            if not text or text.startswith("<"):
                continue
            full = _normalise(text)
            if any(_under(full, root) for root in allowed):
                continue
            if any(_under(full, root) for root in forbidden):
                raise RuntimeError(f"public-run guard: {event} on local-only path {full}")

    sys.addaudithook(hook)
    return f"active (forbid {', '.join(forbidden)}; allow {', '.join(allowed) or 'nothing'})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", choices=suite_scope.PUBLIC_CHECKS,
                        help="run only the named public CI group")
    parser.add_argument("--guard-forbid", action="append", default=[],
                        metavar="PATH", help="local-only root this run may not read")
    parser.add_argument("--guard-allow", action="append", default=[],
                        metavar="PATH", help="root exempted from --guard-forbid")
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args(argv)

    modules = suite_scope.PUBLIC_CHECKS[args.check] if args.check else MODULES
    print(f"guard         : {install_access_guard(args.guard_forbid, args.guard_allow)}")
    print(f"public check  : {args.check or 'all'} ({len(modules)} modules)")
    os.environ[ACTIVE_ENV] = "1"

    runner = unittest.TextTestRunner(verbosity=args.verbose, stream=sys.stderr)
    loader = unittest.TestLoader()
    results: dict[str, unittest.TestResult] = {}
    not_run: list[tuple[str, str]] = []
    for name in modules:
        try:
            suite = loader.loadTestsFromModule(importlib.import_module(name))
        except Exception as exc:  # missing dependency or import-time failure
            not_run.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"NOT-RUN {name}: {type(exc).__name__}: {exc}")
            continue
        if not suite.countTestCases():
            not_run.append((name, "no tests loaded"))
            continue
        results[name] = runner.run(suite)

    failures = sum(len(result.failures) + len(result.errors) for result in results.values())
    skipped = sum(len(result.skipped) for result in results.values())
    total = sum(result.testsRun for result in results.values())
    print(f"\npublic set    : {len(modules)} modules")
    for name, result in results.items():
        state = "FAIL" if not result.wasSuccessful() else "COMPLETED WITH SKIPS" if result.skipped else "PASS"
        print(f"  {state} {name}: {result.testsRun} tests, "
              f"{len(result.failures)} failures, {len(result.errors)} errors, "
              f"{len(result.skipped)} skipped")
        for test, reason in result.skipped:
            print(f"    SKIP {test}: {reason}")
    print(f"ran           : {total} tests, {failures} failures/errors")
    if skipped:
        print(f"not verified  : {skipped} skipped test(s) - see the reasons above")
    if not_run:
        print("not run       :")
        for name, reason in not_run:
            print(f"  {name}: {reason}")
    if failures:
        print("RESULT: FAIL")
        return 1
    if not_run:
        print("RESULT: INCOMPLETE (modules not run)")
        return 2
    if total <= skipped:
        print("RESULT: INCOMPLETE (no tests completed)")
        return 2
    print("RESULT: PASS WITH SKIPS" if skipped else "RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
