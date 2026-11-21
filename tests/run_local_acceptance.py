"""Explicit local-machine acceptance entry point (D-R5).

Runs the checks that need private data owned by this machine: the production
vector index, the real bge-m3 model cache, the production SQLite database and
the local corpus originals.  These checks are deliberately kept out of the
public entry point (``tests/run_public_tests.py``), which must stay runnable in
a clean allowlist-derived code directory.

    .venv/Scripts/python tests/run_local_acceptance.py

Exit codes: 0 = everything ran and passed, 1 = failure, 2 = something could not
be verified (missing private data), reported per module as ``not-run``/``skip``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import suite_scope  # noqa: E402

MODULES: tuple[str, ...] = tuple(suite_scope.LOCAL_MODULES)

# Private resources the local set depends on; missing entries are reported
# instead of being silently skipped.
PRIVATE_RESOURCES: tuple[tuple[str, Path], ...] = (
    ("production vector index", ROOT / "index" / "chroma_db"),
    ("index ledger", ROOT / "index" / "manifest.json"),
    ("local bge-m3 model", ROOT / ".model-cache" / "BAAI__bge-m3"),
    ("production database", ROOT / "直肠癌文献爬取" / "tasks.sqlite"),
    ("raw corpus originals", ROOT / "corpus_raw"),
)


def report_resources() -> list[str]:
    missing: list[str] = []
    print("local resources:")
    for label, path in PRIVATE_RESOURCES:
        present = path.exists()
        if not present:
            missing.append(label)
        print(f"  {'present' if present else 'MISSING'} {label}: {path}")
    return missing


def run_modules(verbosity: int) -> tuple[dict[str, unittest.TestResult], list[tuple[str, str]]]:
    results: dict[str, unittest.TestResult] = {}
    not_run: list[tuple[str, str]] = []
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stderr)
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            suite = unittest.TestLoader().loadTestsFromModule(module)
        except Exception as exc:
            not_run.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"NOT-RUN {name}: {type(exc).__name__}: {exc}")
            continue
        results[name] = runner.run(suite)
    return results, not_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args(argv)

    missing = report_resources()
    if missing:
        print("missing private resources: " + ", ".join(missing))

    results, not_run = run_modules(args.verbose)
    failures = sum(len(r.failures) + len(r.errors) for r in results.values())
    skipped = sum(len(r.skipped) for r in results.values())
    verified = 0
    for name, result in results.items():
        state = "PASS" if result.wasSuccessful() else "FAIL"
        print(f"  {state} {name}: {result.testsRun} tests, "
              f"{len(result.failures)} failures, {len(result.errors)} errors, "
              f"{len(result.skipped)} skipped - {suite_scope.LOCAL_MODULES.get(name, '')}")
        if result.wasSuccessful() and not result.skipped:
            verified += 1
    print(f"local set     : {len(MODULES)} modules, {verified} fully verified")
    if skipped:
        print(f"not verified  : {skipped} skipped test(s) - see reasons above")
    for name, reason in not_run:
        print(f"not run       : {name}: {reason}")
    if failures:
        print("RESULT: FAIL")
        return 1
    if not_run or skipped or verified < len(MODULES):
        print("RESULT: INCOMPLETE (local acceptance not fully verified)")
        return 2
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
