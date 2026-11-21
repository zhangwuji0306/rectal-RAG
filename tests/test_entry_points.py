"""Separation checks for the public and local test entry points (D-R5).

Two proofs live here:

* **static** - every module under ``tests/`` is classified exactly once as
  public, local or support code, and a public module may only contain reviewed
  references to production paths (``PRODUCTION_REFERENCE_REGISTRY``);
* **runtime** - the public entry point runs to completion inside a clean code
  directory assembled from ``docs/发布范围-allowlist.txt``, with every path of
  this workspace refused to the test process except the interpreter itself.

The runtime proof runs from a clean copy rather than in place because the
public set still reaches this machine's copies of the workspace when it is run
inside it: ``tests/b3_fakes.py`` binds the model directory to the repository
root, so ``build_sources_payload`` fingerprints the local ``.model-cache``
whenever it exists.  Refusing the whole workspace except the interpreter is the
stronger check, and it is the situation a clean checkout is in.  Checks that
genuinely need the private data are listed in ``tests/suite_scope.py`` and run
through ``tests/run_local_acceptance.py``.

The guarded run executes the whole public set and is therefore slow; it skips
itself when this module already runs inside a public run (the runner exports
``RAG_PUBLIC_TESTS_ACTIVE=1``), which is what keeps the entry point from
recursing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import release_scope as rs  # noqa: E402
import suite_scope  # noqa: E402

PUBLIC_RUNNER = HERE / "run_public_tests.py"
LOCAL_RUNNER = HERE / "run_local_acceptance.py"
NESTED_ENV = "RAG_PUBLIC_TESTS_ACTIVE"
VENV = ROOT / ".venv"
LOCAL_ONLY_DIRS = ("index/chroma_db", "converted", "corpus_raw", ".model-cache",
                   "直肠癌文献爬取/tasks.sqlite")


def require_not_nested(test: unittest.TestCase) -> None:
    if os.environ.get(NESTED_ENV) == "1":
        test.skipTest("already running inside the public entry point: the guarded "
                      "clean-directory proof runs from a direct invocation")


class ScopeManifestTests(unittest.TestCase):
    def test_every_test_module_is_classified_once(self):
        on_disk = set(suite_scope.all_python_modules())
        public = set(suite_scope.PUBLIC_MODULES)
        local = set(suite_scope.LOCAL_MODULES)
        support = set(suite_scope.SUPPORT_MODULES)
        self.assertEqual(public & local, set(), "a module is both public and local")
        self.assertEqual(public & support, set(), "a module is both public and support")
        self.assertEqual(local & support, set(), "a module is both local and support")
        self.assertEqual(on_disk, public | local | support,
                         "tests/ contains unclassified modules")
        self.assertEqual(set(suite_scope.test_modules()), public | local,
                         "a test module is missing from the classification")

    def test_public_and_local_entry_points_use_the_classified_sets(self):
        import run_local_acceptance
        import run_public_tests

        self.assertEqual(tuple(run_public_tests.MODULES), tuple(suite_scope.PUBLIC_MODULES))
        self.assertEqual(tuple(run_local_acceptance.MODULES), tuple(suite_scope.LOCAL_MODULES))
        self.assertFalse(set(run_public_tests.MODULES) & set(run_local_acceptance.MODULES),
                         "the two entry points must not overlap")
        self.assertTrue(PUBLIC_RUNNER.is_file(), "the public entry point is missing")
        self.assertTrue(LOCAL_RUNNER.is_file(), "the local entry point is missing")

    def test_ci_groups_partition_public_modules(self):
        self.assertTrue(all(suite_scope.PUBLIC_CHECKS.values()))
        assigned = [module for group in suite_scope.PUBLIC_CHECKS.values() for module in group]
        self.assertCountEqual(assigned, suite_scope.PUBLIC_MODULES)

    def test_local_modules_state_a_concrete_private_dependency(self):
        keywords = ("model", "chroma", "manifest", "gold set", "corpus",
                    "database", "queries.jsonl", "ledger")
        for name, reason in suite_scope.LOCAL_MODULES.items():
            with self.subTest(module=name):
                self.assertGreater(len(reason), 20, f"{name} needs a concrete reason")
                self.assertTrue(any(word in reason.lower() for word in keywords),
                                f"{name}: the reason names no private resource: {reason}")

    def test_public_modules_reference_production_paths_only_as_reviewed(self):
        registry = suite_scope.PRODUCTION_REFERENCE_REGISTRY
        for name in suite_scope.PUBLIC_MODULES:
            with self.subTest(module=name):
                recorded = registry.get(name, {"references": {}, "reason": ""})
                self.assertTrue(recorded.get("reason") or not suite_scope.references(name),
                                f"{name} references production paths without a review reason")
                self.assertEqual(
                    suite_scope.references(name), recorded["references"],
                    f"{name}: production-path references changed; review the module and "
                    f"update PRODUCTION_REFERENCE_REGISTRY in tests/suite_scope.py",
                )
        for name in registry:
            self.assertIn(name, suite_scope.PUBLIC_MODULES,
                          f"{name} is registered but not public")

    def test_public_modules_carry_no_absolute_workspace_path(self):
        for name in suite_scope.PUBLIC_MODULES:
            with self.subTest(module=name):
                absolute = suite_scope.references(name).get("absolute_workspace_path", ())
                self.assertEqual(absolute, (),
                                 f"{name} hard-codes a local absolute path: {absolute}")


class CleanDirectoryTests(unittest.TestCase):
    def test_public_entry_point_runs_in_a_clean_allowlist_directory(self):
        require_not_nested(self)
        with tempfile.TemporaryDirectory(prefix="rag-public-") as tmp:
            clean = Path(tmp) / "code"
            clean.mkdir()
            copied = rs.build_release_dir(clean)
            self.assertTrue(copied, "the allowlist produced an empty code directory")
            for name in LOCAL_ONLY_DIRS:
                self.assertFalse((clean / name).exists(),
                                 f"local-only data leaked into the clean code directory: {name}")
            environment = dict(os.environ)
            environment["PYTHONIOENCODING"] = "utf-8"
            environment[NESTED_ENV] = "1"
            environment.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [sys.executable, str(clean / "tests" / "run_public_tests.py"),
                 "--guard-forbid", str(ROOT), "--guard-allow", str(VENV)],
                cwd=clean, capture_output=True, text=True, encoding="utf-8",
                errors="replace", env=environment, timeout=3600,
            )
            self.assertEqual(
                completed.returncode, 0,
                "the public entry point must run in a clean code directory built from "
                f"the allowlist\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr[-4000:]}",
            )
            self.assertIn("RESULT: PASS", completed.stdout)
            self.assertIn("guard         : active", completed.stdout)


if __name__ == "__main__":
    sys.exit(unittest.main())
