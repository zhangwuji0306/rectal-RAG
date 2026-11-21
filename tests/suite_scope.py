"""Classification of the test suite into a public set and a local set (D-R5).

Public modules use synthetic fixtures only: no production vector index, no real
bge-m3 model, no production SQLite database and no local corpus original.  They
must run from a clean code directory assembled out of
``docs/发布范围-allowlist.txt``.

Local modules need data that only exists on this machine (the production index,
the real model, the production database, or the local originals).  They run
through ``tests/run_local_acceptance.py`` and report ``skip``/``not-run`` with a
reason when the private dependency is absent - never a pass.

``PRODUCTION_REFERENCE_REGISTRY`` records the remaining references to
production paths inside public modules.  Every entry is a reviewed one: an
assertion, a mock target, a source-file read or a guard.  A new reference - and
a reference whose line disappeared or changed - fails
``tests/test_entry_points.py``, so a production dependency cannot enter the
public set silently.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Modules that are safe to run in a clean allowlist-derived code directory.
PUBLIC_MODULES: tuple[str, ...] = (
    "test_passage_provenance",
    "test_e2e_evaluation",
    "test_d2_adapters",
    "test_answer_generation",
    "test_evidence_pack",
    "test_query_expansion",
    "test_hybrid_retrieval",
    "test_document_retrieval",
    "test_semantic_reranker",
    "test_a1_schema",
    "test_a2_metadata",
    "test_a3_state_machine",
    "test_a4_oa_resolver",
    "test_a5_fulltext",
    "test_a6_content_validation",
    "test_a7_normalization",
    "test_a8_lease_batch",
    "test_a8_remediation",
    "test_a9_reporting",
    "test_ar1_content_binding",
    "test_ar2_lease_ownership",
    "test_ar3_format_entry",
    "test_ar4_reporting_truth",
    "test_pipeline",
    "test_index_maintenance",
    "test_evaluation_identity",
    "test_evaluation_rankings",
    "test_paper_registry",
    "test_quality_metadata",
    "test_citation_graph",
    "test_citation_calibration",
    "test_evidence_ranking",
    "test_stable_copy",
    "test_b3_continuous_runner",
    "test_b3_passage_representation",
    "test_release_boundary",
    "test_release_scope",
    "test_entry_points",
)

# The full public entry point and the named CI checks share these modules.
_FOCUSED_CHECKS: dict[str, tuple[str, ...]] = {
    "integration": (
        "test_a5_fulltext", "test_a7_normalization", "test_a8_lease_batch",
        "test_a8_remediation", "test_ar1_content_binding", "test_ar2_lease_ownership",
        "test_ar3_format_entry", "test_pipeline",
    ),
    "schema-validation": ("test_a1_schema",),
    "index-integrity": (
        "test_index_maintenance", "test_b3_continuous_runner",
        "test_b3_passage_representation", "test_stable_copy",
    ),
    "retrieval-regression": (
        "test_evaluation_identity", "test_evaluation_rankings", "test_document_retrieval",
        "test_hybrid_retrieval", "test_query_expansion", "test_semantic_reranker",
        "test_evidence_ranking",
    ),
}
_FOCUSED_MODULES = {module for modules in _FOCUSED_CHECKS.values() for module in modules}
PUBLIC_CHECKS: dict[str, tuple[str, ...]] = {
    "unit": tuple(module for module in PUBLIC_MODULES if module not in _FOCUSED_MODULES),
    **_FOCUSED_CHECKS,
}

# Modules that need data owned by this machine, with the concrete reason.
LOCAL_MODULES: dict[str, str] = {
    "test_b1_index_integrity": (
        "reads the production Chroma collection and index/manifest.json and "
        "fingerprints the local .model-cache/bge-m3 model"
    ),
    "test_b2_benchmark": (
        "validates the real Gold Set evaluation/queries.jsonl, which is private "
        "evaluation evidence and deliberately outside the release allowlist"
    ),
    "test_b3_publish_recovery": (
        "publish/recovery paths fingerprint the local .model-cache/bge-m3 model "
        "and raise rag_core.IndexIntegrityError without it"
    ),
    "test_b3_writer_boundary": (
        "publish paths fingerprint the local .model-cache/bge-m3 model and raise "
        "rag_core.IndexIntegrityError without it"
    ),
    "test_ar3_normalized_index": (
        "skips itself unless the real bge-m3 model cache exists and reads the "
        "production index/b3-sources.json ledger"
    ),
}

# Modules under tests/ that are support code or entry points, not test modules.
SUPPORT_MODULES: tuple[str, ...] = (
    "b3_fakes",
    "release_scope",
    "suite_scope",
    "run_public_tests",
    "run_local_acceptance",
)

# Shapes of a reference to local-only data.  A reference is a literal or
# ROOT-relative path pointing at production data, never a synthetic path built
# below a temporary directory.
PRODUCTION_REFERENCE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("absolute_workspace_path", r"[A-Za-z]:[\\/](?:Users|writing-rag)[\\/]"),
    ("production_path_literal",
     r"""["'](?:corpus_raw/|converted/|normalized/|mucinous/|papers/|pdfs_merged/|"""
     r"""flash-failed/|index/chroma_db|index/manifest\.json|\.model-cache/|"""
     r"""直肠癌文献爬取/tasks\.sqlite)"""),
    ("production_corpus_join",
     r"""ROOT\s*/\s*["'](?:corpus_raw|converted|normalized|mucinous|papers|pdfs_merged|"""
     r"""flash-failed|\.model-cache)["']"""),
    ("production_index_join",
     r"""ROOT\s*/\s*["']index["']\s*/\s*["'](?:chroma_db|manifest|b3-|_tmp_eval)"""),
    ("production_crawler_join",
     r"""ROOT\s*/\s*["']直肠癌文献爬取["']\s*/\s*["'](?:tasks\.sqlite|reports|index|"""
     r"""normalized|corpus_raw)["']"""),
)
PRODUCTION_REFERENCE_RES = tuple(
    (label, re.compile(pattern)) for label, pattern in PRODUCTION_REFERENCE_PATTERNS
)

PRODUCTION_REFERENCE_REGISTRY: dict[str, dict] = {
    "test_a9_reporting": {
        "reason": "parses the report CLI defaults and compares path objects only",
        "references": {
            "production_corpus_join": (
                "self.assertEqual(Path(args.raw_root), ROOT / \"corpus_raw\")",
                "self.assertEqual(Path(args.normalized_root), ROOT / \"normalized\")",
            ),
            "production_crawler_join": (
                "self.assertNotEqual(Path(args.raw_root), ROOT / \"直肠癌文献爬取\" / \"corpus_raw\")",
                "self.assertNotEqual(Path(args.normalized_root), ROOT / \"直肠癌文献爬取\" / \"normalized\")",
            ),
        },
    },
    "test_ar3_format_entry": {
        "reason": ("compares the B3 staging ledger before/after a fixture run through "
                   "is_file()/read_bytes(); absent in a clean code directory"),
        "references": {
            "production_index_join": (
                "B3_SOURCES = ROOT / \"index\" / \"b3-sources.json\"",
            ),
        },
    },
    "test_b3_passage_representation": {
        "reason": "asserts staging paths differ from production paths",
        "references": {
            "production_index_join": (
                "production_db = ROOT / \"index\" / \"chroma_db\"",
                "production_manifest = ROOT / \"index\" / \"manifest.json\"",
            ),
        },
    },
    "test_release_boundary": {
        "reason": ("records the untracked local originals and the allowlist wording; "
                   "the on-disk check skips when the originals are absent"),
        "references": {
            "production_path_literal": (
                "\"corpus_raw/PMID_31096736/article.xml\",",
                "\"corpus_raw/PMID_31096736/source.json\",",
                "\"直肠癌文献爬取/tasks.sqlite\",",
                "for expected in (\"corpus_raw/\", \"tasks.sqlite\", \"chroma_db\", \"backups/\"):",
            ),
        },
    },
    "test_release_scope": {
        "reason": ("the inspected history ledger and the ignore-rule checks: these are "
                   "the paths the release boundary must keep out, never reads"),
        "references": {
            "production_path_literal": (
                "\"index/manifest.json\",",
                "\"papers/索引信息.csv\",",
                "\"corpus_raw/PMID_31096736/article.xml\": {",
                "\"corpus_raw/PMID_31096736/source.json\": {",
                "\"index/manifest.json\": {",
                "\"papers/索引信息.csv\": {",
                "\"直肠癌文献爬取/tasks.sqlite\": {",
                "\"corpus_raw/PMID_31096736/article.xml\",",
                "\"corpus_raw/PMID_31096736/source.json\",",
                "\"直肠癌文献爬取/tasks.sqlite\",",
                "\"index/manifest.json\",",
                "\"papers/索引信息.csv\",",
                "\"直肠癌文献爬取/tasks.sqlite\",",
                "self.assertTrue(rs.is_ignored(\"corpus_raw/PMID_31096736/article.xml\"),",
            ),
        },
    },
    "test_index_maintenance": {
        "reason": "creates only a synthetic converted/ archive directory under a temporary fixture root",
        "references": {
            "production_path_literal": (
                "for relative in (\".claude\", \"index\", \"converted/已入库\", \"papers\", \"flash-failed\"):",
            ),
        },
    },    "test_entry_points": {
        "reason": "the guarded-run assertions on the clean code directory",
        "references": {
            "production_path_literal": (
                "LOCAL_ONLY_DIRS = (\"index/chroma_db\", \"converted\", \"corpus_raw\", \".model-cache\",",
                "\"直肠癌文献爬取/tasks.sqlite\")",
            ),
        },
    },
}


def test_modules() -> list[str]:
    """Every importable test module name (``test_*.py``) under tests/."""
    return sorted(path.stem for path in TESTS_DIR.glob("test_*.py"))


def all_python_modules() -> list[str]:
    return sorted(path.stem for path in TESTS_DIR.glob("*.py"))


def module_path(name: str) -> Path:
    return TESTS_DIR / f"{name}.py"


def references(name: str) -> dict[str, tuple[str, ...]]:
    """Production-path references in a module, as label -> matched source lines."""
    found: dict[str, list[str]] = {}
    for line in module_path(name).read_text(encoding="utf-8").splitlines():
        for label, regex in PRODUCTION_REFERENCE_RES:
            if regex.search(line):
                found.setdefault(label, []).append(line.strip())
    return {label: tuple(lines) for label, lines in found.items()}
