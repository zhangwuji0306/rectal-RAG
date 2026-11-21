"""Validate the frozen B2 Gold Set, qrels, split manifest, and test freeze."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APPROVED_TEST_IDS = (
    "B2-Q0001", "B2-Q0004", "B2-Q0008", "B2-Q0010", "B2-Q0014",
    "B2-Q0020", "B2-Q0022", "B2-Q0024", "B2-Q0031", "B2-Q0032",
    "B2-Q0037", "B2-Q0038", "B2-Q0041", "B2-Q0043", "B2-Q0044",
    "B2-Q0050", "B2-Q0053", "B2-Q0058", "B2-Q0061", "B2-Q0066",
    "B2-Q0067", "B2-Q0070", "B2-Q0071", "B2-Q0078", "B2-Q0079",
    "B2-Q0080", "B2-Q0082", "B2-Q0088", "B2-Q0095", "B2-Q0097",
)
EXPECTED_STYLE_TEST = {
    "abbreviation": 6,
    "keyword": 7,
    "long": 6,
    "short": 6,
    "synonym": 5,
}
EXPECTED_REVIEW_TEST = {
    "human_manual": 6,
    "ai_assisted_human_reviewed": 24,
}
EXPECTED_TEST_CATEGORIES = {
    "MRI staging", "EMVI / CRM", "黏液表型", "TNT / CRT", "pCR / TRG",
    "DFS / OS / recurrence", "CEA / MSI", "radiomics", "deep learning",
    "新辅助治疗", "手术", "病理", "guidelines", "trial names", "精确术语",
    "宽泛临床问题", "guidelines / trial names",
}


class GoldSetValidationError(ValueError):
    pass


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise GoldSetValidationError(f"missing JSONL: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise GoldSetValidationError(f"invalid JSON at {path}:{line_number}: {exc}")
    return rows


def _read_freeze(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start = text.find("```json")
    if start < 0:
        raise GoldSetValidationError("test_freeze.md has no json freeze block")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end < 0:
        raise GoldSetValidationError("test_freeze.md has an unterminated json block")
    try:
        value = json.loads(text[start:end])
    except json.JSONDecodeError as exc:
        raise GoldSetValidationError(f"invalid freeze JSON: {exc}")
    if not isinstance(value, dict):
        raise GoldSetValidationError("freeze JSON must be an object")
    return value


def _assert_equal(errors: list[str], label: str, actual, expected) -> None:
    if actual != expected:
        _fail(errors, f"{label}: expected {expected!r}, got {actual!r}")


def validate(root: Path = ROOT) -> dict:
    errors: list[str] = []
    annotation_path = root / "B2_gold_annotations.jsonl"
    qrels_path = root / "evaluation" / "queries.jsonl"
    manifest_path = root / "evaluation" / "split_manifest.json"
    freeze_path = root / "evaluation" / "test_freeze.md"
    annotations = _load_jsonl(annotation_path)
    qrels = _load_jsonl(qrels_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("manifest root must be an object")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise GoldSetValidationError(f"invalid split_manifest.json: {exc}")
    freeze = _read_freeze(freeze_path)

    _assert_equal(errors, "query count", len(annotations), 100)
    _assert_equal(errors, "judgement count", sum(
        len(row.get("judgements", [])) for row in annotations
    ), 300)
    _assert_equal(errors, "qrels row count", len(qrels), len(annotations))

    ids = [row.get("query_id") for row in annotations]
    texts = [row.get("query") for row in annotations]
    if len(set(ids)) != len(ids):
        _fail(errors, "duplicate query_id")
    if len(set(texts)) != len(texts):
        _fail(errors, "duplicate query text")
    if any(not isinstance(value, str) or not value for value in ids):
        _fail(errors, "query_id must be non-empty strings")
    if any(not isinstance(value, str) or not value.strip() for value in texts):
        _fail(errors, "query must be non-empty strings")

    split_counts = Counter(row.get("split") for row in annotations)
    _assert_equal(errors, "split counts", dict(split_counts), {"dev": 70, "test": 30})
    test_ids = [row["query_id"] for row in annotations if row.get("split") == "test"]
    _assert_equal(errors, "approved test IDs", test_ids, list(APPROVED_TEST_IDS))

    test_rows = [row for row in annotations if row.get("split") == "test"]
    _assert_equal(errors, "test style counts",
                  dict(Counter(row.get("style") for row in test_rows)),
                  EXPECTED_STYLE_TEST)
    _assert_equal(errors, "test review_method counts",
                  dict(Counter(row.get("review_method") for row in test_rows)),
                  EXPECTED_REVIEW_TEST)
    categories = Counter(row.get("category") for row in test_rows)
    if set(categories) != EXPECTED_TEST_CATEGORIES or any(
        categories.get(category, 0) < 1 for category in EXPECTED_TEST_CATEGORIES
    ):
        _fail(errors, f"test category coverage mismatch: {dict(categories)}")

    no_positive = []
    strict_negative = []
    partial_only = []
    row_ids = set()
    for row in annotations:
        required = (
            "query_id", "query", "query_source", "category", "style", "split",
            "annotator", "annotated_at", "review_method", "status", "judgements",
        )
        for field in required:
            if field not in row or row[field] in (None, ""):
                _fail(errors, f"{row.get('query_id')}: missing {field}")
        judgements = row.get("judgements")
        if not isinstance(judgements, list) or len(judgements) != 3:
            _fail(errors, f"{row.get('query_id')}: judgement count is not 3")
            continue
        ranks = []
        source_ids = set()
        relevance_values = []
        for judgement in judgements:
            for field in ("row_id", "candidate_rank", "source_id", "source_file",
                          "source_path", "relevance", "rationale", "evidence"):
                if field not in judgement or judgement[field] in (None, ""):
                    _fail(errors, f"{row['query_id']}: judgement missing {field}")
            row_id = judgement.get("row_id")
            if row_id in row_ids:
                _fail(errors, f"duplicate judgement row_id: {row_id}")
            row_ids.add(row_id)
            ranks.append(judgement.get("candidate_rank"))
            source_id = judgement.get("source_id")
            if source_id in source_ids:
                _fail(errors, f"{row['query_id']}: duplicate source_id {source_id}")
            source_ids.add(source_id)
            relevance = judgement.get("relevance")
            if isinstance(relevance, bool) or relevance not in (0, 1, 2, 3):
                _fail(errors, f"{row['query_id']}: invalid relevance {relevance!r}")
            relevance_values.append(relevance)
            source_path = Path(str(judgement.get("source_path", "")))
            if source_path.is_absolute() or ".." in source_path.parts:
                _fail(errors, f"{row['query_id']}: source_path must be relative")
            elif not (root / source_path).is_file():
                _fail(errors, f"{row['query_id']}: missing source_path {source_path}")
            evidence = judgement.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                _fail(errors, f"{row['query_id']}: evidence must be non-empty")
            for item in evidence if isinstance(evidence, list) else []:
                if not isinstance(item, dict) or not str(item.get("path", "")).strip():
                    _fail(errors, f"{row['query_id']}: invalid evidence path")
                    continue
                evidence_path = Path(item["path"])
                if evidence_path.is_absolute() or ".." in evidence_path.parts:
                    _fail(errors, f"{row['query_id']}: evidence path must be relative")
                elif not (root / evidence_path).is_file():
                    _fail(errors, f"{row['query_id']}: missing evidence path {evidence_path}")
                if not str(item.get("excerpt", "")).strip():
                    _fail(errors, f"{row['query_id']}: empty evidence excerpt")
        if sorted(ranks) != [1, 2, 3]:
            _fail(errors, f"{row['query_id']}: candidate ranks are not 1,2,3")
        if max(relevance_values, default=0) < 2:
            no_positive.append(row["query_id"])
            if max(relevance_values, default=0) == 0:
                strict_negative.append(row["query_id"])
            else:
                partial_only.append(row["query_id"])

    _assert_equal(errors, "no-positive query count", len(no_positive), 14)
    _assert_equal(errors, "strict-negative query count", len(strict_negative), 5)
    _assert_equal(errors, "0/1-only query count", len(partial_only), 9)
    _assert_equal(errors, "test no-positive count",
                  sum(query_id in test_ids for query_id in no_positive), 4)
    _assert_equal(errors, "test strict-negative count",
                  sum(query_id in test_ids for query_id in strict_negative), 1)

    annotation_by_id = {row["query_id"]: row for row in annotations}
    qrels_by_id = {row.get("query_id"): row for row in qrels}
    if set(qrels_by_id) != set(annotation_by_id):
        _fail(errors, "qrels query_id set differs from annotations")
    for query_id, row in annotation_by_id.items():
        if qrels_by_id.get(query_id) != row:
            _fail(errors, f"qrels lost or changed annotation fields for {query_id}")

    manifest_test_ids = manifest.get("test_query_ids")
    freeze_test_ids = freeze.get("test_query_ids")
    _assert_equal(errors, "manifest test IDs", manifest_test_ids, list(APPROVED_TEST_IDS))
    _assert_equal(errors, "freeze test IDs", freeze_test_ids, list(APPROVED_TEST_IDS))
    _assert_equal(errors, "manifest query count", manifest.get("query_count"), 100)
    _assert_equal(errors, "manifest judgement count", manifest.get("judgement_count"), 300)
    _assert_equal(errors, "manifest split counts", manifest.get("split_counts"),
                  {"dev": 70, "test": 30})
    _assert_equal(errors, "freeze split counts", freeze.get("split_counts"),
                  {"dev": 70, "test": 30})
    _assert_equal(errors, "manifest/freeze positive threshold",
                  manifest.get("positive_threshold"), 2)
    _assert_equal(errors, "freeze/manifest positive threshold",
                  freeze.get("positive_threshold"), manifest.get("positive_threshold"))

    if errors:
        raise GoldSetValidationError("\n".join(f"- {error}" for error in errors))
    return {
        "query_count": len(annotations),
        "judgement_count": len(row_ids),
        "split_counts": dict(split_counts),
        "test_query_ids": test_ids,
        "no_positive_query_count": len(no_positive),
        "strict_negative_query_count": len(strict_negative),
        "partial_only_query_count": len(partial_only),
        "status": "PASS",
    }


def main() -> int:
    try:
        result = validate()
    except (GoldSetValidationError, OSError) as exc:
        print(f"B2 Gold Set validation: FAIL\n{exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
