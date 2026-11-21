"""Pure-synthetic tests for optional evaluator ranking records."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "index"))
import rag_core  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "evaluation_rankings_evaluator", ROOT / "index" / "evaluate_retrieval.py"
)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def synthetic_hit(source, doi="", score=None):
    hit = {
        "source": source,
        "doi": doi,
        "title": "SYNTHETIC TITLE MUST NOT BE SAVED",
        "snippet": "SYNTHETIC SNIPPET MUST NOT BE SAVED",
        "embedding": [0.125, 0.25],
        "text": "SYNTHETIC BODY MUST NOT BE SAVED",
    }
    if score is not None:
        hit["score"] = score
    return hit


def synthetic_qrels():
    return [
        {
            "query_id": "synthetic-q1",
            "query": "SYNTHETIC QUERY TEXT ONE",
            "split": "dev",
            "judgements": [
                {"source_file": "paper-a.md", "doi": "10/synthetic-a", "relevance": 3},
                {"source_file": "paper-b.md", "doi": "10/synthetic-b", "relevance": 2},
            ],
        },
        {
            "query_id": "synthetic-q2",
            "query": "SYNTHETIC QUERY TEXT TWO",
            "split": "test",
            "judgements": [
                {"source_file": "paper-c.md", "doi": "10/synthetic-c", "relevance": 1},
            ],
        },
    ]


def synthetic_retrieval(query, k, ranking_trace=None, **_kwargs):
    if query == "SYNTHETIC QUERY TEXT ONE":
        hits = [
            synthetic_hit("paper-a.md", score=0.91),
            synthetic_hit("paper-a.md", "10/synthetic-a", 0.88),
            synthetic_hit("paper-x.md", "10/synthetic-x", 0.72),
        ]
    else:
        hits = [synthetic_hit("paper-c.md", "10/synthetic-c", 0.81)]
    selected = hits[:k]
    if ranking_trace is not None:
        ranking_trace.update({
            "candidate_budget": max(40, k * 5),
            "passage_candidates_returned": len(hits),
            "literature_candidates_before_limit": len(hits),
            "returned_count": len(selected),
            "truncation_position": k if len(hits) > k else None,
            "truncated_literature_count": max(0, len(hits) - k),
            "ordered_hits": [
                {"rank": rank, "doi": hit["doi"], "source": hit["source"],
                 "score": hit.get("score")}
                for rank, hit in enumerate(selected, 1)
            ],
        })
    return selected


class EvaluationRankingRecordTests(unittest.TestCase):
    def evaluate_synthetic(self, record):
        calls = []

        def retrieve(query, k, **kwargs):
            calls.append(query)
            return synthetic_retrieval(query, k, **kwargs)

        records = [] if record else None
        with patch.object(evaluator, "retrieve_literature", side_effect=retrieve):
            result = evaluator.evaluate(
                synthetic_qrels(), [5], retrieval_cache={},
                ranking_records=records,
                retrieval_trace_cache={},
            )
        return result, records, calls

    def test_record_toggle_preserves_metrics_and_retrieval_count(self):
        plain, _, plain_calls = self.evaluate_synthetic(record=False)
        recorded, records, recorded_calls = self.evaluate_synthetic(record=True)

        self.assertEqual(plain, recorded)
        self.assertEqual(len(plain_calls), 2)
        self.assertEqual(len(recorded_calls), 2)
        self.assertEqual(records[0]["retrieval"]["candidate_budget"], 250)
        self.assertTrue(records[0]["retrieval"]["truncation_observed"])

    def test_saved_identities_replay_metrics_and_denominators_offline(self):
        original, records, _ = self.evaluate_synthetic(record=True)
        records_by_id = {record["query_id"]: record for record in records}
        offline_cache = {}
        for row in synthetic_qrels():
            offline_cache[row["query"]] = [
                {key: hit[key] for key in ("doi", "source") if hit.get(key)}
                for hit in records_by_id[row["query_id"]]["retrieved_rankings"]
            ]

        with patch.object(evaluator, "retrieve_literature",
                          side_effect=AssertionError("offline replay retrieved again")):
            replayed = evaluator.evaluate(
                synthetic_qrels(), [5], retrieval_cache=offline_cache
            )

        self.assertEqual(replayed, original)
        for field in (
            "total_queries", "evaluable_queries", "excluded_no_positive_queries",
            "ndcg_evaluable_queries", "ndcg_total_queries",
            "strict_negative_queries", "partial_only_queries",
        ):
            self.assertEqual(replayed[field], original[field])
        self.assertEqual(original["evaluable_queries"], 1)
        self.assertEqual(original["excluded_no_positive_queries"], 1)

    def test_artifact_contains_identity_stages_but_no_text_or_vectors(self):
        _, records, _ = self.evaluate_synthetic(record=True)
        record = records[0]
        serialized = json.dumps(evaluator.ranking_artifact(records), ensure_ascii=False)

        self.assertFalse(evaluator.ranking_artifact(records)["ranking_is_passage_level"])
        self.assertEqual(record["query_id"], "synthetic-q1")
        self.assertEqual(record["retrieved_rankings"][0]["pre_f06_identity"],
                         "source:paper-a.md")
        self.assertEqual(record["identity_resolution"][0]["resolved_identity"],
                         "doi:10/synthetic-a")
        self.assertFalse(record["identity_resolution"][1]["kept"])
        self.assertEqual([x["rank"] for x in record["deduplicated_rankings"]], [1, 2])
        self.assertEqual(record["retrieved_rankings"][0]["score"], 0.91)
        for forbidden in (
            "SYNTHETIC QUERY TEXT", "SYNTHETIC TITLE", "SYNTHETIC SNIPPET",
            "SYNTHETIC BODY", "embedding", '"query"', '"title"',
            '"snippet"', '"text"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_core_trace_reports_same_call_budget_and_actual_truncation(self):
        raw = []
        for source, doi, score in (
            ("paper-a.md", "10/synthetic-a", 0.9),
            ("paper-a.md", "10/synthetic-a", 0.8),
            ("paper-b.md", "10/synthetic-b", 0.7),
            ("paper-c.md", "10/synthetic-c", 0.6),
        ):
            raw.append({
                "source": source, "doi": doi, "score": score,
                "title": "synthetic", "year": "2026", "journal": "fixture",
                "authors": "fixture", "mac": "no", "section": "Results",
                "snippet": "private synthetic snippet",
            })
        trace = {}
        with patch.object(rag_core, "retrieve", return_value=raw) as retrieve:
            result = rag_core.retrieve_literature(
                "synthetic trace query", k=2, candidate_k=4, ranking_trace=trace
            )
        expected = rag_core.summary_list(rag_core.dedupe_by_doi(raw)[:2])

        self.assertEqual(retrieve.call_count, 1)
        self.assertEqual(result, expected)
        self.assertEqual(trace["candidate_budget"], 4)
        self.assertEqual(trace["passage_candidates_returned"], 4)
        self.assertEqual(trace["literature_candidates_before_limit"], 3)
        self.assertEqual(trace["returned_count"], 2)
        self.assertEqual(trace["truncation_position"], 2)
        self.assertEqual(trace["truncated_literature_count"], 1)
        self.assertNotIn("snippet", json.dumps(trace))
        self.assertNotIn("embedding", json.dumps(trace))


if __name__ == "__main__":
    unittest.main()
