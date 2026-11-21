"""Offline regression tests for the B2 Gold Set and evaluator."""

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluator = load_module("b2_evaluate_retrieval", ROOT / "index" / "evaluate_retrieval.py")
validator = load_module("b2_validate_gold_set", ROOT / "evaluation" / "validate_gold_set.py")


def judgement(source_file, doi, relevance):
    return {
        "source_id": source_file.removesuffix(".md"),
        "source_file": source_file,
        "doi": doi,
        "relevance": relevance,
    }


def row(query_id, query, split, review_method, judgements):
    return {
        "query_id": query_id,
        "query": query,
        "split": split,
        "review_method": review_method,
        "judgements": judgements,
    }


class B2EvaluatorTests(unittest.TestCase):
    def test_graded_ndcg_and_no_positive_exclusion(self):
        qrels = [
            row("Q1", "graded", "dev", "human_manual", [
                judgement("a.md", "10/a", 3),
                judgement("b.md", "10/b", 2),
                judgement("c.md", "10/c", 1),
            ]),
            row("Q2", "strict negative", "dev", "human_manual", [
                judgement("d.md", "10/d", 0),
                judgement("e.md", "10/e", 0),
                judgement("f.md", "10/f", 0),
            ]),
        ]
        calls = []

        def fake_retrieve(query, **kwargs):
            calls.append((query, kwargs["k"]))
            if query == "graded":
                return [
                    {"source": "a.md", "doi": "10/a"},
                    {"source": "duplicate-a.md", "doi": "10/a"},
                    {"source": "unjudged.md", "doi": "10/x"},
                    {"source": "b.md", "doi": "10/b"},
                ]
            return []

        original = evaluator.retrieve_literature
        evaluator.retrieve_literature = fake_retrieve
        try:
            result = evaluator.evaluate(qrels, [5, 10, 20, 50])
        finally:
            evaluator.retrieve_literature = original

        self.assertEqual(result["total_queries"], 2)
        self.assertEqual(result["evaluable_queries"], 1)
        self.assertEqual(result["excluded_no_positive_queries"], 1)
        self.assertEqual(result["strict_negative_queries"], 1)
        self.assertEqual(result["partial_only_queries"], 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["metrics"]["recall"]["5"], 1.0)
        self.assertEqual(result["metrics"]["mrr_at_10"], 1.0)
        self.assertIsNotNone(result["metrics"]["ndcg"]["10"])
        self.assertLess(result["metrics"]["ndcg"]["10"], 1.0)
        self.assertEqual(evaluator.ndcg([1], [1], 10), None)
        self.assertEqual(evaluator.ndcg([0], [0], 10), None)
        self.assertEqual(evaluator.ndcg([2], [2], 10), 1.0)

    def test_split_views_share_retrieval_cache_and_unique_paper_recall(self):
        qrels = [
            row("Q1", "first", "dev", "human_manual", [
                judgement("a.md", "10/a", 2),
                judgement("b.md", "10/b", 0),
                judgement("c.md", "10/c", 0),
            ]),
            row("Q2", "second", "test", "ai_assisted_human_reviewed", [
                judgement("d.md", "10/d", 2),
                judgement("e.md", "10/e", 0),
                judgement("f.md", "10/f", 0),
            ]),
        ]
        calls = []

        def fake_retrieve(query, **kwargs):
            calls.append(query)
            return [{"source": "a.md", "doi": "10/a"}] if query == "first" else []

        original = evaluator.retrieve_literature
        evaluator.retrieve_literature = fake_retrieve
        try:
            result = evaluator.evaluate_splits(
                qrels,
                ["dev", "test", "overall", "human_manual",
                 "ai_assisted_human_reviewed"],
                [5, 10, 20, 50],
            )
        finally:
            evaluator.retrieve_literature = original

        self.assertEqual(result["retrieval_queries"], 2)
        self.assertEqual(sorted(calls), ["first", "second"])
        self.assertEqual(result["splits"]["dev"]["total_queries"], 1)
        self.assertEqual(result["splits"]["test"]["total_queries"], 1)
        self.assertEqual(
            result["splits"]["human_manual"]["metrics"]
            ["unique_relevant_paper_recall_at_50"], 1.0
        )
        self.assertEqual(
            result["splits"]["ai_assisted_human_reviewed"]["metrics"]
            ["unique_relevant_paper_recall_at_50"], 0.0
        )


class B2GoldSetValidationTests(unittest.TestCase):
    def test_frozen_gold_set_passes(self):
        result = validator.validate(ROOT)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["query_count"], 100)
        self.assertEqual(result["judgement_count"], 300)
        self.assertEqual(result["split_counts"], {"test": 30, "dev": 70})
        self.assertEqual(result["no_positive_query_count"], 14)
        self.assertEqual(result["strict_negative_query_count"], 5)
        self.assertEqual(result["partial_only_query_count"], 9)


if __name__ == "__main__":
    unittest.main()
