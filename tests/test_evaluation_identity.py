"""Pure-synthetic regression tests for evaluator paper identity alignment."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "index"))
SPEC = importlib.util.spec_from_file_location(
    "evaluation_identity_evaluator", ROOT / "index" / "evaluate_retrieval.py"
)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


def judgement(source_file, relevance, doi="", **aliases):
    item = {
        "source_file": source_file,
        "source_id": source_file.removesuffix(".md"),
        "relevance": relevance,
    }
    if doi:
        item["doi"] = doi
    item.update(aliases)
    return item


def qrel(query, judgements):
    return {"query_id": query, "query": query, "judgements": judgements}


class EvaluationIdentityTests(unittest.TestCase):
    def evaluate_hits(self, row, hits):
        with patch.object(evaluator, "retrieve_literature", return_value=hits):
            return evaluator.evaluate([row], [5], retrieval_cache={})["per_query"][0]

    def test_missing_hit_doi_resolves_to_doi_judgement_identity(self):
        result = self.evaluate_hits(
            qrel("missing-hit-doi", [
                judgement("paper-a.md", 3, "10/example-a"),
            ]),
            [{"source": "paper-a.md", "doi": ""}],
        )

        self.assertEqual(result["recall"]["5"], 1.0)
        self.assertEqual(result["unique_relevant_paper_recall_at_50"], 1.0)
        self.assertEqual(result["mrr_at_10"], 1.0)
        self.assertEqual(result["ndcg"]["10"], 1.0)
        self.assertEqual(result["ndcg"]["20"], 1.0)

    def test_missing_judgement_doi_and_normalized_aliases_resolve(self):
        result = self.evaluate_hits(
            qrel("missing-qrel-doi", [
                judgement(
                    "Paper-A.MD", 3, source_path=r"archive\papers\Paper-A.MD"
                ),
            ]),
            [{"source": "retrieved/PAPER-A.md", "doi": "DOI:10/EXAMPLE-A"}],
        )

        self.assertEqual(result["recall"]["5"], 1.0)
        self.assertEqual(result["unique_relevant_paper_recall_at_50"], 1.0)
        self.assertEqual(result["mrr_at_10"], 1.0)
        self.assertEqual(result["ndcg"]["10"], 1.0)
        self.assertEqual(result["ndcg"]["20"], 1.0)

    def test_doi_prefix_and_case_are_normalized(self):
        result = self.evaluate_hits(
            qrel("doi-normalization", [
                judgement("paper-a.md", 3, "  DOI:10/Example-A  "),
            ]),
            [{"source": "elsewhere", "doi": "10/EXAMPLE-A"}],
        )

        self.assertEqual(result["recall"]["5"], 1.0)
        self.assertEqual(result["ndcg"]["10"], 1.0)

    def test_alias_duplicates_count_once_and_distinct_papers_remain_distinct(self):
        result = self.evaluate_hits(
            qrel("dedupe-before-cutoff", [
                judgement("paper-a.md", 3, "10/example-a"),
                judgement("paper-b.md", 2, "10/example-b"),
            ]),
            [
                {"source": r"index\paper-a.md", "doi": ""},
                {"source": "paper-a.md", "doi": "doi:10/EXAMPLE-A"},
                {"source": "paper-b.md", "doi": ""},
            ],
        )

        self.assertEqual(result["positive_papers"], 2)
        self.assertEqual(result["recall"]["5"], 1.0)
        self.assertEqual(result["unique_relevant_paper_recall_at_50"], 1.0)
        self.assertEqual(result["mrr_at_10"], 1.0)
        self.assertEqual(result["ndcg"]["10"], 1.0)
        self.assertLessEqual(result["ndcg"]["10"], 1.0)
        self.assertEqual(result["ndcg"]["20"], 1.0)

    def test_unjudged_first_candidate_keeps_its_rank(self):
        result = self.evaluate_hits(
            qrel("unjudged-rank", [
                judgement("paper-a.md", 3, "10/example-a"),
            ]),
            [
                {},
                {"source": "paper-a.md", "doi": ""},
            ],
        )

        self.assertEqual(result["mrr_at_10"], 0.5)
        self.assertEqual(result["recall"]["5"], 1.0)

    def test_conflicting_aliases_are_unjudged_and_do_not_merge_with_either_paper(self):
        result = self.evaluate_hits(
            qrel("conflicting-aliases", [
                judgement("paper-a.md", 3, "10/example-a"),
                judgement("paper-b.md", 2, "10/example-b"),
            ]),
            [
                {"source": "paper-b.md", "doi": "10/example-a"},
                {"source": "elsewhere", "doi": "10/example-a"},
                {"source": "paper-b.md", "doi": ""},
            ],
        )

        self.assertEqual(result["positive_papers"], 2)
        self.assertEqual(result["recall"]["5"], 1.0)
        self.assertEqual(result["unique_relevant_paper_recall_at_50"], 1.0)
        self.assertEqual(result["mrr_at_10"], 0.5)
        self.assertAlmostEqual(
            result["ndcg"]["10"], (3 / math.log2(3) + 2 / math.log2(4)) / (3 + 2 / math.log2(3))
        )
        self.assertLessEqual(result["ndcg"]["10"], 1.0)

    def test_graded_gain_and_no_positive_na_semantics_remain(self):
        graded = self.evaluate_hits(
            qrel("graded-gain", [
                judgement("paper-a.md", 3, "10/example-a"),
                judgement("paper-b.md", 2, "10/example-b"),
                judgement("paper-c.md", 1, "10/example-c"),
            ]),
            [
                {"source": "paper-b.md", "doi": ""},
                {"source": "paper-a.md", "doi": ""},
            ],
        )
        no_positive = self.evaluate_hits(
            qrel("partial-only", [judgement("paper-c.md", 1, "10/example-c")]),
            [{"source": "paper-c.md", "doi": ""}],
        )

        self.assertEqual(graded["recall"]["5"], 1.0)
        self.assertAlmostEqual(graded["ndcg"]["10"], (2 + 3 / math.log2(3)) / (3 + 2 / math.log2(3)))
        self.assertIsNone(no_positive["recall"]["5"])
        self.assertIsNone(no_positive["unique_relevant_paper_recall_at_50"])
        self.assertIsNone(no_positive["mrr_at_10"])
        self.assertIsNone(no_positive["ndcg"]["10"])
        self.assertIsNone(no_positive["ndcg"]["20"])


if __name__ == "__main__":
    unittest.main()
