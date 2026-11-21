"""Evaluate literature retrieval against the B2 graded Gold Set.

The qrels JSONL keeps the audited query metadata and three graded judgements
per query. Recall, MRR, and UniqueRelevantPaperRecall use relevance >= 2.
nDCG uses the approved graded gains: g(0)=0, g(1)=0, g(2)=2, g(3)=3.
Unjudged papers are not asserted to be irrelevant; they simply have no gain
in this pooled-candidate evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "index"))
from rag_core import retrieve_literature  # noqa: E402

POSITIVE_THRESHOLD = 2
NDCG_CUTOFFS = (10, 20)
UNIQUE_RECALL_CUTOFF = 50
GRADED_GAINS = {0: 0, 1: 0, 2: 2, 3: 3}
FORMAL_SPLITS = (
    "dev",
    "test",
    "overall",
    "human_manual",
    "ai_assisted_human_reviewed",
)


def normalise_key(value: str) -> str:
    value = str(value or "").strip().lower()
    if value.startswith("doi:"):
        value = value[4:].strip()
    return value


def _basename(value: str) -> str:
    return normalise_key(str(value or "").replace("\\", "/").rsplit("/", 1)[-1])


def result_aliases(item: dict) -> set[str]:
    """Return DOI/source aliases that can identify one retrieved paper."""
    aliases = set()
    doi = normalise_key(item.get("doi", ""))
    source = normalise_key(item.get("source", ""))
    if doi:
        aliases.add(doi)
    if source:
        aliases.add(source)
        aliases.add(_basename(source))
    return {alias for alias in aliases if alias}


def judgement_aliases(item: dict) -> set[str]:
    aliases = set()
    doi = normalise_key(item.get("doi", ""))
    if doi:
        aliases.add(doi)
    for field in ("source_id", "source_file", "source_path"):
        value = normalise_key(item.get(field, ""))
        if value:
            aliases.add(value)
            aliases.add(_basename(value))
    return {alias for alias in aliases if alias}


def paper_identity(item: dict, retrieved: bool = False) -> str:
    """Return the DOI-first identity used for unique-paper counting."""
    doi = normalise_key(item.get("doi", ""))
    if doi:
        return f"doi:{doi}"
    if retrieved:
        source = _basename(item.get("source", ""))
    else:
        source = _basename(item.get("source_file", ""))
        source = source or normalise_key(item.get("source_id", ""))
    return f"source:{source}" if source else ""


def row_judgements(row: dict) -> list[dict]:
    """Read B2 graded judgements, with legacy binary qrels compatibility."""
    if isinstance(row.get("judgements"), list):
        return row["judgements"]
    return [
        {"source_file": value, "relevance": POSITIVE_THRESHOLD}
        for value in row.get("relevant", [])
    ]


def _query_relevance(row: dict) -> tuple[dict[str, int], dict[str, set[str]]]:
    """Build canonical paper relevance and all qrel identities per alias."""
    relevance: dict[str, int] = {}
    aliases: dict[str, set[str]] = {}
    for judgement in row_judgements(row):
        identity = paper_identity(judgement)
        if not identity:
            continue
        score = int(judgement.get("relevance", 0))
        relevance[identity] = max(score, relevance.get(identity, 0))
        for alias in judgement_aliases(judgement):
            aliases.setdefault(alias, set()).add(identity)
    return relevance, aliases


def _conflict_identity(hit: dict) -> str:
    """Give an ambiguous hit a stable identity outside every judged paper."""
    doi = normalise_key(hit.get("doi", ""))
    source = _basename(hit.get("source", ""))
    parts = []
    if doi:
        parts.append(f"doi={doi}")
    if source:
        parts.append(f"source={source}")
    return "conflict:" + "|".join(parts)


def _resolved_identity(hit: dict, aliases: dict[str, set[str]]) -> str:
    """Resolve all known hit aliases together, rejecting cross-paper conflicts."""
    matches = {
        identity
        for alias in result_aliases(hit)
        for identity in aliases.get(alias, ())
    }
    if len(matches) == 1:
        return next(iter(matches))
    if len(matches) > 1:
        # A DOI/source disagreement is unjudged. Its conflict key must not
        # collide with either judged identity and is independent of set order.
        return _conflict_identity(hit)
    return paper_identity(hit, retrieved=True)


def _unique_hits(
    hits: list[dict],
    aliases: dict[str, set[str]],
    resolution_trace: list[dict] | None = None,
) -> list[tuple[dict, str]]:
    """Resolve and deduplicate papers before cutoffs, preserving rank order."""
    unique = []
    seen = set()
    for rank, hit in enumerate(hits):
        identity = _resolved_identity(hit, aliases)
        if not identity:
            # A candidate without DOI/source still occupies its returned rank;
            # it has no aliases with which it could safely merge elsewhere.
            identity = f"unidentified:rank:{rank}"
        kept = identity not in seen
        resolved_rank = None
        if not kept:
            if resolution_trace is not None:
                resolution_trace.append({
                    "retrieval_rank": rank + 1,
                    "resolved_identity": identity,
                    "kept": False,
                    "deduplicated_rank": None,
                })
            continue
        seen.add(identity)
        unique.append((hit, identity))
        resolved_rank = len(unique)
        if resolution_trace is not None:
            resolution_trace.append({
                "retrieval_rank": rank + 1,
                "resolved_identity": identity,
                "kept": True,
                "deduplicated_rank": resolved_rank,
            })
    return unique


def _record_score(hit: dict, trace_hit: dict | None = None):
    score = hit.get("score")
    if score is None and trace_hit is not None:
        score = trace_hit.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return score if math.isfinite(float(score)) else None
    return None


def _ranking_record(
    row: dict,
    row_index: int,
    raw_hits: list[dict],
    unique_hits: list[tuple[dict, str]],
    resolution_trace: list[dict],
    retrieval_k: int,
    retrieval_trace: dict | None,
) -> dict:
    trace_hits = (retrieval_trace or {}).get("ordered_hits", [])
    before_resolution = []
    for index, hit in enumerate(raw_hits):
        trace_hit = trace_hits[index] if index < len(trace_hits) else None
        item = {
            "rank": index + 1,
            "pre_f06_identity": paper_identity(hit, retrieved=True) or None,
            "doi": str(hit.get("doi") or ""),
            "source": str(hit.get("source") or ""),
        }
        score = _record_score(hit, trace_hit)
        if score is not None:
            item["score"] = score
        before_resolution.append(item)

    representative_ranks = {
        item["deduplicated_rank"]: item["retrieval_rank"]
        for item in resolution_trace
        if item["kept"]
    }
    after_resolution = []
    for index, (hit, identity) in enumerate(unique_hits):
        retrieval_rank = representative_ranks.get(index + 1, index + 1)
        trace_index = retrieval_rank - 1
        trace_hit = trace_hits[trace_index] if trace_index < len(trace_hits) else None
        item = {
            "rank": index + 1,
            "identity": identity,
            "doi": str(hit.get("doi") or ""),
            "source": str(hit.get("source") or ""),
        }
        score = _record_score(hit, trace_hit)
        if score is not None:
            item["score"] = score
        after_resolution.append(item)

    retrieval = retrieval_trace or {}
    trace_observed = bool(retrieval_trace)
    return {
        "query_id": str(row.get("query_id") or f"row:{row_index + 1}"),
        "retrieval": {
            "returned_count": len(raw_hits),
            "literature_limit": retrieval_k,
            "candidate_budget": retrieval.get("candidate_budget"),
            "candidate_budget_unit": "passages",
            "passage_candidates_returned": retrieval.get(
                "passage_candidates_returned"
            ),
            "literature_candidates_before_limit": retrieval.get(
                "literature_candidates_before_limit"
            ),
            "truncation_position": retrieval.get("truncation_position"),
            "truncated_literature_count": retrieval.get(
                "truncated_literature_count"
            ),
            "truncation_observed": trace_observed,
        },
        "retrieved_rankings": before_resolution,
        "identity_resolution": resolution_trace,
        "deduplicated_rankings": after_resolution,
    }


def ranking_artifact(records: list[dict]) -> dict:
    """Build the optional identity-only local ranking record artifact."""
    return {
        "schema_version": 1,
        "ranking_level": "literature_after_retrieval_aggregation",
        "ranking_is_passage_level": False,
        "identity_stages": {
            "retrieved_rankings": "before evaluator alias resolution",
            "deduplicated_rankings": "after evaluator alias resolution and deduplication",
        },
        "records": records,
    }


def _hit_relevance(identity: str, relevance: dict[str, int]) -> int:
    """Read relevance through the same resolved identity used for deduping."""
    return relevance.get(identity, 0)


def ndcg(graded_relevance: list[int], ideal_relevance: list[int], k: int) -> float | None:
    """Return graded nDCG, or None when IDCG is zero/not evaluable."""
    actual = sum(
        GRADED_GAINS.get(value, 0) / math.log2(index + 2)
        for index, value in enumerate(graded_relevance[:k])
    )
    ideal = sum(
        GRADED_GAINS.get(value, 0) / math.log2(index + 2)
        for index, value in enumerate(sorted(ideal_relevance, reverse=True)[:k])
    )
    return actual / ideal if ideal else None


def _macro(values: list[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(usable) / len(usable), 4) if usable else None


def _selected_rows(qrels: list[dict], split: str) -> list[dict]:
    if split == "overall":
        return list(qrels)
    if split in {"dev", "test"}:
        return [row for row in qrels if row.get("split") == split]
    if split in {"human_manual", "ai_assisted_human_reviewed"}:
        return [row for row in qrels if row.get("review_method") == split]
    raise ValueError(f"unsupported split: {split}")


def evaluate(
    qrels: list[dict],
    ks: list[int],
    retrieval_cache: dict[str, list[dict]] | None = None,
    ranking_records: list[dict] | None = None,
    retrieval_trace_cache: dict[str, dict] | None = None,
    **retrieve_kwargs,
) -> dict:
    """Evaluate one selected qrels list using pooled graded judgements."""
    if not ks:
        raise ValueError("at least one recall cutoff is required")
    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    if not ks:
        raise ValueError("at least one positive recall cutoff is required")
    retrieval_cache = retrieval_cache if retrieval_cache is not None else {}
    retrieval_trace_cache = (
        retrieval_trace_cache if retrieval_trace_cache is not None else {}
    )
    retrieval_k = max(max(ks), UNIQUE_RECALL_CUTOFF)
    per_query = []
    recall_values = {str(k): [] for k in ks}
    mrr_values = []
    ndcg_values = {str(k): [] for k in NDCG_CUTOFFS}
    unique_recall_values = []
    no_positive_ids = []
    strict_negative_ids = []
    partial_only_ids = []

    for row_index, row in enumerate(qrels):
        query_id = row.get("query_id", row.get("query", ""))
        relevance, aliases = _query_relevance(row)
        positive = {identity for identity, score in relevance.items()
                    if score >= POSITIVE_THRESHOLD}
        max_relevance = max(relevance.values(), default=0)
        if not positive:
            no_positive_ids.append(query_id)
            if max_relevance == 0:
                strict_negative_ids.append(query_id)
            else:
                partial_only_ids.append(query_id)
        query = row["query"]
        if query not in retrieval_cache:
            retrieval_trace = {} if ranking_records is not None else None
            call_kwargs = retrieve_kwargs
            if retrieval_trace is not None:
                call_kwargs = dict(retrieve_kwargs)
                call_kwargs["ranking_trace"] = retrieval_trace
            retrieval_cache[query] = retrieve_literature(
                query, k=retrieval_k, **call_kwargs
            )
            if retrieval_trace is not None:
                retrieval_trace_cache[query] = retrieval_trace
        retrieval_trace = retrieval_trace_cache.get(query)
        resolution_trace = [] if ranking_records is not None else None
        hits = _unique_hits(
            retrieval_cache[query], aliases, resolution_trace=resolution_trace
        )
        if ranking_records is not None:
            ranking_records.append(_ranking_record(
                row,
                row_index,
                retrieval_cache[query],
                hits,
                resolution_trace,
                retrieval_k,
                retrieval_trace,
            ))
        graded_hits = [
            (identity, _hit_relevance(identity, relevance))
            for _, identity in hits
        ]
        query_result = {
            "query_id": query_id,
            "query": query,
            "positive_papers": len(positive),
            "max_relevance": max_relevance,
            "excluded_no_positive": not positive,
            "recall": {},
            "ndcg": {},
        }
        for k in ks:
            if positive:
                found = len({
                    identity
                    for identity, score in graded_hits[:k]
                    if score >= POSITIVE_THRESHOLD
                } & positive)
                value = found / len(positive)
                recall_values[str(k)].append(value)
            else:
                value = None
            query_result["recall"][str(k)] = value

        if positive:
            first = next(
                (index + 1 for index, (_, score) in enumerate(graded_hits[:10])
                 if score >= POSITIVE_THRESHOLD),
                None,
            )
            mrr = 1 / first if first else 0.0
            mrr_values.append(mrr)
            unique_found = len({
                identity
                for identity, score in graded_hits[:UNIQUE_RECALL_CUTOFF]
                if score >= POSITIVE_THRESHOLD
            } & positive)
            unique_recall_values.append(unique_found / len(positive))
        else:
            mrr = None
        query_result["mrr_at_10"] = mrr

        ideal_relevance = list(relevance.values())
        for k in NDCG_CUTOFFS:
            value = ndcg([score for _, score in graded_hits], ideal_relevance, k)
            query_result["ndcg"][str(k)] = value
            ndcg_values[str(k)].append(value)
        query_result["unique_relevant_paper_recall_at_50"] = (
            unique_recall_values[-1] if positive else None
        )
        per_query.append(query_result)

    total = len(qrels)
    result = {
        "total_queries": total,
        "evaluable_queries": total - len(no_positive_ids),
        "excluded_no_positive_queries": len(no_positive_ids),
        "ndcg_evaluable_queries": sum(
            value is not None for value in ndcg_values["10"]
        ),
        "ndcg_total_queries": total,
        "strict_negative_queries": len(strict_negative_ids),
        "partial_only_queries": len(partial_only_ids),
        "strict_negative_query_ids": strict_negative_ids,
        "partial_only_query_ids": partial_only_ids,
        "metrics": {
            "recall": {key: _macro(values) for key, values in recall_values.items()},
            "mrr_at_10": _macro(mrr_values),
            "ndcg": {key: _macro(values) for key, values in ndcg_values.items()},
            "unique_relevant_paper_recall_at_50": _macro(unique_recall_values),
        },
        "per_query": per_query,
    }
    return result


def evaluate_splits(
    qrels: list[dict],
    split_names: list[str],
    ks: list[int],
    ranking_records: list[dict] | None = None,
    **retrieve_kwargs,
) -> dict:
    """Evaluate several views in one run, sharing retrieval results by query."""
    cache: dict[str, list[dict]] = {}
    trace_cache: dict[str, dict] = {}
    splits = {}
    records_by_query_id = {}
    for split in split_names:
        split_records = [] if ranking_records is not None else None
        splits[split] = evaluate(
            _selected_rows(qrels, split), ks, retrieval_cache=cache,
            ranking_records=split_records,
            retrieval_trace_cache=trace_cache,
            **retrieve_kwargs
        )
        if split_records is not None:
            for record in split_records:
                records_by_query_id.setdefault(record["query_id"], record)
    if ranking_records is not None:
        ranking_records.extend(records_by_query_id.values())
    return {"splits": splits, "retrieval_queries": len(cache)}


def load_qrels(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate literature retrieval")
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument(
        "--k", default="5,10,20,50", help="comma-separated Recall cutoffs"
    )
    parser.add_argument(
        "--split", default="overall",
        choices=["all", *FORMAL_SPLITS],
        help="overall/dev/test or a review_method diagnostic subset; all runs every view",
    )
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="manifest paired with --db (required to evaluate a staging index)",
    )
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mac-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="write the full result (including per-query values) to this JSON file",
    )
    parser.add_argument(
        "--rankings-output", type=Path, default=None,
        help=("write identity-only literature rankings to a separate local JSON "
              "artifact; does not include query text or passages"),
    )
    args = parser.parse_args()
    ks = sorted({int(value) for value in args.k.split(",") if int(value) > 0})
    qrels = load_qrels(args.qrels)
    kwargs = {
        "db_dir": args.db,
        "model_dir": args.model,
        "device": args.device,
        "mac_only": args.mac_only,
        "manifest_path": args.manifest,
    }
    ranking_records = [] if args.rankings_output else None
    if args.split == "all":
        result = evaluate_splits(
            qrels, list(FORMAL_SPLITS), ks,
            ranking_records=ranking_records,
            **kwargs,
        )
    else:
        result = {
            "splits": {
                args.split: evaluate(
                    _selected_rows(qrels, args.split), ks,
                    ranking_records=ranking_records,
                    **kwargs,
                )
            }
        }
    result["config"] = {
        "qrels": str(args.qrels),
        "db": str(args.db) if args.db else None,
        "manifest": str(args.manifest) if args.manifest else None,
        "model": str(args.model) if args.model else None,
        "device": args.device,
        "mac_only": args.mac_only,
        "k": ks,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.rankings_output:
        args.rankings_output.parent.mkdir(parents=True, exist_ok=True)
        args.rankings_output.write_text(
            json.dumps(ranking_artifact(ranking_records), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
