"""Query the built Chroma index (thin CLI over rag_core).

Usage:
    python index/retrieve.py "问题或英文查询" [-k 8] [--mac-only] [--json]
    python index/retrieve.py "query" --db <chroma_db_dir> --model <model_dir> --device cuda

Any other project can reuse the same index programmatically:
    import sys; sys.path.insert(0, r"E:\\writing-rag\\index")
    from rag_core import retrieve, dedupe_by_doi, summary_list
"""

import argparse
import json
import sys

from rag_core import DEFAULT_DB_DIR, DEFAULT_MODEL_DIR, retrieve, retrieve_literature_mode

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="retrieval query (Chinese or English)")
    ap.add_argument("-k", type=int, default=8, help="top-k results (default 8)")
    ap.add_argument("--mac-only", action="store_true", help="only MAC-classified chunks")
    ap.add_argument("--json", action="store_true", help="dump raw JSON")
    ap.add_argument("--db", default=str(DEFAULT_DB_DIR), help="chroma_db directory")
    ap.add_argument("--model", default=str(DEFAULT_MODEL_DIR), help="local model directory")
    ap.add_argument("--device", default="cpu", help="torch device (cpu|cuda|auto)")
    ap.add_argument('--mode', choices=('dense-only', 'sparse-only', 'hybrid'), default='dense-only')
    ap.add_argument('--sparse-path', help='explicit local BM25 index directory')
    ap.add_argument('--manifest', help='paired dense manifest')
    ap.add_argument('--candidate-k', type=int, default=250)
    ap.add_argument('--dense-k', type=int, default=125)
    ap.add_argument('--sparse-k', type=int, default=125)
    ap.add_argument('--rrf-k', type=int, default=60)
    ap.add_argument('--literature', action='store_true', help='return literature-level results')
    ap.add_argument('--document-path', help='explicit paired document-card index; enables B5')
    ap.add_argument('--target-unique-papers', type=int, help='default: requested k')
    ap.add_argument('--max-chunks-per-paper', type=int, default=3)
    ap.add_argument('--max-candidate-budget', type=int, default=1000)
    ap.add_argument('--fixed-document-budget', action='store_true')
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        import torch  # noqa: PLC0415

        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.document_path:
        if args.mode != 'dense-only' or args.mac_only:
            ap.error('document retrieval requires dense-only and no mac-only filter')
        from document_index import retrieve_document_literature
        out = retrieve_document_literature(args.query, args.document_path, k=args.k,
            target_unique_papers=args.target_unique_papers if args.target_unique_papers is not None else args.k,
            max_chunks_per_paper=args.max_chunks_per_paper, initial_budget=args.candidate_k,
            max_budget=args.max_candidate_budget, dynamic=not args.fixed_document_budget,
            db_dir=args.db, model_dir=args.model, device=device, manifest_path=args.manifest)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    if args.mode != 'dense-only' or args.literature:
        out = retrieve_literature_mode(args.query, k=args.k, mode=args.mode,
            sparse_path=args.sparse_path, candidate_k=args.candidate_k,
            dense_k=args.dense_k, sparse_k=args.sparse_k, rrf_k=args.rrf_k,
            db_dir=args.db, model_dir=args.model, device=device,
            manifest_path=args.manifest, mac_only=args.mac_only)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return
    out = retrieve(
        args.query,
        k=args.k,
        mac_only=args.mac_only,
        db_dir=args.db,
        model_dir=args.model,
        device=device,
        manifest_path=args.manifest,
    )

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return

    for item in out:
        print(f"\n[{item['rank']}] score={item['score']:.3f}  mac={item['mac']}  {item['source']}")
        print(f"    {item['title']} ({item['year']}) {item['journal']} doi:{item['doi']}")
        if item["section"]:
            print(f"    section: {item['section']}")
        print(f"    ---\n    {item['snippet'].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
