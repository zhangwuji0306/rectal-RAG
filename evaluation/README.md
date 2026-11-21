# B2 Gold Retrieval Benchmark

D3 端到端评价独立于本页 B2 Gold，入口为 [D3 rubric 与统计合同](D3-rubric.md)及 `index/e2e_evaluation.py`。原始候选、自动拒绝和最终答案分开评价；智能体诊断不进入人工评分。真实题目、正文、答案和评分只保存在本机私有目录，不能进入普通代码发布物。

`queries.jsonl` is the frozen, graded B2 qrels file. It preserves each
audited query, its `split`, all three `judgements`, and the annotation metadata
from `B2_gold_annotations.jsonl`. `split_manifest.json` and `test_freeze.md`
define and record the locked test split.

Validate the Gold Set before evaluation:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
E:\writing-rag\.venv\Scripts\python.exe E:\writing-rag\evaluation\validate_gold_set.py
```

Run the formal B2 baseline once across all required reporting views:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
E:\writing-rag\.venv\Scripts\python.exe E:\writing-rag\index\evaluate_retrieval.py `
  --qrels E:\writing-rag\evaluation\queries.jsonl `
  --split all --k 5,10,20,50
```

The evaluator supports `--split overall`, `--split dev`, `--split test`,
`--split human_manual`, and `--split ai_assisted_human_reviewed`. Recall@5,
Recall@10, Recall@20, Recall@50, MRR@10, and
UniqueRelevantPaperRecall@50 use `relevance >= 2`; the denominator excludes
queries without a positive judgement and reports total/evaluable/excluded
counts. nDCG@10/20 uses `g(0)=0`, `g(1)=0`, `g(2)=2`, `g(3)=3`; queries with
IDCG=0 are not evaluated or filled with zero.

The benchmark is pooled-candidate evaluation: recall is the recall of relevant
papers within the audited candidate pool, not exhaustive recall over the full
corpus. Unjudged papers are not claimed to be manually irrelevant. The five
all-zero queries and nine queries containing only relevance 0/1 are reported as
diagnostic counts; no abstention accuracy is defined.

The evaluator deduplicates retrieved papers by DOI, falling back to source
filename, and shares retrieval results across the `--split all` views. It never
generates an answer.
