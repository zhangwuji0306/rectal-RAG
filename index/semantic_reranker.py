"""Explicit, local B6 experiment over caller-supplied D250 passage records.

No retrieval or default-path imports. ``text`` must be a complete stored passage,
never a card or snippet; the supplier is responsible for that provenance.
"""
import argparse
import json
import math
import time
from pathlib import Path

MODEL_ID = 'BAAI/bge-reranker-v2-m3'
REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
FIRST_STAGE = 250
POOL_SIZE = 200
OUTPUT_PAPERS = 50
MAX_LENGTH = 512
QUERY_TOKENS = 128
BATCH_SIZE = 4
SECONDS_PER_QUERY = 660
DEV_GATES = {'ndcg10_gain': 0.01, 'mrr10_gain': 0.01, 'recall20_drop': 0.01}


def paper_key(hit):
    return (hit.get('doi') or '').strip().lower() or hit['source']


def validate(candidates):
    if len(candidates) > FIRST_STAGE:
        raise ValueError('D input exceeds 250 passages')
    ids = set()
    previous = math.inf
    for i, hit in enumerate(candidates, 1):
        if (not isinstance(hit.get('id'), str) or not hit['id'] or
                hit['id'] in ids or not isinstance(hit.get('source'), str) or
                not hit['source'] or hit.get('rank') != i):
            raise ValueError('invalid/duplicate identity or non-contiguous D rank')
        score = float(hit['score'])
        if not math.isfinite(score) or score > previous:
            raise ValueError('D scores must be finite and non-increasing')
        ids.add(hit['id'])
        previous = score


def aggregate(candidates, scores):
    """Max passage score; ties retain first D occurrence, including representative."""
    groups = {}
    for hit, score in zip(candidates, scores):
        key = paper_key(hit)
        if key not in groups:
            groups[key] = {'key': key, 'source': hit['source'], 'doi': hit.get('doi', ''),
                           'score': score, 'representative_id': hit['id'], 'member_ids': []}
        group = groups[key]
        group['member_ids'].append(hit['id'])
        if score > group['score']:
            group.update(score=score, representative_id=hit['id'], source=hit['source'],
                         doi=hit.get('doi', ''))
    papers = sorted(groups.values(), key=lambda h: -h['score'])
    return [dict(paper, rank=i) for i, paper in enumerate(papers[:OUTPUT_PAPERS], 1)]


def encode_pairs(tokenizer, query, texts):
    """Right-truncate query first, then passage; pair specials count toward 512."""
    query_ids = tokenizer.encode(query, add_special_tokens=False,
                                 truncation=True, max_length=QUERY_TOKENS)
    capacity = MAX_LENGTH-len(query_ids)-tokenizer.num_special_tokens_to_add(pair=True)
    if capacity < 1:
        raise ValueError('tokenizer leaves no passage capacity')
    features = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                               max_length=capacity)
        # Fixed XLM-R pair format; Transformers 5 removed the legacy builder.
        if (tokenizer.cls_token_id, tokenizer.sep_token_id) != (0, 2):
            raise ValueError('expected fixed XLM-R special tokens')
        pair = [tokenizer.cls_token_id]+query_ids+[tokenizer.sep_token_id]*2+ids+[tokenizer.sep_token_id]
        if len(pair) > MAX_LENGTH:
            raise ValueError('tokenizer exceeded pair budget')
        features.append({'input_ids': pair, 'attention_mask': [1]*len(pair)})
    return features


class LocalScorer:
    """Fixed CPU/float32 Transformers model, no network, no remote code."""
    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        path = Path(model_path)
        identity = json.loads((path/'b6-model.json').read_text(encoding='utf-8'))
        if identity != {'model': MODEL_ID, 'revision': REVISION}:
            raise ValueError('fixed model revision mismatch')
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                      trust_remote_code=False)
        self.tokenizer.truncation_side = 'right'
        self.tokenizer.padding_side = 'right'
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, use_safetensors=True).to('cpu').eval()
        if self.model.config.num_labels != 1:
            raise ValueError('expected single relevance logit')
        self.torch = torch

    def __call__(self, query, texts):
        features = encode_pairs(self.tokenizer, query, texts)
        batch = self.tokenizer.pad(features, padding=True, return_tensors='pt')
        with self.torch.inference_mode():
            return self.model(**batch).logits.reshape(-1).float().tolist()


def experiment(query, d250, *, enabled=False, scorer_factory=None, on_error='raise',
               clock=time.monotonic):
    """Return D250 by default; enabled output includes D200 and D250 controls.

    Model/load/score/timeout failures discard ALL semantic scores. Validation
    errors always raise, so malformed or preview-only inputs cannot look valid.
    Deadline is cooperative between batches, never a hard native-call kill.
    """
    if on_error not in ('raise', 'original'):
        raise ValueError('unknown failure policy')
    validate(d250)
    baseline = aggregate(d250, [float(h['score']) for h in d250])
    result = {'status': 'disabled', 'papers': baseline, 'd250_control': baseline}
    if not enabled:
        return result
    if not isinstance(query, str) or not query.strip():
        raise ValueError('non-empty query required')
    pool = d250[:POOL_SIZE]
    for h in pool:
        if (h.get('text_kind') != 'stored-passage' or
                not isinstance(h.get('text'), str) or not h['text'].strip() or
                h['id'].startswith('card:') or h.get('section') == 'document-card'):
            raise ValueError('complete stored passage text required, no card/snippet fallback')
    original = aggregate(pool, [float(h['score']) for h in pool])
    result.update(papers=original, pool_control=original,
                  pool_ids=[h['id'] for h in pool],
                  dropped_ids=[h['id'] for h in d250[POOL_SIZE:]],
                  passage_order=[h['id'] for h in pool],
                  target_unmet=len(pool) < POOL_SIZE)
    if not pool:
        result['status'] = 'empty'
        return result
    start = clock()
    scores = []
    try:
        if scorer_factory is None:
            raise ValueError('explicit scorer factory required')
        scorer = scorer_factory()
        for offset in range(0, len(pool), BATCH_SIZE):
            if clock()-start >= SECONDS_PER_QUERY:
                raise TimeoutError('B6 query budget exceeded')
            batch = pool[offset:offset+BATCH_SIZE]
            values = list(scorer(query, [h['text'] for h in batch]))
            if len(values) != len(batch):
                raise ValueError('model score count mismatch')
            values = [float(v) for v in values]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('non-finite model score')
            scores.extend(values)
            if clock()-start >= SECONDS_PER_QUERY:
                raise TimeoutError('B6 query budget exceeded')
    except Exception as exc:
        if on_error == 'raise':
            raise
        result.update(status='fallback', error=f'{type(exc).__name__}: {exc}')
        return result
    order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
    result.update(status='reranked', papers=aggregate(pool, scores),
                  passage_order=[pool[i]['id'] for i in order],
                  passage_scores=[{'id': h['id'], 'score': s} for h, s in zip(pool, scores)])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='local JSON: query + d250')
    parser.add_argument('--output', required=True)
    parser.add_argument('--enable-reranker', action='store_true')
    parser.add_argument('--model-path')
    parser.add_argument('--on-error', choices=('raise', 'original'), default='raise')
    args = parser.parse_args()
    if args.enable_reranker and not args.model_path:
        parser.error('--enable-reranker requires --model-path')
    payload = json.loads(Path(args.input).read_text(encoding='utf-8'))
    result = experiment(payload['query'], payload['d250'], enabled=args.enable_reranker,
                        scorer_factory=lambda: LocalScorer(args.model_path), on_error=args.on_error)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
