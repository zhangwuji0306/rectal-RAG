"""Local document cards and bounded passage/document candidate retrieval."""
import json
from pathlib import Path


def paper_key(hit):
    return (hit.get('doi') or '').strip().lower() or hit['source']


def card_text(card):
    lines = [f'{key}: {card.get(key) or "[missing]"}' for key in
             ('title', 'abstract', 'keywords', 'pub_type', 'year', 'journal')]
    if card.get('mesh_terms'):
        # MeSH is a labelled proxy, never free-text keywords.
        lines.append(f'mesh_terms (MeSH): {card["mesh_terms"]}')
    return '\n'.join(lines)


def build(cards, vectors, path, binding, model_fingerprint):
    import numpy as np
    path = Path(path)
    if len({c['source'] for c in cards}) != len(cards) or len(cards) != len(vectors):
        raise ValueError('duplicate/missing document identity')
    path.mkdir(parents=True, exist_ok=False)
    (path/'cards.json').write_text(json.dumps(cards, ensure_ascii=False), encoding='utf-8')
    np.save(path/'vectors.npy', np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    (path/'config.json').write_text(json.dumps({'binding': binding,
        'model_fingerprint': model_fingerprint, 'representation': 'document-card-v2',
        'count': len(cards)}, ensure_ascii=False, indent=2), encoding='utf-8')


class DocumentIndex:
    """Exact cosine over normalized document-card vectors; explicit pair binding."""
    def __init__(self, path, db_dir, manifest_path, model_fingerprint):
        import numpy as np
        from sparse_index import pair_binding
        path = Path(path)
        self.config = json.loads((path/'config.json').read_text(encoding='utf-8'))
        if self.config['binding'] != pair_binding(db_dir, manifest_path):
            raise ValueError('document index belongs to another passage pair')
        if self.config['model_fingerprint'] != model_fingerprint:
            raise ValueError('document model mismatch')
        self.cards = json.loads((path/'cards.json').read_text(encoding='utf-8'))
        self.vectors = np.load(path/'vectors.npy', allow_pickle=False)
        if len(self.cards) != len(self.vectors) or not np.isfinite(self.vectors).all():
            raise ValueError('invalid document vectors')
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
            raise ValueError('document vectors must be normalized')
        self.count = len(self.cards)

    def search_vector(self, vector, k):
        import numpy as np
        if k < 1:
            raise ValueError('document candidate budget must be positive')
        scores = self.vectors @ np.asarray(vector, dtype=np.float32)
        order = sorted(range(self.count), key=lambda i: (-float(scores[i]), self.cards[i]['source']))[:k]
        return [{**{k:v for k,v in self.cards[i].items() if k not in
                    ('abstract', 'keywords', 'abstract_passage_ids',
                     'abstract_passage_ids_historical', 'mesh_terms', 'metadata')},
                 'rank': rank, 'score': float(scores[i]),
                 'id': 'card:'+self.cards[i]['source'], 'section': 'document-card'}
                for rank, i in enumerate(order, 1)]


def merge_candidates(passages, documents, max_chunks_per_paper=3, rrf_k=60):
    """Cap passages per DOI/source; fuse paper ranks without mixing cosine scales."""
    if max_chunks_per_paper < 1 or rrf_k < 1:
        raise ValueError('invalid aggregation settings')
    groups, kept, rejected = {}, [], []
    seen = set()
    for hit in passages:
        if hit['id'] in seen:
            raise ValueError('duplicate passage ID')
        seen.add(hit['id'])
        key = paper_key(hit)
        if len(groups.get(key, [])) >= max_chunks_per_paper:
            rejected.append(hit['id'])
            continue
        groups.setdefault(key, []).append(hit)
        kept.append(hit)
    # Preserve legacy max cosine and stable input order for passage paper ranks.
    passage_papers = sorted(groups, key=lambda key: -max(h['score'] for h in groups[key]))
    merged = {}
    branch_keys = {'passage': [], 'document': []}
    for branch, entries in (('passage', [groups[key][0] for key in passage_papers]),
                            ('document', documents)):
        visited = set()
        for hit in entries:
            key = paper_key(hit)
            if key in visited:
                continue
            visited.add(key)
            rank = len(visited)
            branch_keys[branch].append(key)
            if key not in merged:
                merged[key] = {**hit, 'score': 0., 'contributions': {}, 'key': key}
            merged[key]['contributions'][branch] = {'rank': rank, 'score': 1/(rrf_k+rank)}
            merged[key]['score'] += 1/(rrf_k+rank)
    papers = sorted(merged.values(), key=lambda h: (-h['score'], h['key']))
    for rank, hit in enumerate(papers, 1):
        members = groups.get(hit['key'], [])
        hit.update(rank=rank, member_ids=[h['id'] for h in members], n_hits=len(members),
                   sections=list(dict.fromkeys(h.get('section','') for h in members if h.get('section'))))
    return papers, {'kept_passage_ids': [h['id'] for h in kept], 'capped_passage_ids': rejected,
                    'branch_paper_keys': branch_keys,
                    'passage_mean_chunks_per_paper': len(passages)/len({paper_key(h) for h in passages}) if passages else 0.,
                    'kept_mean_chunks_per_paper': len(kept)/len(groups) if groups else 0.}


def bounded_retrieve(passage_fetch, document_fetch, passage_count, document_count,
                     target_unique_papers=50, max_chunks_per_paper=3,
                     initial_budget=250, max_budget=1000, dynamic=True, k=50):
    """Budget is total branch slots (half passage, half card), not unique papers."""
    if not (1 <= k <= 50 and 1 <= target_unique_papers <= k and
            2 <= initial_budget <= max_budget and passage_count >= 0 and document_count >= 0):
        raise ValueError('invalid document retrieval budget/target')
    budget, previous, rounds = initial_budget, None, []
    while True:
        pk, dk = min(budget//2, passage_count), min(budget-budget//2, document_count)
        passages, documents = passage_fetch(pk) if pk else [], document_fetch(dk) if dk else []
        if len(passages) > pk or len(documents) > dk:
            raise ValueError('candidate provider exceeded budget')
        papers, aggregation = merge_candidates(passages, documents, max_chunks_per_paper)
        identities = ({h['id'] for h in passages}, {h['id'] for h in documents})
        if len(papers) >= target_unique_papers:
            stop = 'target_reached'
        elif pk >= passage_count and dk >= document_count:
            stop = 'corpus_exhausted'
        elif previous is not None and not ((identities[0]-previous[0]) or (identities[1]-previous[1])):
            stop = 'no_new_candidates'
        elif budget >= max_budget:
            stop = 'candidate_cap'
        elif not dynamic:
            stop = 'fixed_budget'
        else:
            stop = None
        rounds.append({'budget': budget, 'requested': {'passage': pk, 'document': dk},
                       'passages': passages, 'documents': documents, 'aggregation': aggregation,
                       'papers': papers, 'stop': stop})
        if stop:
            selected = papers[:k]
            return selected, {'rounds': rounds, 'stop': stop, 'returned_unique_papers': len(selected),
                              'target_unique_papers': target_unique_papers,
                              'target_unmet': len(selected) < target_unique_papers,
                              'ordered_hits': [{'rank': i, 'source': h['source'], 'doi': h.get('doi',''),
                                                'score': h['score']} for i,h in enumerate(selected, 1)]}
        previous = identities
        budget = min(max_budget, budget*2)


def retrieve_document_literature(query, document_path, k=50, target_unique_papers=50,
                                max_chunks_per_paper=3, initial_budget=250, max_budget=1000,
                                dynamic=True, ranking_trace=None, **kwargs):
    import rag_core as core
    if kwargs.pop('mac_only', False):
        raise ValueError('document mac_only is unsupported')
    # Reject missing/foreign cards before preflight can open a persistent client.
    from sparse_index import pair_binding
    db = Path(kwargs.get('db_dir') or core.DEFAULT_DB_DIR).resolve()
    manifest = core._manifest_path(db, kwargs.get('manifest_path'))
    config = json.loads((Path(document_path)/'config.json').read_text(encoding='utf-8'))
    if config['binding'] != pair_binding(db, manifest):
        raise ValueError('document index belongs to another passage pair')
    checked = core.index_preflight(**kwargs)
    index = DocumentIndex(document_path, db, manifest, checked['model_fingerprint'])
    model = core._get_model(kwargs.get('model_dir'), kwargs.get('device','cpu'), kwargs.get('db_dir'))
    vector = model.encode([query], normalize_embeddings=True)[0]
    papers, trace = bounded_retrieve(lambda n: core.retrieve_dense(query, n, **kwargs),
        lambda n: index.search_vector(vector, n), checked['collection_count'], index.count,
        target_unique_papers, max_chunks_per_paper, initial_budget, max_budget, dynamic, k)
    if ranking_trace is not None:
        ranking_trace.update(trace)
    return core.summary_list(papers)
