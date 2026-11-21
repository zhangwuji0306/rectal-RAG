"""C4 offline fixed-pool ranking. No retrieval, network, or production activation."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, ROUND_FLOOR
import json
import math
from pathlib import Path

from citation_graph import readonly, registry
from quality_metadata import MetadataStore, read_registry

SIGNALS = ('journal', 'external', 'corpus', 'publication_type')
FLOOR = Decimal('0.65')
BAND = Decimal('0.01')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def observation(state, value=None, **details):
    if state not in ('available', 'missing', 'failed', 'not_queried', 'unobserved', 'unavailable'):
        raise ValueError('invalid signal state')
    if (state == 'available') != (value is not None):
        raise ValueError('invalid state/value contract')
    return dict(state=state, value=value, **details)


def prior(name, obs):
    """Missing contributes exactly zero; weights are never redistributed."""
    if name not in SIGNALS:
        raise ValueError('unknown signal')
    observation(obs['state'], obs['value'])
    if obs['state'] != 'available':
        return 0.0
    v = obs['value']
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError('invalid numeric signal')
    if name == 'external':
        if v < 0 or int(v) != v:
            raise ValueError('external citations must be a nonnegative integer')
        return math.log1p(min(v, 1000)) / math.log1p(1000) - 0.5
    if name == 'corpus':
        if v <= 0 or int(v) != v:
            raise ValueError('no incoming observation is not a measured zero')
        return 0.5 * math.log1p(min(v, 20)) / math.log1p(20)
    if name == 'publication_type' and v in (-0.5, 0, 0.5):
        return float(v)
    raise ValueError('unsupported available signal')


def type_signal(raw):
    """A transparent article-category preference, never a clinical evidence grade."""
    types = {t.strip().casefold() for t in raw.split(';') if t.strip()}
    if not types:
        return observation('missing', source='C1 publication_type', raw=raw, group='missing')
    if types & {'meta-analysis', 'systematic review'}:
        group, value = 'synthesis', 0.5
    elif 'randomized controlled trial' in types:
        group, value = 'randomized_trial', 0.5
    elif 'case reports' in types:
        group, value = 'case_report', -0.5
    else:
        group, value = 'other', 0.0
    return observation('available', value, source='C1 publication_type', raw=raw, group=group)


def read_graph(path, papers, artifacts, fixed):
    """Consume calibrated-v1 explicitly, retaining unique exact/fuzzy edge classes."""
    c = readonly(path)
    try:
        meta = dict(c.execute('SELECT key,value FROM meta'))
        if meta.get('schema') != 'c3-calibrated-v1' or not meta.get('version'):
            raise ValueError('requires calibrated C3 schema/version')
        if load_rule := json.loads(meta['rule']):
            if load_rule != fixed or meta.get('baseline') != fixed['material_version']:
                raise ValueError('C3 rule/baseline mismatch')
        else:
            raise ValueError('missing C3 rule')
        nodes = {r['paper_id']: json.loads(r['payload']) for r in c.execute('SELECT * FROM nodes')}
        if set(nodes) != set(papers):
            raise ValueError('C3 node identity scope mismatch')
        for pid, p in papers.items():
            if any(nodes[pid][k] != v for k, v in p.items()):
                raise ValueError('C3 registry snapshot mismatch: ' + pid)
        sources = {r['artifact_id']: json.loads(r['payload']) for r in c.execute('SELECT * FROM sources')}
        if set(sources) != {a['artifact_id'] for a in artifacts}:
            raise ValueError('C3 artifact scope mismatch')
        for a in artifacts:
            if any(sources[a['artifact_id']][k] != a[k] for k in ('paper_id', 'source', 'content_sha256', 'role')):
                raise ValueError('C3 artifact identity mismatch')
        edges = set(tuple(r) for r in c.execute('SELECT citing,cited FROM edges'))
        kinds = defaultdict(set)
        for r in c.execute('SELECT citing,cited,kind,payload FROM supports'):
            pair = (r['citing'], r['cited'])
            if r['kind'] not in ('exact', 'fuzzy') or pair not in edges:
                raise ValueError('invalid C3 support class/edge')
            if r['kind'] == 'fuzzy' and json.loads(r['payload'])['fuzzy_evidence']['rule_version'] != fixed['rule_version']:
                raise ValueError('C3 fuzzy support rule mismatch')
            kinds[pair].add(r['kind'])
        if set(kinds) != edges:
            raise ValueError('C3 edge/support mismatch')
        incoming = defaultdict(lambda: defaultdict(set))
        for (src, dst), classes in kinds.items():
            if src not in papers or dst not in papers or src == dst:
                raise ValueError('invalid C3 endpoint')
            incoming[dst]['all'].add(src)
            for kind in classes:
                incoming[dst][kind].add(src)
        for pid in nodes:
            if nodes[pid]['in_degree'] != len(incoming[pid]['all']):
                raise ValueError('C3 indegree mismatch')
        return meta, incoming
    finally:
        c.close()


class Inputs:
    def __init__(self, c1, c2, graph, rule, decision, validation):
        self.papers, artifacts, _ = registry(c1)
        fixed, safety, results = load_json(rule), load_json(decision), load_json(validation)
        if (safety['final_rule'] != fixed or safety['rule_version'] != fixed['rule_version']
                or results['rule'] != fixed or safety['safety_failure'] or not safety['fuzzy_enabled'] or not fixed['enabled']):
            raise ValueError('C3 safety/rule mismatch')
        self.graph_meta, incoming = read_graph(graph, self.papers, artifacts, fixed)
        store = MetadataStore(c2, readonly=True)
        try:
            meta = dict(store.conn.execute('SELECT key,value FROM meta'))
            scope = read_registry(c1)
            if json.loads(meta['registry_snapshot']) != scope:
                raise ValueError('C2 registry snapshot mismatch')
            if {r['paper_id']: json.loads(r['payload']) for r in store.conn.execute('SELECT * FROM papers')} != {p['paper_id']: p for p in scope}:
                raise ValueError('C2 paper snapshot mismatch')
            external = {r['entity']: json.loads(r['payload']) for r in store.conn.execute(
                "SELECT entity,payload FROM observations WHERE level='paper' AND source='openalex' AND metric='cited_by_count' AND year=0")}
            if set(external) != set(self.papers):
                raise ValueError('C2 citation state scope mismatch')
        finally:
            store.close()
        self.aliases = defaultdict(set)
        for a in artifacts:
            self.aliases[a['source'].replace('\\', '/').rsplit('/', 1)[-1].lower()].add(a['paper_id'])
        self.doi = {v: p['paper_id'] for p in scope for v in p['identifiers'].get('doi', [])}
        self.signals = {}
        for pid, p in self.papers.items():
            ext = {k:v for k,v in external[pid].items() if k in
                   ('value','state','reason','rule_version','retrieved_at','source_updated_at','external_id',
                    'request_id','mapping_basis','statistical_cutoff','unit')}
            ext['source'] = 'openalex'
            if ext['rule_version'] != 'c2-v1':
                raise ValueError('C2 observation version mismatch')
            counts = {k: len(incoming[pid][k]) for k in ('all', 'exact', 'fuzzy')}
            corpus = observation('available' if counts['all'] else 'unobserved',
                counts['all'] or None, source='C3 incomplete unique citing papers',
                counts=counts, graph_version=self.graph_meta['version'], rule_version=fixed['rule_version'])
            self.signals[pid] = dict(journal=observation('unavailable', reason='no_supported_journal_quality'),
                external=ext, corpus=corpus, publication_type=type_signal(p['publication_type']))
            for name, obs in self.signals[pid].items():
                prior(name, obs)

    def resolve(self, item):
        source = item['source'].replace('\\', '/').rsplit('/', 1)[-1].lower()
        owners = self.aliases[source]
        if len(owners) != 1:
            raise ValueError('missing/ambiguous candidate source')
        pid = next(iter(owners))
        doi = (item.get('doi') or '').strip().lower()
        if doi and self.doi.get(doi) != pid:
            raise ValueError('candidate DOI/source disagreement')
        return pid


def aggregate(hits):
    if len(hits) != 250 or len({h['id'] for h in hits}) != 250:
        raise ValueError('requires 250 unique saved passages')
    groups = {}
    for i, h in enumerate(hits, 1):
        score = h['score']
        if h['rank'] != i or isinstance(score, bool) or not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError('invalid passage order/score')
        key = (h.get('doi') or '').strip().lower() or h['source']
        if key not in groups:
            groups[key] = dict(key=key, source=h['source'], doi=h.get('doi', ''), score=score,
                               representative_id=h['id'], member_ids=[])
        g = groups[key]
        g['member_ids'].append(h['id'])
        if score > g['score']:
            g.update(score=score, source=h['source'], doi=h.get('doi', ''), representative_id=h['id'])
    return [dict(g, rank=i) for i, g in enumerate(sorted(groups.values(), key=lambda g: -g['score']), 1)]


def rank(pool, weights, floor=FLOOR, band=BAND):
    if set(weights) != set(SIGNALS) or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('invalid weights')
    if not 0 < band or not -1 <= floor <= 1:
        raise ValueError('invalid relevance constraint')
    if len(pool) > 50 or len({p['paper_id'] for p in pool}) != len(pool):
        raise ValueError('fixed pool budget/identity violation')
    output = []
    previous = math.inf
    for i, p in enumerate(pool, 1):
        s = p['score']
        if p['rank'] != i or isinstance(s, bool) or not math.isfinite(s) or not -1 <= s <= min(previous, 1):
            raise ValueError('invalid relevance-only ranking')
        previous = s
        eligible = Decimal(str(s)) >= floor
        bucket = int(((Decimal(str(s)) - floor) / band).to_integral_value(rounding=ROUND_FLOOR)) if eligible else -1
        values = {k: prior(k, p['signals'][k]) for k in SIGNALS}
        contributions = {k: weights[k] * values[k] if eligible else 0.0 for k in SIGNALS}
        output.append(dict(p, baseline_rank=i, eligible=eligible, band=bucket, priors=values,
                           contributions=contributions, adjusted_score=s + sum(contributions.values())))
    output.sort(key=lambda p: (not p['eligible'], -p['band'],
                               -p['adjusted_score'] if p['eligible'] else -p['score'], p['baseline_rank']))
    return [dict(p, rank=i) for i, p in enumerate(output, 1)]


def schemes(alpha):
    zero = dict.fromkeys(SIGNALS, 0.0)
    result = {'relevance_only': zero, 'journal': dict(zero)}
    for name in SIGNALS[1:]:
        result[name] = dict(zero, **{name: alpha})
    result['full'] = dict(zero, external=alpha/3, corpus=alpha/3, publication_type=alpha/3)
    return result


def coverage(pools):
    unique = {p['paper_id']: p for pool in pools for p in pool}
    return dict(unique_papers=len(unique), candidate_occurrences=sum(map(len, pools)),
                signals={k: dict(Counter(p['signals'][k]['state'] for p in unique.values())) for k in SIGNALS})
