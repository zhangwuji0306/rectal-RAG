"""Opt-in local C5 emphasis retrieval; never a source of evidence or facts."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import re
import time

VERSION = 'c5-emphasis-v1'


class ExpansionUnavailable(RuntimeError):
    """Expected generator/retrieval unavailability, distinct from contract errors."""


@dataclass(frozen=True)
class Limits:
    max_subqueries: int = 1
    original_slots: int = 250
    branch_slots: int = 500
    total_slots: int = 750
    generation_seconds: float = 1.0
    retrieval_seconds: float = 60.0
    total_seconds: float = 120.0
    protected: int = 20
    final_k: int = 50

    def __post_init__(self):
        if self.max_subqueries not in (0, 1):
            raise ValueError('v1 supports zero or one subquery')
        if (self.original_slots, self.branch_slots, self.total_slots, self.protected, self.final_k) != (250, 500, 750, 20, 50):
            raise ValueError('unsupported C5 budget/version')
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.generation_seconds, self.retrieval_seconds, self.total_seconds)):
            raise ValueError('invalid deadline')


def generate(query):
    """Repeat an exact suffix while retaining the entire question verbatim."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError('nonempty query required')
    if len(query) > 4000 or not 16 <= len(query.split()) <= 180:
        return []
    if re.search(r'\b(no|not|without|neither|nor)\b', query, re.I):
        return []
    parts = re.split(r' and ', query, flags=re.I)
    if len(parts) < 2:
        return []
    focus = parts[-1].strip().rstrip('?.!;,:')
    if not 3 <= len(focus.split()) <= 40:
        return []
    return [query + '\n' + focus]


def subqueries(query, generated, limit):
    result = []
    for q in generated:
        # The generator contract forbids replacing or deleting any original text.
        if not isinstance(q, str) or not q.strip():
            continue
        if q.strip() == query.strip() or q in result:
            continue
        if not q.startswith(query + '\n') or q[len(query)+1:] not in query:
            raise ValueError('subquery violates exact-original/extractive contract')
        result.append(q)
    return result[:limit]


class Identity:
    def __init__(self, path):
        from citation_graph import registry
        self.papers, artifacts, identifiers = registry(path)
        self.sources = defaultdict(set)
        for a in artifacts:
            self.sources[a['source'].replace('\\', '/').rsplit('/', 1)[-1].lower()].add(a['paper_id'])
        self.dois = {i['value']: i['paper_id'] for i in identifiers
                     if i['scheme'] == 'doi' and i['status'] == 'active'}

    def __call__(self, hit):
        owners = self.sources[hit['source'].replace('\\', '/').rsplit('/', 1)[-1].lower()]
        if len(owners) != 1:
            raise ValueError('missing/ambiguous C1 source')
        pid = next(iter(owners))
        doi = (hit.get('doi') or '').strip().lower()
        if doi and self.dois.get(doi) != pid:
            raise ValueError('C1 DOI/source conflict')
        return pid


def papers(hits, resolve, branch, query):
    groups = {}
    seen = set()
    for i, h in enumerate(hits, 1):
        s = h['score']
        if h['rank'] != i or h['id'] in seen or isinstance(s, bool) or not math.isfinite(s) or not -1 <= s <= 1:
            raise ValueError('invalid passage identity/rank/score')
        seen.add(h['id'])
        pid = resolve(h)
        if pid not in groups:
            groups[pid] = dict(paper_id=pid, source=h['source'], doi=h.get('doi', ''),
                               score=s, representative_id=h['id'], origins=[])
        g = groups[pid]
        g['origins'].append(dict(branch=branch, query=query, passage_id=h['id'],
                                 passage_rank=i, raw_score=s))
        if s > g['score']:
            g.update(score=s, representative_id=h['id'], source=h['source'], doi=h.get('doi', ''))
    ordered = sorted(groups.values(), key=lambda p: -p['score'])
    for rank, p in enumerate(ordered, 1):
        p['branch_rank'] = rank
        for origin in p['origins']:
            origin['paper_rank'] = rank
    return ordered


def merge(original, branches, *, same_budget=False, limits=Limits()):
    """RRF uses ranks only; raw cosine scores remain branch-local diagnostics."""
    union = {}
    for pool in [original, *branches]:
        for p in pool:
            pid = p['paper_id']
            if pid not in union:
                union[pid] = dict(p, origins=[], fusion_score=0.0)
            union[pid]['origins'].extend(p['origins'])
            union[pid]['fusion_score'] += 1 / (60 + p['branch_rank'])
    old = {p['paper_id']: i for i, p in enumerate(original)}
    protected = [p['paper_id'] for p in original[:limits.protected]]
    remainder = [p for pid, p in union.items() if pid not in protected]
    if not branches:
        remainder.sort(key=lambda p: old[p['paper_id']])
    elif same_budget:
        extra_order = {p['paper_id']: i for i, p in enumerate(branches[0])}
        remainder.sort(key=lambda p: (extra_order.get(p['paper_id'], math.inf), old.get(p['paper_id'], math.inf), p['paper_id']))
    else:
        remainder.sort(key=lambda p: (-p['fusion_score'], old.get(p['paper_id'], math.inf), p['paper_id']))
    full = [union[pid] for pid in protected] + remainder
    return dict(union=[dict(p, rank=i) for i, p in enumerate(full, 1)],
                final=[dict(p, rank=i) for i, p in enumerate(full[:limits.final_k], 1)], protected=protected)


def run(query, original_hits, resolve, fetch, *, limits=Limits(), generator=generate, clock=time.monotonic):
    """fetch(query, slots, timeout) must enforce its deadline, e.g. a subprocess.

    Original hits are supplied/reused but fully charged. Never catch programming,
    identity, schema, model, or index errors as expansion failures.
    """
    if len(original_hits) != limits.original_slots:
        raise ValueError('requires complete protected D250 request')
    original = papers(original_hits, resolve, 'original', query)
    start = clock()
    events = []
    qs = []
    generation_calls = int(limits.max_subqueries > 0)
    if generation_calls:
        try:
            generated = generator(query)
            if clock()-start > min(limits.generation_seconds, limits.total_seconds):
                raise TimeoutError('generation deadline exceeded')
            qs = subqueries(query, generated, limits.max_subqueries)
        except (ExpansionUnavailable, TimeoutError) as exc:
            events.append(dict(branch='generation', state='failed', reason=str(exc), error=type(exc).__name__))
    charged = limits.original_slots
    pools = []
    calls = 1
    for i, q in enumerate(qs, 1):
        remaining = limits.total_seconds - (clock()-start)
        if remaining <= 0:
            events.append(dict(branch=f'sub{i}', state='not_started', reason='total deadline'))
            break
        if charged + limits.branch_slots > limits.total_slots:
            raise ValueError('candidate request budget exceeded')
        charged += limits.branch_slots
        calls += 1
        before = clock()
        timeout = min(limits.retrieval_seconds, remaining)
        try:
            hits = fetch(q, limits.branch_slots, timeout)
        except (ExpansionUnavailable, TimeoutError) as exc:
            events.append(dict(branch=f'sub{i}', state='failed', reason=str(exc), error=type(exc).__name__, seconds=clock()-before))
            continue
        # Validate before deciding whether the successful response was too late.
        if len(hits) > limits.branch_slots:
            raise ValueError('retriever exceeded requested slots')
        pool = papers(hits, resolve, f'sub{i}', q)
        if clock()-before > timeout:
            events.append(dict(branch=f'sub{i}', state='failed', reason='retrieval deadline exceeded', seconds=clock()-before))
            continue
        pools.append(pool)
        events.append(dict(branch=f'sub{i}', state='success', seconds=clock()-before, returned=len(hits)))
    merged = merge(original, pools, limits=limits)
    return dict(version=VERSION, query=query, subqueries=qs, **merged, events=events,
                expanded=bool(pools), fallback=bool(events) and not pools,
                accounting=dict(requested_slots=charged, retrieval_calls=calls, generation_calls=generation_calls,
                                reused_original_slots=250, incremental_seconds=clock()-start),
                use='retrieval-only; subqueries are not evidence')


def differences(before, after, grades):
    """grades maps C1 ID to explicit pool labels; absent IDs remain unjudged."""
    a, b = set(before), set(after)
    def group(ids):
        return {name: sorted(pid for pid in ids if
                ('unjudged' if pid not in grades else 'positive' if grades[pid] >= 2 else 'other_judged') == name)
                for name in ('positive', 'other_judged', 'unjudged')}
    return dict(added=group(b-a), lost=group(a-b), retained=group(a&b))
