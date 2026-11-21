"""D1 local evidence organization. No retrieval, model, database or file I/O.

Inputs are C1 registry() snapshots, C2 MetadataStore.rows() observations and
original rag_core.retrieve() hits. Bindings and UTF-8 source bytes must be
provided explicitly: retrieve's 400-character snippet is not a full passage.
Offsets are zero-based Unicode code points, end-exclusive, in decoded bytes
(no newline normalization). v1 requires one continuous original range; v2
explicitly distinguishes ordered original spans from inserted whitespace and
never supplies a continuous envelope for them. Tables describe the same unit.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

SCHEMA = 'd1-evidence-pack-v1'
SEGMENTED_SCHEMA = 'd1-evidence-pack-v2-segmented'
SCORE = 'rag_core.retrieve:round(1-cosine_distance,4)'


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('non-finite JSON number: ' + value)

    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number

    pack = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid,
                      parse_float=finite_float)
    if pack['schema'] not in (SCHEMA, SEGMENTED_SCHEMA):
        raise ValueError('unsupported Evidence Pack schema')
    return pack


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _basename(source):
    return source.replace('\\', '/').rsplit('/', 1)[-1].lower()


def _evidence_id(evidence):
    # Labels/context are metadata; identity includes the exact content and range.
    keys = ('paper_id', 'artifact_id', 'source', 'content_sha256',
            'representation_version', 'chunk', 'start', 'end', 'passage')
    identity = {k: evidence[k] for k in keys}
    if 'source_segments' in evidence:
        identity['source_segments'] = evidence['source_segments']
        identity['location_mode'] = evidence['location_mode']
    return 'EV' + _hash(dumps(identity).encode('utf-8'))


def segmented_text(segments, original):
    """Reconstruct an ordered selection, with inserted whitespace made explicit.

    Each source span uses ORIGINAL decoded codepoints. Gaps are intentionally
    excluded content, not a continuous envelope. This checks location/content,
    not the completeness or clinical semantics of the selected material.
    """
    if not isinstance(segments, list) or not segments:
        raise ValueError('nonempty source segments required')
    parts, last_end, spans = [], -1, 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError('invalid source segment')
        if set(segment) == {'separator'}:
            value = segment['separator']
            if not isinstance(value, str) or not value or not value.isspace():
                raise ValueError('only explicit whitespace separators allowed')
            parts.append(value)
        elif set(segment) == {'start', 'end'}:
            start, end = segment['start'], segment['end']
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(original) or start < last_end):
                raise ValueError('invalid or unordered original source segment')
            parts.append(original[start:end]); last_end = end; spans += 1
        else:
            raise ValueError('invalid source segment fields')
    if not spans:
        raise ValueError('source-backed content required')
    return ''.join(parts)


class Snapshot:
    """Already accepted C1 ownership only; pending relations are never consumed.

    No DOI/title heuristic merging. Distinct artifacts (including confirmed
    duplicates) stay separate evidence units under their C1 owner. Unresolved
    conflict owners are rejected. Empty C2 observations mean explicitly absent,
    rather than a fabricated zero or not_queried C2 observation.
    """
    def __init__(self, papers, artifacts, identifiers, observations=()):
        self.papers = deepcopy(papers)
        self.artifacts = {}
        for artifact in artifacts:
            aid = artifact['artifact_id']
            if aid in self.artifacts or artifact['paper_id'] not in papers:
                raise ValueError('duplicate artifact or unknown paper')
            self.artifacts[aid] = deepcopy(artifact)
        self.identifiers = {}
        self.ids = {pid: {'doi': [], 'pmid': []} for pid in papers}
        for row in identifiers:
            if row['status'] != 'active':
                continue
            key = (row['scheme'], row['value'])
            if key in self.identifiers or row['paper_id'] not in papers:
                raise ValueError('identifier ownership conflict')
            self.identifiers[key] = row['paper_id']
            if row['scheme'] in ('doi', 'pmid'):
                self.ids[row['paper_id']][row['scheme']].append(row['value'])
        self.observations = deepcopy(list(observations))
        for row in self.observations:
            payload = row['payload']
            if (row['level'] not in ('paper', 'journal')
                    or payload['state'] not in ('available', 'missing', 'failed', 'not_queried')
                    or (payload['state'] == 'available') != (payload['value'] is not None)):
                raise ValueError('invalid C2 observation contract')

    def paper(self, pid):
        paper = self.papers[pid]
        if paper['identity_status'] != 'verified':
            raise ValueError('unresolved or merged C1 owner')
        rows = [r for r in self.observations if r['level'] == 'paper' and r['entity'] == pid]
        # Journal observations join only via an available C2 journal_source_id.
        journals = {r['payload']['value'] for r in rows
                    if r['metric'] == 'journal_source_id' and r['payload']['state'] == 'available'}
        rows += [r for r in self.observations if r['level'] == 'journal' and r['entity'] in journals]
        rows = sorted(rows, key=lambda r: (r['level'], r['entity'], r['source'], r['metric'], r['year']))
        return dict(paper_id=pid, title=paper['title'] or None,
                    doi=sorted(self.ids[pid]['doi']), pmid=sorted(self.ids[pid]['pmid']),
                    identity_status=paper['identity_status'],
                    quality_metadata=dict(status='provided' if rows else 'absent',
                                          observations=deepcopy(rows)), evidence=[])


def _unit(hit, binding, snapshot, sources, *, segmented=False):
    artifact = snapshot.artifacts[binding['artifact_id']]
    pid = artifact['paper_id']
    snapshot.paper(pid)  # Fail before accepting content from unresolved owners.
    if _basename(hit['source']) != _basename(artifact['source']):
        raise ValueError('hit/artifact source mismatch')
    owners = {a['paper_id'] for a in snapshot.artifacts.values()
              if _basename(a['source']) == _basename(hit['source'])}
    if owners != {pid}:
        raise ValueError('ambiguous source ownership')
    for scheme in ('doi', 'pmid'):
        value = hit.get(scheme)
        if value:
            # Normalization is the accepted C1 contract, not a new merge rule.
            from paper_registry import normalize_identifier
            value = normalize_identifier(scheme, value)[0]
            if snapshot.identifiers.get((scheme, value)) != pid:
                raise ValueError('hit identifier/source conflict')
    if binding['content_sha256'] != artifact['content_sha256']:
        raise ValueError('binding/artifact version mismatch')
    raw = sources[artifact['artifact_id']]
    if _hash(raw) != artifact['content_sha256']:
        raise ValueError('source bytes/artifact version mismatch')
    text = raw.decode('utf-8')
    start, end = binding['start'], binding['end']
    if 'source_segments' in binding:
        if not segmented or start is not None or end is not None:
            raise ValueError('segmented location requires v2 and no continuous envelope')
        passage = segmented_text(binding['source_segments'], text)
    else:
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text)):
            raise ValueError('invalid source range')
        passage = text[start:end]
    if binding['passage'] != passage:
        raise ValueError('full retrieved passage/source range mismatch')
    if (not binding['representation_version']
            or binding['representation_version'] != hit['embedding_text_version']
            or type(hit['chunk']) is not int or hit['chunk'] < 0):
        raise ValueError('invalid representation/chunk binding')
    if hit['snippet'] != passage[:400]:
        raise ValueError('retrieval snippet/source range mismatch')
    table = deepcopy(binding['table'])
    if table['kind'] not in ('text', 'table', 'unknown'):
        raise ValueError('invalid evidence kind')
    missing = []
    if table['kind'] == 'table':
        for key in ('title', 'columns', 'units', 'context'):
            values = table[key]
            if values is None or values == [] or values == '':
                missing.append(key)
                continue
            if key in ('columns', 'units'):
                if not isinstance(values, list):
                    raise ValueError('table columns/units require lists')
            else:
                values = [values]
            if any(not isinstance(v, str) or not v or v not in passage for v in values):
                raise ValueError('table context outside located evidence unit')
    elif table['kind'] == 'unknown':
        missing.append('kind_and_context')
    table['missing'] = missing
    result = dict(paper_id=pid, artifact_id=artifact['artifact_id'], source=artifact['source'],
                  artifact_role=artifact['role'], content_sha256=artifact['content_sha256'],
                  representation_version=binding['representation_version'], chunk=hit['chunk'],
                  start=start, end=end, offset_unit='unicode_codepoint', section=hit['section'],
                  passage=passage, table=table, truncated=False, verification='verified', origins=[])
    if 'source_segments' in binding:
        result.update(location_mode='ordered_source_segments',
                      source_segments=deepcopy(binding['source_segments']))
    result['evidence_id'] = _evidence_id(result)
    return result


def build(hits, bindings, snapshot, sources, *, max_chars, score_semantics, segmented=False):
    """Keep whole units in original rank order; skip oversized units visibly.

    Budget is Unicode characters of unique selected passages, excluding metadata
    and JSON syntax, not a model token/window guarantee. No partial truncation.
    Bindings are one per hit, carrying artifact_id, content_sha256, start/end,
    representation_version, full passage and table. Missing bindings fail,
    never guess offsets. Full passage must come from the original chunk source,
    not from padding retrieve's snippet with guessed neighboring text.
    Caller explicitly attests original dense score semantics; fused hits rejected.
    """
    if type(max_chars) is not int or max_chars < 0 or score_semantics != SCORE:
        raise ValueError('invalid budget or non-dense score semantics')
    if len(hits) != len(bindings):
        raise ValueError('one explicit source binding required per hit')
    units = {}
    for rank, (hit, binding) in enumerate(zip(hits, bindings), 1):
        score = hit['score']
        if (hit['rank'] != rank or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not -1 <= score <= 1
                or any(k in hit for k in ('contributions', 'fusion_score', 'quality_score'))):
            raise ValueError('invalid original dense hit rank/score')
        unit = _unit(hit, binding, snapshot, sources, segmented=segmented)
        eid = unit['evidence_id']
        origin = dict(rank=rank, semantic_score=score, score_semantics=SCORE,
                      retrieval_id=hit.get('id'), legacy_paper_id=hit.get('paper_id'),
                      source=hit['source'], snippet=hit['snippet'])
        if eid in units and {k:v for k,v in unit.items() if k != 'origins'} != {
                k:v for k,v in units[eid].items() if k != 'origins'}:
            raise ValueError('conflicting metadata at same evidence location')
        units.setdefault(eid, unit)['origins'].append(origin)
    papers, omitted, used = {}, [], 0
    for unit in units.values():
        cost = len(unit['passage'])
        if used + cost > max_chars:
            omitted.append(dict(evidence_id=unit['evidence_id'], paper_id=unit['paper_id'],
                                reason='whole_unit_exceeds_remaining_budget', required_chars=cost,
                                origins=unit['origins']))
            continue
        used += cost
        pid = unit['paper_id']
        papers.setdefault(pid, snapshot.paper(pid))['evidence'].append(unit)
    return dict(schema=SEGMENTED_SCHEMA if segmented else SCHEMA, score_semantics=SCORE, papers=list(papers.values()),
                budget=dict(unit='unicode_codepoints_of_unique_passages', limit=max_chars,
                            used=used, selection='first_hit_order_skip_oversized_whole_units',
                            truncation='disabled', omitted=omitted,
                            over_budget=bool(omitted)),
                interpretation='paper groups are not independent-study adjudication')


def _verify_budget(budget, selected_ids, actual_used):
    fields = {'unit', 'limit', 'used', 'selection', 'truncation', 'omitted', 'over_budget'}
    if not isinstance(budget, dict) or not fields <= budget.keys():
        raise ValueError('missing required budget fields')
    if any(type(budget[k]) is not int or budget[k] < 0 for k in ('limit', 'used')):
        raise ValueError('budget limit/used must be nonnegative integers')
    if (budget['unit'] != 'unicode_codepoints_of_unique_passages'
            or budget['selection'] != 'first_hit_order_skip_oversized_whole_units'
            or budget['truncation'] != 'disabled'):
        raise ValueError('unsupported budget unit/selection/truncation')
    if (not isinstance(budget['omitted'], list) or type(budget['over_budget']) is not bool
            or budget['over_budget'] != bool(budget['omitted'])):
        raise ValueError('invalid budget omissions/over_budget flag')
    if actual_used != budget['used'] or actual_used > budget['limit']:
        raise ValueError('pack budget mismatch')
    omitted_ids = set()
    omitted_fields = {'evidence_id', 'paper_id', 'reason', 'required_chars', 'origins'}
    origin_fields = {'rank', 'semantic_score', 'score_semantics', 'retrieval_id',
                     'legacy_paper_id', 'source', 'snippet'}
    for entry in budget['omitted']:
        if not isinstance(entry, dict) or not omitted_fields <= entry.keys():
            raise ValueError('invalid omitted evidence structure')
        if (any(not isinstance(entry[k], str) or not entry[k] for k in ('evidence_id', 'paper_id'))
                or entry['reason'] != 'whole_unit_exceeds_remaining_budget'
                or type(entry['required_chars']) is not int or entry['required_chars'] <= 0
                or entry['required_chars'] <= budget['limit'] - actual_used
                or not isinstance(entry['origins'], list) or not entry['origins']
                or any(not isinstance(o, dict) or not origin_fields <= o.keys() for o in entry['origins'])):
            raise ValueError('invalid omitted evidence fields')
        eid = entry['evidence_id']
        if eid in selected_ids or eid in omitted_ids:
            raise ValueError('duplicate or selected omitted evidence')
        omitted_ids.add(eid)


def verify(pack, snapshot, sources):
    """Recheck serialized selected evidence against supplied C1 and source bytes.

    Does not assert retrieval quality, clinical support, or completeness of table
    extraction. C1 does not independently attest chunk-to-character mappings;
    offsets and source text are checked, chunk/section remain retrieval labels.
    Budget checks cover the supplied pack's structure and character accounting,
    not completeness of the original retrieval or authenticity of omitted costs.
    """
    if pack['schema'] not in (SCHEMA, SEGMENTED_SCHEMA) or pack['score_semantics'] != SCORE:
        raise ValueError('unsupported pack/score version')
    seen, paper_ids, used = set(), set(), 0
    for paper in pack['papers']:
        if paper['paper_id'] in paper_ids or not paper['evidence']:
            raise ValueError('duplicate or empty paper group')
        paper_ids.add(paper['paper_id'])
        expected = snapshot.paper(paper['paper_id'])
        if {k:v for k,v in paper.items() if k != 'evidence'} != {
                k:v for k,v in expected.items() if k != 'evidence'}:
            raise ValueError('paper identity/metadata snapshot mismatch')
        for unit in paper['evidence']:
            if unit['paper_id'] != paper['paper_id']:
                raise ValueError('evidence/group owner mismatch')
            binding = {k: unit[k] for k in ('artifact_id', 'content_sha256', 'start', 'end',
                                           'representation_version', 'table', 'passage')}
            if 'source_segments' in unit:
                binding['source_segments'] = unit['source_segments']
            for origin in unit['origins']:
                score = origin['semantic_score']
                if (origin['score_semantics'] != SCORE or isinstance(score, bool)
                        or not isinstance(score, (int, float)) or not math.isfinite(score)
                        or not -1 <= score <= 1):
                    raise ValueError('origin score semantics mismatch')
                hit = dict(source=origin['source'], snippet=origin['snippet'], section=unit['section'],
                           chunk=unit['chunk'], embedding_text_version=unit['representation_version'])
                rebuilt = _unit(hit, binding, snapshot, sources,
                                segmented=pack['schema'] == SEGMENTED_SCHEMA)
                if {k:v for k,v in unit.items() if k != 'origins'} != {
                        k:v for k,v in rebuilt.items() if k != 'origins'}:
                    raise ValueError('evidence content/identity mismatch')
            if not unit['origins'] or unit['evidence_id'] in seen:
                raise ValueError('missing origins or duplicate selected evidence')
            seen.add(unit['evidence_id'])
            used += len(unit['passage'])
    _verify_budget(pack['budget'], seen, used)
    return dict(status='verified', evidence=len(seen), papers=len(pack['papers']))
