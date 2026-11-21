"""D2 pure local answer organization; no model, retrieval, network or file I/O.

The built-in verifier interprets ONLY the explicit formal record language below.
It is a limited rule engine, not a prose entailment model. No substring/keyword
test or model-reported support label can promote ordinary prose to supported.
Generator adapters are caller-supplied trusted local code; evidence is data.
"""
from copy import deepcopy
from decimal import Decimal
import json
import math
import re

import evidence_pack as ep

SCHEMA = 'd2-candidate-v1'
FIELDS = ('population', 'condition', 'outcome', 'comparator', 'value', 'unit',
          'direction', 'polarity', 'time', 'relation')
LABEL = r'[a-z][a-z0-9_-]*'
RECORD = re.compile(
    r'population=(?P<population>' + LABEL + r'); condition=(?P<condition>' + LABEL
    + r'); outcome=(?P<outcome>' + LABEL + r'); comparator=(?P<comparator>' + LABEL
    + r'); value=(?P<value>-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?); unit=(?P<unit>' + LABEL
    + r'); direction=(?P<direction>higher|lower|equal); polarity=(?P<polarity>positive|negative); '
    r'time=(?P<time>' + LABEL + r'); relation=(?P<relation>association|causal)\.')


def parse_record(text):
    """Every field is literal; no synonyms, conversions, inference or extraction.

    Population/condition/outcome/comparator/time/unit must agree exactly. A
    negative polarity negates this numeric directional relation. Multiple lines
    and prose are outside this grammar (including compound claims).
    """
    match = RECORD.fullmatch(text)
    if match is None:
        return None
    fact = match.groupdict()
    fact['value'] = Decimal(fact['value'])
    return fact


def _text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('expected nonempty string')
    return value


def selected(pack):
    if pack['schema'] not in (ep.SCHEMA, ep.SEGMENTED_SCHEMA):
        raise ValueError('unsupported Evidence Pack schema')
    units = {}
    for paper in pack['papers']:
        for evidence in paper['evidence']:
            eid = _text(evidence['evidence_id'])
            if eid in units or evidence['paper_id'] != paper['paper_id']:
                raise ValueError('duplicate evidence or inconsistent owner')
            units[eid] = (paper, evidence)
    return units


def validate_candidate(candidate, pack):
    """Exact external schema; model provides no bibliography or support verdict.

    claims is an ordered list of atomic claims. Split compound conclusions into
    separate items; an unparsed item cannot inherit another item's support.
    confidence is a finite self-rating [0,1], never a calibrated probability.
    """
    units = selected(pack)
    if (not isinstance(candidate, dict) or set(candidate) != {'schema', 'claims'}
            or candidate['schema'] != SCHEMA or not isinstance(candidate['claims'], list)):
        raise ValueError('invalid candidate schema')
    seen = set()
    for claim in candidate['claims']:
        if not isinstance(claim, dict) or set(claim) != {'claim', 'evidence_ids', 'confidence'}:
            raise ValueError('invalid claim schema')
        text = _text(claim['claim'])
        if text in seen:
            raise ValueError('duplicate claim; combine its references')
        seen.add(text)
        ids, confidence = claim['evidence_ids'], claim['confidence']
        if (not isinstance(ids, list) or any(not isinstance(i, str) or i not in units for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError('unknown, omitted, foreign or duplicate evidence ID')
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            raise ValueError('confidence must be finite self-rating in [0,1]')
    return deepcopy(candidate)


def assess(claim, passage):
    """Formal-record semantics only. No fixture answers or support labels used.

    Under matching context, (value, direction) identifies the exact measurement
    proposition. Opposite polarities contradict only that SAME proposition.
    Distinct positive measurements are incompatible in this single-measurement
    language. Negating a different proposition proves neither support nor
    contradiction here; no contraposition or cross-proposition inference.
    """
    a, b = parse_record(claim), parse_record(passage)
    if a is None or b is None:
        return dict(status='unresolved', reason='outside_formal_record_language')
    mismatches = [k for k in FIELDS if a[k] != b[k]]
    if not mismatches:
        return dict(status='supported', reason='all_ten_formal_fields_agree')
    context = {'population', 'condition', 'outcome', 'comparator', 'unit', 'time', 'relation'}
    same_proposition = a['value'] == b['value'] and a['direction'] == b['direction']
    incompatible = (same_proposition or a['polarity'] == b['polarity'] == 'positive')
    status = 'contradicted' if not context.intersection(mismatches) and incompatible else 'insufficient'
    return dict(status=status, reason='different_fields:' + ','.join(mismatches))


def citations(ids, units):
    """Group by C1 paper, retain every passage locator; never count studies."""
    groups = {}
    for eid in ids:
        paper, unit = units[eid]
        pid = paper['paper_id']
        group = groups.setdefault(pid, dict(paper_id=pid, title=deepcopy(paper['title']),
                                          doi=deepcopy(paper['doi']), pmid=deepcopy(paper['pmid']),
                                          locations=[]))
        group['locations'].append({k: deepcopy(unit[k]) for k in (
            'evidence_id', 'artifact_id', 'source', 'content_sha256', 'representation_version',
            'chunk', 'start', 'end', 'offset_unit', 'section', 'passage')})
        if 'source_segments' in unit:
            group['locations'][-1].update(location_mode=unit['location_mode'],
                                         source_segments=deepcopy(unit['source_segments']))
    return list(groups.values())


def answer(pack, candidate=None, *, snapshot=None, sources=None, generator=None, question=None):
    """Recheck D1 when dependencies supplied; errors abort instead of downgrading.

    generator(question, pack copy) -> candidate. No default model exists.
    The narrow formal verifier is always programmatic, never a generator verdict.
    Missing dependencies keep claims unpublished even with a formal match.
    No output field in a candidate is accepted as a verification certificate.
    """
    pack = deepcopy(pack)
    units = selected(pack)
    if (snapshot is None) != (sources is None):
        raise ValueError('snapshot and sources must be supplied together')
    location = 'unverified'
    if snapshot is not None:
        ep.verify(pack, snapshot, sources)
        location = 'verified'
    if candidate is not None and generator is not None:
        raise ValueError('provide candidate or generator, not both')
    generation = 'provided_candidate' if candidate is not None else 'unavailable'
    if generator is not None:
        _text(question)
        try:
            candidate = generator(question, deepcopy(pack))
            generation = 'injected_local_adapter'
        except (TimeoutError, ConnectionError) as exc:
            generation = 'unavailable:' + type(exc).__name__
        else:
            if candidate is None:
                raise ValueError('generator must return candidate schema')
    if candidate is None:
        candidate = dict(schema=SCHEMA, claims=[])
    candidate = validate_candidate(candidate, pack)
    claims = []
    for item in candidate['claims']:
        # Compare all selected evidence, including counterevidence the candidate
        # omitted. This is not a claim that original retrieval was complete.
        checks = [dict(evidence_id=eid, **assess(item['claim'], unit['passage']))
                  for eid, (_, unit) in units.items()]
        support = [c['evidence_id'] for c in checks if c['status'] == 'supported']
        oppose = [c['evidence_id'] for c in checks if c['status'] == 'contradicted']
        cited = [c for c in checks if c['evidence_id'] in item['evidence_ids']]
        status = 'insufficient'
        reason = 'no_cited_support'
        if support and oppose:
            status, reason = 'conflict', 'selected_evidence_disagrees'
        elif cited and all(c['status'] == 'supported' for c in cited):
            status, reason = 'supported', 'every_cited_record_supports_atomic_claim'
        elif any(c['status'] == 'contradicted' for c in cited):
            status, reason = 'contradicted', 'cited_record_denies_claim'
        elif any(c['status'] == 'unresolved' for c in cited):
            status, reason = 'unresolved', 'requires_prose_semantic_backend_or_atomic_split'
        publish = status == 'supported' and location == 'verified'
        claims.append(dict(**item, support_status=status, reason=reason,
                           publication='verified_formal_claim' if publish else 'withheld',
                           checks=checks, citations=citations(item['evidence_ids'], units),
                           supporting_A=citations(support, units) if status == 'conflict' else [],
                           supporting_B=citations(oppose, units) if status == 'conflict' else []))
    published = sum(c['publication'] == 'verified_formal_claim' for c in claims)
    state = ('answered' if published == len(claims) else 'partial') if published else 'abstained'
    return dict(schema='d2-answer-v1', state=state, location_status=location,
                generation=generation, verifier='formal-record-rules-v1',
                confidence_semantics='uncalibrated_self_rating_not_used_for_acceptance',
                reason=None if published else ('no_candidate_or_no_evidence' if not claims or not units
                                                else 'unverified_location_or_unsupported_claims'),
                candidate=candidate, claims=claims)


def render(result):
    """Plain text with JSON-quoted data, no generated Markdown/HTML/tools/URLs.

    Withheld candidate text stays in the diagnostic object, never in the final
    answer. Conflicts show each source's own assertion and locator.
    """
    quote = lambda value: json.dumps(value, ensure_ascii=False, allow_nan=False)
    ability = ('人工逐项确认，不代表自动医学文本语义核验。' if result['verifier'] == 'human-reviewed-prose-v1'
               else '仅 formal-record-rules-v1，不代表普通医学文本语义核验。')
    lines = ['程序状态: ' + result['state'], '定位复核: ' + result['location_status'], '核验能力: ' + ability]
    for claim in result['claims']:
        if claim['publication'] in ('verified_formal_claim', 'verified_human_reviewed_claim'):
            label = '经人工确认的文献陈述: ' if claim['publication'] == 'verified_human_reviewed_claim' else '已核验形式化文献陈述: '
            lines += [label + quote(claim['claim']),
                      '引用及原文定位: ' + quote(claim['citations'])]
        else:
            lines.append('程序说明: 候选未发布; ' + claim['support_status'] + '; ' + claim['reason'])
        if claim['support_status'] == 'conflict':
            label = '已定位' if result['location_status'] == 'verified' else '未复核定位'
            lines += [label + '冲突证据 supporting A: ' + quote(claim['supporting_A']),
                      label + '冲突证据 supporting B: ' + quote(claim['supporting_B'])]
    return '\n'.join(lines)
