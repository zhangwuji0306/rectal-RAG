"""D3 descriptive accounting of explicitly supplied, independently scored records.

No retrieval, model, semantic labelling or answer publication. This validates
associations and exact grounds; it cannot authenticate an annotator or entailment.
"""
from collections import Counter
import argparse
import json
import math
from pathlib import Path

VIEWS = ('raw', 'automatic', 'final')
UNKNOWN = (None, 'unresolved', 'missing_original')
SUPPORT = ('supported', 'contradicted', 'insufficient')
ERRORS = ('retrieval_omission', 'pack_missing', 'pack_truncation', 'pack_unbound',
          'generation_omission', 'generation_distortion', 'generation_number',
          'generation_unsupported', 'citation_selection', 'citation_rendering',
          'support_verification', 'conflict_handling', 'abstention', 'human_incomplete')


def prepare_pack(hits, records, locations, snapshot, sources, *, max_chars, paper_limit=50):
    """D3 handoff preserving a dense prefix and every unavailable row.

    records/locations are keyed by retrieval ID. An unavailable location is
    {status:'unbound', reason:...}; a bound one is {status:'bound', location:...}.
    Never drop unbound rows or renumber survivors to satisfy D1. Prefix selection
    stops before the first new paper exceeding the limit; it is not a sparse
    filter of later hits from the retained papers. All original inputs survive.
    """
    from d2_adapters import pack_from_records
    if type(paper_limit) is not int or paper_limit < 1:
        raise ValueError('positive paper limit required')
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError('nonnegative evidence budget required')
    ids = [text(h['id']) for h in hits]
    if len(set(ids)) != len(ids) or [h['rank'] for h in hits] != list(range(1, len(hits)+1)):
        raise ValueError('complete consecutive original dense trace required')
    if set(records) != set(ids) or set(locations) != set(ids):
        raise ValueError('one full row and explicit location status per original hit required')
    for hit in hits:
        score = hit['score']
        if (type(hit['rank']) is not int or type(score) not in (int, float)
                or not math.isfinite(score) or not -1 <= score <= 1
                or any(k in hit for k in ('contributions', 'fusion_score', 'quality_score'))):
            raise ValueError('original dense rank/score semantics required')
        row, loc = records[hit['id']], locations[hit['id']]
        if row['id'] != hit['id']:
            raise ValueError('full row/retrieval association mismatch')
        if loc['status'] == 'unbound':
            if set(loc) != {'status', 'reason'}:
                raise ValueError('unbound rows require an explicit reason')
            text(loc['reason'])
        elif loc['status'] != 'bound' or set(loc) != {'status', 'location'}:
            raise ValueError('invalid explicit location status')
    owners = {}
    for artifact in snapshot.artifacts.values():
        owners.setdefault(artifact['source'], set()).add(artifact['paper_id'])
    papers, prefix, failures = set(), [], []
    for hit in hits:
        owner = owners[hit['source']]
        if len(owner) != 1:
            raise ValueError('ambiguous source ownership')
        pid = next(iter(owner))
        if pid not in papers and len(papers) == paper_limit:
            break
        papers.add(pid); prefix.append(hit)
        loc = locations[hit['id']]
        if loc['status'] == 'unbound':
            failures.append(dict(retrieval_id=hit['id'], rank=hit['rank'], reason=text(loc['reason'])))
    result = dict(status='blocked_unbound' if failures else 'ready', pack=None,
                  selected_retrieval_ids=[h['id'] for h in prefix],
                  unselected_retrieval_ids=ids[len(prefix):], failures=failures)
    if not failures:
        result['pack'] = pack_from_records(prefix, [records[h['id']] for h in prefix],
            [locations[h['id']]['location'] for h in prefix], snapshot, sources, max_chars=max_chars)
    return result


def blank():
    return dict(label=None, assessor=None, basis=None, reason=None, read_scope=None, grounds=[])


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('expected nonempty text')
    return value


def unique(rows, key):
    if not isinstance(rows, list):
        raise ValueError('expected list: ' + key)
    result = {}
    for row in rows:
        value = text(row[key])
        if value in result:
            raise ValueError('duplicate ' + key + ': ' + value)
        result[value] = row
    return result


def judgement(row, labels, artifacts):
    if set(row) != set(blank()):
        raise ValueError('invalid judgement fields')
    label = row['label']
    if label is not None and (type(label) is not str or label not in (*labels, 'unresolved', 'missing_original')):
        raise ValueError('invalid label')
    if label is None:
        if row != blank():
            raise ValueError('unjudged must be blank; retain drafts outside scoring')
        return
    for key in ('assessor', 'reason', 'read_scope'):
        text(row[key])
    if row['basis'] not in ('human', 'agent_diagnostic'):
        raise ValueError('invalid assessment basis')
    if not isinstance(row['grounds'], list):
        raise ValueError('grounds must be a list')
    for g in row['grounds']:
        if set(g) != {'artifact_id', 'version', 'start', 'end', 'quote'}:
            raise ValueError('invalid ground fields')
        a = artifacts[g['artifact_id']]
        if (g['version'] != a['version'] or type(g['start']) is not int
                or type(g['end']) is not int or not 0 <= g['start'] < g['end'] <= len(a['text'])
                or a['text'][g['start']:g['end']] != g['quote']):
            raise ValueError('ground/source version or exact range mismatch')
    if label in ('supported', 'contradicted', 'relevant') and not row['grounds']:
        raise ValueError('located grounds required')


def validate(case):
    text(case['question_id']); text(case['question']); text(case['history'])
    if case['split'] not in ('development', 'test'):
        raise ValueError('invalid split')
    artifacts = unique(case['artifacts'], 'artifact_id')
    for a in artifacts.values():
        text(a['version']); text(a['text'])
    expected = unique(case['expected_points'], 'point_id')
    for p in expected.values():
        text(p['text'])
    judgement(case['answerability'], ('sufficient', 'insufficient', 'conflicting'), artifacts)
    pool = unique(case['pool'], 'paper_id')
    for p in pool.values():
        judgement(p['relevance'], ('relevant', 'irrelevant'), artifacts)
    if not isinstance(case['retrieved_papers'], list):
        raise ValueError('retrieved papers must be a list')
    if type(case['retrieval_complete']) is not bool:
        raise ValueError('explicit retrieval trace status required')
    if not case['retrieval_complete'] and case['retrieved_papers']:
        raise ValueError('incomplete trace must not masquerade as complete paper ranking')
    # Passage duplicates are permitted here, and deduplicated at paper level.
    for pid in case['retrieved_papers']:
        text(pid)
    if set(case['outputs']) != set(VIEWS):
        raise ValueError('raw/automatic/final outputs must all be retained')
    for view, output in case['outputs'].items():
        text(output['artifact_ref'])
        if output['state'] not in ('answered', 'partial', 'abstained', 'failed', 'not_run'):
            raise ValueError('invalid output state')
        if not isinstance(output['text'], str) or not isinstance(output['events'], list):
            raise ValueError('retain output text and automatic events')
        if output['state'] == 'not_run' and (output['text'] or output['claims']):
            raise ValueError('not_run cannot contain an answer or claims')
        if output['state'] in ('answered', 'partial') and not output['text'].strip():
            raise ValueError('answered/partial requires actual output text')
        if output['state'] in ('abstained', 'failed') and output['claims']:
            # A failed raw response can still emit headings with references.
            # Keep those references in the ALL-citation denominator without
            # turning a non-answer into an answered/partial clinical response.
            nonanswer_references = (view == 'raw' and output['state'] == 'failed'
                and all(c['important'] is False and c['citation_required'] is False
                        for c in output['claims']))
            if not nonanswer_references:
                raise ValueError('evaluate withheld candidates in raw, not in failed/refusal prose')
        for event in output['events']:
            text(event['kind']); text(event['detail'])
        claims = unique(output['claims'], 'fact_id')
        for c in claims.values():
            text(c['text']); text(c['parent_claim_id'])
            if type(c['important']) is not bool or type(c['citation_required']) is not bool:
                raise ValueError('explicit claim applicability required')
            judgement(c['support'], SUPPORT, artifacts)
            judgement(c['hallucination'], ('yes', 'no'), artifacts)
            citations = unique(c['citations'], 'citation_id')
            papers = set()
            for citation in citations.values():
                pid = text(citation['paper_id'])
                if pid in papers:
                    raise ValueError('combine same-paper passages into one claim citation')
                papers.add(pid)
                if not citation['evidence_ids'] or len(set(citation['evidence_ids'])) != len(citation['evidence_ids']):
                    raise ValueError('citation requires unique evidence ids')
                for eid in citation['evidence_ids']:
                    text(eid)
                judgement(citation['validity'], ('valid', 'invalid'), artifacts)
                judgement(citation['support'], SUPPORT, artifacts)
        points = unique(output['point_scores'], 'point_id')
        if set(points) != set(expected):
            raise ValueError('each expected point must be scored, including failures/refusals')
        for p in points.values():
            judgement(p['score'], ('covered', 'missing'), artifacts)
            if output['state'] == 'not_run' and p['score']['label'] is not None:
                raise ValueError('not_run requires unjudged point scores')
            if not output['text'].strip() and p['score']['label'] == 'covered':
                raise ValueError('covered requires actual output text')
        judgement(output['conflict'], ('correct', 'incorrect', 'not_applicable'), artifacts)
        judgement(output['refusal'], ('correct', 'incorrect', 'not_applicable'), artifacts)
        if output['state'] == 'not_run' and any(output[k]['label'] is not None for k in ('conflict', 'refusal')):
            raise ValueError('not_run requires unjudged quality scores')
        if output['refusal']['label'] == 'correct' and (
                output['state'] != 'abstained' or not output['text'].strip()):
            raise ValueError('only an actual refusal with text can be scored correct')
        if len(set(output['errors'])) != len(output['errors']) or any(e not in ERRORS for e in output['errors']):
            raise ValueError('invalid or duplicate error attribution')
        if view == 'final' and output['corrected_by']:
            text(output['corrected_by'])
            scores = [c[k] for c in claims.values() for k in ('support', 'hallucination')]
            scores += [p['score'] for p in points.values()]
            scores += [output['conflict'], output['refusal']]
            scores += [c[k] for atom in claims.values() for c in atom['citations'] for k in ('validity', 'support')]
            if any(j['label'] is not None and j['assessor'] == output['corrected_by'] for j in scores):
                raise ValueError('answer corrector cannot independently score final answer')
    return case


def metric(labels, positive, *, inapplicable=0):
    counts = Counter(labels)
    denominator = sum(counts[x] for x in counts if x not in UNKNOWN)
    numerator = sum(counts[x] for x in positive)
    return dict(numerator=numerator, denominator=denominator,
                value=numerator / denominator if denominator else None,
                unjudged=counts[None], unresolved=counts['unresolved'],
                missing_original=counts['missing_original'], inapplicable=inapplicable,
                eligibility_unjudged=0, eligibility_unresolved=0, eligibility_missing_original=0)


def scoped(j, basis):
    return j['label'] if j['label'] is None or j['basis'] == basis else None


def score_case(case, view, basis):
    validate(case)
    out = case['outputs'][view]
    facts = [c for c in out['claims'] if c['important']]
    required = [c for c in facts if c['citation_required']]
    citations = [r for c in out['claims'] for r in c['citations']]
    # Required structural rates keep ALL applicable claims/citations in their
    # denominator, including unjudged. Unknowns produce an interval, not a zero.
    def full(labels, positive, **kwargs):
        m = metric(labels, positive, **kwargs)
        unknown = m['unjudged'] + m['unresolved'] + m['missing_original']
        m['denominator'] += unknown
        m['value'] = m['numerator']/m['denominator'] if m['denominator'] and not unknown else None
        m['lower_bound'] = m['numerator']/m['denominator'] if m['denominator'] else None
        m['upper_bound'] = (m['numerator']+unknown)/m['denominator'] if m['denominator'] else None
        return m
    result = dict(
        invalid_citation_rate=full([scoped(c['validity'], basis) for c in citations], ('invalid',)),
        citation_support=full([scoped(c['support'], basis) for c in citations], ('supported',)),
        claim_support=full([scoped(c['support'], basis) for c in facts], ('supported',)),
        unsupported_claim_rate=full([scoped(c['support'], basis) for c in facts], ('contradicted', 'insufficient')),
        hallucination=full([scoped(c['hallucination'], basis) for c in facts], ('yes',)),
        missing_citation_rate=full(['missing' if not c['citations'] else 'present' for c in required], ('missing',)),
        citation_completeness=full(['missing' if not c['citations'] else 'present' for c in required], ('present',)),
        answer_completeness=full([scoped(p['score'], basis) for p in out['point_scores']], ('covered',)),
    )
    excluded_facts = len(out['claims']) - len(facts)
    for name in ('claim_support', 'unsupported_claim_rate', 'hallucination'):
        result[name]['inapplicable'] = excluded_facts
    for name in ('missing_citation_rate', 'citation_completeness'):
        result[name]['inapplicable'] = len(out['claims']) - len(required)
    a = scoped(case['answerability'], basis)
    def gate(m, condition):
        m['inapplicable'] = int(a not in UNKNOWN and a != condition)
        for label, key in ((None, 'eligibility_unjudged'), ('unresolved', 'eligibility_unresolved'),
                           ('missing_original', 'eligibility_missing_original')):
            m[key] = int(a == label)
        return m
    result['sufficient_answer_rate'] = gate(full(
        [None if out['state'] == 'not_run' else 'yes' if out['state'] in ('answered', 'partial') else 'no']
        if a == 'sufficient' else [], ('yes',),
        ), 'sufficient')
    for name, field, condition in (('correct_refusal_rate', 'refusal', 'insufficient'),
                                    ('conflict_handling', 'conflict', 'conflicting')):
        label = scoped(out[field], basis)
        if a == condition and label == 'not_applicable':
            raise ValueError('applicable question cannot have not_applicable score')
        if a == condition and label == 'correct':
            if field == 'refusal' and out['state'] != 'abstained':
                raise ValueError('only an actual refusal can be scored correct')
            if out['state'] in ('failed', 'not_run'):
                raise ValueError('failed/not_run cannot be scored correct')
        result[name] = gate(full([label] if a == condition else [], ('correct',)), condition)
    relevant = [p['paper_id'] for p in case['pool'] if scoped(p['relevance'], basis) == 'relevant']
    retrieved = set(case['retrieved_papers'])
    result['pooled_candidate_recall'] = metric([
        ('hit' if pid in retrieved else 'miss') if case['retrieval_complete'] else 'retrieval_unknown'
        for pid in relevant], ('hit',))
    recall = result['pooled_candidate_recall']
    recall['retrieval_unknown'] = len(relevant) if not case['retrieval_complete'] else 0
    recall['value'] = recall['value'] if not recall['retrieval_unknown'] else None
    recall['lower_bound'] = recall['numerator']/recall['denominator'] if recall['denominator'] else None
    recall['upper_bound'] = (recall['numerator']+recall['retrieval_unknown'])/recall['denominator'] if recall['denominator'] else None
    result['pooled_candidate_recall']['unjudged'] = sum(scoped(p['relevance'], basis) is None for p in case['pool'])
    result['pooled_candidate_recall']['unresolved'] = sum(scoped(p['relevance'], basis) == 'unresolved' for p in case['pool'])
    result['pooled_candidate_recall']['missing_original'] = sum(scoped(p['relevance'], basis) == 'missing_original' for p in case['pool'])
    return dict(question_id=case['question_id'], split=case['split'], view=view, basis=basis,
                state=out['state'], metrics=result, answerability=a, errors=out['errors'],
                retrieved_unpooled=len(retrieved - {p['paper_id'] for p in case['pool']}))


def evaluate(cases, *, basis='human'):
    if basis not in ('human', 'agent_diagnostic'):
        raise ValueError('invalid basis')
    unique(cases, 'question_id')
    rows = [score_case(c, v, basis) for c in cases for v in VIEWS]
    summaries = []
    for split in ('development', 'test', 'overall'):
        for view in VIEWS:
            selected = [r for r in rows if r['view'] == view and (split == 'overall' or r['split'] == split)]
            totals = {}
            for name in rows[0]['metrics'] if rows else ():
                values = [r['metrics'][name] for r in selected]
                sums = {key: sum(m[key] for m in values) for key in (
                    'numerator', 'denominator', 'unjudged', 'unresolved', 'missing_original', 'inapplicable',
                    'eligibility_unjudged', 'eligibility_unresolved', 'eligibility_missing_original')}
                if name == 'pooled_candidate_recall':
                    sums['retrieval_unknown'] = sum(m['retrieval_unknown'] for m in values)
                valid = [m['value'] for m in values if m['value'] is not None]
                unknown = sums['unjudged'] + sums['unresolved'] + sums['missing_original']
                # Recall is conditional on judged relevant papers; its unknown
                # pool members are reported separately, not invented negatives.
                if name == 'pooled_candidate_recall':
                    unknown = sums['retrieval_unknown']
                complete = not unknown
                totals[name] = dict(**sums, value=sums['numerator']/sums['denominator']
                                    if sums['denominator'] and complete else None,
                                    macro=sum(valid)/len(valid) if valid else None, macro_questions=len(valid))
                totals[name].update(lower_bound=sums['numerator']/sums['denominator'] if sums['denominator'] else None,
                                    upper_bound=(sums['numerator']+unknown)/sums['denominator'] if sums['denominator'] else None)
            summaries.append(dict(split=split, view=view, questions=len(selected), metrics=totals,
                                  answerability=dict(Counter(str(r['answerability']) for r in selected)),
                                  states=dict(Counter(r['state'] for r in selected)),
                                  errors=dict(Counter(e for r in selected for e in r['errors']))))
    return dict(schema='d3-evaluation-v1', basis=basis, disposition='descriptive_only_no_acceptance_threshold',
                per_question=rows, summary=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--basis', choices=('human', 'agent_diagnostic'), default='human')
    args = parser.parse_args()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite number: ' + value)
    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number
    cases = json.loads(args.records.read_text(encoding='utf-8'), object_pairs_hook=pairs,
                       parse_constant=invalid, parse_float=finite)
    result = evaluate(cases, basis=args.basis)
    # Never overwrite prior scores or raw material.
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
