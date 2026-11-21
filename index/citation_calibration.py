"""Offline C3 label checking, calibration, held-out validation and rule application.

The four CLI stages are separate. Only check reads both label groups; calibrate
rejects validation input. Apply consumes a fixed rule and validation decision,
never labels. Original graph, materials and answers are opened read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
from decimal import Decimal
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3

import citation_graph as cg
from paper_registry import normalize_title

ALGORITHM = 'c3-fuzzy-calibration-v1'
LABELS = {'confirmed_match', 'no_confirmed_local_match', 'ambiguous',
          'insufficient_evidence', 'not_reference_or_bad_boundary'}
SCORES = (0.85, 0.90, 0.95, 0.98, 1.00)
MARGINS = (0.05, 0.10)
REJECTIONS = ['original_boundary_not_parsed', 'unsafe_citing_identity',
              'non_main_role', 'not_fuzzy_candidate', 'strong_identifier_present',
              'no_candidates', 'unsafe_target_identity', 'self_loop',
              'tied_top_score', 'year_absent', 'bibliographic_conflict',
              'below_score', 'below_margin']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def validate_answer(answer, item, version, papers):
    if answer['version'] != version or answer['ref_id'] != item['ref_id']:
        raise ValueError('label version/ref_id mismatch: '+item['ref_id'])
    boundary, label = answer['boundary'], answer['human_label']
    if boundary not in ('valid', 'invalid', 'uncertain') or label not in LABELS:
        raise ValueError('invalid boundary/label: '+item['ref_id'])
    if ((boundary == 'invalid') != (label == 'not_reference_or_bad_boundary')
            or (boundary == 'uncertain' and label != 'insufficient_evidence')):
        raise ValueError('inconsistent boundary/label: '+item['ref_id'])
    reason = answer['human_reason']
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 5000:
        raise ValueError('invalid human reason: '+item['ref_id'])
    target = answer['human_target_paper_id']
    if label == 'confirmed_match':
        if boundary != 'valid' or target not in papers or papers[target]['identity_status'] != 'verified':
            raise ValueError('confirmed target must be verified: '+item['ref_id'])
    elif target != '':
        raise ValueError('unconfirmed answer has target: '+item['ref_id'])


def leakage(items, refs, papers, sources, relations):
    """Detect shared identities/content/declared versions and duplicate citations.

    Pending duplicate/version relations are leakage risks only, never identity
    merges. Near duplicates use token Dice >= .95 after entry-number removal.
    """
    adjacency = defaultdict(set)
    for relation in relations:
        if relation['kind'] in ('exact_duplicate', 'near_duplicate', 'alternate_version'):
            a, b = relation['from_paper_id'], relation['to_paper_id']
            adjacency[a].add(b)
            adjacency[b].add(a)
    for p, paper in papers.items():
        if paper.get('duplicate_of'):
            adjacency[p].add(paper['duplicate_of'])
            adjacency[paper['duplicate_of']].add(p)
    components = {}
    for p in papers:
        if p in components:
            continue
        todo, seen = [p], set()
        while todo:
            q = todo.pop()
            if q not in seen:
                seen.add(q)
                todo.extend(adjacency[q]-seen)
        components.update({q: p for q in seen})
    shas = defaultdict(set)
    for source in sources:
        shas[source['paper_id']].add(source['content_sha256'])
    found = []
    for a, b in itertools.combinations(items, 2):
        if a['split'] == b['split']:
            continue
        p, q = a['citing_paper_id'], b['citing_paper_id']
        reasons = []
        if components[p] == components[q]:
            reasons.append('shared_citing_identity_or_version_duplicate_relation')
        if shas[p] & shas[q]:
            reasons.append('shared_citing_content')
        x, y = [normalize_title(cg.NUMBER.sub('', refs[i['ref_id']]['raw'], count=1)) for i in (a, b)]
        if x == y:
            reasons.append('duplicate_reference')
        else:
            tx, ty = set(x.split()), set(y.split())
            if tx and ty and 2*len(tx & ty)/(len(tx)+len(ty)) >= .95:
                reasons.append('near_duplicate_reference_dice_ge_0.95')
        if reasons:
            found.append(dict(ref_ids=[a['ref_id'], b['ref_id']], reasons=reasons))
    return found


def check(materials, labels, calibration_csv, validation_csv, db, output, expected=48):
    package = read_json(materials)
    items = package['items']
    if len(items) != expected or len({i['ref_id'] for i in items}) != expected:
        raise ValueError('material ref_id count/uniqueness mismatch')
    if Counter(i['split'] for i in items) != {'calibration': expected//2, 'validation': expected//2}:
        raise ValueError('material split count mismatch')
    c = cg.open_graph(db)
    try:
        meta = dict(c.execute('SELECT * FROM c3_meta'))
        if package['version'] != meta['citation_graph_version']:
            raise ValueError('material graph version mismatch')
        papers = {r[0]: json.loads(r[1]) for r in c.execute('SELECT * FROM nodes')}
        sources = [json.loads(r[0]) for r in c.execute('SELECT payload FROM sources')]
        refs = {}
        for item in items:
            row = c.execute('SELECT payload FROM refs WHERE ref_id=?', (item['ref_id'],)).fetchone()
            if row is None:
                raise ValueError('material ref_id absent from graph: '+item['ref_id'])
            r = json.loads(row[0])
            if (any(item[k] != r[k] for k in ('ref_id','raw','source','line_start','line_end','parse_status'))
                    or item['citing_paper_id'] != r['paper_id']
                    or sorted(item['candidates']) != sorted(x['paper_id'] for x in r['match']['candidates'])):
                raise ValueError('material/reference mismatch: '+item['ref_id'])
            selected = [x['text'] for x in item['context'] if x['selected']]
            if '\n'.join(selected) != r['raw']:
                raise ValueError('material context mismatch: '+item['ref_id'])
            refs[item['ref_id']] = r
    finally:
        c.close()
    if set(package['catalog']) != set(papers):
        raise ValueError('material catalog identity set mismatch')
    for p, record in package['catalog'].items():
        if any(record[k] != papers[p].get(k, '') for k in record):
            raise ValueError('material catalog metadata mismatch: '+p)
    c = cg.readonly(labels)
    try:
        if [r[0] for r in c.execute('SELECT version FROM scope')] != [package['version']]:
            raise ValueError('labels scope mismatch')
        answers = {}
        for rid, payload in c.execute('SELECT ref_id,payload FROM labels'):
            answer = json.loads(payload)
            if rid != answer['ref_id'] or rid in answers:
                raise ValueError('label row key mismatch/duplicate')
            answers[rid] = answer
    finally:
        c.close()
    if set(answers) != set(refs):
        raise ValueError('labels/material ref_id sets differ')
    for item in items:
        validate_answer(answers[item['ref_id']], item, package['version'], papers)
    for split, path in [('calibration', calibration_csv), ('validation', validation_csv)]:
        if not Path(path).read_bytes().startswith(b'\xef\xbb\xbf'):
            raise ValueError('CSV must be UTF-8 with BOM')
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        subset = {i['ref_id']: i for i in items if i['split'] == split}
        if len(rows) != len(subset) or {r['ref_id'] for r in rows} != set(subset):
            raise ValueError('CSV ref_id set/count mismatch: '+split)
        for row in rows:
            i, a = subset[row['ref_id']], answers[row['ref_id']]
            expected_row = {k: str(v) for k, v in dict(version=package['version'], split=split,
                ref_id=i['ref_id'], citing_paper_id=i['citing_paper_id'], source=i['source'],
                line_start=i['line_start'], line_end=i['line_end'], raw_reference=i['raw'],
                parse_status=i['parse_status'], **{k: a[k] for k in ('boundary','human_target_paper_id','human_label','human_reason')}).items()}
            if row != expected_row:
                raise ValueError('CSV/labels field mismatch: '+row['ref_id'])
    c = cg.readonly(json.loads(meta['inputs'])['c1'])
    try:
        relations = [dict(r) for r in c.execute('SELECT * FROM relations')]
    finally:
        c.close()
    leaks = leakage(items, refs, papers, sources, relations)
    if leaks:
        raise ValueError('cross-group leakage: '+cg.dump(leaks))
    # Challenge names are source sampling strata, not assessed truth.
    original_csv = Path(db).parent/'human-review-unlabelled.csv'
    with original_csv.open(encoding='utf-8-sig', newline='') as stream:
        challenges = {r['ref_id']: r['case'] for r in csv.DictReader(stream)}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    for split in ('calibration','validation'):
        records = [dict(reference=refs[i['ref_id']], answer=answers[i['ref_id']],
                        challenge=challenges[i['ref_id']]) for i in items if i['split'] == split]
        write_json(out/(split+'-input.json'), dict(split=split, material_version=package['version'],
            matcher_rules=meta['rules'], papers=papers, records=records))
    result = dict(status='structural_checks_passed', material_version=package['version'],
                  total=len(items), groups=dict(Counter(i['split'] for i in items)),
                  program_boundaries=dict(Counter(i['parse_status'] for i in items)),
                  csv_equal=True, bom=True, cross_group_leakage=leaks,
                  human_conclusions='read as supplied; not re-certified by agent')
    write_json(out/'structure.json', result)
    return result


def load_input(path, split):
    data = read_json(path)
    if data['split'] != split or data['matcher_rules'] != cg.RULES:
        raise ValueError('wrong split/matcher algorithm for this entry point')
    ids = [x['reference']['ref_id'] for x in data['records']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('empty/duplicate input references')
    for record in data['records']:
        validate_answer(record['answer'], record['reference'], data['material_version'], data['papers'])
    return data


def rule(version, score=None, margin=None):
    if ((score is None) != (margin is None)
            or (score is not None and (score not in SCORES or margin not in MARGINS))):
        raise ValueError('rule outside finite search')
    value = dict(algorithm=ALGORITHM, material_version=version, matcher_rules=cg.RULES,
                 enabled=score is not None, score_threshold=score, margin_threshold=margin,
                 rank_order='score DESC, title_token_coverage DESC, paper_id ASC; saved Matcher candidates',
                 single_candidate_second_score=0, tied_top_score='reject',
                 year='candidate year must occur in reference years', rejection_conditions=REJECTIONS,
                 score_semantics='title similarity, not probability', exact='unchanged')
    value['rule_version'] = 'c3-rule-'+cg.fingerprint(value)[:24]
    return value


def validate_rule(value, version):
    if value['algorithm'] != ALGORITHM or value['material_version'] != version:
        raise ValueError('rule algorithm/material version mismatch')
    if value != rule(version, value['score_threshold'], value['margin_threshold']):
        raise ValueError('rule content/version differs from supported contract')
    if value['enabled'] and (value['score_threshold'] not in SCORES or value['margin_threshold'] not in MARGINS):
        raise ValueError('rule outside finite search')


def predict(ref, papers, fixed):
    """No human answer is accepted by this function. Never rerank by labels."""
    m = ref['match']
    result = dict(ref_id=ref['ref_id'], accepted=False, target=None, kind=None,
                  program_buildable=False, top1=None, score=None, second_score=None,
                  margin=None, rule_version=fixed['rule_version'])
    def reject(reason):
        return dict(result, reason=reason)
    if ref['parse_status'] != 'parsed':
        return reject('original_boundary_not_parsed')
    if papers[ref['paper_id']]['identity_status'] != 'verified':
        return reject('unsafe_citing_identity')
    if ref['role'] not in ('main','version','duplicate'):
        return reject('non_main_role')
    if m['status'] == 'exact':
        target = m['target']
        if papers[target]['identity_status'] != 'verified':
            raise ValueError('unsafe exact target violates Matcher contract')
        if target == ref['paper_id']:
            return reject('self_loop')
        return dict(result, accepted=True, target=target, kind='exact', program_buildable=True, reason='unchanged_exact')
    if m['status'] != 'fuzzy_candidate':
        return reject('not_fuzzy_candidate:'+m['status'])
    if m['identifiers'] or m['declarations'] or cg.declarations(ref['raw']):
        return reject('strong_identifier_present')
    candidates = m['candidates']
    if not candidates:
        raise ValueError('fuzzy_candidate without candidates')
    if len({x['paper_id'] for x in candidates}) != len(candidates):
        raise ValueError('duplicate candidate identity')
    for x in candidates:
        if x['paper_id'] not in papers or not math.isfinite(x['score']) or not 0 <= x['score'] <= 1:
            raise ValueError('invalid saved candidate')
        if any(x[k] != papers[x['paper_id']][k] for k in ('year','title','authors','identity_status')):
            raise ValueError('saved candidate/catalog mismatch')
    ranked = sorted(candidates, key=lambda x:(-x['score'], -x['title_token_coverage'], x['paper_id']))
    top = ranked[0]
    second = ranked[1]['score'] if len(ranked)>1 else 0
    margin = Decimal(str(top['score']))-Decimal(str(second))
    result.update(top1=top['paper_id'], score=top['score'], second_score=second, margin=float(margin), candidates=ranked)
    if papers[top['paper_id']]['identity_status'] != 'verified':
        return reject('unsafe_target_identity')
    if top['paper_id'] == ref['paper_id']:
        return reject('self_loop')
    result['program_buildable'] = True
    if top['score'] == second:
        return reject('tied_top_score')
    years = sorted(set(re.findall(r'\b(?:18|19|20)\d{2}\b',ref['raw'])))
    if top['reference_years'] != years:
        raise ValueError('saved reference years differ')
    if str(top['year']) not in years:
        return reject('year_absent')
    check = cg.bibliographic_check(ref['raw'], papers[top['paper_id']])
    result['bibliographic_check'] = check
    if check['decision'] == 'pending_review':
        return reject('bibliographic_conflict')
    if not fixed['enabled']:
        return reject('fuzzy_disabled')
    if top['score'] < fixed['score_threshold']:
        return reject('below_score')
    if margin < Decimal(str(fixed['margin_threshold'])):
        return reject('below_margin')
    return dict(result, accepted=True, target=top['paper_id'], kind='fuzzy', reason='finite_rule_accepted')


def evaluate(data, fixed):
    rows = []
    for record in data['records']:
        ref, answer = record['reference'], record['answer']
        prediction = predict(ref, data['papers'], fixed)
        if not prediction['accepted']:
            outcome = 'rejected'
        elif answer['human_label'] != 'confirmed_match':
            outcome = 'unconfirmed_acceptance'
        elif prediction['target'] == answer['human_target_paper_id']:
            outcome = 'correct_acceptance'
        else:
            outcome = 'wrong_binding'
        rows.append(dict(prediction=prediction, answer=answer, outcome=outcome,
            challenge=record['challenge'], parse_status=ref['parse_status'],
            positive=answer['human_label']=='confirmed_match',
            target_in_top3=answer['human_label']=='confirmed_match' and answer['human_target_paper_id'] in
                           [x['paper_id'] for x in ref['match']['candidates']],
            locator={k:ref[k] for k in ('source','path','artifact_id','line_start','line_end','raw')}))
    counts = Counter(r['outcome'] for r in rows)
    accepted = sum(r['prediction']['accepted'] for r in rows)
    positives = sum(r['positive'] for r in rows)
    correct, wrong, unconfirmed = [counts[k] for k in ('correct_acceptance','wrong_binding','unconfirmed_acceptance')]
    def ratio(n, d, formula):
        return dict(numerator=n, denominator=d, value=n/d if d else None, formula=formula)
    summary = dict(total=len(rows), program_buildable=sum(r['prediction']['program_buildable'] for r in rows),
        positives=positives, target_in_top3=sum(r['target_in_top3'] for r in rows),
        candidate_misses=sum(r['positive'] and not r['target_in_top3'] for r in rows),
        accepted=accepted, correct=correct, wrong_binding=wrong, unconfirmed_acceptance=unconfirmed,
        rejected=len(rows)-accepted,
        acceptance_precision=ratio(correct, correct+wrong, 'correct / (correct + wrong_binding); confirmed-target accepts only'),
        confirmed_correct_fraction=ratio(correct, accepted, 'correct / all accepts; unconfirmed accepts remain in denominator'),
        end_to_end_recall=ratio(correct, positives, 'correct / all human confirmed_match, including boundary and candidate misses'),
        boundary_comparison=dict(Counter(r['parse_status']+'/'+r['answer']['boundary'] for r in rows)),
        challenges={case:dict(Counter(r['outcome'] for r in rows if r['challenge']==case)) for case in sorted({r['challenge'] for r in rows})})
    summary['program_not_buildable'] = len(rows)-summary['program_buildable']
    return dict(rule=fixed, summary=summary, rows=rows)


def calibrate(input_path, output):
    data = load_input(input_path, 'calibration')
    combinations = [rule(data['material_version'])]+[rule(data['material_version'],s,m) for s,m in itertools.product(SCORES,MARGINS)]
    results = [evaluate(data, value) for value in combinations]
    safe = [r for r in results[1:] if r['summary']['wrong_binding']==r['summary']['unconfirmed_acceptance']==0]
    selected = max(safe, key=lambda r:(r['summary']['correct'],r['rule']['score_threshold'],r['rule']['margin_threshold'])) if safe else results[0]
    if selected['summary']['correct'] == 0:
        selected = results[0]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'all-combinations.json', results)
    write_json(out/'rule.json', selected['rule'])
    decision = dict(selected=selected['rule'], summary=selected['summary'],
        reason='zero wrong and zero unconfirmed; maximize correct, then score, then margin; zero correct disables',
        calibration_ref_ids=sorted(r['reference']['ref_id'] for r in data['records']),
        validation_used=False)
    write_json(out/'selection.json', decision)
    return decision


def validate(input_path, rule_path, output):
    data = load_input(input_path, 'validation')
    fixed = read_json(rule_path)
    validate_rule(fixed, data['material_version'])
    result = evaluate(data, fixed)
    safety_failure = result['summary']['wrong_binding']+result['summary']['unconfirmed_acceptance'] > 0
    enabled = fixed['enabled'] and not safety_failure and result['summary']['correct']>0
    result['decision'] = dict(fuzzy_enabled=enabled, safety_failure=safety_failure,
        rule_version=fixed['rule_version'], material_version=data['material_version'],
        final_rule=fixed if enabled else rule(data['material_version']),
        reason='heldout_support_for_limited_rule' if enabled else 'keep_disabled; no supported heldout acceptance or safety failure',
        thresholds_retuned=False)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'results.json', result)
    write_json(out/'decision.json', dict(result['decision'], selected_rule=fixed,
        counts={k:result['summary'][k] for k in ('correct','wrong_binding','unconfirmed_acceptance')}))
    return dict(summary=result['summary'], decision=result['decision'])


def apply_rule(db, rule_path, decision_path, output):
    fixed, decision = read_json(rule_path), read_json(decision_path)
    c = cg.open_graph(db)
    try:
        meta = dict(c.execute('SELECT * FROM c3_meta'))
        validate_rule(fixed, meta['citation_graph_version'])
        if decision['selected_rule'] != fixed or decision['rule_version'] != fixed['rule_version']:
            raise ValueError('validation/fixed-rule mismatch')
        # Recompute the final safety decision from the recorded held-out counts.
        s = decision['counts']
        enabled = fixed['enabled'] and s['correct']>0 and s['wrong_binding']==s['unconfirmed_acceptance']==0
        if decision['fuzzy_enabled'] != enabled or decision['final_rule'] != (fixed if enabled else rule(meta['citation_graph_version'])):
            raise ValueError('validation decision inconsistent')
        baseline = json.loads(meta['summary'])
    finally:
        c.close()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    if not enabled:
        result = dict(fuzzy_enabled=False, baseline_db=str(Path(db).resolve()),
            baseline_graph_version=meta['citation_graph_version'], baseline=baseline,
            final_rule=rule(meta['citation_graph_version']), new_database=None,
            graph_delta=dict(fuzzy_supports=0, fuzzy_unique_edges=0, overlap_exact=0,
                             exact_edges=0, edges=0, nodes=0, degrees_changed=0, pagerank_max_change=0),
            reason=decision['reason'])
        write_json(out/'application.json', result)
        return result
    return build_candidate_graph(db, fixed, out)


def build_candidate_graph(db, fixed, out):
    data = cg.snapshot(db)
    papers = {r['paper_id']:json.loads(r['payload']) for r in data['nodes']}
    refs = [json.loads(r['payload']) for r in data['refs']]
    accepted = []
    for ref in refs:
        prediction = predict(ref, papers, fixed)
        if prediction['accepted'] and prediction['kind']=='fuzzy':
            ref['fuzzy_evidence'] = prediction
            accepted.append(ref)
    # The existing graph routine retains exact supports and handles roles,
    # self-loops, pair deduplication, isolates and all metrics.
    applied = copy.deepcopy(refs)
    for ref in applied:
        if 'fuzzy_evidence' in ref:
            ref['match'] = dict(ref['match'], status='exact', target=ref['fuzzy_evidence']['target'])
    edges, supports, metrics, numerical = cg.graph(papers, applied)
    exact_supports = {(r['citing'],r['cited'],r['ref_id']) for r in data['supports']}
    baseline_edges, baseline_supports, _, _ = cg.graph(papers, copy.deepcopy(refs))
    if set(baseline_supports) != exact_supports or set(baseline_edges) != {(r['citing'],r['cited']) for r in data['edges']}:
        raise ValueError('saved exact graph differs from existing graph rules')
    if not exact_supports <= set(supports):
        raise ValueError('exact supports changed')
    by_id = {r['ref_id']:r for r in applied}
    for ref in refs:
        ref['edge_class'] = by_id[ref['ref_id']]['edge_class']
    fuzzy_pairs = {(r['paper_id'],r['fuzzy_evidence']['target']) for r in accepted}
    exact_edges = {(r['citing'],r['cited']) for r in data['edges']}
    locator_checks = []
    for ref in accepted:
        lines = Path(ref['path']).read_text(encoding='utf-8-sig').splitlines()
        if '\n'.join(lines[ref['line_start']-1:ref['line_end']]) != ref['raw']:
            raise ValueError('fuzzy support locator mismatch: '+ref['ref_id'])
        locator_checks.append(dict(ref_id=ref['ref_id'], path=ref['path'], line_start=ref['line_start'], line_end=ref['line_end']))
    version = 'c3-calibrated-'+cg.fingerprint(dict(baseline=data['meta']['citation_graph_version'],rule=fixed,refs=refs))[:24]
    output = out/'citation-graph.sqlite3'
    # New schema separates support kind explicitly; no upstream schema migration.
    with output.open('xb'):
        pass
    c = sqlite3.connect(output)
    try:
        c.executescript('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE nodes(paper_id TEXT PRIMARY KEY,payload TEXT); CREATE TABLE sources(artifact_id TEXT PRIMARY KEY,payload TEXT); CREATE TABLE refs(ref_id TEXT PRIMARY KEY,payload TEXT); CREATE TABLE edges(citing TEXT,cited TEXT,PRIMARY KEY(citing,cited)); CREATE TABLE supports(citing TEXT,cited TEXT,ref_id TEXT,kind TEXT,payload TEXT,PRIMARY KEY(citing,cited,ref_id));')
        c.executemany('INSERT INTO meta VALUES (?,?)', [('schema','c3-calibrated-v1'),('version',version),('rule',cg.dump(fixed)),('baseline',data['meta']['citation_graph_version'])])
        c.executemany('INSERT INTO nodes VALUES (?,?)', [(p,cg.dump(dict(papers[p],**metrics[p]))) for p in sorted(papers)])
        c.executemany('INSERT INTO sources VALUES (?,?)', [(r['artifact_id'],r['payload']) for r in data['sources']])
        c.executemany('INSERT INTO refs VALUES (?,?)', [(r['ref_id'],cg.dump(r)) for r in refs])
        c.executemany('INSERT INTO edges VALUES (?,?)', edges)
        original = {r['ref_id']:r for r in refs}
        c.executemany('INSERT INTO supports VALUES (?,?,?,?,?)', [(a,b,rid,'exact' if (a,b,rid) in exact_supports else 'fuzzy',cg.dump(original[rid])) for a,b,rid in supports])
        c.commit()
    finally:
        c.close()
    result = dict(fuzzy_enabled=True, graph_version=version, baseline_graph_version=data['meta']['citation_graph_version'],
        new_database=str(output.resolve()), exact_supports_unchanged=len(exact_supports),
        graph_delta=dict(fuzzy_supports=len(accepted), fuzzy_unique_edges=len(fuzzy_pairs), overlap_exact=len(fuzzy_pairs & exact_edges),
            edges=len(edges)-len(exact_edges), nodes=0,
            degrees_changed=sum(any(papers[p][k]!=metrics[p][k] for k in ('in_degree','out_degree')) for p in papers),
            pagerank_max_change=max(abs(papers[p]['pagerank']-metrics[p]['pagerank']) for p in papers)), numerical=numerical)
    write_json(out/'fuzzy-locators.json', locator_checks)
    write_json(out/'application.json', result)
    return result


def verify_candidate(db, baseline, locators=False):
    """Recompute a candidate graph from its fixed rule, without labels/models."""
    original = cg.snapshot(baseline)
    papers = {r['paper_id']:json.loads(r['payload']) for r in original['nodes']}
    baseline_refs = {r['ref_id']:json.loads(r['payload']) for r in original['refs']}
    c = cg.readonly(db)
    try:
        meta = dict(c.execute('SELECT * FROM meta'))
        if meta['schema'] != 'c3-calibrated-v1' or meta['baseline'] != original['meta']['citation_graph_version']:
            raise ValueError('candidate schema/baseline mismatch')
        fixed = json.loads(meta['rule'])
        validate_rule(fixed, meta['baseline'])
        nodes = {r[0]:json.loads(r[1]) for r in c.execute('SELECT * FROM nodes')}
        sources = [dict(r) for r in c.execute('SELECT * FROM sources ORDER BY 1,2')]
        refs = [json.loads(r[0]) for r in c.execute('SELECT payload FROM refs ORDER BY ref_id')]
        saved_edges = [tuple(r) for r in c.execute('SELECT * FROM edges ORDER BY citing,cited')]
        saved_supports = [tuple(r) for r in c.execute('SELECT * FROM supports ORDER BY citing,cited,ref_id')]
    finally:
        c.close()
    if sources != original['sources'] or set(nodes) != set(papers) or {r['ref_id'] for r in refs} != set(baseline_refs):
        raise ValueError('candidate source/node/reference sets differ')
    applied = []
    fuzzy = []
    for ref in refs:
        old = baseline_refs[ref['ref_id']]
        comparable = {k:v for k,v in ref.items() if k not in ('edge_class','fuzzy_evidence')}
        if comparable != {k:v for k,v in old.items() if k!='edge_class'}:
            raise ValueError('original reference/match changed: '+ref['ref_id'])
        prediction = predict(old, papers, fixed)
        expected = prediction if prediction['accepted'] and prediction['kind']=='fuzzy' else None
        if ref.get('fuzzy_evidence') != expected:
            raise ValueError('fuzzy recomputation differs: '+ref['ref_id'])
        clone = copy.deepcopy(ref)
        if expected:
            fuzzy.append(ref)
            clone['match'] = dict(clone['match'], status='exact',target=expected['target'])
        applied.append(clone)
    edges, supports, metrics, numerical = cg.graph(papers, applied)
    if edges != saved_edges:
        raise ValueError('candidate edges differ')
    exact = {(r['citing'],r['cited'],r['ref_id']) for r in original['supports']}
    if not exact <= set(supports):
        raise ValueError('candidate lost exact support')
    by_id = {r['ref_id']:r for r in refs}
    expected_supports = [(a,b,rid,'exact' if (a,b,rid) in exact else 'fuzzy',cg.dump(by_id[rid])) for a,b,rid in supports]
    if expected_supports != saved_supports:
        raise ValueError('candidate supports/kinds/evidence differ')
    for ref in applied:
        if ref['edge_class'] != by_id[ref['ref_id']]['edge_class']:
            raise ValueError('candidate edge classification differs')
    for p in papers:
        if nodes[p] != dict(papers[p],**metrics[p]):
            raise ValueError('candidate node identity/metrics differ: '+p)
    version = 'c3-calibrated-'+cg.fingerprint(dict(baseline=meta['baseline'],rule=fixed,refs=refs))[:24]
    if version != meta['version']:
        raise ValueError('candidate version mismatch')
    checked = 0
    if locators:
        grouped = defaultdict(list)
        for ref in fuzzy:
            grouped[ref['path']].append(ref)
        for path, entries in grouped.items():
            lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
            for ref in entries:
                if '\n'.join(lines[ref['line_start']-1:ref['line_end']]) != ref['raw']:
                    raise ValueError('candidate fuzzy locator mismatch: '+ref['ref_id'])
                checked += 1
    return dict(status='PASS', graph_version=version,nodes=len(nodes),edges=len(edges),supports=len(supports),
        exact_supports_unchanged=len(exact),exact_unique_edges=len(original['edges']),fuzzy_supports=len(fuzzy),
        fuzzy_locators_checked=checked,isolates=sum(m['in_degree']==m['out_degree']==0 for m in metrics.values()),
        references_unchanged=len(refs),sources_unchanged=len(sources),numerical=numerical)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('check')
    for name in ('materials','labels','calibration-csv','validation-csv','db','output'):
        p.add_argument('--'+name, required=True)
    p = subs.add_parser('calibrate')
    p.add_argument('--calibration-input', required=True)
    p.add_argument('--output', required=True)
    p = subs.add_parser('validate')
    p.add_argument('--validation-input', required=True)
    p.add_argument('--rule', required=True)
    p.add_argument('--output', required=True)
    p = subs.add_parser('apply')
    for name in ('db','rule','decision','output'):
        p.add_argument('--'+name, required=True)
    p = subs.add_parser('verify')
    p.add_argument('--db', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--locators', action='store_true')
    args = parser.parse_args(argv)
    if args.command=='check':
        result = check(args.materials,args.labels,args.calibration_csv,args.validation_csv,args.db,args.output)
    elif args.command=='calibrate':
        result = calibrate(args.calibration_input,args.output)
    elif args.command=='validate':
        result = validate(args.validation_input,args.rule,args.output)
    elif args.command=='apply':
        result = apply_rule(args.db,args.rule,args.decision,args.output)
    else:
        result = verify_candidate(args.db,args.baseline,args.locators)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
