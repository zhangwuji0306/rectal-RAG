"""Reproducible C4 dev-only experiment on accepted saved D250 rankings."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import html
import json
import math
from pathlib import Path
import re

from evidence_ranking import Inputs, aggregate, coverage, load_json, rank, schemes


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def evaluator():
    """Use existing evaluator functions without importing its retrieval dependency."""
    path = Path(__file__).with_name('evaluate_retrieval.py')
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    constants = {'POSITIVE_THRESHOLD', 'NDCG_CUTOFFS', 'UNIQUE_RECALL_CUTOFF', 'GRADED_GAINS', 'FORMAL_SPLITS'}
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name not in
             ('main', 'load_qrels', 'evaluate_splits')) or (isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id in constants for t in n.targets))]
    def forbidden(*args, **kwargs):
        raise RuntimeError('retrieval prohibited in saved-pool evaluation')
    ns = dict(math=math, Path=Path, json=json, retrieve_literature=forbidden)
    nodes.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), 'exec'), ns)
    return ns


def dev_rows(path):
    rows = []
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            # Do not deserialize any locked-test row.
            if re.search(r'"split"\s*:\s*"dev"', line):
                row = json.loads(line)
                if row['split'] != 'dev':
                    raise ValueError('invalid dev split')
                rows.append(row)
    if not rows or len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('empty/duplicate dev input')
    return sorted(rows, key=lambda r: r['query_id'])


def definition():
    return dict(version='c4-dev-v1', candidate_budget='250 saved dense passages; DOI/source max; first 50 papers',
        identity='C1 source and active DOI must agree; duplicate paper_id rejected; no C1 adjudication',
        relevance='saved rounded cosine score, max per DOI/source, stable first passage ties; not a probability',
        constraint='cosine >= 0.65; immutable floor((cosine-0.65)/0.01) bands; below floor original order',
        signal_rules=dict(journal='unavailable, weight 0; no JIF/SJR/journal percentile',
            external='OpenAlex cumulative cited_by_count as observed at C2 retrieved_at; min(log1p(n)/log1p(1000),1)-0.5',
            corpus='unique incoming C3 citing papers; union exact/fuzzy, separate counts; positive-only 0.5*min(log1p(n)/log1p(20),1); no incoming = unobserved',
            publication_type='C1 semicolon classes: synthesis or RCT +0.5, case report -0.5, other 0, absent missing; precedence synthesis/RCT/case/other'),
        missing='all nonavailable priors exactly 0, unchanged weights; real external zero=-0.5; real other type=0',
        weights=[0.005, 0.01], ablations='single signals at alpha; full alpha/3 each external/corpus/type, journal 0',
        selection='full only: nDCG10 delta >=0.01 and MRR10 delta >=0.01; each Recall delta >=-0.01 and nDCG20 delta >=-0.01; maximize nDCG10 then MRR10 then smaller alpha; otherwise relevance only',
        deployment='disabled regardless of dev result; independent review and human diagnosis outstanding',
        strata='positive judged papers grouped by C1 year <=2019/2020-2023/>=2024/missing and type synthesis/randomized_trial/case_report/other/missing; query-macro pooled metrics within group, same full ranking',
        human='40 dev queries chosen by largest full-model maximum rank displacement then query_id; purposive diagnosis only; no generated human answers',
        test='no locked test deserialization, evaluation or selection')


def fixed_pool(row, saved, requests, inputs):
    qid = row['query_id']
    if saved['query_id'] != qid or saved['query'] != row['query'] or not saved['complete']:
        raise ValueError('saved dev query mismatch')
    req = requests['requests']['D250']
    if requests['query_id'] != qid or req['owner'] != 'D' or req['n'] != 250 or req['branch'] != 'passage' or req['hits'] != saved['d250']:
        raise ValueError('accepted D250 request mismatch')
    full = aggregate(saved['d250'])
    if full != saved['all_papers']['D250'] or full[:50] != saved['top50']['D250']:
        raise ValueError('saved D250 aggregation mismatch')
    result = []
    for p in full[:50]:
        pid = inputs.resolve(p)
        result.append(dict(p, paper_id=pid, signals=inputs.signals[pid]))
    # Validate the pool even for a zero-weight run.
    rank(result, schemes(0)['relevance_only'])
    return result


def flat(metrics):
    return {**{'Recall@'+k: v for k, v in metrics['recall'].items()},
            **{'nDCG@'+k: v for k, v in metrics['ndcg'].items()}, 'MRR@10': metrics['mrr_at_10']}


def comparison(before, after):
    b, a = flat(before['metrics']), flat(after['metrics'])
    per = []
    for old, new in zip(before['per_query'], after['per_query']):
        if old['query_id'] != new['query_id']:
            raise ValueError('query comparison order mismatch')
        x, y = flat(old), flat(new)
        per.append(dict(query_id=old['query_id'], excluded=old['excluded_no_positive'],
            before=x, after=y, delta={k: y[k]-x[k] if x[k] is not None else None for k in x}))
    return dict(before=b, after=a, delta={k:a[k]-b[k] if b[k] is not None else None for k in b},
        total=after['total_queries'], evaluable=after['evaluable_queries'], excluded=after['excluded_no_positive_queries'],
        ndcg_evaluable=after['ndcg_evaluable_queries'], per_query=per,
        changes={k:dict(Counter('excluded' if p['delta'][k] is None else 'improved' if p['delta'][k]>0 else
             'regressed' if p['delta'][k]<0 else 'unchanged' for p in per)) for k in b})


def year_group(year):
    if not str(year).isdigit():
        return 'missing'
    return '<=2019' if int(year)<=2019 else '2020-2023' if int(year)<=2023 else '>=2024'


def diagnose(rows, rankings, inputs, ev):
    """Mechanical rank changes, not human judgements of bias or relevance."""
    output = {}
    for name, queries in rankings.items():
        changes = []
        for row in rows:
            rel, aliases = ev['_query_relevance'](row)
            for p in queries[row['query_id']]:
                delta = p['rank']-p['baseline_rank']
                ext = p['signals']['external']
                newer = year_group(inputs.papers[p['paper_id']]['year']) == '>=2024'
                grade = rel.get(ev['_resolved_identity'](p,aliases))
                low = ext['state']=='available' and ext['value']<=10
                if delta and (newer or (grade is not None and grade>=2 and low)):
                    changes.append(dict(query_id=row['query_id'],paper_id=p['paper_id'],baseline_rank=p['baseline_rank'],
                        rank=p['rank'],delta=delta,newer_2024_onward=newer,pooled_grade=grade,
                        external_count=ext['value'],external_state=ext['state'],observed_low_citations=low))
        output[name] = dict(kind='program/agent diagnostic; no new human labels',
            high_journal_irrelevant_rise='not estimable: journal quality unavailable',
            new_demoted_occurrences=sum(c['newer_2024_onward'] and c['delta']>0 for c in changes),
            positive_low_citation_demoted_occurrences=sum(c['pooled_grade'] is not None and c['pooled_grade']>=2 and c['observed_low_citations'] and c['delta']>0 for c in changes),
            changes=changes)
    return output


def stratify(rows, rankings, inputs, ev):
    groups = {axis: defaultdict(list) for axis in ('year', 'publication_type')}
    counts = {axis: Counter() for axis in groups}
    for row in rows:
        buckets = {axis: defaultdict(list) for axis in groups}
        for j in row['judgements']:
            if j['relevance'] < 2:
                continue
            pid = inputs.resolve(dict(source=j.get('source_file') or j.get('source_path') or j['source_id'], doi=j.get('doi','')))
            paper = inputs.papers[pid]
            g = dict(year=year_group(paper['year']), publication_type=inputs.signals[pid]['publication_type']['group'])
            for axis, key in g.items():
                buckets[axis][key].append(j)
                counts[axis][key] += 1
        for axis in groups:
            for key, judgements in buckets[axis].items():
                groups[axis][key].append(dict(row, judgements=judgements))
    output = {}
    for axis in groups:
        output[axis] = {}
        # Explicit empty missing group, never silently omitted.
        groups[axis].setdefault('missing', [])
        for key, subset in sorted(groups[axis].items()):
            scored = {name:ev['evaluate'](subset,[5,10,20,50],retrieval_cache={r['query']:ranks[r['query_id']] for r in subset})
                      for name, ranks in rankings.items()}
            output[axis][key] = dict(positive_judgement_occurrences=counts[axis][key],
                comparisons={name:comparison(scored['relevance_only'], val) for name,val in scored.items()})
    return output


def run(args):
    out = Path(args.output)
    out.mkdir(exist_ok=False, parents=True)
    # Persist the complete definition before loading qrels or comparing weights.
    save(out/'definition.json', definition())
    inputs = Inputs(args.c1,args.c2,args.graph,args.rule,args.decision,args.validation)
    rows, ev = dev_rows(args.dev_qrels), evaluator()
    if len(rows) != 70:
        raise ValueError('accepted dev requires 70 queries')
    pools = {}
    for row in rows:
        qid = row['query_id']
        saved = load_json(Path(args.saved)/f'{qid}.json')
        requests = load_json(Path(args.requests)/qid/'requests.json')
        pools[qid] = fixed_pool(row,saved,requests,inputs)
    save(out/'coverage.json',coverage(pools.values()))
    save(out/'candidate-pools.json',pools)
    save(out/'inputs.json',dict(c1=args.c1,c2=args.c2,graph=args.graph,rule=args.rule,decision=args.decision,
        validation=args.validation, graph_metadata=inputs.graph_meta, saved=args.saved, requests=args.requests, dev_qrels=args.dev_qrels,
        query_ids=list(pools), validated_queries=len(pools)))
    results, all_rankings = {}, {}
    baseline = None
    for alpha in definition()['weights']:
        ranks, evaluations = {}, {}
        for name, weights in schemes(alpha).items():
            ranks[name] = {qid:rank(pool,weights) for qid,pool in pools.items()}
            evaluations[name] = ev['evaluate'](rows,[5,10,20,50],retrieval_cache={r['query']:ranks[name][r['query_id']] for r in rows})
            if baseline is None:
                baseline = evaluations[name]
            # Existing accepted per-query D metrics must replay exactly.
            if name == 'relevance_only':
                for row, per in zip(rows,evaluations[name]['per_query']):
                    expected = load_json(Path(args.saved)/f"{row['query_id']}.json")['evaluation']['D250']['metrics']
                    if {k:v for k,v in per.items() if k!='query'} != expected:
                        raise ValueError('accepted evaluator replay mismatch')
        key = str(alpha)
        fields = ('paper_id','rank','baseline_rank','score','eligible','band','priors','contributions','adjusted_score')
        save(out/f'rankings-{key}.json',{n:{qid:[{k:p[k] for k in fields} for p in ps]
             for qid,ps in rs.items()} for n,rs in ranks.items()})
        save(out/f'evaluations-{key}.json',evaluations)
        results[key] = {name:comparison(baseline,val) for name,val in evaluations.items()}
        save(out/f'strata-{key}.json',stratify(rows,ranks,inputs,ev))
        save(out/f'program-diagnostics-{key}.json',diagnose(rows,ranks,inputs,ev))
        all_rankings[key] = ranks
    options=[]
    for alpha, comp in results.items():
        d=comp['full']['delta']
        if d['nDCG@10']>=.01 and d['MRR@10']>=.01 and all(d[k]>=-.01 for k in ('Recall@5','Recall@10','Recall@20','Recall@50','nDCG@20')):
            options.append((comp['full']['after']['nDCG@10'],comp['full']['after']['MRR@10'],-float(alpha),alpha))
    selected = max(options)[3] if options else None
    save(out/'comparison.json',results)
    save(out/'decision.json',dict(selected='full@'+selected if selected else 'relevance_only', enable_quality=False,
        human_diagnosis='pending; 40-query materials provided', stage_complete=False))
    diagnostics=[]
    for r in rows:
        qid=r['query_id']
        full=all_rankings['0.01']['full'][qid]
        diagnostics.append((max(abs(p['rank']-p['baseline_rank']) for p in full),qid,r))
    chosen=sorted(diagnostics,key=lambda x:(-x[0],x[1]))[:40]
    materials=[]
    for displacement,qid,row in chosen:
        candidates=[]
        for p in all_rankings['0.01']['full'][qid]:
            paper=inputs.papers[p['paper_id']]
            candidates.append(dict(p,title=paper['title'],year=paper['year'],publication_type=paper['publication_type'],
                all_scheme_ranks={a:{n:next(x['rank'] for x in rs[qid] if x['paper_id']==p['paper_id']) for n,rs in ranks.items()} for a,ranks in all_rankings.items()}))
        materials.append(dict(query_id=qid,query=row['query'],max_displacement=displacement,candidates=candidates,
            pooled_judgements=row['judgements'],notice='unjudged is unknown; agent/program diagnostic, human review pending'))
    save(out/'human-materials.json',materials)
    page = ['<!doctype html><meta charset="utf-8"><title>C4人工偏差诊断</title>',
        '<style>body{font:15px sans-serif;margin:24px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:6px}h2{margin-top:48px}a{color:#1359a6}</style>',
        '<h1>C4：40题人工偏差诊断（未完成）</h1><p>按最大位移选择；不是总体效果估计。期刊信号不可用；空白判断不等于无关。填写同目录 human-review.csv。新文献定义为2024年起，程序低引用观察为OpenAlex累计≤10，均不是质量或偏差裁决。</p>']
    for item in materials:
        page.append('<h2>'+html.escape(item['query_id'])+'</h2><p>'+html.escape(item['query'])+'</p>')
        page.append('<details><summary>已有pooled judgements（其他文献未知）</summary><pre>'+html.escape(json.dumps(item['pooled_judgements'],ensure_ascii=False,indent=2))+'</pre></details>')
        page.append('<table><tr><th>paper_id / 书目</th><th>年份 / 类型</th><th>baseline → full@.01</th><th>OpenAlex引用</th><th>C3入度 all/exact/fuzzy</th><th>其他方案排名（.005 / .01）</th></tr>')
        for p in sorted(item['candidates'],key=lambda x:x['baseline_rank']):
            ext,corpus=p['signals']['external'],p['signals']['corpus']
            cells=[p['paper_id']+' / '+p['title']+' / '+p['doi'],str(p['year'])+' / '+p['publication_type'],
                str(p['baseline_rank'])+' → '+str(p['rank']),str(ext['value'])+' ('+ext['state']+')',str(corpus['counts']),
                json.dumps(p['all_scheme_ranks'],ensure_ascii=False)]
            page.append('<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in cells)+'</tr>')
        page.append('</table>')
    (out/'human-review.html').write_text('\n'.join(page),encoding='utf-8')
    with (out/'human-review.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['query_id','high_journal_irrelevant_rise','new_paper_suppressed','relevant_low_citation_demoted','paper_ids','reason','reviewer'])
        writer.writeheader()
        writer.writerows(dict(query_id=qid) for _,qid,_ in chosen)
    (out/'human-instructions.md').write_text('人工偏差诊断：40题，按full@0.01最大位移降序及query_id选择，仅为有目的诊断。\n'
        '逐题对照human-materials.json中的题意、书目、各方案排名/贡献及已有pooled判断；未标注不等于无关。\n'
        '期刊信号不可用，该列填“不适用”；其余填“是/否/不确定”，记录paper_id、理由及reviewer；缺信息时保留不确定。\n'
        '若判定相关性或研究类型需要摘要/全文，由人类在授权本地来源核实；程序未生成任何人工答案。\n'
        '填写human-review.csv。40题均需复核；C3引用绑定标签不适用于此诊断。\n',encoding='utf-8')
    print(json.dumps(dict(queries=len(rows),coverage=coverage(pools.values()),decision=load_json(out/'decision.json')),ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('c1','c2','graph','rule','decision','validation','saved','requests','dev-qrels','output'):
        p.add_argument('--'+key,required=True)
    run(p.parse_args())


if __name__=='__main__':
    main()
