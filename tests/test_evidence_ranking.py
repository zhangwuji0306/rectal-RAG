"""C4 synthetic ranking, read-only identity and pooled evaluation contracts."""
import copy
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'index'))
import evidence_ranking as er
import evidence_experiment as ex
import paper_registry as pr
import quality_metadata as qm
from citation_graph import registry, readonly


def signals():
    return {name:er.observation('missing') for name in er.SIGNALS}


def pool(scores=(.659,.658,.65,.649,.64)):
    return [dict(paper_id=str(i),source=f'{i}.md',doi='',rank=i,score=s,signals=signals())
            for i,s in enumerate(scores,1)]


class RankingTests(unittest.TestCase):
    def test_floor_and_band_hold_under_extreme_weights(self):
        ps=pool((.671,.669,.651,.65,.649,.648))
        for p in ps:
            p['signals']['external']=er.observation('available',0)
        for i in (1,3,5):
            ps[i]['signals']['external']=er.observation('available',10**100)
        ranked=er.rank(ps,dict(er.schemes(0)['relevance_only'],external=1e12))
        self.assertEqual([p['paper_id'] for p in ranked],['1','2','4','3','5','6'])
        self.assertTrue(all(not any(p['contributions'].values()) for p in ranked if not p['eligible']))

    def test_fixed_collection_and_no_input_mutation(self):
        ps=pool(); before=copy.deepcopy(ps)
        ranked=er.rank(ps,er.schemes(.01)['full'])
        self.assertEqual(ps,before)
        self.assertEqual({p['paper_id'] for p in ps},{p['paper_id'] for p in ranked})

    def test_budget_duplicate_and_bad_relevance_fail(self):
        for ps in (pool(tuple([.7]*51)),[pool()[0]]*2,pool((.6,.7)),pool((float('nan'),))):
            with self.subTest(ps=len(ps)),self.assertRaises(ValueError):
                er.rank(ps,er.schemes(0)['full'])

    def test_neutral_missing_distinct_from_real_zero(self):
        for state in ('missing','failed','not_queried','unobserved','unavailable'):
            self.assertEqual(er.prior('external',er.observation(state)),0)
        self.assertEqual(er.prior('external',er.observation('available',0)),-.5)
        self.assertEqual(er.prior('publication_type',er.observation('available',0)),0)

    def test_invalid_external_values_and_state(self):
        for val in (-1,.5,True,float('inf'),float('nan'),'3'):
            with self.subTest(val=val),self.assertRaises(ValueError):
                er.prior('external',er.observation('available',val))
        for state,val in [('missing',0),('available',None),('unknown',None)]:
            with self.assertRaises(ValueError):er.observation(state,val)

    def test_normalization_caps_and_recomputable_contributions(self):
        self.assertEqual(er.prior('external',er.observation('available',1000)),.5)
        self.assertEqual(er.prior('external',er.observation('available',10000)),.5)
        self.assertEqual(er.prior('corpus',er.observation('available',20)),.5)
        ps=pool(); ps[1]['signals']['external']=er.observation('available',10)
        w=er.schemes(.01)['full']; ranked=er.rank(ps,w)
        p=next(p for p in ranked if p['paper_id']=='2')
        self.assertAlmostEqual(p['contributions']['external'],w['external']*(math.log1p(10)/math.log1p(1000)-.5))
        self.assertEqual(p['adjusted_score'],p['score']+sum(p['contributions'].values()))
        self.assertEqual(p['contributions']['corpus'],0)

    def test_zero_weights_all_missing_and_ties_replay(self):
        ps=pool((.66,.66,.65,.64))
        for w in (er.schemes(.01)['full'],er.schemes(0)['full']):
            self.assertEqual([p['paper_id'] for p in er.rank(ps,w)],[p['paper_id'] for p in ps])
        ps[1]['signals']['external']=er.observation('available',1000)
        self.assertEqual([p['paper_id'] for p in er.rank(ps,er.schemes(0)['full'])],['1','2','3','4'])

    def test_type_classification_is_explicit(self):
        for raw,group,val in [('Systematic Review; Journal Article','synthesis',.5),
            ('Randomized Controlled Trial','randomized_trial',.5),('Case Reports','case_report',-.5),
            ('Journal Article','other',0),('','missing',None)]:
            s=er.type_signal(raw); self.assertEqual((s['group'],s['value']),(group,val))

    def test_unavailable_journal_has_zero_weight(self):
        self.assertTrue(all(w['journal']==0 for w in er.schemes(.01).values()))
        self.assertEqual(er.rank(pool(),er.schemes(.01)['journal']),er.rank(pool(),er.schemes(0)['full']))
        with self.assertRaises(ValueError):er.prior('journal',er.observation('available',.8))

    def test_coverage_reports_unique_and_occurrence_denominators(self):
        ps=pool()
        c=er.coverage([ps,ps]); self.assertEqual((c['unique_papers'],c['candidate_occurrences']),(5,10))
        self.assertEqual(c['signals']['external'],{'missing':5})

    def test_saved_aggregation_max_members_and_stable_ties(self):
        hits=[dict(id=str(i),rank=i+1,score=.7 if i<2 else .6,source=f'{i%2}.md',doi='') for i in range(250)]
        a=er.aggregate(hits)
        self.assertEqual([p['source'] for p in a],['0.md','1.md'])
        self.assertEqual([len(p['member_ids']) for p in a],[125,125])
        self.assertEqual([p['representative_id'] for p in a],['0','1'])
        with self.assertRaises(ValueError):er.aggregate(hits[:249])

    def test_saved_pool_provenance_budget_and_duplicate_identity(self):
        hits=[dict(id=str(i),rank=i+1,score=.7 if i<2 else .6,source=f'{i%2}.md',doi='') for i in range(250)]
        full=er.aggregate(hits)
        saved=dict(query_id='q',query='question',complete=True,d250=hits,all_papers={'D250':full},top50={'D250':full})
        req=dict(query_id='q',requests={'D250':dict(owner='D',n=250,branch='passage',hits=hits)})
        class Inputs:
            signals={'a':signals()}
            def resolve(self,item):return 'a'
        row=dict(query_id='q',query='question')
        with self.assertRaisesRegex(ValueError,'identity'):
            ex.fixed_pool(row,saved,req,Inputs())
        req['requests']['D250']['n']=200
        with self.assertRaisesRegex(ValueError,'request'):
            ex.fixed_pool(row,saved,req,Inputs())

    def test_invalid_weights_and_constraints_fail(self):
        for w in ({},dict(er.schemes(0)['full'],external=-1),dict(er.schemes(0)['full'],corpus=float('nan'))):
            with self.assertRaises(ValueError):er.rank(pool(),w)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.c1=self.root/'c1.sqlite3';self.c2=self.root/'c2.sqlite3';self.graph=self.root/'g.sqlite3'
        with pr.PaperRegistry(self.c1,create=True) as r:
            for i in range(3):
                r.register(dict(source=f'{i}.md',metadata=dict(title=f'Title {i}'),identifiers=dict(doi=f'10.1234/{i}'),content_sha256=str(i)*64))
        papers,artifacts,_=registry(self.c1);self.papers=papers;self.artifacts=artifacts
        self.pids=sorted(papers); a,b,c=self.pids
        store=qm.MetadataStore(self.c2)
        try:
            store.bind(qm.read_registry(self.c1),[2025])
            for pid in self.pids:
                store.put('paper',pid,'cited_by_count',0,state='available',retrieved_at='2026-11-01',unit='citing works')
            store.conn.commit()
        finally:store.close()
        self.fixed=dict(rule_version='rule1',material_version='base1',enabled=True)
        self.rule=self.write('rule.json',self.fixed)
        self.decision=self.write('decision.json',dict(final_rule=self.fixed,rule_version='rule1',safety_failure=False,fuzzy_enabled=True))
        self.validation=self.write('validation.json',dict(rule=self.fixed))
        with closing(sqlite3.connect(self.graph)) as conn:
            conn.executescript('CREATE TABLE meta(key,value); CREATE TABLE nodes(paper_id,payload); CREATE TABLE sources(artifact_id,payload); CREATE TABLE edges(citing,cited); CREATE TABLE supports(citing,cited,kind,payload);')
            conn.executemany('INSERT INTO meta VALUES (?,?)',[('schema','c3-calibrated-v1'),('version','graph1'),('rule',json.dumps(self.fixed)),('baseline','base1')])
            conn.executemany('INSERT INTO nodes VALUES (?,?)',[(pid,json.dumps(dict(p,in_degree=int(pid==b)))) for pid,p in papers.items()])
            conn.executemany('INSERT INTO sources VALUES (?,?)',[(x['artifact_id'],json.dumps(x)) for x in artifacts])
            conn.execute('INSERT INTO edges VALUES (?,?)',(a,b))
            conn.executemany('INSERT INTO supports VALUES (?,?,?,?)',[(a,b,'exact','{}'),(a,b,'fuzzy',json.dumps(dict(fuzzy_evidence=dict(rule_version='rule1'))))])
            conn.commit()

    def write(self,name,obj):
        p=self.root/name;p.write_text(json.dumps(obj),encoding='utf-8');return p

    def inputs(self):
        return er.Inputs(self.c1,self.c2,self.graph,self.rule,self.decision,self.validation)

    def change(self,path,sql,params=()):
        with closing(sqlite3.connect(path)) as c:
            c.execute(sql,params);c.commit()

    def test_readonly_inputs_and_exact_fuzzy_union(self):
        before={p:p.read_bytes() for p in (self.c1,self.c2,self.graph)}
        x=self.inputs(); s=x.signals[self.pids[1]]['corpus']
        self.assertEqual(s['counts'],dict(all=1,exact=1,fuzzy=1))
        self.assertEqual(s['graph_version'],'graph1'); self.assertEqual(s['rule_version'],'rule1')
        self.assertEqual(x.signals[self.pids[2]]['corpus']['state'],'unobserved')
        self.assertEqual(before,{p:p.read_bytes() for p in before})
        c=readonly(self.c1)
        try:
            with self.assertRaises(sqlite3.OperationalError):c.execute('DELETE FROM papers')
        finally:c.close()

    def test_wrong_schema_rejected_without_writes(self):
        for path,sql in [(self.c1,"UPDATE registry_meta SET value='old' WHERE key='schema_version'"),
                         (self.c2,"UPDATE meta SET value='old' WHERE key='schema'"),
                         (self.graph,"UPDATE meta SET value='c3-v1' WHERE key='schema'")]:
            old=path.read_bytes();self.change(path,sql);bad=path.read_bytes()
            with self.subTest(path=path),self.assertRaises(ValueError):self.inputs()
            self.assertEqual(path.read_bytes(),bad);path.write_bytes(old)

    def test_missing_input_not_created(self):
        missing=self.root/'absent.sqlite3'
        with self.assertRaises(sqlite3.OperationalError):er.Inputs(missing,self.c2,self.graph,self.rule,self.decision,self.validation)
        self.assertFalse(missing.exists())

    def test_snapshot_and_version_contract_breaks_fail(self):
        for sql,params in [("UPDATE meta SET value='other' WHERE key='baseline'",()),
            ("UPDATE supports SET kind='unknown'",()),
            ("UPDATE supports SET payload=? WHERE kind='fuzzy'",(json.dumps(dict(fuzzy_evidence=dict(rule_version='other'))),)),
            ("DELETE FROM nodes WHERE paper_id=?",(self.pids[2],))]:
            old=self.graph.read_bytes();self.change(self.graph,sql,params)
            with self.subTest(sql=sql),self.assertRaises(ValueError):self.inputs()
            self.graph.write_bytes(old)
        self.write('validation.json',dict(rule=dict(self.fixed,enabled=False)))
        with self.assertRaises(ValueError):self.inputs()

    def test_c2_scope_break_and_identity_disagreement_fail(self):
        self.change(self.c2,"UPDATE meta SET value='[]' WHERE key='registry_snapshot'")
        with self.assertRaises(ValueError):self.inputs()
        # Consumer resolve must reject contradictory DOI/source even if both exist.
        x=object.__new__(er.Inputs);x.aliases={'a.md':{'a'}};x.doi={'10.1234/b':'b'}
        with self.assertRaises(ValueError):x.resolve(dict(source='a.md',doi='10.1234/b'))


class EvaluationTests(unittest.TestCase):
    def test_dev_only_does_not_deserialize_test(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'q.jsonl'
            p.write_text('{"split":"test", INVALID JSON}\n'+json.dumps(dict(split='dev',query_id='q',query='q')),encoding='utf-8')
            self.assertEqual(len(ex.dev_rows(p)),1)

    def test_pooled_evaluator_denominators_and_unknown(self):
        ev=ex.evaluator()
        rows=[dict(query_id=str(i),query=str(i),judgements=[dict(source_file='a.md',relevance=grade)]) for i,grade in enumerate((3,1,0))]
        ranks={str(i):[dict(source='unjudged.md'),dict(source='a.md')] for i in range(3)}
        scored=ev['evaluate'](rows,[5,10,20,50],retrieval_cache=ranks)
        self.assertEqual((scored['total_queries'],scored['evaluable_queries'],scored['excluded_no_positive_queries']),(3,1,2))
        self.assertEqual(scored['per_query'][0]['mrr_at_10'],.5)
        self.assertIsNone(scored['per_query'][1]['ndcg']['10'])
        c=ex.comparison(scored,scored)
        self.assertEqual(c['changes']['nDCG@10'],dict(unchanged=1,excluded=2))

    def test_strata_unit_missing_and_denominators(self):
        class Inputs:
            papers={'a':dict(year='2025'),'b':dict(year='')}
            signals={'a':dict(publication_type=dict(group='other')),'b':dict(publication_type=dict(group='missing'))}
            def resolve(self,item):return item['source'].split('.')[0]
        rows=[dict(query_id='q',query='q',judgements=[dict(source_file='a.md',relevance=3),dict(source_file='b.md',relevance=2)])]
        rankings={'relevance_only':{'q':[dict(source='a.md'),dict(source='b.md')]}}
        s=ex.stratify(rows,rankings,Inputs(),ex.evaluator())
        for axis,key in [('year','>=2024'),('year','missing'),('publication_type','missing')]:
            self.assertEqual(s[axis][key]['positive_judgement_occurrences'],1)
            self.assertEqual(s[axis][key]['comparisons']['relevance_only']['evaluable'],1)


if __name__=='__main__':unittest.main()
