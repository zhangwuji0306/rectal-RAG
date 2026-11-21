"""Synthetic C3 label, fixed-rule and held-out safety contracts."""
import copy
import csv
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'index'))
import citation_calibration as cc
import citation_graph as cg
from tests import test_citation_graph as graph_tests


def fixture():
    papers = {p:dict(paper_id=p, title=t, year='2020', authors='Smith A', journal='Alpha Journal',
                    identity_status='verified') for p,t in [('a','Source'),('b','Rectal imaging'),('c','Other study')]}
    candidates = [dict(paper_id=p, score=s, title_token_coverage=1, reference_years=['2020'],
                       **{k:papers[p][k] for k in ('title','year','authors','identity_status')})
                  for p,s in [('b',1.0),('c',.5)]]
    ref = dict(ref_id='r1',paper_id='a', role='main',parse_status='parsed',
        raw='1. Smith A. Rectal imaging. Alpha Journal 2020;1:2.', source='a.md',path='synthetic.md',
        artifact_id='art-a',line_start=1,line_end=1,
        match=dict(status='fuzzy_candidate', target=None, identifiers={},declarations=[], candidates=candidates))
    answer = dict(version='v',ref_id='r1',boundary='valid',human_label='confirmed_match',
                  human_target_paper_id='b',human_reason='Synthetic title/year agreement\nwith newline')
    data = dict(split='calibration',material_version='v',matcher_rules=cg.RULES,papers=papers,
                records=[dict(reference=ref,answer=answer,challenge='near_title')])
    return data,ref,papers,answer


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.data,self.ref,self.papers,self.answer=fixture()
        self.fixed=cc.rule('v',.95,.1)

    def save(self,name,value):
        path=self.root/name; cc.write_json(path,value); return path

    def test_label_version_ref_target_and_reason(self):
        cc.validate_answer(self.answer,self.ref,'v',self.papers)
        for field,value in [('version','bad'),('ref_id','bad'),('human_target_paper_id','missing'),
                            ('human_label','bad'),('boundary','bad'),('human_reason',' ')]:
            a=dict(self.answer,**{field:value})
            with self.subTest(field=field),self.assertRaises(ValueError):
                cc.validate_answer(a,self.ref,'v',self.papers)
        self.papers['b']['identity_status']='pending'
        with self.assertRaises(ValueError):cc.validate_answer(self.answer,self.ref,'v',self.papers)

    def test_boundary_label_combinations(self):
        for boundary,label in [('invalid','no_confirmed_local_match'),('uncertain','confirmed_match'),('valid','not_reference_or_bad_boundary')]:
            a=dict(self.answer,boundary=boundary,human_label=label)
            with self.assertRaises(ValueError):cc.validate_answer(a,self.ref,'v',self.papers)
        a=dict(self.answer,boundary='uncertain',human_label='insufficient_evidence',human_target_paper_id='')
        cc.validate_answer(a,self.ref,'v',self.papers)
        a['human_target_paper_id']='b'
        with self.assertRaises(ValueError):cc.validate_answer(a,self.ref,'v',self.papers)

    def test_calibration_rejects_validation_entry(self):
        self.data['split']='validation'
        path=self.save('validation.json',self.data)
        with self.assertRaises(ValueError):cc.calibrate(path,self.root/'cal')
        self.assertFalse((self.root/'cal').exists())
        with self.assertRaises(SystemExit):
            with redirect_stdout(io.StringIO()):cc.main(['calibrate','--validation-input',str(path),'--output','x'])

    def test_validation_rejects_calibration_entry(self):
        path=self.save('calibration.json',self.data)
        rule=self.save('rule.json',self.fixed)
        with self.assertRaises(ValueError):cc.validate(path,rule,self.root/'val')

    def test_ranking_uses_scores_not_display_or_human_target(self):
        self.ref['match']['candidates'].reverse()
        self.answer['human_target_paper_id']='c'
        result=cc.evaluate(self.data,self.fixed)
        self.assertEqual(result['rows'][0]['prediction']['target'],'b')
        self.assertEqual(result['summary']['wrong_binding'],1)

    def test_tie_rejected_even_different_year_and_coverage(self):
        self.ref['match']['candidates'][1]['score']=1
        self.ref['match']['candidates'][1]['title_token_coverage']=.2
        self.assertEqual(cc.predict(self.ref,self.papers,self.fixed)['reason'],'tied_top_score')

    def test_margin_decimal_boundary_and_single_candidate(self):
        self.ref['match']['candidates'][0]['score']=.95
        self.ref['match']['candidates'][1]['score']=.85
        self.assertTrue(cc.predict(self.ref,self.papers,self.fixed)['accepted'])
        self.ref['match']['candidates'][1]['score']=.850001
        self.assertEqual(cc.predict(self.ref,self.papers,self.fixed)['reason'],'below_margin')
        self.ref['match']['candidates'].pop()
        self.assertEqual(cc.predict(self.ref,self.papers,self.fixed)['second_score'],0)

    def test_year_required_and_not_replaced_by_human(self):
        self.ref['raw']=self.ref['raw'].replace('2020','2021')
        for c in self.ref['match']['candidates']:c['reference_years']=['2021']
        self.assertEqual(cc.predict(self.ref,self.papers,self.fixed)['reason'],'year_absent')

    def test_program_boundary_not_changed_by_human(self):
        self.ref['parse_status']='boundary_unverified'
        r=cc.evaluate(self.data,self.fixed)
        self.assertEqual(r['summary']['program_buildable'],0)
        self.assertEqual(r['summary']['end_to_end_recall']['denominator'],1)
        self.assertEqual(r['summary']['end_to_end_recall']['value'],0)

    def test_role_identity_self_and_no_skipping_top1(self):
        for field,value,reason in [('role','supplement','non_main_role')]:
            self.ref[field]=value
            self.assertEqual(cc.predict(self.ref,self.papers,self.fixed)['reason'],reason)
        self.ref['role']='main'; self.papers['a']['identity_status']='pending'
        self.assertEqual(cc.predict(self.ref,self.papers,self.fixed)['reason'],'unsafe_citing_identity')
        self.papers['a']['identity_status']='verified';self.ref['paper_id']='b'
        self.assertEqual(cc.predict(self.ref,self.papers,self.fixed)['reason'],'self_loop')
        self.ref['paper_id']='a'; self.papers['b']['identity_status']='pending'
        self.ref['match']['candidates'][0]['identity_status']='pending'
        self.assertEqual(cc.predict(self.ref,self.papers,self.fixed)['reason'],'unsafe_target_identity')

    def test_strong_identifier_and_f1_never_promoted(self):
        self.ref['raw']+=' DOI: unreadable'
        self.assertEqual(cc.predict(self.ref,self.papers,self.fixed)['reason'],'strong_identifier_present')
        for status in ('bibliographic_conflict','unresolved','conflict','ambiguous','external_or_unresolved'):
            self.ref['match']['status']=status
            self.assertFalse(cc.predict(self.ref,self.papers,self.fixed)['accepted'])

    def test_fuzzy_bibliographic_contradiction_screen(self):
        self.ref['raw']='1. Jones B. Completely different title. Beta Journal 2020;1:2.'
        self.assertEqual(cc.predict(self.ref,self.papers,self.fixed)['reason'],'bibliographic_conflict')

    def test_exact_unchanged_when_fuzzy_disabled(self):
        self.ref['match']=dict(status='exact',target='b')
        r=cc.predict(self.ref,self.papers,cc.rule('v'))
        self.assertTrue(r['accepted']);self.assertEqual(r['kind'],'exact')

    def test_finite_search_and_tie_selection_deterministic(self):
        path=self.save('cal.json',self.data)
        a=cc.calibrate(path,self.root/'first')
        b=cc.calibrate(path,self.root/'second')
        self.assertEqual(a,b)
        self.assertEqual((a['selected']['score_threshold'],a['selected']['margin_threshold']),(1,.1))
        self.assertEqual(len(cc.read_json(self.root/'first/all-combinations.json')),11)

    def test_unconfirmed_acceptance_disables_all_unsafe_rules(self):
        self.answer.update(human_label='no_confirmed_local_match',human_target_paper_id='')
        path=self.save('cal.json',self.data)
        result=cc.calibrate(path,self.root/'cal')
        self.assertFalse(result['selected']['enabled'])
        for r in cc.read_json(self.root/'cal/all-combinations.json')[1:]:
            self.assertEqual(r['summary']['unconfirmed_acceptance'],1)
            self.assertIsNone(r['summary']['acceptance_precision']['value'])
            self.assertEqual(r['summary']['confirmed_correct_fraction']['value'],0)

    def test_zero_correct_safe_rules_still_disable(self):
        self.ref['match']['candidates'][0]['score']=.8
        result=cc.calibrate(self.save('cal.json',self.data),self.root/'cal')
        self.assertFalse(result['selected']['enabled'])
        self.assertEqual(result['summary']['correct'],0)

    def test_outside_candidates_and_zero_denominators(self):
        self.answer['human_target_paper_id']='a'
        result=cc.evaluate(self.data,self.fixed)
        self.assertEqual(result['summary']['candidate_misses'],1)
        self.assertEqual(result['summary']['end_to_end_recall']['value'],0)
        self.answer.update(human_label='ambiguous',human_target_paper_id='')
        result=cc.evaluate(self.data,cc.rule('v'))
        for k in ('acceptance_precision','confirmed_correct_fraction','end_to_end_recall'):
            self.assertIsNone(result['summary'][k]['value'])

    def test_heldout_failure_disables_without_retuning(self):
        self.data['split']='validation';self.answer['human_target_paper_id']='c'
        path=self.save('validation.json',self.data); fixed=self.save('rule.json',self.fixed)
        before=fixed.read_bytes()
        result=cc.validate(path,fixed,self.root/'val')
        self.assertTrue(result['decision']['safety_failure'])
        self.assertFalse(result['decision']['fuzzy_enabled'])
        self.assertFalse(result['decision']['thresholds_retuned'])
        self.assertEqual(before,fixed.read_bytes())
        self.assertEqual(cc.read_json(self.root/'val/results.json')['summary']['wrong_binding'],1)

    def test_unconfirmed_heldout_failure_preserved(self):
        self.data['split']='validation';self.answer.update(human_label='insufficient_evidence',human_target_paper_id='')
        result=cc.validate(self.save('val.json',self.data),self.save('rule.json',self.fixed),self.root/'out')
        self.assertTrue(result['decision']['safety_failure'])
        self.assertEqual(result['summary']['unconfirmed_acceptance'],1)

    def test_rule_corruption_and_output_overwrite_fail(self):
        for field,value in [('rule_version','bad'),('material_version','bad'),('score_threshold',.7)]:
            bad=dict(self.fixed,**{field:value})
            with self.assertRaises(ValueError):cc.validate_rule(bad,'v')
        path=self.save('input.json',self.data);before=path.read_bytes()
        with self.assertRaises(FileExistsError):cc.write_json(path,{})
        self.assertEqual(path.read_bytes(),before)

    def test_saved_candidate_invalid_fails(self):
        for score in (float('nan'),1.01):
            self.ref['match']['candidates'][0]['score']=score
            with self.assertRaises(ValueError):cc.predict(self.ref,self.papers,self.fixed)

    def test_duplicate_version_and_reference_leakage(self):
        i=[dict(ref_id='r1',split='calibration',citing_paper_id='a'),dict(ref_id='r2',split='validation',citing_paper_id='b')]
        refs={'r1':self.ref,'r2':dict(self.ref,ref_id='r2')}
        relation=[dict(kind='alternate_version',from_paper_id='a',to_paper_id='b',status='pending')]
        found=cc.leakage(i,refs,self.papers,[],relation)
        self.assertIn('duplicate_reference',found[0]['reasons'])
        self.assertIn('shared_citing_identity_or_version_duplicate_relation',found[0]['reasons'])
        refs['r2']['raw']='Entirely different study in another year 2010'
        self.assertEqual(cc.leakage(i,refs,self.papers,[],[]),[])


class GraphApplicationTests(unittest.TestCase):
    def setUp(self):
        # Reuse the existing synthetic C1 builder, without inheriting its tests.
        self.storage=graph_tests.StorageTests(); self.storage.setUp(); self.addCleanup(self.storage.doCleanups)
        self.storage.build();self.root=self.storage.root;self.db=self.storage.out

    def test_disabled_application_no_database_and_readonly(self):
        version=cg.snapshot(self.db)['meta']['citation_graph_version']
        fixed=cc.rule(version,.95,.1)
        rp=self.root/'rule.json';cc.write_json(rp,fixed)
        decision=dict(selected_rule=fixed,rule_version=fixed['rule_version'],fuzzy_enabled=False,
            final_rule=cc.rule(version), counts=dict(correct=0,wrong_binding=1,unconfirmed_acceptance=0),reason='safety failure')
        dp=self.root/'decision.json';cc.write_json(dp,decision)
        before=self.db.read_bytes()
        result=cc.apply_rule(self.db,rp,dp,self.root/'applied')
        self.assertFalse(result['fuzzy_enabled']);self.assertIsNone(result['new_database'])
        self.assertEqual(before,self.db.read_bytes())
        self.assertFalse((self.root/'applied/citation-graph.sqlite3').exists())

    def package(self):
        data=cg.snapshot(self.db)
        version=data['meta']['citation_graph_version']
        nodes={r['paper_id']:json.loads(r['payload']) for r in data['nodes']}
        first=json.loads(data['refs'][0]['payload'])
        second=dict(first,ref_id='second',paper_id=first['match']['target'],raw='2. Different study. 2011.')
        c=sqlite3.connect(self.db)
        try:
            c.execute('INSERT INTO refs VALUES (?,?)',(second['ref_id'],cg.dump(second)));c.commit()
        finally:c.close()
        items=[];answers=[]
        for split,ref in [('calibration',first),('validation',second)]:
            items.append(dict(split=split,citing_paper_id=ref['paper_id'],
                **{k:ref[k] for k in ('ref_id','raw','source','line_start','line_end','parse_status')},
                candidates=sorted(x['paper_id'] for x in ref['match']['candidates']),
                context=[dict(text=ref['raw'],selected=True)]))
            answers.append(dict(version=version,ref_id=ref['ref_id'],boundary='valid',human_label='no_confirmed_local_match',
                                human_target_paper_id='',human_reason='Synthetic reason\nwith newline'))
        catalog={p:{k:r[k] for k in ('paper_id','title','authors','year','journal','identity_status')} for p,r in nodes.items()}
        mp=self.root/'materials.json';cc.write_json(mp,dict(version=version,items=items,catalog=catalog))
        lp=self.root/'labels.sqlite3';c=sqlite3.connect(lp)
        try:
            c.executescript('CREATE TABLE scope(version); CREATE TABLE labels(ref_id TEXT PRIMARY KEY,payload TEXT);')
            c.execute('INSERT INTO scope VALUES (?)',(version,))
            c.executemany('INSERT INTO labels VALUES (?,?)',[(a['ref_id'],cg.dump(a)) for a in answers]);c.commit()
        finally:c.close()
        csvs=[]
        for i,a in zip(items,answers):
            row={k:i[k] for k in ('split','ref_id','citing_paper_id','source','line_start','line_end','parse_status')}
            row.update(a,raw_reference=i['raw'])
            path=self.root/(i['split']+'.csv')
            with path.open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            csvs.append(path)
        with (self.root/'human-review-unlabelled.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['ref_id','case']);writer.writeheader()
            writer.writerows([dict(ref_id=i['ref_id'],case='synthetic') for i in items])
        return mp,lp,*csvs

    def test_structure_csv_newlines_readonly_and_separated_exports(self):
        mp,lp,cal,val=self.package()
        before={p:p.read_bytes() for p in (mp,lp,cal,val,self.db,self.storage.c1)}
        result=cc.check(mp,lp,cal,val,self.db,self.root/'checked',expected=2)
        self.assertEqual(result['total'],2)
        self.assertEqual(cc.read_json(self.root/'checked/calibration-input.json')['split'],'calibration')
        self.assertEqual(cc.read_json(self.root/'checked/validation-input.json')['split'],'validation')
        for p,raw in before.items():self.assertEqual(p.read_bytes(),raw)

    def test_csv_mismatch_fails_before_output_or_input_writes(self):
        mp,lp,cal,val=self.package();before=lp.read_bytes()
        text=cal.read_text(encoding='utf-8-sig').replace('Synthetic reason','Changed reason')
        cal.write_text(text,encoding='utf-8-sig',newline='')
        with self.assertRaisesRegex(ValueError,'CSV/labels field mismatch'):
            cc.check(mp,lp,cal,val,self.db,self.root/'checked',expected=2)
        self.assertFalse((self.root/'checked').exists());self.assertEqual(before,lp.read_bytes())

    def test_fuzzy_support_locator_kind_dedup_and_stability(self):
        # Extend the synthetic baseline with two saved fuzzy references to the
        # same exact pair; neither label values nor ref-id whitelists are used.
        data=cg.snapshot(self.db)
        nodes={r['paper_id']:json.loads(r['payload']) for r in data['nodes']}
        exact=json.loads(data['refs'][0]['payload'])
        target=exact['match']['target'];a=exact['paper_id']
        raw='1. Smith A. Rectal cancer staging. 2020.'
        source=self.root/'fuzzy.md';source.write_text(raw,encoding='utf-8')
        m=cg.Matcher(nodes,[]).match(raw)
        m['candidates']=[x for x in m['candidates'] if x['paper_id']==target]
        c=sqlite3.connect(self.db)
        try:
            for rid,role in [('fuzzy1','version'),('fuzzy2','duplicate')]:
                r=dict(exact,ref_id=rid,role=role,raw=raw,path=str(source),line_start=1,line_end=1,match=m,edge_class='excluded')
                c.execute('INSERT INTO refs VALUES (?,?)',(rid,cg.dump(r)))
            c.commit()
        finally:c.close()
        fixed=cc.rule(data['meta']['citation_graph_version'],1,.1)
        before=self.db.read_bytes()
        results=[]
        for name in ('one','two'):
            out=self.root/name;out.mkdir()
            results.append(cc.build_candidate_graph(self.db,fixed,out))
        self.assertEqual(results[0]['graph_version'],results[1]['graph_version'])
        self.assertEqual(results[0]['graph_delta']['fuzzy_supports'],2)
        self.assertEqual(results[0]['graph_delta']['fuzzy_unique_edges'],1)
        self.assertEqual(results[0]['graph_delta']['overlap_exact'],1)
        self.assertEqual(results[0]['graph_delta']['edges'],0)
        self.assertEqual(before,self.db.read_bytes())
        c=sqlite3.connect(self.root/'one/citation-graph.sqlite3')
        try:
            self.assertEqual(c.execute('SELECT kind,count(*) FROM supports GROUP BY kind ORDER BY kind').fetchall(),[('exact',1),('fuzzy',2)])
            p=json.loads(c.execute("SELECT payload FROM supports WHERE ref_id='fuzzy1'").fetchone()[0])
            self.assertEqual(p['fuzzy_evidence']['rule_version'],fixed['rule_version'])
            self.assertEqual(p['line_start'],1)
        finally:c.close()
        verified=cc.verify_candidate(self.root/'one/citation-graph.sqlite3',self.db,locators=True)
        self.assertEqual(verified['fuzzy_locators_checked'],2)
        source.write_text('changed',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'locator mismatch'):
            cc.verify_candidate(self.root/'one/citation-graph.sqlite3',self.db,locators=True)

    def test_candidate_writer_refuses_existing_database(self):
        out=self.root/'occupied';out.mkdir()
        path=out/'citation-graph.sqlite3';path.write_bytes(self.storage.c1.read_bytes())
        before=path.read_bytes()
        fixed=cc.rule(cg.snapshot(self.db)['meta']['citation_graph_version'],1,.1)
        with self.assertRaises(FileExistsError):cc.build_candidate_graph(self.db,fixed,out)
        self.assertEqual(before,path.read_bytes())


if __name__=='__main__':unittest.main()
