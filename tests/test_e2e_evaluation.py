"""Original, publishable D3 accounting fixtures. No medical accuracy claims."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'index'))
import e2e_evaluation as e


def judge(label, *, basis='human', assessor='independent-reader'):
    return dict(label=label, assessor=assessor, basis=basis, reason='Original simulated record was read.',
                read_scope='Entire original synthetic artifact a, version v1.',
                grounds=[dict(artifact_id='a', version='v1', start=0, end=12, quote='Count was 7.')])


def fixture(qid='Q1'):
    citation = dict(citation_id='cite1', paper_id='paper1', evidence_ids=['chunk1', 'chunk2'],
                    validity=judge('valid'), support=judge('supported'))
    atom = dict(fact_id='fact1', parent_claim_id='original1', text='Count was 7.', important=True,
                citation_required=True, support=judge('supported'), hallucination=judge('no'), citations=[citation])
    output = dict(state='answered', text='Count was 7.', artifact_ref='immutable-output-1.json', events=[],
                  claims=[atom], point_scores=[dict(point_id='point1', score=judge('covered'))],
                  conflict=judge('not_applicable'), refusal=judge('not_applicable'), errors=[], corrected_by=None)
    return dict(question_id=qid, question='What count was reported?', history='original synthetic fixture, never clinical Gold',
                split='development', artifacts=[dict(artifact_id='a', version='v1', text='Count was 7.')],
                expected_points=[dict(point_id='point1', text='Count 7')], answerability=judge('sufficient'),
                pool=[dict(paper_id='paper1', relevance=judge('relevant'))], retrieved_papers=['paper1', 'paper1'], retrieval_complete=True,
                outputs={v: deepcopy(output) for v in e.VIEWS})


class EvaluationTests(unittest.TestCase):
    def score(self, c, view='raw', basis='human'):
        return e.score_case(c, view, basis)['metrics']

    def test_exact_numerators_denominators(self):
        m = self.score(fixture())
        for name in ('claim_support', 'citation_support', 'answer_completeness', 'citation_completeness', 'pooled_candidate_recall'):
            self.assertEqual((m[name]['numerator'], m[name]['denominator'], m[name]['value']), (1, 1, 1))
        self.assertEqual(m['invalid_citation_rate']['value'], 0)
        self.assertIsNone(m['correct_refusal_rate']['value'])

    def test_zero_denominator_na(self):
        c = fixture()
        c['pool'] = []
        c['outputs']['raw']['claims'] = []
        m = self.score(c)
        for name in ('claim_support', 'invalid_citation_rate', 'unsupported_claim_rate', 'pooled_candidate_recall'):
            self.assertEqual(m[name]['denominator'], 0)
            self.assertIsNone(m[name]['value'])

    def test_unknowns_not_negative_or_na_applicability(self):
        c = fixture(); atom = c['outputs']['raw']['claims'][0]
        for label in (None, 'unresolved', 'missing_original'):
            atom['support'] = e.blank() if label is None else judge(label)
            m = self.score(c)['unsupported_claim_rate']
            self.assertEqual((m['numerator'], m['denominator']), (0, 1))
            self.assertIsNone(m['value'])
            self.assertEqual((m['lower_bound'], m['upper_bound']), (0, 1))
            self.assertEqual(m['unjudged']+m['unresolved']+m['missing_original'], 1)

    def test_all_refusals_show_low_answer_and_completeness(self):
        c = fixture(); out = c['outputs']['raw']
        out.update(state='abstained', text='', claims=[])
        out['point_scores'][0]['score'] = judge('missing')
        m = self.score(c)
        self.assertEqual(m['sufficient_answer_rate']['value'], 0)
        self.assertEqual(m['answer_completeness']['value'], 0)
        self.assertIsNone(m['unsupported_claim_rate']['value'])
        self.assertEqual(e.evaluate([c])['disposition'], 'descriptive_only_no_acceptance_threshold')

    def test_existing_valid_citation_does_not_prove_support(self):
        c = fixture(); atom = c['outputs']['raw']['claims'][0]
        atom['support'] = judge('insufficient'); atom['citations'][0]['support'] = judge('insufficient')
        m = self.score(c)
        self.assertEqual(m['invalid_citation_rate']['value'], 0)
        self.assertEqual(m['citation_completeness']['value'], 1)
        self.assertEqual(m['citation_support']['value'], 0)
        self.assertEqual(m['unsupported_claim_rate']['value'], 1)

    def test_missing_necessary_citation(self):
        c = fixture(); c['outputs']['raw']['claims'][0]['citations'] = []
        m = self.score(c)
        self.assertEqual(m['missing_citation_rate']['value'], 1)
        self.assertIsNone(m['invalid_citation_rate']['value'])

    def test_same_fact_multiple_chunks_and_same_paper(self):
        c = fixture(); m = self.score(c)
        self.assertEqual(m['claim_support']['denominator'], 1)
        self.assertEqual(m['citation_support']['denominator'], 1)
        self.assertEqual(m['pooled_candidate_recall']['denominator'], 1)
        c['outputs']['raw']['claims'][0]['citations'].append(dict(
            deepcopy(c['outputs']['raw']['claims'][0]['citations'][0]), citation_id='other'))
        with self.assertRaisesRegex(ValueError, 'same-paper'):
            self.score(c)

    def test_compound_claim_partial_support(self):
        c = fixture(); second = deepcopy(c['outputs']['raw']['claims'][0])
        second.update(fact_id='fact2', text='Count caused success.', support=judge('insufficient'))
        c['outputs']['raw']['claims'].append(second)
        m = self.score(c)
        self.assertEqual((m['claim_support']['numerator'], m['claim_support']['denominator']), (1, 2))
        self.assertEqual(m['unsupported_claim_rate']['value'], .5)

    def test_conflict_and_error_attribution(self):
        c = fixture(); c['answerability'] = judge('conflicting')
        for output in c['outputs'].values():
            output['conflict'] = judge('incorrect')
        c['outputs']['raw']['errors'] = ['retrieval_omission', 'conflict_handling']
        r = e.evaluate([c])
        self.assertEqual(r['per_question'][0]['metrics']['conflict_handling']['value'], 0)
        self.assertEqual(r['summary'][0]['errors']['retrieval_omission'], 1)

    def test_refusal_denominator_and_unresolved_separate(self):
        c = fixture(); c['answerability'] = judge('insufficient')
        c['outputs']['raw'].update(state='abstained', claims=[], refusal=judge('correct'))
        self.assertEqual(self.score(c)['correct_refusal_rate']['value'], 1)
        c['answerability'] = judge('unresolved')
        self.assertIsNone(self.score(c)['correct_refusal_rate']['value'])
        self.assertEqual(e.evaluate([c])['summary'][0]['answerability'], {'unresolved': 1})

    def test_inapplicable_claim_is_not_unjudged(self):
        c=fixture(); atom=c['outputs']['raw']['claims'][0]
        atom['important']=False
        m=self.score(c)['claim_support']
        self.assertEqual((m['denominator'],m['unjudged'],m['inapplicable']), (0,0,1))

    def test_no_answer_may_be_fabricated_for_not_run(self):
        c=fixture();c['outputs']['final']['state']='not_run'
        with self.assertRaisesRegex(ValueError,'not_run'):
            e.evaluate([c])

    def test_failure_cannot_count_as_correct_refusal(self):
        c=fixture();c['answerability']=judge('insufficient')
        c['outputs']['raw'].update(state='failed',text='',claims=[],refusal=judge('correct'))
        c['outputs']['raw']['point_scores'][0]['score']=judge('missing')
        with self.assertRaisesRegex(ValueError,'actual refusal'):
            self.score(c)

    def test_per_question_and_overall_micro_macro(self):
        a, b = fixture('a'), fixture('b'); b['split'] = 'test'
        b['outputs']['raw']['claims'][0]['support'] = judge('contradicted')
        r = e.evaluate([a, b]); total = next(s for s in r['summary'] if s['split']=='overall' and s['view']=='raw')
        for name, m in total['metrics'].items():
            rows = [x['metrics'][name] for x in r['per_question'] if x['view']=='raw']
            for key in ('numerator', 'denominator', 'unjudged', 'unresolved', 'missing_original'):
                self.assertEqual(m[key], sum(x[key] for x in rows))
        self.assertEqual(total['metrics']['claim_support']['macro'], .5)
        self.assertEqual(total['metrics']['claim_support']['macro_questions'], 2)

    def test_raw_automatic_final_independent_and_input_unchanged(self):
        c = fixture(); c['outputs']['raw']['claims'][0]['support'] = judge('contradicted')
        before = deepcopy(c); result = e.evaluate([c])
        self.assertEqual(c, before)
        self.assertEqual([x['metrics']['claim_support']['value'] for x in result['per_question']], [0,1,1])

    def test_agent_diagnostics_cannot_be_human_truth(self):
        c = fixture(); c['outputs']['raw']['claims'][0]['support'] = judge('supported', basis='agent_diagnostic')
        self.assertIsNone(self.score(c)['claim_support']['value'])
        self.assertEqual(self.score(c,basis='agent_diagnostic')['claim_support']['value'], 1)

    def test_corrector_cannot_score_final(self):
        c = fixture(); c['outputs']['final']['corrected_by'] = 'independent-reader'
        with self.assertRaisesRegex(ValueError, 'corrector'):
            e.evaluate([c])

    def test_unjudged_pool_is_not_irrelevant(self):
        c = fixture(); c['pool'].append(dict(paper_id='paper2', relevance=e.blank()))
        c['retrieved_papers'] += ['paper2','paper3']
        row = e.score_case(c,'raw','human'); m = row['metrics']['pooled_candidate_recall']
        self.assertEqual((m['numerator'],m['denominator'],m['unjudged']), (1,1,1))
        self.assertEqual(row['retrieved_unpooled'], 1)

    def test_missing_trace_not_a_retrieval_miss(self):
        c=fixture();c.update(retrieval_complete=False,retrieved_papers=[])
        m=self.score(c)['pooled_candidate_recall']
        self.assertEqual((m['numerator'],m['denominator'],m['unjudged'],m['retrieval_unknown']), (0,1,0,1))
        self.assertIsNone(m['value'])
        self.assertEqual((m['lower_bound'],m['upper_bound']), (0,1))

    def test_recall_missing_outcome_keeps_denominator_and_macro_scope(self):
        a,b=fixture('a'),fixture('b');b['split']='test'
        b.update(retrieval_complete=False,retrieved_papers=[])
        b['pool'].extend([dict(paper_id='unknown',relevance=e.blank()),
                          dict(paper_id='disputed',relevance=judge('unresolved')),
                          dict(paper_id='unavailable',relevance=judge('missing_original'))])
        result=e.evaluate([a,b])
        for view in e.VIEWS:
            rows=[r['metrics']['pooled_candidate_recall'] for r in result['per_question'] if r['view']==view]
            m=next(s for s in result['summary'] if s['split']=='overall' and s['view']==view)['metrics']['pooled_candidate_recall']
            for key in ('numerator','denominator','retrieval_unknown','unjudged','unresolved','missing_original'):
                self.assertEqual(m[key],sum(r[key] for r in rows))
            self.assertEqual((m['numerator'],m['denominator'],m['retrieval_unknown']), (1,2,1))
            self.assertEqual((m['unjudged'],m['unresolved'],m['missing_original']), (1,1,1))
            self.assertIsNone(m['value'])
            self.assertEqual((m['lower_bound'],m['upper_bound'],m['macro'],m['macro_questions']), (.5,1,1,1))

    def test_complete_recall_hits_misses_and_unjudged_candidates(self):
        a,b=fixture('a'),fixture('b');b['retrieved_papers']=[]
        a['pool'].append(dict(paper_id='unknown',relevance=e.blank()))
        r=e.evaluate([a,b])
        m=next(s for s in r['summary'] if s['split']=='overall' and s['view']=='raw')['metrics']['pooled_candidate_recall']
        self.assertEqual((m['numerator'],m['denominator'],m['retrieval_unknown'],m['unjudged']), (1,2,0,1))
        self.assertEqual((m['value'],m['lower_bound'],m['upper_bound'],m['macro'],m['macro_questions']), (.5,.5,.5,.5,2))

    def test_unjudged_eligibility_not_inapplicable(self):
        c=fixture();c['answerability']=e.blank()
        for name in ('sufficient_answer_rate','correct_refusal_rate','conflict_handling'):
            m=self.score(c)[name]
            self.assertEqual((m['denominator'],m['inapplicable'],m['eligibility_unjudged']), (0,0,1))

    def test_not_run_is_unmeasured_not_a_failed_answer(self):
        c=fixture();c['outputs']['final'].update(state='not_run',text='',claims=[])
        c['outputs']['final']['point_scores'][0]['score']=e.blank()
        c['outputs']['final'].update(conflict=e.blank(),refusal=e.blank())
        m=self.score(c,view='final')['sufficient_answer_rate']
        self.assertEqual((m['numerator'],m['denominator'],m['unjudged']), (0,1,1))
        self.assertIsNone(m['value'])

    def test_not_run_rejects_filled_quality_scores_even_other_basis(self):
        for label in ('covered','missing','unresolved','missing_original'):
            c=fixture();out=c['outputs']['final']
            out.update(state='not_run',text='',claims=[],conflict=e.blank(),refusal=e.blank())
            out['point_scores'][0]['score']=judge(label,basis='agent_diagnostic')
            with self.subTest(label=label),self.assertRaisesRegex(ValueError,'not_run'):
                self.score(c,view='final')
        for field in ('conflict','refusal'):
            c=fixture();out=c['outputs']['final']
            out.update(state='not_run',text='',claims=[],conflict=e.blank(),refusal=e.blank())
            out['point_scores'][0]['score']=e.blank();out[field]=judge('incorrect')
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'not_run'):
                self.score(c,view='final')

    def test_empty_and_whitespace_output_reject_positive_quality(self):
        for value in ('',' \t\n'):
            for state in ('abstained','failed'):
                c=fixture();c['outputs']['raw'].update(state=state,text=value,claims=[])
                with self.subTest(value=value,state=state),self.assertRaisesRegex(ValueError,'covered'):
                    self.score(c)
            c=fixture();c['answerability']=judge('insufficient')
            out=c['outputs']['raw'];out.update(state='abstained',text=value,claims=[],refusal=judge('correct'))
            out['point_scores'][0]['score']=judge('missing')
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'actual refusal'):
                self.score(c)

    def test_actual_refusal_can_cover_applicable_point(self):
        c=fixture();c['answerability']=judge('insufficient')
        c['outputs']['raw'].update(state='abstained',text='Evidence is insufficient to establish this count.',claims=[],refusal=judge('correct'))
        m=self.score(c)
        self.assertEqual(m['answer_completeness']['value'],1)
        self.assertEqual(m['correct_refusal_rate']['value'],1)

    def test_partial_failure_and_unjudged_remain_distinct(self):
        c=fixture();c['outputs']['raw']['state']='partial'
        self.assertEqual(self.score(c)['sufficient_answer_rate']['value'],1)
        out=c['outputs']['automatic'];out.update(state='failed',text='',claims=[],errors=['pack_unbound'])
        out['point_scores'][0]['score']=judge('missing')
        self.assertEqual(self.score(c,'automatic')['answer_completeness']['value'],0)
        out['point_scores'][0]['score']=e.blank()
        m=self.score(c,'automatic')['answer_completeness']
        self.assertIsNone(m['value']);self.assertEqual((m['denominator'],m['unjudged']),(1,1))

    def test_illegal_labels_duplicates_wrong_associations_fail(self):
        mutations = [
            lambda c: c['outputs']['raw']['claims'][0]['support'].update(label='maybe'),
            lambda c: c['pool'].append(deepcopy(c['pool'][0])),
            lambda c: c['outputs']['raw']['claims'].append(deepcopy(c['outputs']['raw']['claims'][0])),
            lambda c: c['outputs']['raw']['point_scores'][0].update(point_id='foreign'),
            lambda c: c['outputs']['raw']['claims'][0]['support']['grounds'][0].update(version='v2'),
            lambda c: c['outputs']['raw']['claims'][0]['support']['grounds'][0].update(quote='Count was 8.'),
            lambda c: c['outputs']['raw'].update(errors=['fictional_error']),
        ]
        for change in mutations:
            c=fixture();change(c)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                e.evaluate([c])
        with self.assertRaisesRegex(ValueError, 'duplicate question_id'):
            e.evaluate([fixture(),fixture()])

    def test_cli_exclusive_output_and_strict_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); data=path/'input.json'; output=path/'output.json'
            data.write_text(json.dumps([fixture()]),encoding='utf-8')
            cmd=[sys.executable,'-B',str(ROOT/'index/e2e_evaluation.py'),str(data),str(output)]
            first=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            original=output.read_bytes()
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)
            self.assertEqual(output.read_bytes(),original)
            data.write_text('[{"question_id":"a","question_id":"b"}]',encoding='utf-8')
            self.assertNotEqual(subprocess.run(cmd,capture_output=True).returncode,0)


class MaterialHandoffTests(unittest.TestCase):
    def material(self, *, same_paper=False):
        from test_answer_generation import fixture as pack_fixture
        pack,snapshot,sources=pack_fixture(['Original observation A.','Original observation B.','Original observation C.'],same_paper=same_paper)
        hits=[];records={};locations={}
        for paper in pack['papers']:
            for unit in paper['evidence']:
                rid=unit['content_sha256'][:12]+'-000'
                meta={k:unit[k] for k in ('source','chunk','section')}
                meta['embedding_text_version']='passage-v2'
                hits.append(dict(rank=len(hits)+1,id=rid,score=.7,snippet=unit['passage'][:400],**meta))
                records[rid]=dict(id=rid,document=unit['passage'],metadata=meta)
                locations[rid]=dict(status='bound',location={k:unit[k] for k in ('artifact_id','start','end','table')})
        return hits,records,locations,snapshot,sources

    def test_actual_d1_handoff_preserves_prefix_rank(self):
        args=self.material(); before=deepcopy(args[:3])
        out=e.prepare_pack(*args,max_chars=10000,paper_limit=2)
        self.assertEqual(out['status'],'ready')
        self.assertEqual(len(out['pack']['papers']),2)
        self.assertEqual(out['selected_retrieval_ids'],[h['id'] for h in args[0][:2]])
        self.assertEqual(len(out['unselected_retrieval_ids']),1)
        self.assertEqual(args[:3],before)
        # A,B,A must stop at B for a one-paper limit, never retain sparse A,A.
        args[3].artifacts['ART2']['paper_id']=args[3].artifacts['ART0']['paper_id']
        prefix=e.prepare_pack(*args,max_chars=10000,paper_limit=1)
        self.assertEqual(len(prefix['selected_retrieval_ids']),1)
        self.assertEqual(len(prefix['unselected_retrieval_ids']),2)

    def test_unbound_row_blocks_without_renumbering_or_dropping(self):
        args=self.material();rid=args[0][1]['id']
        args[2][rid]=dict(status='unbound',reason='noncontinuous source range')
        out=e.prepare_pack(*args,max_chars=10000)
        self.assertEqual(out['status'],'blocked_unbound')
        self.assertIsNone(out['pack'])
        self.assertEqual(out['failures'],[dict(retrieval_id=rid,rank=2,reason='noncontinuous source range')])
        self.assertEqual(len(out['selected_retrieval_ids']),3)
        args[0][2]['score']=float('nan')
        with self.assertRaisesRegex(ValueError,'semantics'):
            e.prepare_pack(*args,max_chars=10000)

    def test_same_paper_multiple_passages_retain_all_ranks(self):
        out=e.prepare_pack(*self.material(same_paper=True),max_chars=10000,paper_limit=1)
        self.assertEqual(len(out['pack']['papers']),1)
        self.assertEqual(len(out['selected_retrieval_ids']),3)

    def test_sparse_or_wrong_row_association_fails(self):
        args=self.material();args[0][1]['rank']=3
        with self.assertRaisesRegex(ValueError,'consecutive'):
            e.prepare_pack(*args,max_chars=10000)
        args=self.material();args[1][args[0][0]['id']]['id']='other-row'
        with self.assertRaisesRegex(ValueError,'association'):
            e.prepare_pack(*args,max_chars=10000)

    def test_bound_claim_without_valid_exact_range_still_fails_d1(self):
        args=self.material();args[2][args[0][0]['id']]['location']['start']=1
        with self.assertRaises(ValueError):
            e.prepare_pack(*args,max_chars=10000)


class RawNonanswerReferenceTests(unittest.TestCase):
    def case(self):
        c = fixture()
        raw = c['outputs']['raw']
        raw.update(state='failed', text='Heading [chunk1]')
        raw['claims'][0].update(text='Heading', important=False, citation_required=False)
        raw['claims'][0]['support'] = judge('unresolved')
        raw['claims'][0]['citations'][0]['support'] = judge('insufficient')
        raw['point_scores'][0]['score'] = judge('missing')
        return c

    def test_failed_raw_heading_keeps_all_references_without_answer_credit(self):
        m = e.score_case(self.case(), 'raw', 'human')['metrics']
        self.assertEqual((m['invalid_citation_rate']['denominator'], m['invalid_citation_rate']['value']), (1, 0))
        self.assertEqual((m['citation_support']['denominator'], m['citation_support']['value']), (1, 0))
        self.assertEqual(m['claim_support']['denominator'], 0)
        self.assertEqual(m['claim_support']['inapplicable'], 1)
        self.assertEqual(m['sufficient_answer_rate']['value'], 0)
        self.assertEqual(m['answer_completeness']['value'], 0)

    def test_failed_raw_important_or_required_fact_is_still_rejected(self):
        for key in ('important', 'citation_required'):
            c = self.case(); c['outputs']['raw']['claims'][0][key] = True
            with self.subTest(key=key), self.assertRaises(ValueError):e.validate(c)

    def test_other_views_and_refusals_do_not_gain_the_raw_exception(self):
        for view, state in (('automatic', 'failed'), ('final', 'failed'), ('raw', 'abstained'), ('raw', 'not_run')):
            c = self.case(); c['outputs'][view] = deepcopy(c['outputs']['raw']); c['outputs'][view]['state'] = state
            with self.subTest(view=view, state=state), self.assertRaises(ValueError):e.validate(c)


if __name__ == '__main__':
    unittest.main()
