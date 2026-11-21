"""Synthetic C5 contracts; no corpus, model, index, or external service."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'index'))
import query_expansion as qe
from evidence_experiment import evaluator

Q = 'In patients with rectal cancer what evidence compares magnetic resonance imaging for staging and prediction of treatment response?'


def hits(n=250, offset=0):
    return [dict(id=str(i), rank=r, source=f'p{i % 70}.md', doi='', score=1-r/10000)
            for r, i in enumerate(range(offset, offset+n), 1)]


def resolve(h):
    return h['source']


class ExpansionTests(unittest.TestCase):
    def test_trigger_preserves_entire_question_and_exact_focus(self):
        q = qe.generate(Q)[0]
        self.assertTrue(q.startswith(Q+'\n'))
        self.assertIn(q.split('\n')[1], Q)

    def test_no_trigger_short_negative_long_no_conjunction(self):
        for q in ('short query', Q.replace(' and ', ' versus '), Q+' without chemotherapy', 'word '*181+Q):
            self.assertEqual(qe.generate(q), [])

    def test_empty_and_invalid_query_fail(self):
        for q in ('', None, ' '):
            with self.assertRaises(ValueError): qe.generate(q)

    def test_dedupe_empty_and_limit(self):
        q = qe.generate(Q)[0]
        self.assertEqual(qe.subqueries(Q, ['', Q, q, q], 1), [q])
        self.assertEqual(qe.subqueries(Q, [q], 0), [])

    def test_intent_contract_error_not_fallback(self):
        with self.assertRaises(ValueError):
            qe.run(Q, hits(), resolve, None, generator=lambda _: ['invented population'])

    def test_no_expansion_exact_original_order(self):
        r = qe.run('short query', hits(), resolve, lambda *a: self.fail('fetch'))
        original = qe.papers(hits(), resolve, 'original', 'short query')
        self.assertEqual([p['paper_id'] for p in r['final']], [p['paper_id'] for p in original[:50]])
        self.assertEqual(r['accounting']['requested_slots'], 250)
        self.assertFalse(r['expanded'])

    def test_success_budget_protection_and_all_origins(self):
        r = qe.run(Q, hits(), resolve, lambda q, n, t: hits(n, 1000))
        self.assertEqual(r['accounting']['requested_slots'], 750)
        self.assertEqual(r['accounting']['retrieval_calls'], 2)
        self.assertEqual(r['accounting']['generation_calls'], 1)
        self.assertEqual([p['paper_id'] for p in r['final'][:20]], r['protected'])
        self.assertEqual(sum(len(p['origins']) for p in r['union']), 750)
        self.assertEqual({o['branch'] for p in r['union'] for o in p['origins']}, {'original', 'sub1'})
        self.assertEqual(len({p['paper_id'] for p in r['union']}), 70)

    def test_expected_generation_failure_original(self):
        def fail(_): raise qe.ExpansionUnavailable('offline')
        r = qe.run(Q, hits(), resolve, None, generator=fail)
        self.assertTrue(r['fallback'])
        self.assertEqual(r['accounting']['requested_slots'], 250)

    def test_generation_timeout(self):
        times = iter([0, 2, 2])
        r = qe.run(Q, hits(), resolve, None, clock=lambda: next(times))
        self.assertTrue(r['fallback'])
        self.assertEqual(r['events'][0]['error'], 'TimeoutError')

    def test_expected_subquery_failure_charged_no_retry(self):
        def fail(*_): raise TimeoutError('hard deadline')
        r = qe.run(Q, hits(), resolve, fail)
        self.assertTrue(r['fallback'])
        self.assertEqual(r['accounting']['requested_slots'], 750)
        self.assertEqual(r['accounting']['retrieval_calls'], 2)

    def test_total_deadline_prevents_subquery(self):
        times = iter([0, .5, 121, 121])
        r = qe.run(Q, hits(), resolve, lambda *a: self.fail('fetch'), clock=lambda: next(times))
        self.assertTrue(r['fallback'])
        self.assertEqual(r['accounting']['retrieval_calls'], 1)

    def test_late_response_discarded(self):
        times = iter([0, 0, 0, 0, 61, 61, 61])
        r = qe.run(Q, hits(), resolve, lambda *a: hits(500, 1000), clock=lambda: next(times))
        self.assertTrue(r['fallback'])
        self.assertFalse(r['expanded'])

    def test_identity_and_internal_error_propagate(self):
        for exc in (ValueError('identity'), RuntimeError('index'), KeyError('schema')):
            def fail(*_, exc=exc): raise exc
            with self.assertRaises(type(exc)): qe.run(Q, hits(), resolve, fail)

    def test_incomplete_original_and_overbudget_response_fail(self):
        with self.assertRaises(ValueError): qe.run(Q, hits(249), resolve, None)
        with self.assertRaises(ValueError): qe.run(Q, hits(), resolve, lambda *a: hits(501))

    def test_bad_rank_duplicate_or_nonfinite_fail(self):
        for key, value in (('rank', 2), ('score', float('nan')), ('id', '1')):
            h = hits(); h[0][key] = value
            with self.assertRaises(ValueError): qe.papers(h, resolve, 'o', Q)

    def test_stable_ties_and_no_input_mutation(self):
        h = hits(); b = hits(500, 1000)
        saved = copy.deepcopy([h, b])
        one = qe.run(Q, h, resolve, lambda *a: b)
        two = qe.run(Q, h, resolve, lambda *a: b)
        self.assertEqual(one['union'], two['union'])
        self.assertEqual([h, b], saved)

    def test_zero_subqueries_has_zero_generation_calls(self):
        r = qe.run(Q, hits(), resolve, None, limits=replace(qe.Limits(), max_subqueries=0))
        self.assertEqual(r['accounting']['generation_calls'], 0)

    def test_invalid_limits_fail(self):
        for kw in (dict(max_subqueries=2), dict(total_slots=500), dict(total_seconds=0), dict(retrieval_seconds=float('inf'))):
            with self.assertRaises(ValueError): qe.Limits(**kw)

    def test_identity_schema_mismatch_fails(self):
        with patch('citation_graph.registry', side_effect=ValueError('C1 v2 required')):
            with self.assertRaises(ValueError): qe.Identity('synthetic')

    def test_identity_aliases_agree_or_fail(self):
        data = ({'p': {}}, [dict(source='a.md',paper_id='p'),dict(source='alias.md',paper_id='p')],
                [dict(value='10/x',paper_id='p',scheme='doi',status='active')])
        with patch('citation_graph.registry', return_value=data): identity = qe.Identity('synthetic')
        self.assertEqual(identity(dict(source='alias.md',doi='10/X')), 'p')
        for h in (dict(source='absent.md'), dict(source='a.md',doi='10/wrong')):
            with self.assertRaises(ValueError): identity(h)

    def test_differences_do_not_call_unjudged_irrelevant(self):
        d = qe.differences(['positive','low','old'], ['positive','new','low'], {'positive':3,'low':1})
        self.assertEqual(d['retained']['positive'], ['positive'])
        self.assertEqual(d['retained']['other_judged'], ['low'])
        self.assertEqual(d['added']['unjudged'], ['new'])
        self.assertEqual(d['lost']['unjudged'], ['old'])

    def test_metric_denominator_and_nulls(self):
        ev = evaluator()
        rows = [dict(query_id='a',query='a',judgements=[dict(doi='10/a',relevance=3)]),
                dict(query_id='b',query='b',judgements=[dict(doi='10/b',relevance=1)])]
        result = ev['evaluate'](rows, [5,10,20,50], retrieval_cache={'a':[dict(doi='10/a',source='a')], 'b':[]})
        self.assertEqual(result['evaluable_queries'], 1)
        self.assertEqual(result['excluded_no_positive_queries'], 1)
        self.assertEqual(result['metrics']['recall']['50'], 1)
        self.assertIsNone(result['per_query'][1]['recall']['50'])


if __name__ == '__main__':
    unittest.main()
