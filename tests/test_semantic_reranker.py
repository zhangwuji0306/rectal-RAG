"""B6 synthetic tests: no corpus, DB, network or real model."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'index'))
import semantic_reranker as rr


def candidates(n=8):
    return [dict(id=f'p{i}', source=f's{i}.md', rank=i+1, score=1-i/1000,
                 doi='', text=f'full synthetic passage {i}', text_kind='stored-passage')
            for i in range(n)]


def run(hits, values=None, **kwargs):
    scores = iter(values if values is not None else range(len(hits)))
    scorer = lambda q, texts: [next(scores) for text in texts]
    return rr.experiment('synthetic query', hits, enabled=True,
                         scorer_factory=lambda: scorer, **kwargs)


class Tokenizer:
    cls_token_id = 0
    sep_token_id = 2
    def encode(self, text, **kw):
        return list(range(len(text.split())))[:kw['max_length']]
    def num_special_tokens_to_add(self, pair):
        return 4


class RerankerTests(unittest.TestCase):
    def test_fixed_contract(self):
        self.assertEqual((rr.FIRST_STAGE, rr.POOL_SIZE, rr.OUTPUT_PAPERS), (250, 200, 50))
        self.assertEqual((rr.BATCH_SIZE, rr.MAX_LENGTH, rr.QUERY_TOKENS), (4, 512, 128))
        self.assertEqual(rr.DEV_GATES, dict(ndcg10_gain=.01, mrr10_gain=.01, recall20_drop=.01))

    def test_disabled_never_loads_or_requires_text(self):
        hits = candidates()
        for hit in hits:
            del hit['text']
        factory = Mock(side_effect=AssertionError('must not load'))
        result = rr.experiment('q', hits, scorer_factory=factory)
        self.assertEqual(result['status'], 'disabled')
        self.assertEqual([p['source'] for p in result['papers']], [h['source'] for h in hits])
        factory.assert_not_called()

    def test_default_retrieval_has_no_reranker_import(self):
        root = Path(__file__).resolve().parents[1]/'index'
        for name in ('rag_core.py', 'retrieve.py', 'document_index.py'):
            self.assertNotIn('semantic_reranker', (root/name).read_text(encoding='utf-8'))

    def test_pool_and_both_controls(self):
        hits = candidates(250)
        result = run(hits)
        self.assertEqual(result['pool_ids'], [h['id'] for h in hits[:200]])
        self.assertEqual(result['dropped_ids'], [h['id'] for h in hits[200:]])
        self.assertEqual(len(set(result['passage_order'])), 200)
        self.assertEqual(set(result['pool_ids']), set(result['passage_order']))
        self.assertEqual(len(result['papers']), 50)
        self.assertEqual(result['pool_control'][0]['representative_id'], 'p0')
        self.assertEqual(result['d250_control'][0]['representative_id'], 'p0')

    def test_no_mutation(self):
        hits = candidates()
        before = copy.deepcopy(hits)
        run(hits)
        self.assertEqual(hits, before)

    def test_aggregation_and_paper_ties(self):
        hits = candidates(4)
        hits[0]['doi'] = ' DOI '
        hits[2]['doi'] = 'doi'
        result = run(hits, [1, 5, 5, 2])
        paper = result['papers'][0]
        self.assertEqual(paper['key'], 'doi')
        self.assertEqual(paper['representative_id'], 'p2')
        self.assertEqual(paper['member_ids'], ['p0', 'p2'])
        self.assertEqual(len(result['papers']), 3)

    def test_tied_representative_and_stability(self):
        hits = candidates()
        hits[1]['source'] = hits[0]['source']
        a = run(hits, [3]*8)
        self.assertEqual(a, run(hits, [3]*8))
        self.assertEqual(a['passage_order'], [h['id'] for h in hits])
        self.assertEqual(a['papers'][0]['representative_id'], 'p0')
        self.assertEqual(a['papers'][0]['member_ids'], ['p0', 'p1'])

    def test_empty_does_not_load(self):
        factory = Mock(side_effect=AssertionError())
        result = rr.experiment('q', [], enabled=True, scorer_factory=factory)
        self.assertEqual(result['status'], 'empty')
        self.assertEqual(result['papers'], [])
        factory.assert_not_called()

    def test_single_is_scored(self):
        result = run(candidates(1), [9])
        self.assertEqual(result['papers'][0]['score'], 9)

    def test_short_pool_not_padded(self):
        result = run(candidates(3))
        self.assertTrue(result['target_unmet'])
        self.assertEqual(len(result['pool_ids']), 3)

    def test_batching(self):
        sizes = []
        def scorer(query, texts):
            sizes.append(len(texts))
            return [0]*len(texts)
        rr.experiment('q', candidates(9), enabled=True, scorer_factory=lambda: scorer)
        self.assertEqual(sizes, [4, 4, 1])

    def test_truncation_includes_pair_specials(self):
        features = rr.encode_pairs(Tokenizer(), 'q '*1000, ['p '*2000, 'short'])
        self.assertEqual(len(features[0]['input_ids']), 512)
        self.assertEqual(len(features[1]['input_ids']), 133)
        self.assertEqual(features[0]['input_ids'][1:129], list(range(128)))
        self.assertEqual(features[0]['input_ids'][131:-1], list(range(380)))

    def test_bad_identity_budget_order(self):
        bads = [candidates(251)]
        a = candidates(); a[1]['id'] = a[0]['id']; bads.append(a)
        a = candidates(); a[1]['rank'] = 9; bads.append(a)
        a = candidates(); a[1]['score'] = float('nan'); bads.append(a)
        a = candidates(); a[1]['score'] = 10; bads.append(a)
        for hits in bads:
            with self.subTest(hits=len(hits)), self.assertRaises(ValueError):
                run(hits, on_error='original')

    def test_no_preview_or_card(self):
        for patch in ({'text': ''}, {'text_kind': 'snippet'}, {'id': 'card:s'},
                      {'section': 'document-card'}, {'text_kind': None}):
            hits = candidates(); hits[0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                run(hits, on_error='original')

    def test_bad_query(self):
        with self.assertRaises(ValueError):
            rr.experiment('', candidates(), enabled=True)

    def test_score_anomalies_atomic_fallback(self):
        for value in (float('nan'), float('inf'), -float('inf'), 'bad'):
            result = run(candidates(), [4, 3, 2, 1, value, 8, 9, 10], on_error='original')
            self.assertEqual(result['status'], 'fallback')
            self.assertEqual(result['papers'], result['pool_control'])
            self.assertEqual(result['passage_order'], result['pool_ids'])
            self.assertNotIn('passage_scores', result)

    def test_load_error_and_raise(self):
        def factory():
            raise RuntimeError('synthetic loading error')
        result = rr.experiment('q', candidates(), enabled=True,
                               scorer_factory=factory, on_error='original')
        self.assertEqual(result['status'], 'fallback')
        with self.assertRaises(RuntimeError):
            rr.experiment('q', candidates(), enabled=True, scorer_factory=factory)

    def test_wrong_count(self):
        with self.assertRaises(ValueError):
            rr.experiment('q', candidates(), enabled=True,
                          scorer_factory=lambda: lambda q, t: [])

    def test_partial_batch_error_falls_back_all(self):
        calls = []
        def score(q, texts):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('synthetic batch error')
            return [4, 3, 2, 1]
        result = rr.experiment('q', candidates(), enabled=True,
                               scorer_factory=lambda: score, on_error='original')
        self.assertEqual(result['papers'], result['pool_control'])
        self.assertEqual(result['passage_order'], result['pool_ids'])

    def test_timeout_discards_successful_batch(self):
        times = iter([0, 0, rr.SECONDS_PER_QUERY+1])
        result = run(candidates(), on_error='original', clock=lambda: next(times))
        self.assertEqual(result['status'], 'fallback')
        self.assertIn('TimeoutError', result['error'])
        self.assertEqual(result['papers'], result['pool_control'])

    def test_cli_disabled_and_explicit_missing_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = root/'input.json'; output = root/'out.json'
            payload.write_text(json.dumps(dict(query='q', d250=candidates())), encoding='utf-8')
            cmd = [sys.executable, str(Path(rr.__file__)), '--input', str(payload), '--output', str(output)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(json.loads(output.read_text())['status'], 'disabled')
            proc = subprocess.run(cmd+['--enable-reranker'], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('requires --model-path', proc.stderr)


if __name__ == '__main__':
    unittest.main()
