"""Synthetic B4 invariants; no model or real database access."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'index'))
import rag_core as core
import sparse_index as sparse
import evaluate_retrieval as evaluator

def hit(i, source=None, doi='', score=1):
    return dict(id=str(i), rank=1, score=score, source=source or str(i)+'.md', chunk=0,
                doi=doi, title='', year='', journal='', authors='', mac='', section='x')

class HybridTests(unittest.TestCase):
    def test_tokenization(self):
        self.assertEqual(sparse.tokenize('ＡBC 123,直肠癌! aBc'), ['abc','123','直','肠','癌','abc'])
        self.assertEqual(sparse.tokenize(' ！？'), [])

    def test_bm25_hand_zero_repeat_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'s'
            rows=[(str(i),t,{'source':str(i),'embedding_text_version':'passage-v2'})
                  for i,t in enumerate(('alpha alpha','beta beta','alpha alpha'))]
            binding={'db_dir':'synthetic','manifest_path':'synthetic'}
            sparse.build(rows,path,binding)
            with patch.object(sparse,'pair_binding',return_value=binding): idx=sparse.SparseIndex(path)
            try:
                a=idx.search('ALPHA alpha',250)
                self.assertEqual(a,idx.search('alpha',250))
                self.assertEqual([h['id'] for h in a],['0','2'])
                self.assertAlmostEqual(a[0]['score'],math.log(1+1.5/2.5)*2*2.2/3.2)
                self.assertEqual(idx.search('absentxyz',250),[])
                self.assertEqual(idx.search('',250),[])
            finally: idx.close()

    def test_rrf_once_tie_duplicate(self):
        result=core.fuse_rrf([hit('b'),hit('a')],[hit('a'),hit('b')])
        self.assertEqual([h['id'] for h in result],['a','b'])
        self.assertAlmostEqual(result[0]['score'],1/61+1/62)
        with self.assertRaises(ValueError): core.fuse_rrf([hit('a'),hit('a')],[])
        with self.assertRaises(ValueError): core.fuse_rrf([hit('a')],[hit('a',source='bad')])

    def test_aggregation_budget_truncation(self):
        dense=[hit(i,doi='same' if i<2 else '',score=1-i/1000) for i in range(125)]
        with patch.object(core,'retrieve_dense',return_value=dense) as d, patch.object(core,'retrieve_sparse',return_value=dense) as s:
            trace={}
            result=core.retrieve_literature_mode('q',mode='hybrid',sparse_path='synthetic',ranking_trace=trace)
            self.assertEqual(d.call_args.args[1],125); self.assertEqual(s.call_args.args[1],125)
            self.assertEqual(len(result),50)
            self.assertEqual(trace['passage_candidates_returned'],125)
            self.assertEqual(len(trace['aggregates']),124)
            self.assertEqual(trace['aggregates'][0]['n_hits'],2)
            self.assertEqual(trace['truncated_literature_count'],74)
        with self.assertRaises(ValueError): core.retrieve_literature_mode('q',mode='hybrid',dense_k=250,sparse_k=250)

    def test_missing_sparse_no_fallback(self):
        with patch.object(core,'retrieve_dense') as d:
            with self.assertRaises(ValueError): core.retrieve_literature_mode('q',mode='hybrid')
            d.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(Exception): sparse.SparseIndex(tmp)

    def test_pair_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'db').mkdir(); (p/'m').write_text('{}')
            (p/'db/.publish-identity.json').write_text('{"manifest_sha256":"wrong"}')
            (p/'m.ready.json').write_text('{"manifest_sha256":"wrong"}')
            with self.assertRaises(ValueError): sparse.pair_binding(p/'db',p/'m')

    def test_alias_no_extra_document(self):
        with tempfile.TemporaryDirectory() as tmp:
            sparse.build([('a','text',{'source':'canonical','embedding_text_version':'passage-v2'})],Path(tmp)/'s',{},expected_counts={'canonical':1})
            with self.assertRaises(ValueError):
                sparse.build([('a','x',{'source':'alias','embedding_text_version':'passage-v2'})],Path(tmp)/'bad',{},expected_counts={'canonical':1})

    def test_dense_default_compatible(self):
        hits=[hit('a'),hit('b',score=.8)]
        with patch.object(core,'retrieve',return_value=hits): old=core.retrieve_literature('q',k=2,candidate_k=250)
        with patch.object(core,'retrieve_dense',return_value=hits): self.assertEqual(old,core.retrieve_literature_mode('q',k=2))

    def test_dense_trace_does_not_change_legacy_payload(self):
        from unittest.mock import Mock
        collection=Mock()
        collection.count.return_value=300
        collection.query.return_value={'ids':[['actual-id']], 'documents':[['body']],
            'metadatas':[[{'source':'a.md','chunk':7,'doi':'10/a'}]],'distances':[[.123456]]}
        model=Mock(); model.encode.return_value=[[1.,0.]]
        with patch.object(core,'index_preflight'), patch.object(core,'_get_collection',return_value=collection), patch.object(core,'_get_model',return_value=model):
            legacy=core.retrieve('q',candidate_k=250)
            trace={}; traced=core.retrieve('q',candidate_k=250,ranking_trace=trace)
            self.assertEqual(legacy,traced)
            self.assertNotIn('id',legacy[0])
            self.assertEqual(trace['candidates'][0]['id'],'actual-id')
            self.assertEqual(trace['candidates'][0]['score'],.8765)
            self.assertEqual(trace['candidates'][0]['chunk'],7)
            core.retrieve_dense('q',125)
            self.assertEqual(collection.query.call_args.kwargs['n_results'],125)

    def test_mode_cache_and_identity_conflict(self):
        row={'query_id':'q','query':'same','judgements':[{'source_file':'a.md','doi':'10/a','relevance':3},{'source_file':'b.md','doi':'10/b','relevance':2}]}
        a=evaluator.evaluate([row],[5,10,20,50],retrieval_cache={'same':[hit('a',source='a.md',doi='10/a')]})
        b=evaluator.evaluate([row],[5,10,20,50],retrieval_cache={'same':[hit('b',source='a.md',doi='10/b')]})
        self.assertEqual(a['metrics']['recall']['5'],.5)
        self.assertEqual(b['metrics']['recall']['5'],0)

if __name__=='__main__': unittest.main()
