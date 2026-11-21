"""Original synthetic replay/mapping cases, with the actual chunker function."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import re
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'index'))
import passage_provenance as pp
import evidence_pack as ep
import d2_adapters as da
import answer_generation as ag


class Tokenizer:
    def __call__(self, text, **kwargs):
        return dict(input_ids=self.encode(text),
                    offset_mapping=[(i, i + 1) for i in range(len(text))])

    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, ids, **kwargs):
        return ''.join(ids)


def chunker():
    # Compile the unchanged function only; no DB/model imports in public tests.
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'index/build_index.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'chunk_markdown')
    ns = dict(re=re, CHUNK_MAX_TOKENS=800, CHUNK_OVERLAP_TOKENS=80, MIN_CHUNK_TOKENS=16)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-chunker>', 'exec'), ns)
    return ns['chunk_markdown']


def fixture():
    original = '# Results\r\nCount was seven.\r\n![excluded](x)\r\nFollow-up was one year.\r\n'
    raw = original.encode('utf-8'); digest = hashlib.sha256(raw).hexdigest()
    snap = ep.Snapshot({'p': dict(title='Synthetic', identity_status='verified')},
        [dict(artifact_id='a', paper_id='p', source='a.md', content_sha256=digest, role='main')], [])
    mapped = pp.replay(original, Tokenizer(), chunker())[0]
    hit = dict(id=digest[:12]+'-000', rank=1, score=.8, source='a.md', chunk=0,
               section='Results', embedding_text_version='passage-v2', snippet=mapped['text'][:400])
    record = dict(id=hit['id'], document=mapped['text'], metadata={k: hit[k] for k in (
        'source', 'chunk', 'section', 'embedding_text_version')})
    loc = dict(artifact_id='a', start=None, end=None, source_segments=mapped['source_segments'], table=dict(kind='text'))
    return snap, {'a': raw}, hit, record, loc


class ProvenanceTests(unittest.TestCase):
    def pack(self):
        s, b, h, r, l = fixture()
        return da.pack_from_records([h], [r], [l], s, b, max_chars=1000), s, b

    def test_crlf_skip_and_explicit_noncontinuous_location(self):
        pack, snap, sources = self.pack()
        self.assertEqual(pack['schema'], ep.SEGMENTED_SCHEMA)
        unit = pack['papers'][0]['evidence'][0]
        self.assertIsNone(unit['start']); self.assertIsNone(unit['end'])
        self.assertNotIn('excluded', unit['passage'])
        self.assertEqual(ep.segmented_text(unit['source_segments'], sources['a'].decode()), unit['passage'])
        self.assertEqual(ep.verify(ep.loads(ep.dumps(pack)), snap, sources)['evidence'], 1)

    def test_repeat_resolved_by_chunk_index_not_global_search(self):
        text = '# A\nSame long repeated sentence.\n# B\nSame long repeated sentence.\n'
        chunks = pp.replay(text, Tokenizer(), chunker())
        self.assertEqual(chunks[0]['text'], chunks[1]['text'])
        self.assertNotEqual(chunks[0]['source_segments'], chunks[1]['source_segments'])
        for c in chunks:
            self.assertEqual(ep.segmented_text(c['source_segments'], text), c['text'])

    def test_long_paragraph_sentence_join_and_overlap(self):
        text = '# A\n' + 'First sentence here.\nSecond sentence here. ' * 6 + '\n\nAnother paragraph here.\n'
        chunks = pp.replay(text, Tokenizer(), chunker(), max_tokens=40, overlap=8)
        self.assertGreater(len(chunks), 2)
        for c in chunks:
            self.assertEqual(c['status'], 'bound')
            self.assertEqual(ep.segmented_text(c['source_segments'], text), c['text'])

    def test_unlocatable_token_decode_is_explicit_failure(self):
        class Changed(Tokenizer):
            def decode(self, ids, **kwargs):
                return 'invented-character-\u2603'
        chunks = pp.replay('z'*90, Changed(), chunker(), max_tokens=30, overlap=5)
        self.assertTrue(all(c['status']=='unbound' for c in chunks))
        self.assertTrue(all(c['source_segments'] is None for c in chunks))

    def test_normalized_window_cannot_bind_to_adjacent_exact_match(self):
        class Normalizing(Tokenizer):
            def decode(self, ids, **kwargs):
                return super().decode(ids, **kwargs).replace('Ａ', 'A')

        text = 'Ａ' * 8 + 'A' * 8 + 'b' * 8
        first, second, _ = pp.replay(text, Normalizing(), chunker(), max_tokens=8, overlap=0)
        self.assertEqual(first['text'], second['text'])
        self.assertEqual(first['status'], 'unbound')
        self.assertIsNone(first['source_segments'])
        self.assertEqual(second['status'], 'bound')
        self.assertEqual(second['source_segments'], [dict(start=8, end=16)])

    def test_overlapping_windows_bind_to_their_own_original_codepoints(self):
        prefix = '# A\r\n'
        text = prefix + '🦀' * 40
        chunks = pp.replay(text, Tokenizer(), chunker(), max_tokens=16, overlap=4)
        self.assertEqual([c['status'] for c in chunks], ['bound'] * 3)
        self.assertEqual([c['source_segments'] for c in chunks], [
            [dict(start=5, end=21)], [dict(start=17, end=33)], [dict(start=29, end=45)],
        ])

    def test_unverifiable_offsets_reject_even_exact_window_text(self):
        class NoOffsets(Tokenizer):
            def __call__(self, text, **kwargs):
                raise NotImplementedError

        class MismatchedIds(Tokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                result['input_ids'][0] = '!'
                return result

        class InvalidOffsets(Tokenizer):
            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                result['offset_mapping'][0] = (0, 0)
                return result

        for tokenizer in (NoOffsets(), MismatchedIds(), InvalidOffsets()):
            with self.subTest(tokenizer=type(tokenizer).__name__):
                chunks = pp.replay('abcdefghijklmnopqrstuvwxyz', tokenizer, chunker(),
                                   max_tokens=10, overlap=2)
                self.assertEqual(chunks[0]['text'], 'abcdefghij')
                self.assertTrue(all(c['status'] == 'unbound' and c['source_segments'] is None
                                    for c in chunks))

    def test_reference_mismatch_fails_before_any_mapping_returned(self):
        with self.assertRaisesRegex(ValueError, 'differs'):
            pp.replay('a'*20, Tokenizer(), lambda *a, **kw: [])

    def test_fences_comments_unicode_and_original_offsets(self):
        text = '# A\r\n\u4e2d\U0001f980' + 'a'*20 + '\r\n```\r\nhidden\r\n```\r\n<!--hidden-->\r\n' + 'b'*20
        c = pp.replay(text, Tokenizer(), chunker())[0]
        self.assertEqual(ep.segmented_text(c['source_segments'], text), c['text'])
        self.assertEqual(c['source_segments'][0]['start'], 5)
        self.assertNotIn('hidden', c['text'])

    def test_tampered_ranges_order_separators_and_envelope_rejected(self):
        for change in ('range', 'order', 'separator', 'envelope'):
            pack, snap, sources = self.pack(); unit=pack['papers'][0]['evidence'][0]
            if change == 'range':unit['source_segments'][0]['start'] += 1
            elif change == 'order':unit['source_segments'].reverse()
            elif change == 'separator':unit['source_segments'][1] = dict(separator='forged conclusion')
            else:unit['start']=0; unit['end']=len(sources['a'].decode())
            with self.subTest(change=change), self.assertRaises(ValueError):ep.verify(pack,snap,sources)

    def test_v1_cannot_silently_accept_v2_mapping(self):
        pack, snap, sources = self.pack(); pack['schema']=ep.SCHEMA
        with self.assertRaisesRegex(ValueError,'requires v2'):ep.verify(pack,snap,sources)

    def test_wrong_artifact_version_and_full_row_rejected(self):
        for change in ('source', 'row', 'id'):
            snap, sources, hit, record, loc = fixture()
            if change=='source':sources['a']+=b'changed'
            elif change=='row':record['document']+='wrong'
            else:record['id']='wrong-000'
            with self.subTest(change=change), self.assertRaises(ValueError):
                da.pack_from_records([hit],[record],[loc],snap,sources,max_chars=1000)

    def test_d2_candidate_gate_and_citations_preserve_segments(self):
        pack, snap, sources = self.pack(); eid=pack['papers'][0]['evidence'][0]['evidence_id']
        candidate=dict(schema=ag.SCHEMA,claims=[dict(claim='Count was seven.',evidence_ids=[eid],confidence=.9)])
        result=ag.answer(pack,candidate,snapshot=snap,sources=sources)
        self.assertEqual(result['state'],'abstained')
        location=result['claims'][0]['citations'][0]['locations'][0]
        self.assertEqual(location['source_segments'],pack['papers'][0]['evidence'][0]['source_segments'])
        self.assertIsNone(location['start'])
        self.assertNotIn('verified_formal_claim', ag.render(result))

    def test_no_source_content_or_bool_offsets_rejected(self):
        for segments in ([dict(separator=' ')],[dict(start=False,end=5)], [dict(start=2,end=1)]):
            with self.assertRaises(ValueError):ep.segmented_text(segments,'123456')


if __name__ == '__main__':
    unittest.main()
