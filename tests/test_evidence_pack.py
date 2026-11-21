"""D1 behavioral tests from two original synthetic papers, never private data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'index'))
import evidence_pack as ep
import paper_registry as pr
from citation_graph import registry


def fixture():
    articles = json.loads((Path(__file__).parent / 'fixtures/d1_articles.json').read_text(encoding='utf-8'))
    papers, artifacts, identifiers, sources, hits, bindings = {}, [], [], {}, [], []
    for i, article in enumerate(articles.values(), 1):
        pid = f'PAP{i:06d}'
        papers[pid] = dict(paper_id=pid, title=article['title'], identity_status='verified')
        raw = ''.join(article['units']).encode('utf-8')
        sha = hashlib.sha256(raw).hexdigest()
        aid = pr.artifact_id_for(article['source'], sha)
        artifacts.append(dict(artifact_id=aid, paper_id=pid, source=article['source'],
                              content_sha256=sha, role='main'))
        sources[aid] = raw
        for scheme in ('doi', 'pmid'):
            if article[scheme]:
                identifiers.append(dict(paper_id=pid, scheme=scheme, value=article[scheme], status='active'))
        start = 0
        for chunk, passage in enumerate(article['units']):
            table = dict(kind='text')
            if i == 1 and chunk == 1:
                table = dict(kind='table', title='Table 1. Synthetic sizes', columns=['Group', 'Size (mm)', 'n'],
                             units=['mm'], context='untreated patients only; exploratory comparison, no causal inference.')
            if i == 2 and chunk == 1:
                table = dict(kind='table', title=None, columns=['Outcome', 'Value'], units=None, context=None)
            hits.append(dict(rank=len(hits)+1, score=round(.92-len(hits)*.03, 4), source=article['source'],
                             paper_id='source:legacy', doi=article['doi'], section=passage.splitlines()[0],
                             chunk=chunk, embedding_text_version='passage-v2', snippet=passage[:400]))
            bindings.append(dict(artifact_id=aid, content_sha256=sha, representation_version='passage-v2',
                                 start=start, end=start+len(passage), passage=passage, table=table))
            start += len(passage)
    observation = dict(level='paper', entity='PAP000001', source='openalex', metric='cited_by_count', year=0,
                       payload=dict(value=7, state='available', reason='', retrieved_at='2026-11-08T08:00:00Z',
                                    source_updated_at=None, rule_version='c2-v1'), last_valid=None)
    return papers, artifacts, identifiers, sources, hits, bindings, [observation]


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.papers, self.artifacts, self.ids, self.sources, self.hits, self.bindings, self.obs = fixture()

    def snapshot(self):
        return ep.Snapshot(self.papers, self.artifacts, self.ids, self.obs)

    def build(self, **kwargs):
        return ep.build(self.hits, self.bindings, self.snapshot(), self.sources,
                        max_chars=kwargs.get('max_chars', 10000), score_semantics=kwargs.get('score_semantics', ep.SCORE))

    def units(self, pack):
        return [e for p in pack['papers'] for e in p['evidence']]

    def test_deterministic_roundtrip_and_reversible_originals(self):
        pack = self.build()
        self.assertEqual(ep.dumps(pack), ep.dumps(self.build()))
        decoded = ep.loads(ep.dumps(pack))
        self.assertEqual(pack, decoded)
        self.assertEqual(ep.verify(decoded, self.snapshot(), self.sources),
                         dict(status='verified', evidence=5, papers=2))
        for unit in self.units(decoded):
            raw = self.sources[unit['artifact_id']].decode('utf-8')
            self.assertEqual(raw[unit['start']:unit['end']], unit['passage'])
            self.assertIn(unit['artifact_id'], {a['artifact_id'] for a in self.artifacts if a['paper_id'] == unit['paper_id']})

    def test_input_immutability_ranking_and_raw_scores(self):
        before = deepcopy((self.hits, self.bindings, self.papers, self.artifacts, self.ids, self.obs, self.sources))
        pack = self.build()
        origins = sorted((o for e in self.units(pack) for o in e['origins']), key=lambda o:o['rank'])
        self.assertEqual([o['semantic_score'] for o in origins], [h['score'] for h in self.hits])
        self.assertTrue(all(o['score_semantics'] == ep.SCORE for o in origins))
        self.assertEqual(before, (self.hits, self.bindings, self.papers, self.artifacts, self.ids, self.obs, self.sources))
        pack['papers'][0]['quality_metadata']['observations'][0]['payload']['value'] = 999
        self.assertEqual(self.obs[0]['payload']['value'], 7)

    def test_same_paper_multiple_passages_and_duplicate_origins(self):
        self.hits.append(dict(self.hits[0], rank=6, score=.12))
        self.bindings.append(deepcopy(self.bindings[0]))
        pack = self.build()
        self.assertEqual([len(p['evidence']) for p in pack['papers']], [3, 2])
        first = pack['papers'][0]['evidence'][0]
        self.assertEqual([o['rank'] for o in first['origins']], [1, 6])
        self.assertEqual([o['semantic_score'] for o in first['origins']], [.92, .12])
        self.assertEqual(pack['budget']['used'], sum(len(e['passage']) for e in self.units(pack)))

    def test_confirmed_registry_alias_groups_but_does_not_drop_passages(self):
        alias = dict(self.artifacts[0], source='alpha-alias.md', role='duplicate')
        alias['artifact_id'] = pr.artifact_id_for(alias['source'], alias['content_sha256'])
        self.artifacts.append(alias)
        self.sources[alias['artifact_id']] = self.sources[self.artifacts[0]['artifact_id']]
        self.hits.append(dict(self.hits[0], source=alias['source'], rank=6))
        self.bindings.append(dict(self.bindings[0], artifact_id=alias['artifact_id']))
        pack = self.build()
        self.assertEqual(len(pack['papers']), 2)
        self.assertEqual(len(pack['papers'][0]['evidence']), 4)
        self.assertEqual(pack['papers'][0]['evidence'][-1]['artifact_role'], 'duplicate')

    def test_titles_missing_doi_and_pending_relationship_do_not_merge(self):
        self.papers['PAP000002']['pending_duplicate_of'] = 'PAP000001'
        pack = self.build()
        self.assertEqual(len(pack['papers']), 2)
        self.assertEqual(pack['papers'][1]['doi'], [])
        self.assertEqual(pack['papers'][1]['pmid'], ['202'])

    def test_conflicted_owner_and_hit_identifier_conflict_rejected(self):
        self.papers['PAP000001']['identity_status'] = 'conflict'
        with self.assertRaisesRegex(ValueError, 'unresolved'): self.build()
        self.papers['PAP000001']['identity_status'] = 'verified'
        self.hits[0]['pmid'] = '202'
        with self.assertRaisesRegex(ValueError, 'conflict'): self.build()

    def test_source_ambiguity_is_not_automatically_adjudicated(self):
        self.artifacts.append(dict(self.artifacts[0], artifact_id='different', paper_id='PAP000002'))
        with self.assertRaisesRegex(ValueError, 'ambiguous'): self.build()

    def test_version_update_changes_id_and_old_pack_cannot_validate(self):
        old = self.build()
        aid = self.artifacts[0]['artifact_id']
        self.sources[aid] += b'\nNew version.'
        with self.assertRaisesRegex(ValueError, 'version mismatch'):
            ep.verify(old, self.snapshot(), self.sources)
        sha = hashlib.sha256(self.sources[aid]).hexdigest()
        self.artifacts[0]['content_sha256'] = sha
        for b in self.bindings[:3]: b['content_sha256'] = sha
        updated = self.build()
        self.assertNotEqual(self.units(old)[0]['evidence_id'], self.units(updated)[0]['evidence_id'])
        self.assertEqual(self.units(old)[3]['evidence_id'], self.units(updated)[3]['evidence_id'])
        with self.assertRaisesRegex(ValueError, 'version mismatch'):
            ep.verify(old, self.snapshot(), self.sources)

    def test_tampered_text_location_identity_and_metadata_fail_verification(self):
        pack = self.build()
        for key, value in [('passage', 'forged'), ('start', 1), ('evidence_id', 'old'), ('paper_id', 'PAP000002')]:
            with self.subTest(key=key):
                bad = deepcopy(pack)
                bad['papers'][0]['evidence'][0][key] = value
                with self.assertRaises(ValueError): ep.verify(bad, self.snapshot(), self.sources)
        bad = deepcopy(pack)
        bad['papers'][0]['quality_metadata']['observations'][0]['payload']['value'] = 8
        with self.assertRaisesRegex(ValueError, 'snapshot'): ep.verify(bad, self.snapshot(), self.sources)

    def test_table_context_preserved_and_missing_explicit(self):
        pack = self.build()
        table = self.units(pack)[1]
        self.assertEqual(table['table']['missing'], [])
        for token in ('Table 1.', 'Size (mm)', 'untreated patients only', 'no causal inference'):
            self.assertIn(token, table['passage'])
        incomplete = self.units(pack)[4]
        self.assertEqual(incomplete['table']['missing'], ['title', 'units', 'context'])
        self.assertIsNone(incomplete['table']['title'])
        self.bindings[1]['table']['context'] = 'invented patients'
        with self.assertRaisesRegex(ValueError, 'outside'): self.build()

    def test_budget_exact_boundary_overflow_and_no_lost_qualifiers(self):
        size = len(self.bindings[0]['passage'])
        exact = self.build(max_chars=size)
        first = self.units(exact)[0]
        self.assertEqual(first['passage'], self.bindings[0]['passage'])
        self.assertIn('does not establish causality', first['passage'])
        self.assertFalse(first['truncated'])
        small = self.build(max_chars=size-1)
        self.assertNotIn(first['evidence_id'], [e['evidence_id'] for e in self.units(small)])
        omission = small['budget']['omitted'][0]
        self.assertEqual(omission['evidence_id'], first['evidence_id'])
        self.assertEqual(omission['required_chars'], size)
        self.assertTrue(small['budget']['over_budget'])
        self.assertLessEqual(small['budget']['used'], size-1)

    def test_zero_budget_empty_results_and_unknown_table_context(self):
        pack = self.build(max_chars=0)
        self.assertEqual(pack['papers'], [])
        self.assertEqual(len(pack['budget']['omitted']), 5)
        empty = ep.build([], [], self.snapshot(), {}, max_chars=0, score_semantics=ep.SCORE)
        self.assertFalse(empty['budget']['over_budget'])
        self.assertEqual(ep.verify(empty, self.snapshot(), {} )['evidence'], 0)
        self.bindings[0]['table'] = dict(kind='unknown')
        self.assertEqual(self.units(self.build())[0]['table']['missing'], ['kind_and_context'])

    def test_missing_metadata_states_are_not_fabricated(self):
        for state in ('missing', 'failed', 'not_queried'):
            with self.subTest(state=state):
                self.obs[0]['payload'].update(state=state, value=None, reason='synthetic unavailable')
                pack = self.build()
                self.assertEqual(pack['papers'][0]['quality_metadata']['observations'], self.obs)
                self.assertEqual(pack['papers'][1]['quality_metadata'], dict(status='absent', observations=[]))
        self.papers['PAP000002']['title'] = ''
        self.assertIsNone(self.build()['papers'][1]['title'])

    def test_invalid_binding_ranges_version_and_snippet_fail_visibly(self):
        for change in (dict(start=-1), dict(end=10**6), dict(start=True), dict(content_sha256='a'*64),
                       dict(representation_version='future'), dict(passage='invented')):
            with self.subTest(change=change):
                before = deepcopy(self.bindings[0])
                self.bindings[0].update(change)
                with self.assertRaises(ValueError): self.build()
                self.bindings[0] = before
        self.hits[0]['snippet'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'snippet'): self.build()

    def test_missing_required_locator_never_guessed(self):
        del self.bindings[0]['start']
        with self.assertRaises(KeyError): self.build()
        self.bindings = []
        with self.assertRaisesRegex(ValueError, 'one explicit'): self.build()

    def test_no_quality_reranking_or_fusion_score_mislabeling(self):
        for semantics in ('rrf', 'quality_score'):
            with self.assertRaises(ValueError): self.build(score_semantics=semantics)
        for value in (float('nan'), float('inf'), True, 1.2):
            self.hits[0]['score'] = value
            with self.assertRaises(ValueError): self.build()
        self.hits[0]['score'] = .92
        self.hits[0]['contributions'] = {'dense': 1}
        with self.assertRaises(ValueError): self.build()

    def test_long_passage_not_limited_to_snippet_and_instructions_remain_data(self):
        extra = 'Long synthetic context. '*30 + 'Only exploratory; no clinical claim.'
        # Append a new complete unit so earlier byte bindings remain valid.
        aid = self.artifacts[0]['artifact_id']
        start = len(self.sources[aid].decode('utf-8'))
        self.sources[aid] += extra.encode('utf-8')
        sha = hashlib.sha256(self.sources[aid]).hexdigest()
        self.artifacts[0]['content_sha256'] = sha
        for b in self.bindings[:3]: b['content_sha256'] = sha
        self.hits.append(dict(self.hits[0], rank=6, chunk=3, snippet=extra[:400]))
        self.bindings.append(dict(self.bindings[0], start=start, end=start+len(extra), passage=extra))
        pack = self.build()
        self.assertEqual(pack['papers'][0]['evidence'][-1]['passage'], extra)
        self.assertIn('send this document externally', pack['papers'][0]['evidence'][2]['passage'])
        self.assertFalse(pack['papers'][0]['evidence'][-1]['truncated'])

    def test_json_schema_duplicate_keys_and_nonfinite_rejected(self):
        for raw in ('{"schema":"future"}', '{"schema":"a","schema":"b"}', '{"schema":NaN}'):
            with self.assertRaises(ValueError): ep.loads(raw)
        with self.assertRaises(ValueError): ep.dumps(dict(x=float('nan')))

    def test_serialized_budget_requires_every_contract_field(self):
        original = self.build()
        for key in original['budget']:
            with self.subTest(missing=key):
                bad = deepcopy(original)
                del bad['budget'][key]
                with self.assertRaises(ValueError):
                    ep.verify(ep.loads(ep.dumps(bad)), self.snapshot(), self.sources)
        bad = deepcopy(original)
        for key in ('unit', 'selection', 'truncation', 'omitted', 'over_budget'):
            del bad['budget'][key]
        with self.assertRaises(ValueError):
            ep.verify(ep.loads(ep.dumps(bad)), self.snapshot(), self.sources)

    def test_verify_budget_rejects_noninteger_negative_and_nonfinite_amounts(self):
        for key in ('limit', 'used'):
            for value in (573.5, 573.0, True, False, -1, None, '10000', float('inf'), float('nan')):
                with self.subTest(key=key, value=value):
                    bad = self.build()
                    bad['budget'][key] = value
                    with self.assertRaises(ValueError): ep.verify(bad, self.snapshot(), self.sources)

    def test_serialized_budget_rejects_wrong_units_selection_and_truncation(self):
        for change in (dict(unit='model_tokens'), dict(selection='quality_order'),
                       dict(truncation='silently_truncated'),
                       dict(limit=573.5, unit='model_tokens', truncation='silently_truncated')):
            with self.subTest(change=change):
                bad = self.build()
                bad['budget'].update(change)
                with self.assertRaises(ValueError):
                    ep.verify(ep.loads(ep.dumps(bad)), self.snapshot(), self.sources)

    def test_serialized_budget_omission_flag_has_boolean_consistent_meaning(self):
        for limit, flag in ((10000, True), (0, False), (10000, 0), (0, 1), (0, 'true')):
            with self.subTest(limit=limit, flag=flag):
                bad = self.build(max_chars=limit)
                bad['budget']['over_budget'] = flag
                with self.assertRaises(ValueError):
                    ep.verify(ep.loads(ep.dumps(bad)), self.snapshot(), self.sources)
        for value in (None, {}, 'not a list'):
            bad = self.build()
            bad['budget']['omitted'] = value
            with self.assertRaises(ValueError): ep.verify(bad, self.snapshot(), self.sources)

    def test_serialized_budget_omitted_records_require_structure_and_cost(self):
        original = self.build(max_chars=0)
        variants = [None, [], {}, {'evidence_id': 'EVmissing'}]
        entry = original['budget']['omitted'][0]
        for key in entry:
            missing = deepcopy(entry)
            del missing[key]
            variants.append(missing)
        for change in (dict(required_chars=True), dict(required_chars=0), dict(required_chars=-1),
                       dict(required_chars=1.5), dict(reason='quality_filter'), dict(paper_id=None),
                       dict(evidence_id=''), dict(origins=[]), dict(origins=[None]), dict(origins=[{}])):
            variants.append(dict(entry, **change))
        for variant in variants:
            with self.subTest(variant=variant):
                bad = deepcopy(original)
                bad['budget']['omitted'][0] = variant
                with self.assertRaises(ValueError):
                    ep.verify(ep.loads(ep.dumps(bad)), self.snapshot(), self.sources)

    def test_budget_rejects_duplicate_selected_or_unnecessary_omissions(self):
        pack = self.build(max_chars=120)
        for change in ('duplicate', 'selected', 'fits_remaining'):
            with self.subTest(change=change):
                bad = deepcopy(pack)
                if change == 'duplicate':
                    bad['budget']['omitted'].append(deepcopy(bad['budget']['omitted'][0]))
                elif change == 'selected':
                    bad['budget']['omitted'][0]['evidence_id'] = self.units(bad)[0]['evidence_id']
                else:
                    bad['budget']['limit'] += 1
                    bad['budget']['omitted'][0]['required_chars'] = 1
                with self.assertRaises(ValueError):
                    ep.verify(ep.loads(ep.dumps(bad)), self.snapshot(), self.sources)

    def test_budget_actual_passage_sum_still_checked(self):
        for change in (dict(used=572), dict(limit=572)):
            bad = self.build()
            bad['budget'].update(change)
            with self.assertRaisesRegex(ValueError, 'budget mismatch'):
                ep.verify(ep.loads(ep.dumps(bad)), self.snapshot(), self.sources)

    def test_json_rejects_exponent_overflow_in_budget_or_nested_metadata(self):
        original = ep.dumps(self.build())
        for exponent in ('1e999', '-1e999', 'NaN', 'Infinity', '-Infinity'):
            with self.subTest(exponent=exponent):
                wire = original.replace('"limit":10000', '"limit":' + exponent)
                with self.assertRaisesRegex(ValueError, 'non-finite JSON number'): ep.loads(wire)
        with self.assertRaisesRegex(ValueError, 'non-finite JSON number'):
            ep.loads('{"schema":"' + ep.SCHEMA + '","nested":{"value":1e999}}')
        self.assertEqual(ep.loads('{"schema":"' + ep.SCHEMA + '","finite":1e3}')['finite'], 1000.)

    def test_budget_roundtrips_empty_zero_and_partially_selected_packs(self):
        packs = [self.build(), self.build(max_chars=0), self.build(max_chars=120),
                 ep.build([], [], self.snapshot(), {}, max_chars=0, score_semantics=ep.SCORE)]
        for pack in packs:
            with self.subTest(budget=pack['budget']):
                before = deepcopy(pack)
                result = ep.verify(ep.loads(ep.dumps(pack)), self.snapshot(), self.sources)
                self.assertEqual(result['evidence'], len(self.units(pack)))
                self.assertEqual(pack, before)

    def test_group_ownership_and_conflicting_location_labels_rejected(self):
        pack = self.build()
        pack['papers'][1]['evidence'].append(pack['papers'][0]['evidence'].pop())
        with self.assertRaisesRegex(ValueError, 'group owner'):
            ep.verify(pack, self.snapshot(), self.sources)
        self.hits.append(dict(self.hits[0], rank=6, section='Different label'))
        self.bindings.append(deepcopy(self.bindings[0]))
        with self.assertRaisesRegex(ValueError, 'conflicting metadata'): self.build()

    def test_invalid_budgets_and_input_rank_rejected(self):
        for limit in (-1, True, 1.2):
            with self.assertRaises(ValueError): self.build(max_chars=limit)
        self.hits[1]['rank'] = 8
        with self.assertRaises(ValueError): self.build()

    def test_unicode_offsets_use_codepoints_without_newline_normalization(self):
        text = '标题\r\n结果：仅限合成患者。😀\r\n'
        raw = text.encode('utf-8')
        sha = hashlib.sha256(raw).hexdigest()
        artifact = dict(self.artifacts[0], content_sha256=sha)
        binding = dict(self.bindings[0], content_sha256=sha, start=4, end=len(text), passage=text[4:])
        hit = dict(self.hits[0], snippet=text[4:])
        snapshot = ep.Snapshot(self.papers, [artifact], self.ids)
        pack = ep.build([hit], [binding], snapshot, {artifact['artifact_id']:raw}, max_chars=100, score_semantics=ep.SCORE)
        unit = self.units(pack)[0]
        self.assertEqual(unit['passage'], text[4:])
        self.assertEqual(pack['budget']['used'], len(text[4:]))
        self.assertEqual(ep.verify(pack, snapshot, {artifact['artifact_id']:raw})['evidence'], 1)

    def test_actual_c2_rows_preserve_current_failure_last_valid_and_journal(self):
        from quality_metadata import MetadataStore
        with tempfile.TemporaryDirectory() as tmp:
            store = MetadataStore(Path(tmp) / 'metadata.sqlite3')
            try:
                store.put('paper', 'PAP000001', 'cited_by_count', 7, state='available', retrieved_at='old')
                store.put('paper', 'PAP000001', 'cited_by_count', state='failed', reason='http_429', retrieved_at='new')
                store.put('paper', 'PAP000001', 'journal_source_id', 'S1', state='available', retrieved_at='new')
                store.put('journal', 'S1', 'annual_works_count', 8, state='available', year=2025, retrieved_at='new')
                store.put('journal', 'S2', 'annual_works_count', 999, state='available', year=2025, retrieved_at='new')
                self.obs = store.rows()
            finally:
                store.close()
        pack = self.build()
        rows = pack['papers'][0]['quality_metadata']['observations']
        self.assertEqual(len(rows), 3)
        failure = next(r for r in rows if r['metric'] == 'cited_by_count')
        self.assertEqual(failure['payload']['state'], 'failed')
        self.assertIsNone(failure['payload']['value'])
        self.assertEqual(failure['last_valid']['value'], 7)
        self.assertEqual(failure['payload']['retrieved_at'], 'new')
        self.assertEqual(failure['last_valid']['retrieved_at'], 'old')
        self.assertEqual(ep.verify(ep.loads(ep.dumps(pack)), self.snapshot(), self.sources)['papers'], 2)

    def test_real_c1_synthetic_registry_snapshot_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'registry.sqlite3'
            store = pr.PaperRegistry(path, create=True)
            try:
                for artifact in self.artifacts:
                    pid = artifact['paper_id']
                    store.register(dict(source=artifact['source'], content_sha256=artifact['content_sha256'],
                                        identifiers={i['scheme']:i['value'] for i in self.ids if i['paper_id'] == pid},
                                        metadata=dict(title=self.papers[pid]['title'])))
                alias = dict(self.artifacts[0], source='confirmed-alias.md')
                store.register(dict(source=alias['source'], content_sha256=alias['content_sha256'],
                                    identifiers={'doi': '10.1234/synthetic.alpha'},
                                    duplicate_of={'kind': 'source', 'value': 'alpha.md'}))
            finally:
                store.close()
            snapshot = ep.Snapshot(*registry(path))
            alias_artifact = next(a for a in snapshot.artifacts.values() if a['source'] == 'confirmed-alias.md')
            self.sources[alias_artifact['artifact_id']] = self.sources[self.artifacts[0]['artifact_id']]
            self.hits.append(dict(self.hits[0], rank=6, source='confirmed-alias.md'))
            self.bindings.append(dict(self.bindings[0], artifact_id=alias_artifact['artifact_id']))
            pack = ep.build(self.hits, self.bindings, snapshot, self.sources, max_chars=10000, score_semantics=ep.SCORE)
            self.assertEqual(ep.verify(pack, snapshot, self.sources)['papers'], 2)
            self.assertEqual(len(pack['papers'][0]['evidence']), 4)


if __name__ == '__main__':
    unittest.main()
