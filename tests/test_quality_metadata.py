"""C2 tests use only fabricated records, temporary databases and fake HTTP."""
import io
from contextlib import closing, redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'index'))
import quality_metadata as qm
import paper_registry as pr


def paper(pid='PAP000001', doi='10.1234/a', **ids):
    return {'paper_id': pid, 'identity_status': 'verified',
            'identifiers': dict({'doi': [doi]} if doi else {}, **{k: [v] for k, v in ids.items()})}


def work(doi='10.1234/a', count=0, **extra):
    return dict({'id': 'https://openalex.org/W1', 'doi': 'https://doi.org/' + doi,
                 'ids': {}, 'cited_by_count': count, 'publication_date': '2020-01-01',
                 'publication_year': 2020, 'type': 'article',
                 'citation_normalized_percentile': {'value': 0.2},
                 'primary_topic': {'subfield': {'id': 'subfields/1'}},
                 'topics': [{'subfield': {'id': 'subfields/1'}}, {'subfield': {'id': 'subfields/2'}}],
                 'primary_location': {'source': {'id': 'https://openalex.org/S1', 'issn_l': '1234-5678', 'type': 'journal'}}}, **extra)


def request(records, status='available', reason=''):
    return {'request_id': 1, 'retrieved_at': '2026-11-01T00:00:00+00:00',
            'status': status, 'reason': reason, 'payload': {'results': records}}


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = qm.MetadataStore(Path(self.tmp.name) / 'meta.sqlite3')
        self.addCleanup(self.store.close)

    def test_zero_and_all_other_states(self):
        for idx, state in enumerate(qm.STATES):
            self.store.put('paper', str(idx), 'cited_by_count', 0 if idx == 0 else None,
                           state, '' if idx == 0 else 'explicit')
        self.assertEqual([r['payload']['value'] for r in self.store.rows()], [0, None, None, None])
        with self.assertRaises(ValueError):
            self.store.put('paper', 'x', 'cited_by_count', 0, 'missing', 'x')

    def test_source_year_separation_and_idempotence(self):
        for source in ('a', 'b'):
            for year in (2021, 2022):
                for _ in range(2):
                    self.store.put('journal', 'S1', 'annual_works_count', 0, 'available', year=year, source=source)
        self.assertEqual(len(self.store.rows()), 4)

    def test_levels_rejected(self):
        with self.assertRaises(ValueError):
            self.store.put('journal', 'S1', 'cited_by_count', 1, 'available')
        with self.assertRaises(ValueError):
            self.store.put('paper', 'p1', 'annual_works_count', 1, 'available')

    def test_refresh_failure_keeps_previous_valid_and_current_failure(self):
        self.store.put('paper', 'p1', 'cited_by_count', 7, 'available', retrieved_at='old')
        self.store.put('paper', 'p1', 'cited_by_count', state='failed', reason='http_429', retrieved_at='new')
        row = self.store.rows()[0]
        self.assertEqual(row['payload']['state'], 'failed')
        self.assertIsNone(row['payload']['value'])
        self.assertEqual(row['last_valid']['value'], 7)
        self.assertEqual(row['last_valid']['retrieved_at'], 'old')

    def test_multidiscipline_is_preserved_without_aggregation(self):
        qm.write_paper(self.store, paper(), work(), 'available', 'exact_doi', request([]))
        percentile = next(r for r in self.store.rows() if r['metric'] == 'field_normalized_percentile')
        self.assertEqual(len(percentile['payload']['subjects']), 2)
        self.assertEqual(percentile['payload']['normalization']['primary_subfield']['id'], 'subfields/1')

    def test_missing_date_and_missing_normalization(self):
        qm.write_paper(self.store, paper(), work(publication_date=None, primary_topic=None), 'available', 'exact_doi', request([]))
        observations = {r['metric']: r['payload'] for r in self.store.rows()}
        self.assertEqual(observations['cumulative_annualized_citation_rate']['state'], 'missing')
        self.assertEqual(observations['field_normalized_percentile']['state'], 'missing')

    def test_journal_years_no_backfill_and_zero(self):
        rec = {'id': 'https://openalex.org/S1', 'issn_l': '1234-5678', 'type': 'journal',
               'counts_by_year': [{'year': 2021, 'works_count': 0, 'cited_by_count': 12}]}
        qm.write_journal(self.store, rec['id'], [{'issn_l': rec['issn_l']}], [rec], [2021, 2022], request([rec]))
        data = {(r['metric'], r['year']): r['payload'] for r in self.store.rows()}
        self.assertEqual(data['annual_works_count', 2021]['value'], 0)
        self.assertEqual(data['annual_works_count', 2022]['state'], 'missing')
        self.assertEqual(data['journal_field_normalized_percentile', 2021]['state'], 'missing')

    def test_journal_identifier_mismatch(self):
        rec = {'id': 'https://openalex.org/S1', 'issn_l': '9999-0000', 'type': 'journal'}
        qm.write_journal(self.store, rec['id'], [{'issn_l': '1234-5678'}], [rec], [2021], request([rec]))
        self.assertTrue(all(r['payload']['state'] == 'failed' for r in self.store.rows()))

    def test_request_resume_retry_and_refresh(self):
        class Fake:
            calls = 0
            def get(self, url):
                self.calls += 1
                return None, 'failed', 'network_error', [{'reason': 'network_error'}]
        fake = Fake()
        self.store.request('public', fake)
        self.store.request('public', fake)
        self.assertEqual(fake.calls, 1)
        self.store.request('public', fake, retry_failed=True)
        self.assertEqual(fake.calls, 2)
        self.store.request('public', fake, refresh=True)
        self.assertEqual(fake.calls, 3)

    def test_collect_covers_all_and_cached_rerun(self):
        class Fake:
            calls = 0
            def get(self, url):
                self.calls += 1
                records = [work()] if '/works?' in url else [{'id': 'https://openalex.org/S1', 'issn_l': '1234-5678', 'type': 'journal'}]
                return {'results': records}, 'available', '', [{'status': 200}]
        fake = Fake()
        papers = [paper(), paper('PAP000002', '10.1234/missing'), paper('PAP000003', None)]
        first = qm.collect(self.store, papers, [2021], fake)
        rows = self.store.rows()
        second = qm.collect(self.store, papers, [2021], fake)
        self.assertEqual(first, second)
        self.assertEqual(rows, self.store.rows())
        self.assertEqual(fake.calls, 2)
        self.assertEqual(first['paper_fields_total'], 12)
        self.assertEqual(sum(first['paper_projected_journal_status']['annual_works_count:2021'].values()), 3)

    def test_registry_readonly_and_identity_change_rejected(self):
        path = Path(self.tmp.name) / 'registry.sqlite3'
        registry = pr.PaperRegistry(path, create=True)
        registry.register({'source': 'synthetic.md', 'identifiers': {'doi': '10.1234/a'}, 'content_sha256': 'a' * 64})
        registry.close()
        before = path.read_bytes()
        papers = qm.read_registry(path)
        self.assertEqual(before, path.read_bytes())
        self.store.bind(papers, [2021])
        with self.assertRaises(ValueError):
            self.store.bind([paper('other')], [2021])


class MatchingTests(unittest.TestCase):
    def test_exact_normalized_doi(self):
        self.assertEqual(qm.match_work(paper(), [work()], {})[1], 'available')
        self.assertEqual(qm.match_work(paper(), [work('10.1234/A')], {})[1], 'available')

    def test_contradictory_external_doi_fields(self):
        self.assertEqual(qm.match_work(paper(), [work(ids={'doi': '10.1234/b'})], {})[1], 'failed')

    def test_unrelated_invalid_record_does_not_poison_exact_match(self):
        bad = work('10.1234/b', id='https://openalex.org/W2', ids={'doi': '10.1234/c'})
        records = [work(), bad]
        self.assertEqual(qm.match_work(paper(), records, {})[1:], ('available', 'exact_doi'))
        for doi in ('10.1234/b', '10.1234/c'):
            self.assertEqual(qm.match_work(paper(doi=doi), records, {})[1:],
                             ('failed', 'invalid_external_identifiers'))

    def test_related_invalid_candidate_cannot_be_discarded_for_valid_one(self):
        bad = work('10.1234/a', id='https://openalex.org/W2', ids={'doi': '10.1234/b'})
        for records in ([work(), bad], [bad, work()]):
            self.assertEqual(qm.match_work(paper(), records, {})[1:],
                             ('failed', 'invalid_external_identifiers'))

    def test_invalid_record_related_by_secondary_identifier_is_rejected(self):
        bad = work('10.1234/b', id='https://openalex.org/W2', ids={'doi': '10.1234/c', 'pmid': '1'})
        self.assertEqual(qm.match_work(paper(pmid='1'), [work(), bad], {})[1], 'failed')

    def test_review_reproducer_with_shared_fixture_record_id(self):
        bad = work('10.1234/b', ids={'doi': '10.1234/c'})
        self.assertEqual(qm.match_work(paper(), [work(), bad], {})[1:], ('available', 'exact_doi'))

    def test_unrelated_malformed_identifier_is_isolated(self):
        bad = work('10.1234/b', id='https://openalex.org/W2', ids={'pmid': 'invalid'})
        self.assertEqual(qm.match_work(paper(), [bad, work()], {})[1], 'available')
        self.assertEqual(qm.match_work(paper(doi='10.1234/b'), [bad, work()], {})[1], 'failed')

    def test_no_record(self):
        self.assertEqual(qm.match_work(paper(), [], {})[1], 'missing')

    def test_conflicting_pmid(self):
        self.assertEqual(qm.match_work(paper(pmid='1'), [work(ids={'pmid': 'https://pubmed.ncbi.nlm.nih.gov/2/'})], {})[1], 'failed')

    def test_foreign_owner(self):
        self.assertEqual(qm.match_work(paper(), [work(ids={'pmid': '1'})], {('pmid', '1'): 'PAP999999'})[1], 'failed')

    def test_ambiguous_external_records(self):
        self.assertEqual(qm.match_work(paper(), [work(), work()], {})[1], 'failed')

    def test_mismatched_doi_rejected(self):
        self.assertEqual(qm.match_work(paper(pmid='1'), [work('10.1234/b', ids={'pmid': '1'})], {})[1], 'failed')

    def test_no_identifier_or_identity_conflict_not_queried(self):
        p = paper()
        p['identity_status'] = 'conflict'
        self.assertEqual(qm.safe_reason(p), 'registry_identity_conflict')
        self.assertEqual(qm.safe_reason(paper(doi=None)), 'no_public_identifier')

    def test_pmid_and_pmcid_normalization(self):
        self.assertEqual(qm.record_ids({'ids': {'pmid': 'https://pubmed.ncbi.nlm.nih.gov/123/', 'pmcid': 'https://www.ncbi.nlm.nih.gov/pmc/articles/456/'}}), {'pmid': '123', 'pmcid': 'PMC456'})


class DatabaseAccessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def invoke(self, command, path, *extra):
        args = [command, '--db', str(path)]
        if command == 'export':
            args += ['--out', str(self.root / 'export.jsonl')]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return qm.main(args + list(extra))

    def assert_rejected_unchanged(self, path):
        before = path.read_bytes()
        with closing(sqlite3.connect(path)) as conn:
            schema = conn.execute('SELECT * FROM sqlite_master ORDER BY name').fetchall()
        for command in ('status', 'export', 'collect'):
            with self.subTest(command=command):
                extra = ['--registry', str(self.root / 'unused.sqlite3')] if command == 'collect' else []
                with self.assertRaises(ValueError):
                    self.invoke(command, path, *extra)
                self.assertEqual(path.read_bytes(), before)
                with closing(sqlite3.connect(path)) as conn:
                    self.assertEqual(conn.execute('SELECT * FROM sqlite_master ORDER BY name').fetchall(), schema)

    def test_c1_database_rejected_without_writes(self):
        path = self.root / 'registry.sqlite3'
        registry = pr.PaperRegistry(path, create=True)
        registry.close()
        self.assert_rejected_unchanged(path)

    def test_other_sqlite_rejected_without_writes(self):
        path = self.root / 'other.sqlite3'
        with closing(sqlite3.connect(path)) as conn:
            conn.execute('CREATE TABLE unrelated(value TEXT)')
        self.assert_rejected_unchanged(path)

    def test_unknown_version_rejected_without_writes(self):
        path = self.root / 'future.sqlite3'
        store = qm.MetadataStore(path)
        store.conn.execute("UPDATE meta SET value='future' WHERE key='schema'")
        store.conn.commit()
        store.close()
        self.assert_rejected_unchanged(path)

    def test_incomplete_c2_structure_rejected_without_writes(self):
        path = self.root / 'broken.sqlite3'
        store = qm.MetadataStore(path)
        store.conn.execute('DROP TABLE observations')
        store.close()
        self.assert_rejected_unchanged(path)

    def test_readonly_commands_and_connection_preserve_database(self):
        path = self.root / 'valid.sqlite3'
        store = qm.MetadataStore(path)
        store.bind([paper()], [2021])
        store.put('paper', paper()['paper_id'], 'cited_by_count', 0, 'available')
        store.conn.commit()
        store.close()
        before = path.read_bytes()
        for command in ('status', 'export'):
            self.assertEqual(self.invoke(command, path), 0)
            self.assertEqual(path.read_bytes(), before)
        readonly = qm.MetadataStore(path, readonly=True)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                readonly.conn.execute("INSERT INTO meta VALUES('forbidden','write')")
        finally:
            readonly.close()
        self.assertEqual(path.read_bytes(), before)

    def test_readonly_missing_database_is_not_created(self):
        path = self.root / 'absent.sqlite3'
        for command in ('status', 'export'):
            with self.assertRaises(SystemExit):
                self.invoke(command, path)
        self.assertFalse(path.exists())

    def test_new_collection_store_creation_and_writable_resume(self):
        path = self.root / 'new.sqlite3'
        store = qm.MetadataStore(path)
        store.bind([paper()], [2021])
        store.close()
        store = qm.MetadataStore(path)
        store.put('paper', 'PAP000001', 'cited_by_count', 0, 'available')
        store.conn.commit()
        store.close()
        self.assertTrue(path.is_file())

    def test_output_cannot_overwrite_input_database(self):
        path = self.root / 'valid.sqlite3'
        store = qm.MetadataStore(path)
        store.bind([paper()], [2021])
        store.close()
        before = path.read_bytes()
        for command in ('status', 'export'):
            with self.assertRaises(SystemExit):
                self.invoke(command, path, '--out', str(path))
            self.assertEqual(path.read_bytes(), before)


class RateTests(unittest.TestCase):
    def test_formula_and_minimum_denominator(self):
        value, state, _, info = qm.citation_rate(10, '2026-11-01', '2026-11-01T00:00:00+00:00')
        self.assertEqual((value, state, info['denominator']), (10, 'available', 1))
        value, _, _, info = qm.citation_rate(10, '2020-01-01', '2026-11-01T00:00:00+00:00')
        self.assertAlmostEqual(value, 10 / info['denominator'])

    def test_zero_is_real_zero(self):
        self.assertEqual(qm.citation_rate(0, '2020-01-01', '2026-11-01T00:00:00+00:00')[:2], (0, 'available'))

    def test_bad_dates_and_numerators(self):
        for count, published, expected in [(1, None, 'missing'), (1, 'bad', 'failed'), (1, '2027-01-01', 'failed'), (-1, '2020-01-01', 'failed'), (float('nan'), '2020-01-01', 'failed'), (None, '2020-01-01', 'missing'), (True, '2020-01-01', 'failed')]:
            self.assertEqual(qm.citation_rate(count, published, '2026-11-01T00:00:00+00:00')[1], expected)


class HttpTests(unittest.TestCase):
    def client(self, replies):
        self.calls, self.sleeps = [], []
        def opener(req, timeout):
            self.calls.append(req)
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            response = io.BytesIO(reply)
            response.status = 200
            return response
        return qm.OpenAlexClient(opener=opener, sleep=self.sleeps.append, monotonic=lambda: 1, key='')

    def test_network_and_rate_limit_retry_then_success(self):
        rate = HTTPError('public', 429, '', {'Retry-After': '2'}, None)
        cli = self.client([URLError('offline'), rate, b'{"meta":{"count":0},"results":[]}'])
        self.assertEqual(cli.get(qm.batch_url('works', 'doi', ['10.1234/a']))[1], 'available')
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(any(s >= 2 for s in self.sleeps))

    def test_parse_failure_bounded(self):
        cli = self.client([b'bad'] * 3)
        self.assertEqual(cli.get(qm.batch_url('works', 'doi', ['10.1234/a']))[1:3], ('failed', 'parse_or_validation_error'))
        self.assertEqual(len(self.calls), 3)

    def test_count_mismatch_is_not_success(self):
        cli = self.client([b'{"meta":{"count":2},"results":[]}'] * 3)
        self.assertEqual(cli.get(qm.batch_url('works', 'doi', ['10.1234/a']))[1], 'failed')

    def test_long_retry_after_deferred(self):
        cli = self.client([HTTPError('public', 429, '', {'Retry-After': '120'}, None)])
        self.assertEqual(cli.get(qm.batch_url('works', 'doi', ['10.1234/a']))[2], 'http_429:retry_deferred')
        self.assertEqual(len(self.calls), 1)

    def test_small_evidence_excludes_abstract_and_links(self):
        record = work(abstract_inverted_index={'private': [1]}, unrelated='large')
        record['primary_location']['pdf_url'] = 'not-retained'
        small = qm.trim_record(record, 'works')
        self.assertNotIn('abstract_inverted_index', small)
        self.assertNotIn('pdf_url', small['primary_location'])


if __name__ == '__main__':
    unittest.main()
