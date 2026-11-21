"""C2 isolated bibliometric metadata; no retrieval, corpus or model imports.

Registry schema v2 has no read-only public constructor. The adapter reads its
documented paper_identity view and active identifiers through SQLite mode=ro.
Only exact public identifiers leave the machine. Default collection resumes
successful requests; --retry-failed retries failures and --refresh re-fetches.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from paper_registry import RegistryError, normalize_identifier

RULE = 'c2-v1'
SOURCE = 'openalex'
STATES = ('available', 'missing', 'failed', 'not_queried')
WORK_FIELDS = ('cited_by_count', 'field_normalized_percentile',
               'cumulative_annualized_citation_rate', 'journal_source_id')
JOURNAL_FIELDS = ('annual_works_count', 'annual_cited_by_count',
                  'journal_field_normalized_percentile')
WORK_SELECT = 'id,doi,ids,type,publication_date,publication_year,primary_topic,topics,cited_by_count,citation_normalized_percentile,primary_location,updated_date'
SOURCE_SELECT = 'id,issn_l,issn,type,topics,counts_by_year,updated_date'


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def read_registry(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        version = conn.execute("SELECT value FROM registry_meta WHERE key='schema_version'").fetchone()
        if not version or version[0] != 'p09-c1-v2':
            raise ValueError('requires accepted C1 v2 registry')
        papers = {r['paper_id']: {'paper_id': r['paper_id'],
                  'identity_status': r['identity_status'], 'identifiers': {}}
                  for r in conn.execute('SELECT paper_id,identity_status FROM paper_identity ORDER BY paper_id')}
        for r in conn.execute("SELECT paper_id,scheme,value FROM identifiers WHERE status='active'"):
            papers[r['paper_id']]['identifiers'].setdefault(r['scheme'], []).append(r['value'])
        return list(papers.values())
    finally:
        conn.close()


def record_ids(record):
    ids = record.get('ids') or {}
    if record.get('doi') and ids.get('doi') and normalize_identifier('doi', record['doi'])[0] != normalize_identifier('doi', ids['doi'])[0]:
        raise RegistryError('contradictory external DOI fields')
    result = {}
    for scheme in ('doi', 'pmid', 'pmcid'):
        raw = record.get('doi') if scheme == 'doi' else ids.get(scheme)
        if not raw:
            raw = ids.get(scheme)
        if raw:
            if scheme == 'pmid':
                raw = str(raw).rstrip('/').rsplit('/', 1)[-1]
            if scheme == 'pmcid':
                raw = str(raw).rstrip('/').rsplit('/', 1)[-1]
                if raw.isdigit():
                    raw = 'PMC' + raw
            result[scheme] = normalize_identifier(scheme, raw)[0]
    return result


def identifier_claims(record):
    """Keep independently valid claims even when the record contradicts itself."""
    if not isinstance(record, dict):
        return set()
    ids = record.get('ids')
    ids = ids if isinstance(ids, dict) else {}
    claims = set()
    for scheme, raw in [('doi', record.get('doi')), *ids.items()]:
        if scheme not in ('doi', 'pmid', 'pmcid') or not raw:
            continue
        try:
            claims.update(record_ids({'ids': {scheme: raw}}).items())
        except (RegistryError, TypeError, AttributeError):
            pass
    return claims


def match_work(paper, records, owners):
    """Require an exact query-key match and reject ANY contradictory identifier."""
    scheme, value = query_identifier(paper)
    parsed, invalid = [], []
    local_claims = {(s, v) for s, values in paper['identifiers'].items() for v in values}
    for record in records:
        try:
            parsed.append((record, record_ids(record)))
        except (RegistryError, TypeError, AttributeError):
            invalid.append((record, identifier_claims(record)))
    hits = [r for r, ids in parsed if ids.get(scheme) == value]
    if any(claims & local_claims for r, claims in invalid):
        return None, 'failed', 'invalid_external_identifiers'
    if not hits:
        for r, ids in parsed:
            if any(v in paper['identifiers'].get(s, []) for s, v in ids.items()):
                return None, 'failed', 'query_identifier_mismatch'
        return None, 'missing', 'no_exact_record'
    if len(hits) != 1:
        return None, 'failed', 'ambiguous_external_records'
    record = hits[0]
    for kind, external in record_ids(record).items():
        local = paper['identifiers'].get(kind, [])
        if local and (len(local) != 1 or external != local[0]):
            return None, 'failed', 'identifier_conflict:' + kind
        owner = owners.get((kind, external))
        if owner and owner != paper['paper_id']:
            return None, 'failed', 'foreign_identifier_owner:' + kind
    if not re.fullmatch(r'https://openalex.org/W\d+', record.get('id', '')):
        return None, 'failed', 'invalid_work_id'
    return record, 'available', 'exact_' + scheme


def query_identifier(paper):
    for scheme in ('doi', 'pmid', 'pmcid'):
        values = paper['identifiers'].get(scheme, [])
        if values:
            return scheme, values[0]
    return '', ''


def safe_reason(paper):
    if paper['identity_status'] != 'verified':
        return 'registry_identity_' + paper['identity_status']
    if any(len(v) > 1 for v in paper['identifiers'].values()):
        return 'multiple_active_identifiers'
    if not query_identifier(paper)[0]:
        return 'no_public_identifier'
    return ''


def numeric(value, integer=False, percentile=False):
    if value is None:
        return None, 'missing', 'field_absent'
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None, 'failed', 'invalid_numeric_value'
    if integer and not isinstance(value, int):
        return None, 'failed', 'non_integer_count'
    if percentile and value > 1:
        return None, 'failed', 'percentile_out_of_range'
    return value, 'available', ''


def citation_rate(count, published, retrieved_at):
    """Cumulative citations / max(age days / 365.2425, 1 year)."""
    value, state, reason = numeric(count, integer=True)
    info = {'formula': 'cited_by_count / max(age_days / 365.2425, 1)',
            'rule_version': RULE, 'numerator': count, 'denominator_unit': 'years',
            'minimum_denominator': 1.0, 'date_basis': 'OpenAlex publication_date',
            'window_start': published, 'window_end': retrieved_at,
            'cutoff_basis': 'retrieval observation; source graph cutoff not supplied'}
    if state != 'available':
        return None, state, 'numerator_' + reason, info
    if not published:
        return None, 'missing', 'publication_date_absent', info
    try:
        start = date.fromisoformat(published)
        end = datetime.fromisoformat(retrieved_at).date()
    except (TypeError, ValueError):
        return None, 'failed', 'invalid_date', info
    days = (end - start).days
    if days < 0:
        return None, 'failed', 'future_publication_date', info
    denominator = max(days / 365.2425, 1.0)
    info.update(age_days=days, denominator=denominator)
    return value / denominator, 'available', '', info


class MetadataStore:
    def __init__(self, path, readonly=False):
        path = Path(path).resolve()
        if path.exists() or readonly:
            self.conn = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
            self.conn.row_factory = sqlite3.Row
            try:
                self._validate(require_bound=readonly)
                if not readonly:
                    self.conn.close()
                    self.conn = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True)
                    self.conn.row_factory = sqlite3.Row
                    self._validate()
            except Exception:
                self.conn.close()
                raise
            return
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript('''
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS papers(paper_id TEXT PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS requests(
          request_id INTEGER PRIMARY KEY,url TEXT NOT NULL,retrieved_at TEXT NOT NULL,
          status TEXT NOT NULL,reason TEXT NOT NULL,payload TEXT,attempts TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations(
          level TEXT NOT NULL,entity TEXT NOT NULL,source TEXT NOT NULL,
          metric TEXT NOT NULL,year INTEGER NOT NULL,payload TEXT NOT NULL,last_valid TEXT,
          PRIMARY KEY(level,entity,source,metric,year));
        ''')
        row = self.conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if row and row[0] != RULE:
            raise ValueError('unsupported metadata schema')
        self.conn.execute("INSERT OR IGNORE INTO meta VALUES('schema',?)", (RULE,))
        self.conn.execute("INSERT OR IGNORE INTO meta VALUES('created_at',?)", (now(),))
        self.conn.commit()

    def _validate(self, require_bound=False):
        required = {
            'meta': {'key', 'value'}, 'papers': {'paper_id', 'payload'},
            'requests': {'request_id', 'url', 'retrieved_at', 'status', 'reason', 'payload', 'attempts'},
            'observations': {'level', 'entity', 'source', 'metric', 'year', 'payload', 'last_valid'},
        }
        for table, columns in required.items():
            kind = self.conn.execute('SELECT type FROM sqlite_master WHERE name=?', (table,)).fetchone()
            actual = {r['name'] for r in self.conn.execute('PRAGMA table_info(' + table + ')')}
            if not kind or kind[0] != 'table' or not columns <= actual:
                raise ValueError('not a C2 database: missing structure ' + table)
        meta = dict(self.conn.execute('SELECT key,value FROM meta'))
        if meta.get('schema') != RULE or not meta.get('created_at'):
            raise ValueError('unsupported or incomplete metadata schema')
        if require_bound:
            try:
                years = json.loads(meta['years'])
                papers = json.loads(meta['registry_snapshot'])
                if not isinstance(years, list) or not years or not all(type(y) is int for y in years) or not isinstance(papers, list):
                    raise ValueError('invalid metadata scope')
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError('C2 database has no valid bound scope') from exc

    def close(self):
        self.conn.close()

    def bind(self, papers, years):
        payload = dumps(papers)
        row = self.conn.execute("SELECT value FROM meta WHERE key='registry_snapshot'").fetchone()
        if row and row[0] != payload:
            raise ValueError('registry identity changed; use a new isolated store')
        years_json = dumps(years)
        previous = self.conn.execute("SELECT value FROM meta WHERE key='years'").fetchone()
        if previous and previous[0] != years_json:
            raise ValueError('year scope changed; use a new isolated store')
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO meta VALUES('registry_snapshot',?)", (payload,))
            self.conn.execute("INSERT OR IGNORE INTO meta VALUES('years',?)", (years_json,))
            self.conn.executemany('INSERT OR IGNORE INTO papers VALUES(?,?)',
                                  [(p['paper_id'], dumps(p)) for p in papers])

    def put(self, level, entity, metric, value=None, state='missing', reason='',
            year=0, source=SOURCE, **details):
        if state not in STATES or (state != 'available' and value is not None):
            raise ValueError('invalid observation state/value')
        if state == 'available' and value is None:
            raise ValueError('available requires a value')
        allowed = WORK_FIELDS if level == 'paper' else JOURNAL_FIELDS if level == 'journal' else ()
        if metric not in allowed:
            raise ValueError('metric/level mismatch')
        if state != 'available' and not reason:
            raise ValueError('non-available requires reason')
        payload = dict(value=value, state=state, reason=reason, rule_version=RULE,
                       metric_year=year or None, **details)
        key = (level, entity, source, metric, year)
        old = self.conn.execute('SELECT payload,last_valid FROM observations WHERE level=? AND entity=? AND source=? AND metric=? AND year=?', key).fetchone()
        valid = dumps(payload) if state == 'available' else (old['last_valid'] if old else None)
        self.conn.execute('INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?)',
                          (*key, dumps(payload), valid))

    def rows(self):
        return [dict(r, payload=json.loads(r['payload']),
                     last_valid=json.loads(r['last_valid']) if r['last_valid'] else None)
                for r in self.conn.execute('SELECT * FROM observations ORDER BY level,entity,metric,year')]

    def request(self, url, client, refresh=False, retry_failed=False):
        old = self.conn.execute('SELECT * FROM requests WHERE url=? ORDER BY request_id DESC LIMIT 1', (url,)).fetchone()
        if old and not refresh and (old['status'] == 'available' or not retry_failed):
            return dict(old, payload=json.loads(old['payload']) if old['payload'] else None)
        payload, state, reason, attempts = client.get(url)
        stamp = now()
        with self.conn:
            cursor = self.conn.execute('INSERT INTO requests(url,retrieved_at,status,reason,payload,attempts) VALUES(?,?,?,?,?,?)',
                (url, stamp, state, reason, dumps(payload) if payload is not None else None, dumps(attempts)))
        return dict(request_id=cursor.lastrowid, url=url, retrieved_at=stamp,
                    status=state, reason=reason, payload=payload)


def trim_record(record, endpoint):
    fields = WORK_SELECT.split(',') if endpoint == 'works' else SOURCE_SELECT.split(',')
    small = {k: record.get(k) for k in fields}
    if record.get('topics') is not None and not isinstance(record['topics'], list):
        raise ValueError('invalid_topics')
    if endpoint == 'works':
        for key in ('ids', 'primary_topic', 'citation_normalized_percentile', 'primary_location'):
            if record.get(key) is not None and not isinstance(record[key], dict):
                raise ValueError('invalid_work_field_shape')
        loc = record.get('primary_location') or {}
        source = loc.get('source')
        small['primary_location'] = {'source': {k: source.get(k) for k in
            ('id', 'issn_l', 'issn', 'type')} if isinstance(source, dict) else None}
    else:
        small['counts_by_year'] = [{k: row.get(k) for k in ('year', 'works_count', 'cited_by_count')}
                                  for row in record.get('counts_by_year') or []]
    return small


class OpenAlexClient:
    """Serial GETs, >=1.1s between starts; at most two transient retries."""
    def __init__(self, opener=urlopen, sleep=time.sleep, monotonic=time.monotonic,
                 key=None):
        self.opener, self.sleep, self.clock = opener, sleep, monotonic
        self.key = key if key is not None else os.environ.get('OPENALEX_API_KEY')
        self.last_start = None

    def get(self, url):
        attempts = []
        for attempt in range(3):
            if self.last_start is not None:
                self.sleep(max(0, 1.1 - (self.clock() - self.last_start)))
            self.last_start = self.clock()
            headers = {'User-Agent': 'writing-rag-C2/1.0', 'Accept': 'application/json'}
            if self.key:
                headers['Authorization'] = 'Bearer ' + self.key
            delay = 0
            try:
                with self.opener(Request(url, headers=headers), timeout=45) as response:
                    if response.status != 200:
                        raise ValueError('unexpected_http_status')
                    data = json.load(response)
                if not isinstance(data, dict) or data.get('error') or not isinstance(data.get('results'), list):
                    raise ValueError('invalid_api_envelope')
                results, meta = data['results'], data.get('meta') or {}
                count = meta.get('count')
                if not isinstance(count, int) or count != len(results) or count > 100:
                    raise ValueError('incomplete_or_invalid_result_count')
                endpoint = 'works' if '/works?' in url else 'sources'
                if not all(isinstance(r, dict) and isinstance(r.get('id'), str) for r in results):
                    raise ValueError('invalid_record_shape')
                small = {'meta': {'count': count}, 'results': [trim_record(r, endpoint) for r in results]}
                attempts.append({'status': 200})
                return small, 'available', '', attempts
            except HTTPError as exc:
                reason = 'http_' + str(exc.code)
                retry = exc.code in (408, 429, 500, 502, 503, 504)
                raw = exc.headers.get('Retry-After', '') if exc.headers else ''
                try:
                    delay = float(raw)
                except ValueError:
                    try:
                        delay = max(0, (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds())
                    except (ValueError, TypeError):
                        pass
                if delay > 60:
                    attempts.append({'reason': reason, 'retry_after_seconds': delay})
                    return None, 'failed', reason + ':retry_deferred', attempts
            except (URLError, TimeoutError, OSError):
                reason, retry = 'network_error', True
            except (ValueError, TypeError, KeyError, AttributeError):
                reason, retry = 'parse_or_validation_error', True
            attempts.append({'reason': reason})
            if not retry or attempt == 2:
                return None, 'failed', reason, attempts
            self.sleep(max(delay, 2 ** attempt + random.uniform(0.4, 0.6)))
        raise AssertionError('unreachable')


def batch_url(endpoint, field, values):
    if not values or len(values) > 25:
        raise ValueError('batch must contain 1..25 identifiers')
    select = WORK_SELECT if endpoint == 'works' else SOURCE_SELECT
    return 'https://api.openalex.org/' + endpoint + '?' + urlencode({
        'filter': field + ':' + '|'.join(values), 'per_page': 100, 'select': select})


def write_paper(store, paper, record, state, reason, request=None):
    stamp = request['retrieved_at'] if request else store.conn.execute("SELECT value FROM meta WHERE key='created_at'").fetchone()[0]
    detail = {'retrieved_at': stamp, 'source_updated_at': (record or {}).get('updated_date'),
              'external_id': (record or {}).get('id'),
              'request_id': request['request_id'] if request else None,
              'mapping_basis': reason, 'classification_system': 'OpenAlex topics/subfields',
              'subjects': (record or {}).get('topics') or [],
              'statistical_cutoff': 'not supplied by source; observed at retrieved_at'}
    pid = paper['paper_id']
    if record is None:
        units = {'cited_by_count': 'citing works', 'field_normalized_percentile': 'fraction [0,1]',
                 'cumulative_annualized_citation_rate': 'citing works/year', 'journal_source_id': 'OpenAlex source ID'}
        for field in WORK_FIELDS:
            store.put('paper', pid, field, state=state, reason=reason,
                      source=SOURCE + ':local' if field == 'cumulative_annualized_citation_rate' else SOURCE,
                      unit=units[field], **detail)
        return
    count = record.get('cited_by_count')
    value, st, why = numeric(count, integer=True)
    store.put('paper', pid, 'cited_by_count', value, st, why, unit='citing works', **detail)
    percentile = record.get('citation_normalized_percentile') or {}
    value, st, why = numeric(percentile.get('value'), percentile=True)
    primary = record.get('primary_topic') or {}
    subfield = primary.get('subfield')
    if st == 'available' and (not subfield or not record.get('publication_year') or not record.get('type')):
        value, st, why = None, 'missing', 'normalization_reference_metadata_absent'
    normalization = {'origin': 'source_provided', 'reference': 'same publication year/type/primary subfield',
                     'reference_population_size': None, 'reference_population_size_reason': 'not supplied',
                     'publication_year': record.get('publication_year'), 'work_type': record.get('type'),
                     'primary_subfield': subfield, 'citation_window': 'publication year and three following years; immature years partial',
                     'flags': percentile}
    store.put('paper', pid, 'field_normalized_percentile', value, st, why,
              unit='fraction [0,1]', normalization=normalization, **detail)
    value, st, why, formula = citation_rate(count, record.get('publication_date'), stamp)
    store.put('paper', pid, 'cumulative_annualized_citation_rate', value, st, why,
              source=SOURCE + ':local', unit='citing works/year', calculation=formula, **detail)
    journal = (record.get('primary_location') or {}).get('source') or {}
    jid = journal.get('id', '')
    if journal.get('type') != 'journal':
        jid, st, why = None, 'missing', 'primary_source_not_journal_or_absent'
    elif not re.fullmatch(r'https://openalex.org/S\d+', jid or '') or not journal.get('issn_l'):
        jid, st, why = None, 'missing', 'journal_id_or_issn_l_absent'
    else:
        st, why = 'available', ''
    store.put('paper', pid, 'journal_source_id', jid, st, why,
              unit='OpenAlex source ID', journal_identity=journal, **detail)


def write_journal(store, jid, expected, records, years, request):
    hits = [r for r in records if r.get('id') == jid]
    record = hits[0] if len(hits) == 1 else None
    state, reason = request['status'], request['reason']
    if state == 'available':
        if len(hits) > 1:
            state, reason = 'failed', 'ambiguous_source_records'
        elif record is None:
            state, reason = 'missing', 'no_exact_source_record'
        elif record.get('type') != 'journal' or not record.get('issn_l') or any(
                e['issn_l'] != record.get('issn_l') for e in expected):
            state, reason = 'failed', 'journal_identifier_conflict'
    detail = {'retrieved_at': request['retrieved_at'], 'source_updated_at': (record or {}).get('updated_date'),
              'external_id': jid, 'request_id': request['request_id'],
              'mapping_basis': 'exact OpenAlex source ID and matching ISSN-L',
              'journal_identity': {k: (record or {}).get(k) for k in ('id', 'issn_l', 'issn', 'type')},
              'classification_system': 'OpenAlex topics/subfields', 'subjects': (record or {}).get('topics') or []}
    for year in years:
        annual = [r for r in (record or {}).get('counts_by_year') or [] if r.get('year') == year]
        for metric, field, unit in (('annual_works_count', 'works_count', 'works'),
                                    ('annual_cited_by_count', 'cited_by_count', 'citations received in year')):
            if state != 'available':
                value, st, why = None, state, reason
            elif len(annual) > 1:
                value, st, why = None, 'failed', 'duplicate_metric_year'
            elif not annual:
                value, st, why = None, 'missing', 'year_absent_no_zero_imputation'
            else:
                value, st, why = numeric(annual[0].get(field), integer=True)
            store.put('journal', jid, metric, value, st, why, year=year,
                      unit=unit, window={'start': f'{year}-01-01', 'end': f'{year}-12-31',
                      'scope': 'primary-location works; cited count includes works from all publication years'}, **detail)
        store.put('journal', jid, 'journal_field_normalized_percentile',
                  state='missing' if state == 'available' else state,
                  reason='not_provided_by_source' if state == 'available' else reason,
                  year=year, unit='percentile', **detail)


def collect(store, papers, years, client, refresh=False, retry_failed=False):
    store.bind(papers, years)
    owners = {(s, v): p['paper_id'] for p in papers for s, vs in p['identifiers'].items() for v in vs}
    groups = defaultdict(list)
    for paper in papers:
        reason = safe_reason(paper)
        if reason:
            write_paper(store, paper, None, 'not_queried', reason)
        else:
            groups[query_identifier(paper)[0]].append(paper)
    for scheme, group in sorted(groups.items()):
        for start in range(0, len(group), 25):
            batch = group[start:start + 25]
            url = batch_url('works', scheme, [query_identifier(p)[1] for p in batch])
            request = store.request(url, client, refresh, retry_failed)
            records = (request['payload'] or {}).get('results', [])
            with store.conn:
                for paper in batch:
                    if request['status'] == 'available':
                        record, state, reason = match_work(paper, records, owners)
                    else:
                        record, state, reason = None, 'failed', request['reason']
                    write_paper(store, paper, record, state, reason, request)
            print(dumps({'processed_scheme': scheme, 'processed': min(start + 25, len(group)), 'total': len(group)}), flush=True)
    journals = defaultdict(list)
    for row in store.rows():
        if row['metric'] == 'journal_source_id' and row['payload']['state'] == 'available':
            journals[row['payload']['value']].append(row['payload']['journal_identity'])
    ids = sorted(journals)
    for start in range(0, len(ids), 25):
        batch = ids[start:start + 25]
        url = batch_url('sources', 'openalex', [v.rsplit('/', 1)[-1] for v in batch])
        request = store.request(url, client, refresh, retry_failed)
        records = (request['payload'] or {}).get('results', [])
        with store.conn:
            for jid in batch:
                write_journal(store, jid, journals[jid], records, years, request)
        print(dumps({'journals_processed': min(start + 25, len(ids)), 'total': len(ids)}), flush=True)
    store.conn.commit()
    return summary(store)


def summary(store):
    rows = store.rows()
    counts, reasons = defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        key = ':'.join((row['level'], row['source'], row['metric'], str(row['year'])))
        counts[key][row['payload']['state']] += 1
        if row['payload']['reason']:
            reasons[key][row['payload']['reason']] += 1
    requests = list(store.conn.execute('SELECT status,attempts FROM requests'))
    paper_rows = [r for r in rows if r['level'] == 'paper']
    years = json.loads(store.conn.execute("SELECT value FROM meta WHERE key='years'").fetchone()[0])
    journal_rows = {(r['entity'], r['metric'], r['year']): r['payload'] for r in rows if r['level'] == 'journal'}
    projected = defaultdict(Counter)
    for row in paper_rows:
        if row['metric'] != 'journal_source_id':
            continue
        mapping = row['payload']
        for year in years:
            for metric in JOURNAL_FIELDS:
                obs = journal_rows.get((mapping['value'], metric, year)) if mapping['state'] == 'available' else mapping
                projected[metric + ':' + str(year)][obs['state'] if obs else 'not_queried'] += 1
    return {'registry_papers': store.conn.execute('SELECT count(*) FROM papers').fetchone()[0],
            'years': years,
            'paper_projected_journal_status': {k: {s: v[s] for s in STATES} for k, v in sorted(projected.items())},
            'journal_entities': len({r['entity'] for r in rows if r['level'] == 'journal'}),
            'field_status': {k: {s: c[s] for s in STATES} for k, c in sorted(counts.items())},
            'reasons': {k: dict(v) for k, v in sorted(reasons.items())},
            'requests': dict(Counter(r['status'] for r in requests)),
            'http_attempts': sum(len(json.loads(r['attempts'])) for r in requests),
            'retried_requests': sum(len(json.loads(r['attempts'])) > 1 for r in requests),
            'preserved_valid_after_failure': sum(r['payload']['state'] == 'failed' and r['last_valid'] is not None for r in rows),
            'paper_fields_total': len(paper_rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('command', choices=('collect', 'status', 'export'))
    parser.add_argument('--db', required=True)
    parser.add_argument('--registry')
    parser.add_argument('--as-of-year', type=int, default=datetime.now(timezone.utc).year)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--out')
    args = parser.parse_args(argv)
    if args.command != 'collect' and not Path(args.db).is_file():
        parser.error('metadata database does not exist')
    if args.command == 'collect' and (not args.registry or Path(args.db).resolve() == Path(args.registry).resolve()):
        parser.error('requires separate --registry and --db')
    if args.out:
        output = Path(args.out)
        for source in [args.db, args.registry] if args.registry else [args.db]:
            target = Path(source)
            if output.resolve() == target.resolve() or (output.exists() and target.exists() and output.samefile(target)):
                parser.error('--out must not overwrite an input database')
    store = MetadataStore(args.db, readonly=args.command != 'collect')
    try:
        if args.command == 'collect':
            result = collect(store, read_registry(args.registry), list(range(args.as_of_year - 5, args.as_of_year)),
                             OpenAlexClient(), args.refresh, args.retry_failed)
        elif args.command == 'export':
            if not args.out:
                parser.error('export requires --out')
            Path(args.out).write_text(''.join(dumps(r) + '\n' for r in store.rows()), encoding='utf-8')
            return 0
        else:
            result = summary(store)
        if args.out:
            Path(args.out).write_text(dumps(result) + '\n', encoding='utf-8')
        print(dumps(result))
        return 0
    finally:
        store.close()


if __name__ == '__main__':
    raise SystemExit(main())
