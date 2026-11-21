"""Local passage BM25. Stores term counts and identity metadata, never text."""
from array import array
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import unicodedata

K1 = 1.2
B = 0.75
TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002fa1f]")
FIELDS = ('source', 'chunk', 'doi', 'paper_id', 'title', 'year', 'journal',
          'authors', 'mac', 'section', 'section_type', 'embedding_text_version',
          'pub_type', 'license')


def tokenize(text):
    return TOKEN.findall(unicodedata.normalize('NFKC', text).casefold())


def pair_binding(db_dir, manifest_path):
    db = Path(db_dir).resolve()
    manifest = Path(manifest_path).resolve()
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    identity = json.loads((db / '.publish-identity.json').read_text(encoding='utf-8'))
    ready = json.loads(Path(str(manifest) + '.ready.json').read_text(encoding='utf-8'))
    if identity['manifest_sha256'] != digest or ready['manifest_sha256'] != digest:
        raise ValueError('sparse target pair mismatch')
    if not (db / 'chroma.sqlite3').is_file():
        raise ValueError('sparse target DB missing')
    return {'db_dir': str(db), 'manifest_path': str(manifest),
            'manifest_sha256': digest, 'pair_id': identity['pair_id']}


def build(rows, sparse_path, binding, expected_counts=None):
    """Consume (id, complete stored document, metadata) once, in batches upstream."""
    path = Path(sparse_path)
    path.mkdir(parents=True, exist_ok=False)
    postings = defaultdict(lambda: array('I'))
    docs, lengths, counts, seen = [], [], Counter(), set()
    for passage_id, document, metadata in rows:
        if passage_id in seen:
            raise ValueError('duplicate passage ID')
        seen.add(passage_id)
        source = metadata['source']
        if metadata.get('embedding_text_version') != 'passage-v2':
            raise ValueError('passage representation mismatch')
        counts[source] += 1
        # Same representation as build_index.build_embedding_text, without its imports.
        title = (metadata.get('title') or '').strip() or 'Unknown title'
        section = (metadata.get('section') or '').strip() or 'other'
        terms = Counter(tokenize(f'Title: {title}\nSection: {section}\nContent:\n{document}'))
        number = len(docs)
        docs.append({'id': passage_id, **{k: metadata[k] for k in FIELDS if k in metadata}})
        lengths.append(sum(terms.values()))
        for term, frequency in terms.items():
            postings[term].extend((number, frequency))
    if not docs or (expected_counts is not None and dict(counts) != expected_counts):
        raise ValueError('sparse source counts do not match dense ledger')
    connection = sqlite3.connect(path / 'bm25.sqlite3')
    try:
        connection.execute('CREATE TABLE config (payload TEXT)')
        config = {'binding': binding, 'n': len(docs), 'avgdl': sum(lengths)/len(docs),
                  'k1': K1, 'b': B, 'tokenizer': 'NFKC-casefold-ascii-han-v1',
                  'source_counts': dict(counts), 'byteorder': __import__('sys').byteorder}
        connection.execute('INSERT INTO config VALUES (?)', (json.dumps(config),))
        connection.execute('CREATE TABLE docs (number INTEGER PRIMARY KEY, payload TEXT, dl INTEGER)')
        connection.executemany('INSERT INTO docs VALUES (?, ?, ?)',
                               ((i, json.dumps(d, ensure_ascii=False), lengths[i]) for i,d in enumerate(docs)))
        connection.execute('CREATE TABLE terms (term TEXT PRIMARY KEY, postings BLOB)')
        connection.executemany('INSERT INTO terms VALUES (?, ?)',
                               ((t, p.tobytes()) for t,p in postings.items()))
        connection.commit()
    finally:
        connection.close()
    return config


class SparseIndex:
    def __init__(self, sparse_path, db_dir=None, manifest_path=None):
        path = Path(sparse_path) / 'bm25.sqlite3'
        self.connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        self.config = json.loads(self.connection.execute('SELECT payload FROM config').fetchone()[0])
        binding = self.config['binding']
        actual = pair_binding(db_dir or binding['db_dir'], manifest_path or binding['manifest_path'])
        if actual != binding:
            self.connection.close()
            raise ValueError('sparse index belongs to another dense pair')
        if self.config['byteorder'] != __import__('sys').byteorder:
            raise ValueError('sparse postings byte order mismatch')
        rows = self.connection.execute('SELECT payload, dl FROM docs ORDER BY number').fetchall()
        self.docs = [json.loads(row[0]) for row in rows]
        self.lengths = [row[1] for row in rows]

    def search(self, query, candidate_k):
        if candidate_k < 1:
            raise ValueError('candidate_k must be positive')
        scores = defaultdict(float)
        n, avgdl = self.config['n'], self.config['avgdl']
        for term in sorted(set(tokenize(query))):
            row = self.connection.execute('SELECT postings FROM terms WHERE term=?', (term,)).fetchone()
            if row is None:
                continue
            values = array('I')
            values.frombytes(row[0])
            df = len(values)//2
            idf = math.log(1 + (n-df+.5)/(df+.5))
            for i in range(0, len(values), 2):
                number, tf = values[i:i+2]
                scores[number] += idf * tf * (K1+1)/(tf+K1*(1-B+B*self.lengths[number]/avgdl))
        ordered = sorted((i for i in scores if scores[i] > 0),
                         key=lambda i: (-scores[i], self.docs[i]['id']))[:candidate_k]
        return [{**self.docs[i], 'rank': r, 'score': scores[i], 'section': self.docs[i].get('section', ''),
                 **{k: self.docs[i].get(k, '') for k in ('title','year','journal','doi','authors','mac')}}
                for r,i in enumerate(ordered, 1)]

    def close(self):
        self.connection.close()
