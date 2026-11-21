"""Paper registry: immutable internal paper ids with mutable identifier aliases.

P09/C1 moves literature identity from filename-centric to paper-centric.  The
registry keeps three things apart:

* ``paper_id`` -- an internal, opaque, immutable id (``PAP000001``).  It is
  allocated once and never derives from, or changes with, a DOI/PMID/PMCID.
* ``identifiers`` -- DOI/PMID/PMCID aliases.  They are *additive and
  correctable*: a later DOI fills a missing alias, a wrong alias is
  superseded (kept for audit) by ``correct_identifier``.  Aliases never rename
  a paper.
* ``artifacts`` -- the concrete files (Markdown sources) that carry content.
  Supplements and alternate versions are artifacts of the main paper, not new
  papers.  ``content_sha256`` identifies duplicated *content*; it never proves
  the same *research identity*, so byte-identical files with conflicting
  DOI/PMID stay separate until a human merges them.

Identity lookup order is DOI -> PMID -> PMCID for every identifier supplied on
a record; all owners are compared before a decision is taken, and a
disagreement is recorded as a pending conflict instead of silently picking the
first hit.  Normalized title hashes only produce candidates.  Relations
(exact duplicate, near duplicate, alternate version, supplement, correction,
retraction) are stored explicitly and remain ``pending`` when the evidence is
insufficient.

Only the standard library plus :mod:`document_metadata` (identifier
normalization) is used; no network, model or vector-store access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from document_metadata import PMID_CARD, normalize_doi, normalize_pmid

SCHEMA_VERSION = 'p09-c1-v2'
MIGRATABLE_SCHEMA_VERSIONS = ('p09-c1-v1',)

IDENTIFIER_SCHEMES = ('doi', 'pmid', 'pmcid')
ARTIFACT_ROLES = ('main', 'supplement', 'version', 'duplicate')
RELATION_KINDS = (
    'exact_duplicate',
    'near_duplicate',
    'alternate_version',
    'supplement',
    'correction',
    'retraction',
)
IDENTITY_STATUSES = ('verified', 'conflict', 'merged')
METADATA_FIELDS = ('title', 'authors', 'year', 'journal', 'publication_type', 'license', 'version')
LEGACY_KINDS = ('b3_paper_id', 'source_stem')
PAPER_ID_PREFIX = 'PAP'

_SHA256_RE = re.compile(r'[0-9a-f]{64}')
_PMCID_RE = re.compile(r'PMC(\d+)')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS registry_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    title_hash TEXT NOT NULL DEFAULT '',
    authors TEXT NOT NULL DEFAULT '',
    year TEXT NOT NULL DEFAULT '',
    journal TEXT NOT NULL DEFAULT '',
    publication_type TEXT NOT NULL DEFAULT '',
    license TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '',
    is_supplement INTEGER NOT NULL DEFAULT 0,
    duplicate_of TEXT REFERENCES papers(paper_id),
    content_sha256 TEXT NOT NULL DEFAULT '',
    metadata_updated_at TEXT NOT NULL,
    metadata_provenance TEXT NOT NULL DEFAULT '{}',
    identity_status TEXT NOT NULL DEFAULT 'verified'
        CHECK (identity_status IN ('verified', 'conflict', 'merged')),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS identifiers (
    scheme TEXT NOT NULL CHECK (scheme IN ('doi', 'pmid', 'pmcid')),
    value TEXT NOT NULL,
    value_raw TEXT NOT NULL DEFAULT '',
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'superseded')),
    source TEXT NOT NULL DEFAULT '',
    evidence TEXT NOT NULL DEFAULT '{}',
    added_at TEXT NOT NULL,
    superseded_at TEXT,
    superseded_by TEXT,
    PRIMARY KEY (scheme, value)
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    source TEXT NOT NULL,
    content_sha256 TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL CHECK (role IN ('main', 'supplement', 'version', 'duplicate')),
    is_supplement INTEGER NOT NULL DEFAULT 0,
    evidence TEXT NOT NULL DEFAULT '{}',
    added_at TEXT NOT NULL,
    UNIQUE (paper_id, source, content_sha256)
);
CREATE TABLE IF NOT EXISTS relations (
    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    to_paper_id TEXT NOT NULL REFERENCES papers(paper_id),
    kind TEXT NOT NULL CHECK (kind IN (
        'exact_duplicate', 'near_duplicate', 'alternate_version',
        'supplement', 'correction', 'retraction')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed')),
    evidence TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (from_paper_id, to_paper_id, kind)
);
CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'resolved')),
    resolution TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS legacy_ids (
    legacy_kind TEXT NOT NULL CHECK (legacy_kind IN ('b3_paper_id', 'source_stem')),
    legacy_id TEXT NOT NULL,
    raw_ids TEXT NOT NULL DEFAULT '[]',
    paper_id TEXT REFERENCES papers(paper_id),
    artifact_source TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'resolved'
        CHECK (status IN ('resolved', 'ambiguous', 'unresolved')),
    resolution TEXT NOT NULL DEFAULT '',
    evidence TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (legacy_kind, legacy_id)
);
CREATE INDEX IF NOT EXISTS artifacts_source_idx ON artifacts(source);
CREATE INDEX IF NOT EXISTS artifacts_sha_idx ON artifacts(content_sha256);
CREATE INDEX IF NOT EXISTS identifiers_paper_idx ON identifiers(paper_id);
CREATE INDEX IF NOT EXISTS legacy_paper_idx ON legacy_ids(paper_id);
CREATE VIEW IF NOT EXISTS paper_identity AS
SELECT
    p.paper_id,
    p.title,
    p.title_hash,
    p.authors,
    p.year,
    p.journal,
    p.publication_type,
    p.license,
    p.version,
    p.source,
    p.is_supplement,
    p.duplicate_of,
    p.content_sha256,
    p.identity_status,
    p.metadata_updated_at,
    (SELECT i.value FROM identifiers i
        WHERE i.paper_id = p.paper_id AND i.scheme = 'doi' AND i.status = 'active'
        ORDER BY i.added_at, i.value LIMIT 1) AS doi,
    (SELECT i.value FROM identifiers i
        WHERE i.paper_id = p.paper_id AND i.scheme = 'pmid' AND i.status = 'active'
        ORDER BY i.added_at, i.value LIMIT 1) AS pmid,
    (SELECT i.value FROM identifiers i
        WHERE i.paper_id = p.paper_id AND i.scheme = 'pmcid' AND i.status = 'active'
        ORDER BY i.added_at, i.value LIMIT 1) AS pmcid
FROM papers p;
"""


class RegistryError(ValueError):
    """Raised when a registry operation is refused or an input is invalid."""


# ---------------------------------------------------------------------------
# value helpers
# ---------------------------------------------------------------------------


def utcnow():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def normalize_pmcid(value):
    """Return ``PMC12345`` for a user supplied PMCID, else ``''``."""
    text = re.sub(r'\s+', '', str(value or '')).upper()
    match = _PMCID_RE.search(text)
    return 'PMC' + match.group(1) if match else ''


def normalize_identifier(scheme, value):
    """Normalize an identifier value; return ``(normalized, raw)``."""
    scheme = str(scheme or '').strip().lower()
    if scheme not in IDENTIFIER_SCHEMES:
        raise RegistryError('unsupported identifier scheme: %r' % (scheme,))
    raw = str(value or '').strip()
    if scheme == 'doi':
        normalized = normalize_doi(raw)
    elif scheme == 'pmid':
        normalized = normalize_pmid(raw)
    else:
        normalized = normalize_pmcid(raw)
    if not normalized:
        raise RegistryError('invalid %s identifier: %r' % (scheme, raw))
    return normalized, raw


def normalize_title(title):
    text = str(title or '').casefold()
    text = re.sub(r'[^\w]+', ' ', text, flags=re.UNICODE)
    return re.sub(r'\s+', ' ', text).strip()


def title_fingerprint(title):
    normalized = normalize_title(title)
    if not normalized:
        return ''
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def artifact_id_for(source, content_sha256):
    seed = '%s\x00%s' % (source or '', content_sha256 or '')
    return 'ART' + hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]


def source_sort_key(source):
    name = Path(str(source or '')).name
    return (0 if name.startswith('PMID_') else 1, name)


def stem_of(source):
    name = Path(str(source or '').replace('\\', '/')).name
    return name[:-3] if name.lower().endswith('.md') else name


def normalize_legacy_id(legacy_kind, legacy_id):
    """Return the stable lookup key for a historical id.

    The key follows the accepted B2 identity rules: the value is compared
    case-insensitively and reduced to its basename.  Internal whitespace is
    *not* collapsed, because a file name with consecutive spaces is a distinct
    historical key and collapsing it could silently merge two sources.  The
    original spelling is kept next to the key (``legacy_ids.raw_ids``) for
    audit and for resolving exact matches.
    """
    text = str(legacy_id or '').strip()
    if not text:
        return ''
    if legacy_kind == 'source_stem':
        return stem_of(text).casefold()
    head, sep, tail = text.partition(':')
    if sep and head.strip().lower() in IDENTIFIER_SCHEMES:
        return '%s:%s' % (head.strip().lower(), tail.strip().casefold())
    if sep and head.strip().lower() == 'source':
        return 'source:' + stem_of(tail).casefold()
    return stem_of(text).casefold()


def b2_identity(item):
    """Mirror the accepted B2 literature identity (``evaluate_retrieval``).

    Kept local so the registry never imports retrieval code: DOI wins, else
    the lower-cased basename of the source file *including* its ``.md``
    suffix; the stored value is never rewritten.
    """
    doi = str(item.get('doi') or '').strip()
    if doi:
        return 'doi:' + doi.strip().lower()
    for key in ('source_file', 'source', 'source_id'):
        value = str(item.get(key) or '').strip()
        if value:
            name = value.replace('\\', '/').rsplit('/', 1)[-1].strip().lower()
            return 'source:' + name
    return ''


def _clean_text(value):
    return ' '.join(str(value or '').split())


def _same_text(left, right):
    return _clean_text(left).casefold() == _clean_text(right).casefold()


def _dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def _recorded_candidates(evidence):
    """Paper ids recorded inside an ambiguous legacy mapping."""
    try:
        payload = json.loads(evidence or '{}')
    except ValueError:
        return []
    if not isinstance(payload, dict):
        return []
    candidates = payload.get('candidates')
    if not candidates:
        detail = payload.get('detail')
        candidates = detail.get('candidates') if isinstance(detail, dict) else None
    return [str(item) for item in candidates or () if item]


def _raw_legacy_ids(row):
    """Original spellings recorded for one legacy row (audit trail)."""
    if row is None:
        return []
    try:
        raws = json.loads(row['raw_ids'] or '[]')
    except (KeyError, IndexError, ValueError):
        return []
    return [str(item) for item in raws or () if item]


def _restored_legacy_spelling(legacy_kind, legacy_id, artifact_source):
    """Recover a historical id whose internal whitespace v1 folded away.

    v1 wrote ``legacy_ids.legacy_id`` after ``_clean_text`` had collapsed every
    whitespace run, so a file name with consecutive spaces (or a non-breaking
    space) lost its exact spelling and could no longer resolve to its paper.
    ``legacy_ids.artifact_source`` kept the spelling as found on disk.  The two
    are the same id only when the retained name folds back onto the stored
    text; any other difference means the evidence describes another identity
    and the stored value is returned untouched.  Case is ignored because the
    lookup key is case-folded anyway; the comparison exists to prove that only
    whitespace runs changed, never the words themselves.  This is an exact
    comparison of the recorded spellings, never a similarity match.
    """
    stored = str(legacy_id or '').strip()
    stem = stem_of(artifact_source)
    if not stored or not stem:
        return ''
    head, sep, tail = stored.partition(':')
    if legacy_kind == 'source_stem':
        candidate, expected = stem, stored
    elif legacy_kind == 'b3_paper_id' and sep and head.strip().lower() == 'source':
        candidate, expected = 'source:' + stem, tail
    else:
        return ''
    if _clean_text(stem).casefold() != _clean_text(expected).casefold():
        return ''
    return candidate


def validate_record(record):
    """Validate one registration record and return a normalized copy.

    Raises :class:`RegistryError` (a ``ValueError``) without touching the
    database when the record cannot be trusted as-is.
    """
    if not isinstance(record, dict):
        raise RegistryError('record must be an object')
    source = str(record.get('source') or '').strip()
    if not source:
        raise RegistryError('record.source is required')
    sha = str(record.get('content_sha256') or '').strip().lower()
    if sha and not _SHA256_RE.fullmatch(sha):
        raise RegistryError('record.content_sha256 must be a hex sha256 digest')

    identifiers_in = record.get('identifiers') or {}
    if not isinstance(identifiers_in, dict):
        raise RegistryError('record.identifiers must be an object')
    unknown = sorted(set(identifiers_in) - set(IDENTIFIER_SCHEMES))
    if unknown:
        raise RegistryError('unsupported identifier schemes: %s' % ', '.join(unknown))
    identifiers = {}
    for scheme, value in identifiers_in.items():
        if not str(value or '').strip():
            continue
        identifiers[scheme] = normalize_identifier(scheme, value)

    metadata_in = record.get('metadata') or {}
    if not isinstance(metadata_in, dict):
        raise RegistryError('record.metadata must be an object')
    unknown = sorted(set(metadata_in) - set(METADATA_FIELDS))
    if unknown:
        raise RegistryError('unsupported metadata fields: %s' % ', '.join(unknown))
    metadata = {}
    for field in METADATA_FIELDS:
        text = _clean_text(metadata_in.get(field))
        if text:
            metadata[field] = text

    provenance_in = record.get('provenance') or {}
    if not isinstance(provenance_in, dict):
        raise RegistryError('record.provenance must be an object')
    provenance = {str(key): _clean_text(value) for key, value in provenance_in.items()}

    duplicate_of = _reference(record.get('duplicate_of'), 'duplicate_of')
    supplement_of = _reference(record.get('supplement_of'), 'supplement_of')

    relations = []
    for entry in record.get('relations') or ():
        if not isinstance(entry, dict):
            raise RegistryError('relation entries must be objects')
        kind = str(entry.get('kind') or '').strip().lower()
        if kind not in RELATION_KINDS:
            raise RegistryError('unsupported relation kind: %r' % (kind,))
        target = _reference(entry.get('target'), 'relation target')
        relations.append(
            {
                'kind': kind,
                'target': target,
                'evidence': _clean_text(entry.get('evidence')),
            }
        )

    legacy_in = record.get('legacy') or {}
    if not isinstance(legacy_in, dict):
        raise RegistryError('record.legacy must be an object')
    unknown = sorted(set(legacy_in) - set(LEGACY_KINDS))
    if unknown:
        raise RegistryError('unsupported legacy kinds: %s' % ', '.join(unknown))
    legacy = {
        kind: str(value).strip()
        for kind, value in legacy_in.items()
        if str(value).strip()
    }

    return {
        'source': source,
        'content_sha256': sha,
        'identifiers': identifiers,
        'metadata': metadata,
        'provenance': provenance,
        'duplicate_of': duplicate_of,
        'supplement_of': supplement_of,
        'relations': relations,
        'legacy': legacy,
    }


def _reference(value, label):
    if value in (None, '', {}):
        return None
    if isinstance(value, str):
        return {'kind': 'source', 'value': value.strip()}
    if not isinstance(value, dict):
        raise RegistryError('%s must be an object' % label)
    kind = str(value.get('kind') or '').strip().lower()
    if kind not in ('doi', 'pmid', 'pmcid', 'source', 'paper'):
        raise RegistryError('%s has unsupported kind: %r' % (label, kind))
    text = str(value.get('value') or '').strip()
    if not text:
        raise RegistryError('%s requires a value' % label)
    return {'kind': kind, 'value': text}


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


class PaperRegistry:
    """SQLite-backed registry of papers, identifiers, artifacts and relations."""

    def __init__(self, path, create=False):
        self.path = Path(path)
        if not create and not self.path.exists():
            raise RegistryError('registry not found: %s' % self.path)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute('PRAGMA foreign_keys = ON')
        if create:
            self._create()
        self._check_schema()

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _create(self):
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            'INSERT OR IGNORE INTO registry_meta (key, value) VALUES (?, ?)',
            ('schema_version', SCHEMA_VERSION),
        )
        self._conn.execute(
            'INSERT OR IGNORE INTO registry_meta (key, value) VALUES (?, ?)',
            ('created_at', utcnow()),
        )
        self._conn.execute(
            'INSERT OR IGNORE INTO registry_meta (key, value) VALUES (?, ?)',
            ('paper_seq', '0'),
        )

    def _check_schema(self):
        row = self._conn.execute(
            "SELECT value FROM registry_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            raise RegistryError('not a paper registry: %s' % self.path)
        version = row['value']
        if version == SCHEMA_VERSION:
            return
        if version in MIGRATABLE_SCHEMA_VERSIONS:
            with self._transaction():
                self._migrate_v1_to_v2()
            return
        raise RegistryError(
            'unsupported registry schema %r (expected %r)' % (version, SCHEMA_VERSION)
        )

    def _migrate_v1_to_v2(self):
        """Upgrade a v1 library in place without reallocating paper ids.

        v2 makes three corrections: historical ids use the B2 lookup key
        (case-folded basename, exact spelling recovered from the retained
        artifact name and kept as a raw alias), legacy ambiguity carries its
        candidates and can only be cleared explicitly, and the two ESGAR
        Part I/II files stop being labelled as supplements because only their
        bibliographic folder-DOI link was ever recorded.
        """
        version = self._meta('schema_version')
        if version != 'p09-c1-v1':
            raise RegistryError('no v1 -> v2 migration for %r' % version)
        columns = {
            row['name']
            for row in self._conn.execute('PRAGMA table_info(legacy_ids)')
        }
        if 'raw_ids' not in columns:
            self._conn.execute(
                "ALTER TABLE legacy_ids ADD COLUMN raw_ids TEXT NOT NULL DEFAULT '[]'"
            )
        if 'resolution' not in columns:
            self._conn.execute(
                "ALTER TABLE legacy_ids ADD COLUMN resolution TEXT NOT NULL DEFAULT ''"
            )
        rows = self._conn.execute('SELECT * FROM legacy_ids ORDER BY rowid').fetchall()
        _, entries = self._legacy_rewrite_plan(rows)
        self._apply_legacy_plan(
            entries,
            'key recovered from the retained artifact spelling',
            utcnow(),
        )
        # Part I / Part II were never evidenced as supplements; they are the
        # same article as their DOI-matched card, so they become content
        # versions.  Their papers keep their own identity.
        affected = {}
        for row in self._conn.execute(
            "SELECT artifact_id, paper_id, source FROM artifacts "
            "WHERE role = 'supplement' AND is_supplement = 1"
        ).fetchall():
            stem = stem_of(row['source'])
            if stem not in ('ESGAR202601 PARTI', 'ESGAR202601 PARTII'):
                continue
            self._conn.execute(
                "UPDATE artifacts SET role = 'version', is_supplement = 0, evidence = ? "
                'WHERE artifact_id = ?',
                (
                    _dump(
                        {
                            'role_correction': 'folder-DOI link is a bibliographic alias, '
                            'not evidence of supplementary material',
                            'previous_role': 'supplement',
                        }
                    ),
                    row['artifact_id'],
                ),
            )
            affected[row['paper_id']] = True
        for paper_id in affected:
            self._conn.execute(
                'UPDATE papers SET is_supplement = 0 WHERE paper_id = ?', (paper_id,)
            )
            self._refresh_primary_artifact(paper_id)
        self._set_meta('schema_version', SCHEMA_VERSION)
        self._set_meta('migrated_from', 'p09-c1-v1')
        self._set_meta('migrated_at', utcnow())

    def repair_legacy_spellings(self, dry_run=False, now=None):
        """Restore source spellings a previous v1 -> v2 migration folded away.

        The migration above recovers the exact spelling from the retained
        ``artifact_source``.  A library that the *old* migration already
        upgraded to v2 keeps its folded keys, because ``_check_schema`` returns
        early for a current schema and never replays the upgrade; this pass
        offers the same regrouping explicitly and is idempotent.  It rewrites
        only ``legacy_ids``: paper ids, artifacts and content bindings are left
        alone, the folded spelling stays behind as a raw alias, and a key
        collision becomes an explicit ambiguous mapping with a pending
        conflict instead of a silent merge.
        """
        now = now or utcnow()
        rows = self._conn.execute('SELECT * FROM legacy_ids ORDER BY rowid').fetchall()
        report, entries = self._legacy_rewrite_plan(rows)
        if dry_run or not report['changed']:
            return report
        with self._transaction():
            self._apply_legacy_plan(
                entries, 'key recovered from the retained artifact spelling', now
            )
            self._set_meta('legacy_repaired_at', now)
        return report

    def _legacy_rewrite_plan(self, rows):
        """Group stored legacy rows by the key the retained evidence implies.

        Returns ``(report, entries)`` without writing anything, so the same
        plan serves the migration and the explicit repair.  A stored id whose
        artifact spelling differs by more than the known whitespace fold is
        left on its own key; two rows that land on one key become ambiguous and
        keep every candidate paper id.
        """
        merged = {}
        for row in rows:
            kind = row['legacy_kind']
            stored = row['legacy_id']
            restored = _restored_legacy_spelling(kind, stored, row['artifact_source'])
            key = normalize_legacy_id(kind, restored or stored) or stored
            raws = {stored}
            if restored:
                raws.add(restored)
            raws.update(_raw_legacy_ids(row))
            entry = merged.get((kind, key))
            if entry is None:
                merged[(kind, key)] = {
                    'legacy_kind': kind,
                    'legacy_id': key,
                    'raw_ids': raws,
                    'paper_id': row['paper_id'],
                    'artifact_source': row['artifact_source'],
                    'status': row['status'],
                    'resolution': row['resolution'],
                    'evidence': row['evidence'],
                    'created_at': row['created_at'],
                    'rekeyed_from': [stored] if key != stored else [],
                    'candidates': [],
                    'collision': False,
                }
                continue
            entry['raw_ids'] |= raws
            if key != stored:
                entry['rekeyed_from'].append(stored)
            candidates = sorted(
                ({entry['paper_id'], row['paper_id']} - {None})
                | set(entry['candidates'])
                | set(_recorded_candidates(entry['evidence']))
                | set(_recorded_candidates(row['evidence']))
            )
            entry['candidates'] = candidates
            entry['status'] = 'ambiguous'
            entry['paper_id'] = None
            entry['collision'] = True
        report = {
            'rows_before': len(rows),
            'rows_after': len(merged),
            'rekeyed': [
                {
                    'legacy_kind': entry['legacy_kind'],
                    'stored': stored,
                    'key': entry['legacy_id'],
                    'paper_id': entry['paper_id'],
                }
                for entry in merged.values()
                for stored in entry['rekeyed_from']
            ],
            'collisions': [
                {
                    'legacy_kind': entry['legacy_kind'],
                    'key': entry['legacy_id'],
                    'candidates': entry['candidates'],
                }
                for entry in merged.values()
                if entry['collision']
            ],
        }
        report['changed'] = bool(
            report['rekeyed']
            or report['collisions']
            or report['rows_after'] != report['rows_before']
        )
        return report, merged

    def _apply_legacy_plan(self, entries, note, now):
        """Write a rewrite plan back and keep every ambiguity visible."""
        self._conn.execute('DELETE FROM legacy_ids')
        ambiguous = []
        for entry in entries.values():
            status = 'ambiguous' if entry['collision'] else entry['status']
            paper_id = entry['paper_id']
            evidence = entry['evidence']
            if status == 'ambiguous':
                paper_id = None
                candidates = sorted(
                    set(entry['candidates'])
                    | set(_recorded_candidates(evidence))
                )
                entry['candidates'] = candidates
                evidence = _dump({'candidates': candidates, 'migration': note})
                ambiguous.append((entry['legacy_kind'], entry['legacy_id'], candidates))
            self._conn.execute(
                'INSERT INTO legacy_ids (legacy_kind, legacy_id, raw_ids, paper_id, '
                'artifact_source, status, resolution, evidence, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    entry['legacy_kind'],
                    entry['legacy_id'],
                    _dump(sorted(entry['raw_ids'] - {entry['legacy_id']})),
                    paper_id,
                    entry['artifact_source'],
                    status,
                    entry['resolution'],
                    evidence,
                    entry['created_at'],
                ),
            )
        for legacy_kind, legacy_key, candidates in ambiguous:
            conflict_id = self._upsert_legacy_conflict(
                legacy_kind, legacy_key, candidates, now
            )
            row = self._conn.execute(
                'SELECT evidence FROM legacy_ids WHERE legacy_kind = ? AND legacy_id = ?',
                (legacy_kind, legacy_key),
            ).fetchone()
            payload = json.loads(row['evidence'] or '{}')
            payload['conflict_id'] = conflict_id
            self._conn.execute(
                'UPDATE legacy_ids SET evidence = ? WHERE legacy_kind = ? AND legacy_id = ?',
                (_dump(payload), legacy_kind, legacy_key),
            )

    @contextmanager
    def _transaction(self):
        if self._conn.in_transaction:
            raise RegistryError('nested registry transaction is not supported')
        self._conn.execute('BEGIN IMMEDIATE')
        try:
            yield
        except BaseException:
            self._conn.rollback()
            raise
        else:
            self._conn.commit()

    def _meta(self, key, default=''):
        row = self._conn.execute('SELECT value FROM registry_meta WHERE key = ?', (key,)).fetchone()
        return row['value'] if row else default

    def _set_meta(self, key, value):
        self._conn.execute(
            'INSERT INTO registry_meta (key, value) VALUES (?, ?) '
            'ON CONFLICT(key) DO UPDATE SET value = excluded.value',
            (key, str(value)),
        )

    # -- readers -----------------------------------------------------------

    def paper(self, paper_id):
        row = self._conn.execute(
            'SELECT * FROM paper_identity WHERE paper_id = ?', (paper_id,)
        ).fetchone()
        return dict(row) if row else None

    def _paper_row(self, paper_id):
        return self._conn.execute('SELECT * FROM papers WHERE paper_id = ?', (paper_id,)).fetchone()

    def _require_paper(self, paper_id):
        row = self._paper_row(paper_id)
        if row is None:
            raise RegistryError('unknown paper: %s' % (paper_id,))
        return row

    def identifiers(self, paper_id):
        rows = self._conn.execute(
            'SELECT * FROM identifiers WHERE paper_id = ? ORDER BY scheme, value', (paper_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def artifacts(self, paper_id):
        rows = self._conn.execute(
            'SELECT * FROM artifacts WHERE paper_id = ? ORDER BY added_at, artifact_id', (paper_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def artifact_by_source(self, source, content_sha256=None):
        if content_sha256 is None:
            row = self._conn.execute(
                'SELECT * FROM artifacts WHERE source = ? ORDER BY added_at, artifact_id LIMIT 1',
                (source,),
            ).fetchone()
        else:
            row = self._conn.execute(
                'SELECT * FROM artifacts WHERE source = ? AND content_sha256 = ?',
                (source, content_sha256),
            ).fetchone()
        return dict(row) if row else None

    def relations(self, paper_id=None):
        if paper_id is None:
            rows = self._conn.execute(
                'SELECT * FROM relations ORDER BY relation_id'
            ).fetchall()
        else:
            rows = self._conn.execute(
                'SELECT * FROM relations WHERE from_paper_id = ? OR to_paper_id = ? '
                'ORDER BY relation_id',
                (paper_id, paper_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def conflicts(self, status=None):
        if status is None:
            rows = self._conn.execute('SELECT * FROM conflicts ORDER BY conflict_id').fetchall()
        else:
            rows = self._conn.execute(
                'SELECT * FROM conflicts WHERE status = ? ORDER BY conflict_id', (status,)
            ).fetchall()
        return [dict(row) for row in rows]

    def legacy_rows(self, legacy_kind=None):
        if legacy_kind is None:
            rows = self._conn.execute(
                'SELECT * FROM legacy_ids ORDER BY legacy_kind, legacy_id'
            ).fetchall()
        else:
            rows = self._conn.execute(
                'SELECT * FROM legacy_ids WHERE legacy_kind = ? ORDER BY legacy_id',
                (legacy_kind,),
            ).fetchall()
        return [dict(row) for row in rows]

    # -- identifier resolution --------------------------------------------

    def owner(self, scheme, value):
        """Active owner of an identifier, or ``None``."""
        try:
            normalized, _raw = normalize_identifier(scheme, value)
        except RegistryError:
            return None
        row = self._conn.execute(
            'SELECT paper_id, status FROM identifiers WHERE scheme = ? AND value = ?',
            (scheme, normalized),
        ).fetchone()
        if row is None or row['status'] != 'active':
            return None
        return row['paper_id']

    def resolve(self, scheme, value):
        return self.owner(scheme, value)

    def paper_ids_for(self, identifiers):
        """All distinct active owners for a mapping of scheme -> value."""
        owners = {}
        for scheme in IDENTIFIER_SCHEMES:
            if scheme in identifiers:
                value = identifiers[scheme]
                if isinstance(value, (tuple, list)):
                    value = value[0]
                owner = self.owner(scheme, value)
                if owner:
                    owners[scheme] = owner
        return owners

    def resolve_alias(self, alias):
        """Resolve a historical alias such as ``doi:...`` or ``source:file.md``.

        Historical keys are compared with the accepted B2 normalization
        (case-folded basename) so an old ``source:pmid_16632979.md`` resolves
        while the stored file name is left untouched.
        """
        text = str(alias or '').strip()
        if not text:
            return None
        head, sep, tail = text.partition(':')
        head = head.strip().lower()
        if sep and head in IDENTIFIER_SCHEMES:
            owner = self.resolve(head, tail)
            if owner:
                return owner
            return self._legacy_paper('b3_paper_id', text)
        if sep and head == 'source':
            stem = stem_of(tail)
            owner = self._legacy_paper('source_stem', stem)
            if owner:
                return owner
            return self._legacy_paper('b3_paper_id', 'source:' + stem)
        return self._legacy_paper('source_stem', text)

    def resolve_many(self, aliases):
        """Return ``(paper_ids, unresolved_aliases)`` for a set of aliases."""
        found = set()
        unresolved = []
        for alias in aliases:
            if not str(alias or '').strip():
                continue
            paper_id = self.resolve_alias(alias)
            if paper_id:
                found.add(paper_id)
            else:
                unresolved.append(str(alias))
        return found, unresolved

    def _legacy_row(self, legacy_kind, legacy_id):
        key = normalize_legacy_id(legacy_kind, legacy_id)
        row = self._conn.execute(
            'SELECT * FROM legacy_ids WHERE legacy_kind = ? AND legacy_id = ?',
            (legacy_kind, key),
        ).fetchone()
        if row is not None:
            return row
        text = str(legacy_id or '').strip()
        for candidate in self._conn.execute(
            'SELECT * FROM legacy_ids WHERE legacy_kind = ?', (legacy_kind,)
        ).fetchall():
            try:
                raws = json.loads(candidate['raw_ids'] or '[]')
            except ValueError:
                raws = []
            if text in raws or text.casefold() in [str(item).casefold() for item in raws]:
                return candidate
        return None

    def _legacy_paper(self, legacy_kind, legacy_id):
        row = self._legacy_row(legacy_kind, legacy_id)
        if row is None or row['status'] != 'resolved':
            return None
        return row['paper_id']

    def _legacy_status(self, identity):
        """Status of the legacy row behind one B2-style identity string."""
        text = str(identity or '').strip()
        if not text:
            return 'unresolved'
        lowered = text.lower()
        if lowered.startswith(('doi:', 'pmid:', 'pmcid:')):
            row = self._legacy_row('b3_paper_id', text)
        elif lowered.startswith('source:'):
            row = self._legacy_row('source_stem', text[7:])
        else:
            row = self._legacy_row('source_stem', text)
        return row['status'] if row is not None else 'unresolved'

    def resolve_legacy(self, legacy_id, legacy_kind=None):
        """Resolve an old B3 paper id / source stem, reporting ambiguity."""
        text = str(legacy_id or '').strip()
        kinds = (legacy_kind,) if legacy_kind else LEGACY_KINDS
        rows = []
        for kind in kinds:
            row = self._legacy_row(kind, text)
            if row is not None:
                rows.append(row)
        if not rows:
            return {
                'legacy_id': text,
                'status': 'unresolved',
                'paper_id': None,
                'candidates': [],
            }

        papers = sorted({row['paper_id'] for row in rows if row['paper_id']})
        statuses = {row['status'] for row in rows}
        if len(papers) > 1 or 'ambiguous' in statuses:
            candidates = set(papers)
            for row in rows:
                recorded = _recorded_candidates(row['evidence'])
                candidates.update(recorded)
            return {
                'legacy_id': text,
                'status': 'ambiguous',
                'paper_id': None,
                'candidates': sorted(candidates),
            }
        if 'resolved' not in statuses:
            return {
                'legacy_id': text,
                'status': 'unresolved',
                'paper_id': None,
                'candidates': papers,
            }
        return {
            'legacy_id': text,
            'status': 'resolved',
            'paper_id': papers[0] if papers else None,
            'candidates': papers,
        }

    # -- writes ------------------------------------------------------------

    def _allocate_paper_id(self):
        seq = int(self._meta('paper_seq', '0'))
        while True:
            seq += 1
            candidate = '%s%06d' % (PAPER_ID_PREFIX, seq)
            if self._paper_row(candidate) is None:
                break
        self._set_meta('paper_seq', seq)
        return candidate

    def _add_conflict(self, kind, detail, now):
        fingerprint = hashlib.sha256(
            ('%s\x00%s' % (kind, _dump(detail))).encode('utf-8')
        ).hexdigest()
        row = self._conn.execute(
            'SELECT conflict_id FROM conflicts WHERE fingerprint = ?', (fingerprint,)
        ).fetchone()
        if row is not None:
            return row['conflict_id']
        cursor = self._conn.execute(
            'INSERT INTO conflicts (fingerprint, kind, detail, status, created_at) '
            "VALUES (?, ?, ?, 'pending', ?)",
            (fingerprint, kind, _dump(detail), now),
        )
        return cursor.lastrowid

    def add_conflict(self, kind, detail, now=None):
        with self._transaction():
            return self._add_conflict(kind, detail, now or utcnow())

    def resolve_conflict(self, conflict_id, resolution, now=None):
        now = now or utcnow()
        with self._transaction():
            row = self._conn.execute(
                'SELECT conflict_id FROM conflicts WHERE conflict_id = ?', (conflict_id,)
            ).fetchone()
            if row is None:
                raise RegistryError('unknown conflict: %s' % (conflict_id,))
            self._conn.execute(
                "UPDATE conflicts SET status = 'resolved', resolution = ?, resolved_at = ? "
                'WHERE conflict_id = ?',
                (_clean_text(resolution), now, conflict_id),
            )
        return {'conflict_id': conflict_id, 'status': 'resolved'}

    def _add_relation(self, from_paper_id, to_paper_id, kind, status, evidence, now):
        if kind not in RELATION_KINDS:
            raise RegistryError('unsupported relation kind: %r' % (kind,))
        if status not in ('pending', 'confirmed'):
            raise RegistryError('unsupported relation status: %r' % (status,))
        if from_paper_id == to_paper_id:
            raise RegistryError('self relations are not stored; the artifact role carries it')
        self._require_paper(from_paper_id)
        self._require_paper(to_paper_id)
        payload = _dump(evidence or {})
        row = self._conn.execute(
            'SELECT relation_id, status FROM relations WHERE from_paper_id = ? '
            'AND to_paper_id = ? AND kind = ?',
            (from_paper_id, to_paper_id, kind),
        ).fetchone()
        if row is not None:
            if status == 'confirmed' and row['status'] != 'confirmed':
                self._conn.execute(
                    "UPDATE relations SET status = 'confirmed', evidence = ? WHERE relation_id = ?",
                    (payload, row['relation_id']),
                )
            return row['relation_id']
        cursor = self._conn.execute(
            'INSERT INTO relations (from_paper_id, to_paper_id, kind, status, evidence, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            (from_paper_id, to_paper_id, kind, status, payload, now),
        )
        return cursor.lastrowid

    def add_relation(self, from_paper_id, to_paper_id, kind, status='pending', evidence=None,
                     now=None):
        with self._transaction():
            relation_id = self._add_relation(
                from_paper_id, to_paper_id, kind, status, evidence, now or utcnow()
            )
        return {'relation_id': relation_id, 'kind': kind, 'status': status}

    def _add_identifier(self, paper_id, scheme, value, source, evidence, now,
                        reactivate=False):
        normalized, raw = normalize_identifier(scheme, value)
        row = self._conn.execute(
            'SELECT * FROM identifiers WHERE scheme = ? AND value = ?', (scheme, normalized)
        ).fetchone()
        payload = _dump(evidence or {'source': source})
        if row is not None:
            if row['status'] == 'active':
                if row['paper_id'] == paper_id:
                    return {'status': 'existing', 'paper_id': paper_id, 'value': normalized}
                conflict_id = self._add_conflict(
                    'identifier_conflict',
                    {
                        'scheme': scheme,
                        'value': normalized,
                        'owner': row['paper_id'],
                        'attempted': paper_id,
                        'source': source,
                    },
                    now,
                )
                return {
                    'status': 'conflict',
                    'paper_id': row['paper_id'],
                    'value': normalized,
                    'conflict_id': conflict_id,
                }
            if row['paper_id'] != paper_id:
                conflict_id = self._add_conflict(
                    'identifier_conflict',
                    {
                        'scheme': scheme,
                        'value': normalized,
                        'owner': row['paper_id'],
                        'attempted': paper_id,
                        'source': source,
                        'detail': 'superseded alias claimed by another paper',
                    },
                    now,
                )
                return {
                    'status': 'conflict',
                    'paper_id': row['paper_id'],
                    'value': normalized,
                    'conflict_id': conflict_id,
                }
            if not reactivate:
                # A normal import never undoes a correction: the alias stays
                # superseded (kept for audit) until an explicit recovery.
                return {'status': 'superseded_kept', 'paper_id': paper_id, 'value': normalized}
            previous = {'superseded_at': row['superseded_at'], 'superseded_by': row['superseded_by']}
            payload = _dump(dict(evidence or {'source': source}, reactivated_from=previous))
            self._conn.execute(
                "UPDATE identifiers SET status = 'active', superseded_at = NULL, "
                "superseded_by = NULL, source = ?, evidence = ? "
                'WHERE scheme = ? AND value = ?',
                (source, payload, scheme, normalized),
            )
            return {'status': 'reactivated', 'paper_id': paper_id, 'value': normalized}
        self._conn.execute(
            'INSERT INTO identifiers (scheme, value, value_raw, paper_id, status, source, '
            "evidence, added_at) VALUES (?, ?, ?, ?, 'active', ?, ?, ?)",
            (scheme, normalized, raw, paper_id, source, payload, now),
        )
        return {'status': 'added', 'paper_id': paper_id, 'value': normalized}

    def add_identifier(self, paper_id, scheme, value, source='', evidence=None, now=None,
                       reactivate=False):
        """Attach an identifier alias on an existing paper.

        A superseded alias is left untouched unless ``reactivate=True`` is
        passed explicitly; that explicit recovery records the previous
        supersede event in the alias evidence.
        """
        self._require_paper(paper_id)
        with self._transaction():
            return self._add_identifier(
                paper_id, scheme, value, source, evidence, now or utcnow(), reactivate=reactivate
            )

    def correct_identifier(self, scheme, old_value, new_value, paper_id=None, source='',
                           evidence=None, now=None):
        """Correct one alias in place; the old value is retained as superseded."""
        now = now or utcnow()
        old_norm, _ = normalize_identifier(scheme, old_value)
        new_norm, new_raw = normalize_identifier(scheme, new_value)
        with self._transaction():
            row = self._conn.execute(
                'SELECT * FROM identifiers WHERE scheme = ? AND value = ?', (scheme, old_norm)
            ).fetchone()
            if row is None:
                raise RegistryError('unknown identifier: %s:%s' % (scheme, old_norm))
            owner = row['paper_id']
            if paper_id and paper_id != owner:
                raise RegistryError(
                    'identifier %s:%s belongs to %s, not %s'
                    % (scheme, old_norm, owner, paper_id)
                )
            if row['status'] != 'active':
                raise RegistryError('identifier %s:%s is not active' % (scheme, old_norm))
            if old_norm == new_norm:
                return {'status': 'noop', 'paper_id': owner, 'value': new_norm}
            existing = self._conn.execute(
                'SELECT * FROM identifiers WHERE scheme = ? AND value = ?', (scheme, new_norm)
            ).fetchone()
            if existing is not None and existing['paper_id'] != owner:
                # Checked for *any* status, before the old row is touched: a
                # superseded alias still belongs to its paper and must not be
                # silently re-homed by a correction.
                raise RegistryError(
                    'new identifier %s:%s already belongs to %s'
                    % (scheme, new_norm, existing['paper_id'])
                )
            payload = _dump(evidence or {'source': source, 'correction_of': old_norm})
            self._conn.execute(
                "UPDATE identifiers SET status = 'superseded', superseded_at = ?, "
                'superseded_by = ? WHERE scheme = ? AND value = ?',
                (now, new_norm, scheme, old_norm),
            )
            if existing is not None:
                self._conn.execute(
                    "UPDATE identifiers SET status = 'active', superseded_at = NULL, "
                    'superseded_by = NULL, source = ?, evidence = ?, added_at = ? '
                    'WHERE scheme = ? AND value = ?',
                    (source, payload, now, scheme, new_norm),
                )
            else:
                self._conn.execute(
                    'INSERT INTO identifiers (scheme, value, value_raw, paper_id, status, '
                    "source, evidence, added_at) VALUES (?, ?, ?, ?, 'active', ?, ?, ?)",
                    (scheme, new_norm, new_raw, owner, source, payload, now),
                )
        return {
            'status': 'corrected',
            'paper_id': owner,
            'old_value': old_norm,
            'value': new_norm,
        }

    # -- registration ------------------------------------------------------

    def _create_paper(self, record, now, identity_status='verified'):
        paper_id = self._allocate_paper_id()
        metadata = record['metadata']
        provenance = {
            field: record['provenance'].get(field, record['source']) for field in metadata
        }
        self._conn.execute(
            'INSERT INTO papers (paper_id, title, title_hash, authors, year, journal, '
            'publication_type, license, source, version, is_supplement, duplicate_of, '
            'content_sha256, metadata_updated_at, metadata_provenance, identity_status, '
            'created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)',
            (
                paper_id,
                metadata.get('title', ''),
                title_fingerprint(metadata.get('title')),
                metadata.get('authors', ''),
                metadata.get('year', ''),
                metadata.get('journal', ''),
                metadata.get('publication_type', ''),
                metadata.get('license', ''),
                record['source'],
                metadata.get('version', ''),
                1 if record['supplement_of'] else 0,
                record['content_sha256'],
                now,
                _dump(provenance),
                identity_status,
                now,
            ),
        )
        title_hash = title_fingerprint(metadata.get('title'))
        if title_hash:
            others = self._conn.execute(
                'SELECT paper_id, source FROM papers WHERE title_hash = ? AND paper_id != ?',
                (title_hash, paper_id),
            ).fetchall()
            if others:
                # Same normalized title only makes a candidate; never merge on it.
                self._add_conflict(
                    'title_candidate_collision',
                    {
                        'paper_id': paper_id,
                        'source': record['source'],
                        'title_hash': title_hash,
                        'candidates': [dict(row) for row in others],
                    },
                    now,
                )
        return paper_id

    def _refresh_primary_artifact(self, paper_id):
        rows = self._conn.execute(
            'SELECT source, content_sha256, role FROM artifacts WHERE paper_id = ?', (paper_id,)
        ).fetchall()
        if not rows:
            return
        primary = min(
            rows,
            key=lambda row: (row['role'] == 'supplement', source_sort_key(row['source'])),
        )
        self._conn.execute(
            'UPDATE papers SET source = ?, content_sha256 = ? WHERE paper_id = ?',
            (primary['source'], primary['content_sha256'], paper_id),
        )

    def _attach_artifact(self, paper_id, record, role, now):
        artifact_id = artifact_id_for(record['source'], record['content_sha256'])
        row = self._conn.execute(
            'SELECT * FROM artifacts WHERE artifact_id = ?', (artifact_id,)
        ).fetchone()
        if row is not None:
            if row['paper_id'] != paper_id:
                self._add_conflict(
                    'artifact_reuse_mismatch',
                    {
                        'artifact_id': artifact_id,
                        'source': record['source'],
                        'owner': row['paper_id'],
                        'attempted': paper_id,
                    },
                    now,
                )
            return artifact_id, False, row['role']
        evidence = {
            'sha256': record['content_sha256'],
            'declared_duplicate_of': record['duplicate_of'],
            'declared_supplement_of': record['supplement_of'],
        }
        self._conn.execute(
            'INSERT INTO artifacts (artifact_id, paper_id, source, content_sha256, role, '
            'is_supplement, evidence, added_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                artifact_id,
                paper_id,
                record['source'],
                record['content_sha256'],
                role,
                1 if role == 'supplement' else 0,
                _dump(evidence),
                now,
            ),
        )
        return artifact_id, True, role

    def _merge_metadata(self, paper_id, record, now):
        row = self._paper_row(paper_id)
        updates = {}
        provenance = json.loads(row['metadata_provenance'] or '{}')
        for field in METADATA_FIELDS:
            incoming = record['metadata'].get(field)
            if not incoming:
                continue
            current = row[field]
            if not current:
                updates[field] = incoming
                provenance[field] = record['provenance'].get(field, record['source'])
            elif not _same_text(current, incoming):
                self._add_conflict(
                    'metadata_drift',
                    {
                        'paper_id': paper_id,
                        'field': field,
                        'existing': current,
                        'incoming': incoming,
                        'source': record['source'],
                    },
                    now,
                )
        if not updates:
            return
        assignments = ', '.join('%s = ?' % field for field in updates)
        self._conn.execute(
            'UPDATE papers SET %s, metadata_updated_at = ?, metadata_provenance = ? '
            'WHERE paper_id = ?' % assignments,
            (*updates.values(), now, _dump(provenance), paper_id),
        )

    def _resolve_reference(self, reference, now):
        kind = reference['kind']
        value = reference['value']
        if kind == 'paper':
            if self._paper_row(value) is None:
                raise RegistryError('unknown reference paper: %s' % value)
            return value
        if kind == 'source':
            artifact = self.artifact_by_source(value)
            if artifact is not None:
                return artifact['paper_id']
            paper_id = self._legacy_paper('source_stem', stem_of(value))
            if paper_id:
                return paper_id
            raise RegistryError('unknown reference source: %s' % value)
        paper_id = self.resolve(kind, value)
        if paper_id:
            return paper_id
        paper_id = self._legacy_paper('b3_paper_id', '%s:%s' % (kind, value))
        if paper_id:
            return paper_id
        raise RegistryError('unknown reference identifier: %s:%s' % (kind, value))

    def _register_conflict(self, record, anchors, existing_artifact, issues, now, target=None):
        """Hold a contradictory registration aside without mixing identities.

        No new alias and no owner metadata is written on this path: the
        artifact is kept with an isolated pending conflict, and the
        disagreement lists every anchor that was compared.  A declared
        ``duplicate_of`` still records the shared content as a pending relation
        to its target, because that statement is about bytes, not identity.
        """
        conflict_record = dict(record)
        conflict_record['supplement_of'] = None
        if existing_artifact is not None:
            paper_id = existing_artifact['paper_id']
            created = False
            artifact_id = existing_artifact['artifact_id']
            role = existing_artifact['role']
        else:
            paper_id = self._create_paper(conflict_record, now, identity_status='conflict')
            created = True
            artifact_id, _new, role = self._attach_artifact(
                paper_id, conflict_record, 'main', now
            )
            self._merge_metadata(paper_id, conflict_record, now)
            self._refresh_primary_artifact(paper_id)
        relation_ids = []
        if record['duplicate_of'] and target is not None and target != paper_id:
            relation_id = self._add_relation(
                paper_id,
                target,
                'exact_duplicate',
                'pending',
                {
                    'source': record['source'],
                    'content_sha256': record['content_sha256'],
                    'why_pending': 'identifiers disagree; sha256 alone is not identity',
                },
                now,
            )
            self._add_conflict(
                'content_duplicate_identifier_mismatch',
                {
                    'source': record['source'],
                    'content_sha256': record['content_sha256'],
                    'paper_id': paper_id,
                    'candidate': target,
                    'relation_id': relation_id,
                },
                now,
            )
            relation_ids.append(relation_id)
        conflict_id = self._add_conflict(
            'identifier_conflict',
            {
                'source': record['source'],
                'content_sha256': record['content_sha256'],
                'identifiers': {
                    scheme: value for scheme, (value, _raw) in record['identifiers'].items()
                },
                'owners': anchors,
                'issues': issues,
            },
            now,
        )
        self._conn.execute(
            "UPDATE papers SET identity_status = 'conflict' WHERE paper_id = ?", (paper_id,)
        )
        return {
            'status': 'conflict',
            'paper_id': paper_id,
            'artifact_id': artifact_id,
            'role': role,
            'created': created,
            'conflicts': [conflict_id],
            'relations': relation_ids,
            'issues': issues,
        }

    def _isolated_hold(self, paper_id):
        """True for a paper created only to hold one contradictory artifact.

        Such a paper owns no identifier alias, so it cannot disagree with the
        record it already holds: replaying that record must compare the incoming
        identifiers with the real owners again instead of reporting the hold
        itself as a second disagreement.
        """
        row = self._conn.execute(
            'SELECT identity_status FROM papers WHERE paper_id = ?', (paper_id,)
        ).fetchone()
        if row is None or row['identity_status'] != 'conflict':
            return False
        return (
            self._conn.execute(
                'SELECT 1 FROM identifiers WHERE paper_id = ? LIMIT 1', (paper_id,)
            ).fetchone()
            is None
        )

    def _identity_issues(self, record, owners, existing_artifact, target):
        """Compare every identity anchor before choosing an owner or writing."""
        issues = []
        anchors = set(owners.values())
        if existing_artifact is not None and not self._isolated_hold(
            existing_artifact['paper_id']
        ):
            anchors.add(existing_artifact['paper_id'])
        if record['supplement_of'] and target is not None:
            anchors.add(target)
        if len(anchors) > 1:
            issues.append({'kind': 'owner_disagreement', 'owners': sorted(anchors)})
            return issues
        if not anchors:
            # Nothing is registered for these identifiers yet.  A declared
            # byte-duplicate then keeps its own paper plus a pending relation.
            return issues
        # ``duplicate_of`` is a statement about content, not about identity:
        # the file still has to agree with the identifiers of the paper it
        # would join, otherwise it is held aside instead of being merged.
        owner = next(iter(anchors))
        for scheme, (value, _raw) in record['identifiers'].items():
            if owners.get(scheme) == owner:
                continue
            current = self._conn.execute(
                'SELECT paper_id, status FROM identifiers WHERE scheme = ? AND value = ?',
                (scheme, value),
            ).fetchone()
            if (
                current is not None
                and current['status'] == 'superseded'
                and current['paper_id'] == owner
            ):
                # The value was corrected on this very paper earlier; replaying
                # the old record must not undo the correction or force a merge.
                continue
            others = [
                row['value']
                for row in self._conn.execute(
                    "SELECT value FROM identifiers WHERE paper_id = ? AND scheme = ? "
                    "AND status = 'active'",
                    (owner, scheme),
                )
            ]
            if any(item != value for item in others):
                issues.append(
                    {
                        'kind': 'identifier_mismatch',
                        'scheme': scheme,
                        'incoming': value,
                        'owner': owner,
                        'existing': sorted(others),
                    }
                )
        for scheme, value in (
            (scheme, value) for scheme, (value, _raw) in record['identifiers'].items()
        ):
            row = self._conn.execute(
                'SELECT paper_id, status FROM identifiers WHERE scheme = ? AND value = ?',
                (scheme, value),
            ).fetchone()
            if row is not None and row['status'] == 'superseded' and row['paper_id'] != owner:
                issues.append(
                    {
                        'kind': 'superseded_alias_conflict',
                        'scheme': scheme,
                        'value': value,
                        'owner': row['paper_id'],
                        'attempted': owner,
                    }
                )
        return issues

    def register(self, record, now=None):
        """Register one artifact record; safe to replay with the same input."""
        now = now or utcnow()
        rec = validate_record(record)
        with self._transaction():
            ident_values = {
                scheme: value for scheme, (value, _raw) in rec['identifiers'].items()
            }
            owners = self.paper_ids_for(ident_values)
            existing_artifact = self.artifact_by_source(
                rec['source'], rec['content_sha256']
            )
            target = None
            if rec['supplement_of']:
                target = self._resolve_reference(rec['supplement_of'], now)
            elif rec['duplicate_of']:
                target = self._resolve_reference(rec['duplicate_of'], now)

            issues = self._identity_issues(rec, owners, existing_artifact, target)
            if issues:
                anchors = set(owners.values())
                if existing_artifact is not None and not self._isolated_hold(
                    existing_artifact['paper_id']
                ):
                    anchors.add(existing_artifact['paper_id'])
                if target and rec['supplement_of']:
                    anchors.add(target)
                return self._register_conflict(
                    rec, sorted(anchors), existing_artifact, issues, now, target=target
                )

            anchors = set(owners.values())
            if existing_artifact is not None:
                anchors.add(existing_artifact['paper_id'])
            if rec['supplement_of']:
                anchors.add(target)
            distinct = sorted(anchors)

            created = False
            if existing_artifact is not None:
                paper_id = existing_artifact['paper_id']
                artifact_id = existing_artifact['artifact_id']
                role = existing_artifact['role']
                status = 'existing'
            elif rec['supplement_of']:
                paper_id = target
                artifact_id, _new, role = self._attach_artifact(
                    paper_id, rec, 'supplement', now
                )
                status = 'attached'
            elif rec['duplicate_of']:
                paper_id = distinct[0] if distinct else None
                if paper_id is None:
                    paper_id = self._create_paper(rec, now)
                    created = True
                artifact_id, _new, role = self._attach_artifact(paper_id, rec, 'duplicate', now)
                status = 'created' if created else 'attached'
            else:
                if distinct:
                    paper_id = distinct[0]
                    role = 'version'
                    status = 'attached'
                else:
                    paper_id = self._create_paper(rec, now)
                    created = True
                    role = 'main'
                    status = 'created'
                artifact_id, _new, role = self._attach_artifact(paper_id, rec, role, now)
                status = 'created' if created else status

            conflict_ids = []
            for scheme in IDENTIFIER_SCHEMES:
                if scheme not in rec['identifiers']:
                    continue
                value, _raw = rec['identifiers'][scheme]
                result = self._add_identifier(
                    paper_id,
                    scheme,
                    value,
                    rec['source'],
                    {'source': rec['source'], 'provenance': rec['provenance'].get(scheme, '')},
                    now,
                )
                if result['status'] == 'conflict':
                    conflict_ids.append(result['conflict_id'])

            self._merge_metadata(paper_id, rec, now)
            if created or role != 'supplement':
                self._refresh_primary_artifact(paper_id)

            if conflict_ids:
                self._conn.execute(
                    "UPDATE papers SET identity_status = 'conflict' WHERE paper_id = ?",
                    (paper_id,),
                )
                status = 'conflict'

            if rec['duplicate_of'] and target != paper_id:
                # Byte-identical content with diverging identifiers: keep the
                # papers apart and leave the duplicate relation pending.
                relation_id = self._add_relation(
                    paper_id,
                    target,
                    'exact_duplicate',
                    'pending',
                    {
                        'source': rec['source'],
                        'content_sha256': rec['content_sha256'],
                        'why_pending': 'identifiers disagree; sha256 alone is not identity',
                    },
                    now,
                )
                self._add_conflict(
                    'content_duplicate_identifier_mismatch',
                    {
                        'source': rec['source'],
                        'content_sha256': rec['content_sha256'],
                        'paper_id': paper_id,
                        'candidate': target,
                        'relation_id': relation_id,
                    },
                    now,
                )
                if not conflict_ids:
                    self._conn.execute(
                        "UPDATE papers SET identity_status = 'conflict' WHERE paper_id = ?",
                        (paper_id,),
                    )

            relation_ids = self._apply_relations(paper_id, rec, now)
            legacy_outcomes = {}
            for legacy_kind in LEGACY_KINDS:
                legacy_value = rec['legacy'].get(legacy_kind)
                if legacy_value:
                    legacy_outcomes[legacy_kind] = self._set_legacy(
                        legacy_kind, legacy_value, paper_id, rec['source'], None, now
                    )
            return {
                'status': status,
                'paper_id': paper_id,
                'artifact_id': artifact_id,
                'role': role,
                'created': created,
                'conflicts': conflict_ids,
                'relations': relation_ids,
                'legacy': legacy_outcomes,
            }

    def _apply_relations(self, paper_id, record, now):
        relation_ids = []
        for relation in record['relations']:
            target = self._resolve_reference(relation['target'], now)
            status = 'confirmed' if relation['evidence'] else 'pending'
            relation_id = self._add_relation(
                paper_id,
                target,
                relation['kind'],
                status,
                {'source': record['source'], 'evidence': relation['evidence']},
                now,
            )
            relation_ids.append(relation_id)
        return relation_ids

    def merge_papers(self, duplicate, canonical, evidence=None, now=None):
        """Explicit human merge of two papers that were kept apart."""
        now = now or utcnow()
        with self._transaction():
            dup = self._paper_row(duplicate)
            if dup is None:
                raise RegistryError('unknown paper: %s' % (duplicate,))
            self._require_paper(canonical)
            if duplicate == canonical:
                raise RegistryError('cannot merge a paper with itself')
            if dup['identity_status'] == 'merged':
                raise RegistryError('paper %s is already merged' % (duplicate,))
            self._conn.execute(
                'UPDATE identifiers SET paper_id = ? WHERE paper_id = ?', (canonical, duplicate)
            )
            rows = self._conn.execute(
                'SELECT artifact_id FROM artifacts WHERE paper_id = ?', (duplicate,)
            ).fetchall()
            for row in rows:
                try:
                    self._conn.execute(
                        'UPDATE artifacts SET paper_id = ? WHERE artifact_id = ?',
                        (canonical, row['artifact_id']),
                    )
                except sqlite3.IntegrityError:
                    self._conn.execute(
                        'DELETE FROM artifacts WHERE artifact_id = ?', (row['artifact_id'],)
                    )
            self._conn.execute(
                'UPDATE legacy_ids SET paper_id = ? WHERE paper_id = ?', (canonical, duplicate)
            )
            ambiguous = self._conn.execute(
                "SELECT legacy_kind, legacy_id, evidence FROM legacy_ids WHERE status = 'ambiguous'"
            ).fetchall()
            for row in ambiguous:
                candidates = set(_recorded_candidates(row['evidence']))
                if not candidates or not candidates <= {duplicate, canonical}:
                    continue
                try:
                    payload = json.loads(row['evidence'] or '{}')
                except ValueError:
                    payload = {}
                self._conn.execute(
                    "UPDATE legacy_ids SET paper_id = ?, status = 'resolved', evidence = ? "
                    'WHERE legacy_kind = ? AND legacy_id = ?',
                    (
                        canonical,
                        _dump(
                            {
                                'resolved_by_merge': [duplicate, canonical],
                                'previous_candidates': sorted(candidates),
                            }
                        ),
                        row['legacy_kind'],
                        row['legacy_id'],
                    ),
                )
                conflict_id = payload.get('conflict_id') if isinstance(payload, dict) else None
                if conflict_id:
                    self._conn.execute(
                        "UPDATE conflicts SET status = 'resolved', resolution = ?, resolved_at = ? "
                        "WHERE conflict_id = ? AND status = 'pending'",
                        ('merged %s into %s' % (duplicate, canonical), now, conflict_id),
                    )
            self._add_relation(
                canonical,
                duplicate,
                'exact_duplicate',
                'confirmed',
                {'merged': True, 'evidence': evidence or ''},
                now,
            )
            self._conn.execute(
                "UPDATE papers SET duplicate_of = ?, identity_status = 'merged', "
                'metadata_updated_at = ? WHERE paper_id = ?',
                (canonical, now, duplicate),
            )
            self._refresh_primary_artifact(canonical)
        return {'status': 'merged', 'paper_id': canonical, 'merged_from': duplicate}

    # -- legacy ids --------------------------------------------------------

    def set_legacy(self, legacy_kind, legacy_id, paper_id, artifact_source='', evidence=None,
                   now=None):
        """Record an old-id -> paper mapping; disagreement becomes ambiguous."""
        with self._transaction():
            return self._set_legacy(
                legacy_kind, legacy_id, paper_id, artifact_source, evidence, now or utcnow()
            )

    def _upsert_legacy_conflict(self, legacy_kind, legacy_key, candidates, now, conflict_id=None):
        detail = {
            'legacy_kind': legacy_kind,
            'legacy_id': legacy_key,
            'candidates': sorted(candidates),
        }
        fingerprint = hashlib.sha256(
            ('%s\x00%s' % ('legacy_id_ambiguous', _dump(detail))).encode('utf-8')
        ).hexdigest()
        if conflict_id:
            row = self._conn.execute(
                'SELECT conflict_id, status FROM conflicts WHERE conflict_id = ?', (conflict_id,)
            ).fetchone()
            if row is not None and row['status'] == 'pending':
                self._conn.execute(
                    'UPDATE conflicts SET fingerprint = ?, detail = ? WHERE conflict_id = ?',
                    (fingerprint, _dump(detail), conflict_id),
                )
                return conflict_id
        return self._add_conflict('legacy_id_ambiguous', detail, now)

    def _set_legacy(self, legacy_kind, legacy_id, paper_id, artifact_source, evidence, now):
        if legacy_kind not in LEGACY_KINDS:
            raise RegistryError('unsupported legacy kind: %r' % (legacy_kind,))
        raw = str(legacy_id or '').strip()
        if not raw:
            raise RegistryError('legacy_id is required')
        key = normalize_legacy_id(legacy_kind, raw)
        if not key:
            raise RegistryError('legacy_id is required')
        self._require_paper(paper_id)
        row = self._legacy_row(legacy_kind, raw)
        raws = sorted(set(_raw_legacy_ids(row)) | {raw}) if row is not None else [raw]
        if row is None:
            self._conn.execute(
                'INSERT INTO legacy_ids (legacy_kind, legacy_id, raw_ids, paper_id, '
                'artifact_source, status, resolution, evidence, created_at) '
                "VALUES (?, ?, ?, ?, ?, 'resolved', '', ?, ?)",
                (
                    legacy_kind,
                    key,
                    _dump(raws),
                    paper_id,
                    artifact_source,
                    _dump(evidence or {'artifact_source': artifact_source}),
                    now,
                ),
            )
            return {'status': 'resolved', 'paper_id': paper_id}
        payload = {}
        try:
            payload = json.loads(row['evidence'] or '{}')
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        conflict_id = payload.get('conflict_id')
        if row['status'] == 'ambiguous':
            # Ambiguity is sticky: normal registration may only add candidates,
            # never turn the row back into a resolved mapping.
            candidates = set(_recorded_candidates(row['evidence']))
            if row['paper_id']:
                candidates.add(row['paper_id'])
            candidates.add(paper_id)
            conflict_id = self._upsert_legacy_conflict(
                legacy_kind, key, candidates, now, conflict_id
            )
            self._conn.execute(
                "UPDATE legacy_ids SET raw_ids = ?, paper_id = NULL, status = 'ambiguous', "
                'artifact_source = ?, evidence = ? WHERE legacy_kind = ? AND legacy_id = ?',
                (
                    _dump(raws),
                    artifact_source,
                    _dump({'candidates': sorted(candidates), 'conflict_id': conflict_id}),
                    legacy_kind,
                    row['legacy_id'],
                ),
            )
            return {
                'status': 'ambiguous',
                'conflict_id': conflict_id,
                'candidates': sorted(candidates),
            }
        if row['paper_id'] != paper_id:
            candidates = {row['paper_id'], paper_id} | set(_recorded_candidates(row['evidence']))
            candidates.discard(None)
            conflict_id = self._upsert_legacy_conflict(
                legacy_kind, key, candidates, now, conflict_id
            )
            self._conn.execute(
                "UPDATE legacy_ids SET raw_ids = ?, paper_id = NULL, status = 'ambiguous', "
                'artifact_source = ?, evidence = ? WHERE legacy_kind = ? AND legacy_id = ?',
                (
                    _dump(raws),
                    artifact_source,
                    _dump({'candidates': sorted(candidates), 'conflict_id': conflict_id}),
                    legacy_kind,
                    row['legacy_id'],
                ),
            )
            return {
                'status': 'ambiguous',
                'conflict_id': conflict_id,
                'candidates': sorted(candidates),
            }
        self._conn.execute(
            'UPDATE legacy_ids SET raw_ids = ?, artifact_source = ? '
            'WHERE legacy_kind = ? AND legacy_id = ?',
            (_dump(raws), artifact_source, legacy_kind, row['legacy_id']),
        )
        return {'status': 'resolved', 'paper_id': paper_id}

    def adjudicate_legacy(self, legacy_kind, legacy_id, paper_id, resolution, now=None):
        """Explicitly settle an ambiguous old id; the only way to clear it."""
        now = now or utcnow()
        if legacy_kind not in LEGACY_KINDS:
            raise RegistryError('unsupported legacy kind: %r' % (legacy_kind,))
        if not _clean_text(resolution):
            raise RegistryError('an explicit resolution note is required')
        with self._transaction():
            self._require_paper(paper_id)
            row = self._legacy_row(legacy_kind, legacy_id)
            if row is None:
                raise RegistryError(
                    'unknown legacy id: %s:%s' % (legacy_kind, legacy_id)
                )
            candidates = set(_recorded_candidates(row['evidence']))
            if row['paper_id']:
                candidates.add(row['paper_id'])
            if paper_id not in candidates:
                raise RegistryError(
                    'paper %s is not a candidate for %s:%s (candidates: %s)'
                    % (paper_id, legacy_kind, legacy_id, ', '.join(sorted(candidates)))
                )
            self._conn.execute(
                "UPDATE legacy_ids SET paper_id = ?, status = 'resolved', resolution = ?, "
                'evidence = ? WHERE legacy_kind = ? AND legacy_id = ?',
                (
                    paper_id,
                    _clean_text(resolution),
                    _dump(
                        {
                            'resolved_by': 'adjudication',
                            'resolution': _clean_text(resolution),
                            'previous_candidates': sorted(candidates),
                        }
                    ),
                    legacy_kind,
                    row['legacy_id'],
                ),
            )
            for conflict in self._conn.execute(
                "SELECT conflict_id, detail FROM conflicts "
                "WHERE kind = 'legacy_id_ambiguous' AND status = 'pending'"
            ).fetchall():
                try:
                    detail = json.loads(conflict['detail'] or '{}')
                except ValueError:
                    continue
                if (
                    detail.get('legacy_kind') == legacy_kind
                    and detail.get('legacy_id') == row['legacy_id']
                ):
                    self._conn.execute(
                        "UPDATE conflicts SET status = 'resolved', resolution = ?, "
                        'resolved_at = ? WHERE conflict_id = ?',
                        (_clean_text(resolution), now, conflict['conflict_id']),
                    )
        return {'status': 'resolved', 'paper_id': paper_id}

    # -- reporting ---------------------------------------------------------

    def stats(self):
        def count(sql, params=()):
            return self._conn.execute(sql, params).fetchone()[0]

        artifacts_by_role = {
            row['role']: row['n']
            for row in self._conn.execute(
                'SELECT role, COUNT(*) AS n FROM artifacts GROUP BY role ORDER BY role'
            )
        }
        identifiers_by_scheme = {
            row['scheme']: row['n']
            for row in self._conn.execute(
                'SELECT scheme, COUNT(*) AS n FROM identifiers GROUP BY scheme ORDER BY scheme'
            )
        }
        legacy_by_kind_status = {
            '%s/%s' % (row['legacy_kind'], row['status']): row['n']
            for row in self._conn.execute(
                'SELECT legacy_kind, status, COUNT(*) AS n FROM legacy_ids '
                'GROUP BY legacy_kind, status ORDER BY legacy_kind, status'
            )
        }
        relations_by_kind_status = {
            '%s/%s' % (row['kind'], row['status']): row['n']
            for row in self._conn.execute(
                'SELECT kind, status, COUNT(*) AS n FROM relations '
                'GROUP BY kind, status ORDER BY kind, status'
            )
        }
        conflicts_by_kind_status = {
            '%s/%s' % (row['kind'], row['status']): row['n']
            for row in self._conn.execute(
                'SELECT kind, status, COUNT(*) AS n FROM conflicts '
                'GROUP BY kind, status ORDER BY kind, status'
            )
        }
        return {
            'schema_version': self._meta('schema_version'),
            'papers': count('SELECT COUNT(*) FROM papers'),
            'papers_by_status': {
                row['identity_status']: row['n']
                for row in self._conn.execute(
                    'SELECT identity_status, COUNT(*) AS n FROM papers '
                    'GROUP BY identity_status ORDER BY identity_status'
                )
            },
            'papers_with_multiple_artifacts': count(
                'SELECT COUNT(*) FROM (SELECT paper_id FROM artifacts '
                'GROUP BY paper_id HAVING COUNT(*) > 1)'
            ),
            'papers_without_identifiers': count(
                'SELECT COUNT(*) FROM papers p WHERE NOT EXISTS '
                '(SELECT 1 FROM identifiers i WHERE i.paper_id = p.paper_id)'
            ),
            'content_sha256_shared_by_several_papers': count(
                'SELECT COUNT(*) FROM (SELECT content_sha256 FROM artifacts '
                "WHERE content_sha256 != '' GROUP BY content_sha256 "
                'HAVING COUNT(DISTINCT paper_id) > 1)'
            ),
            'artifacts': count('SELECT COUNT(*) FROM artifacts'),
            'artifacts_by_role': artifacts_by_role,
            'identifiers': count('SELECT COUNT(*) FROM identifiers'),
            'identifiers_by_scheme': identifiers_by_scheme,
            'identifiers_active': count(
                "SELECT COUNT(*) FROM identifiers WHERE status = 'active'"
            ),
            'relations': count('SELECT COUNT(*) FROM relations'),
            'relations_by_kind_status': relations_by_kind_status,
            'conflicts': count('SELECT COUNT(*) FROM conflicts'),
            'conflicts_by_kind_status': conflicts_by_kind_status,
            'legacy_ids': count('SELECT COUNT(*) FROM legacy_ids'),
            'legacy_by_kind_status': legacy_by_kind_status,
        }

    def multi_source_papers(self):
        rows = self._conn.execute(
            'SELECT a.paper_id, COUNT(*) AS n FROM artifacts a '
            'GROUP BY a.paper_id HAVING COUNT(*) > 1 ORDER BY a.paper_id'
        ).fetchall()
        output = []
        for row in rows:
            output.append(
                {
                    'paper_id': row['paper_id'],
                    'artifacts': [
                        {
                            'source': artifact['source'],
                            'role': artifact['role'],
                            'content_sha256': artifact['content_sha256'],
                        }
                        for artifact in self.artifacts(row['paper_id'])
                    ],
                }
            )
        return output

    def verify(self, inventory=None, qrels=None, b2_cards=None, b2_rankings=None):
        """Read-only verification of coverage, aliases, conflicts and old ids."""
        checks = []

        def check(name, status, detail):
            checks.append({'name': name, 'status': status, 'detail': detail})

        papers = self._conn.execute('SELECT * FROM papers').fetchall()
        orphans = [
            row['paper_id']
            for row in papers
            if row['identity_status'] != 'merged'
            and self._conn.execute(
                'SELECT 1 FROM artifacts WHERE paper_id = ?', (row['paper_id'],)
            ).fetchone()
            is None
        ]
        check(
            'papers_have_artifacts',
            'pass' if not orphans else 'fail',
            {'orphans': orphans[:20], 'total_orphans': len(orphans)},
        )

        active_conflicts = self._conn.execute(
            "SELECT COUNT(*) FROM conflicts WHERE status = 'pending'"
        ).fetchone()[0]
        check(
            'pending_conflicts_visible',
            'pass',
            {'pending': active_conflicts, 'note': 'conflicts stay open for human review'},
        )

        shared = self._conn.execute(
            'SELECT content_sha256 FROM artifacts WHERE content_sha256 != \'\' '
            'GROUP BY content_sha256 HAVING COUNT(DISTINCT paper_id) > 1'
        ).fetchall()
        # Each byte-identical group must be linked to its recorded duplicate
        # relation; a star around the canonical file is enough, so this is a
        # connectivity check rather than a pairwise one.
        unlinked = []
        for row in shared:
            members = {
                item['paper_id']
                for item in self._conn.execute(
                    'SELECT DISTINCT paper_id FROM artifacts WHERE content_sha256 = ?',
                    (row['content_sha256'],),
                )
            }
            edges = [
                (item['from_paper_id'], item['to_paper_id'])
                for item in self._conn.execute(
                    "SELECT from_paper_id, to_paper_id FROM relations "
                    "WHERE kind = 'exact_duplicate'"
                )
            ]
            reachable = {sorted(members)[0]}
            grew = True
            while grew:
                grew = False
                for left, right in edges:
                    if left in reachable and right in members and right not in reachable:
                        reachable.add(right)
                        grew = True
                    if right in reachable and left in members and left not in reachable:
                        reachable.add(left)
                        grew = True
            if reachable != members:
                unlinked.append(
                    {
                        'content_sha256': row['content_sha256'],
                        'papers': sorted(members),
                        'unreached': sorted(members - reachable),
                    }
                )
        check(
            'shared_content_is_flagged',
            'pass' if not unlinked else 'fail',
            {
                'shared_groups': len(shared),
                'unlinked_groups': unlinked[:20],
                'total_unlinked_groups': len(unlinked),
            },
        )

        if inventory is not None:
            missing = []
            mismatched = []
            extra = []
            inventory_sources = set()
            for source, sha in inventory:
                inventory_sources.add(source)
                artifact = self.artifact_by_source(source)
                if artifact is None:
                    missing.append(source)
                    continue
                if sha and artifact['content_sha256'] != sha:
                    mismatched.append(
                        {
                            'source': source,
                            'inventory_sha256': sha,
                            'registry_sha256': artifact['content_sha256'],
                        }
                    )
            for row in self._conn.execute('SELECT source FROM artifacts'):
                if row['source'] not in inventory_sources:
                    extra.append(row['source'])
            check(
                'inventory_coverage',
                'pass' if not missing and not mismatched and not extra else 'fail',
                {
                    'inventory_entries': len(inventory_sources),
                    'missing': missing[:20],
                    'total_missing': len(missing),
                    'sha_mismatch': mismatched[:20],
                    'total_sha_mismatch': len(mismatched),
                    'not_in_inventory': extra[:20],
                    'total_not_in_inventory': len(extra),
                },
            )

        legacy = self.legacy_rows()
        resolved = [row for row in legacy if row['status'] == 'resolved']
        ambiguous = [row for row in legacy if row['status'] == 'ambiguous']
        unresolved = [row for row in legacy if row['status'] == 'unresolved']
        check(
            'legacy_resolution',
            'pass' if not ambiguous and not unresolved else 'fail',
            {
                'resolved': len(resolved),
                'ambiguous': [row['legacy_id'] for row in ambiguous][:20],
                'total_ambiguous': len(ambiguous),
                'unresolved': [row['legacy_id'] for row in unresolved][:20],
                'total_unresolved': len(unresolved),
            },
        )

        conflict_rows = self._conn.execute(
            "SELECT conflict_id, detail, status FROM conflicts "
            "WHERE kind = 'legacy_id_ambiguous' AND status = 'pending'"
        ).fetchall()
        inconsistent = []
        for row in conflict_rows:
            try:
                detail = json.loads(row['detail'] or '{}')
            except ValueError:
                continue
            legacy_kind = detail.get('legacy_kind')
            legacy_id = detail.get('legacy_id')
            if not legacy_kind or not legacy_id:
                continue
            legacy_row = self._legacy_row(legacy_kind, legacy_id)
            if legacy_row is None or legacy_row['status'] != 'ambiguous':
                inconsistent.append(
                    {
                        'conflict_id': row['conflict_id'],
                        'legacy_kind': legacy_kind,
                        'legacy_id': legacy_id,
                        'row_status': legacy_row['status'] if legacy_row else None,
                    }
                )
        check(
            'legacy_conflict_consistency',
            'pass' if not inconsistent else 'fail',
            {
                'pending_legacy_conflicts': len(conflict_rows),
                'inconsistent': inconsistent[:20],
                'total_inconsistent': len(inconsistent),
            },
        )

        if b2_cards is not None or b2_rankings is not None:
            identities = _b2_identities(b2_cards, b2_rankings)
            failing_b2 = []
            for label, identity in identities:
                paper = self.resolve_alias(identity)
                if paper is None:
                    failing_b2.append(
                        {
                            'label': label,
                            'identity': identity,
                            'status': self._legacy_status(identity),
                        }
                    )
            check(
                'b2_identity',
                'pass' if not failing_b2 else 'fail',
                {
                    'identities': len(identities),
                    'failing': failing_b2[:20],
                    'total_failing': len(failing_b2),
                },
            )

        if qrels is not None:
            judgements = _qrels_judgements(qrels)
            failing = []
            for judgement in judgements:
                aliases = _judgement_aliases(judgement)
                found, unresolved_aliases = self.resolve_many(aliases)
                if len(found) != 1 or unresolved_aliases:
                    failing.append(
                        {
                            'row_id': judgement.get('row_id'),
                            'source_id': judgement.get('source_id'),
                            'resolved': sorted(found),
                            'unresolved': unresolved_aliases,
                        }
                    )
            check(
                'qrels_identity',
                'pass' if not failing else 'fail',
                {
                    'judgements': len(judgements),
                    'failing': failing[:20],
                    'total_failing': len(failing),
                },
            )

        return {
            'checks': checks,
            'stats': self.stats(),
            'status': 'PASS' if all(item['status'] == 'pass' for item in checks) else 'FAIL',
        }


def _b2_identities(cards_path=None, rankings_path=None):
    """Collect B2-style identities from cards and saved rankings (read-only)."""
    identities = []
    seen = set()

    def add(label, identity):
        identity = str(identity or '').strip()
        if not identity or identity in seen:
            return
        seen.add(identity)
        identities.append((str(label or ''), identity))

    if cards_path is not None:
        payload = _load_json(cards_path)
        if not isinstance(payload, list):
            raise RegistryError('B2 cards must be a list')
        for card in payload:
            add('card:%s' % (card.get('source') if isinstance(card, dict) else ''), b2_identity(card))
    if rankings_path is not None:
        payload = _load_json(rankings_path)
        records = payload.get('records') if isinstance(payload, dict) else payload
        for record in records or ():
            if not isinstance(record, dict):
                continue
            query_id = record.get('query_id')
            evaluation = record.get('evaluation') or {}
            for item in evaluation.get('retrieved_rankings') or ():
                if not isinstance(item, dict):
                    continue
                identity = item.get('pre_f06_identity') or b2_identity(item)
                add(query_id, identity)
    return identities


def _qrels_judgements(qrels):
    """Accept a qrels path, a list of query rows or a list of judgements."""
    if isinstance(qrels, (str, Path)):
        rows = []
        with open(qrels, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return _qrels_judgements(rows)
    judgements = []
    for row in qrels:
        if not isinstance(row, dict):
            raise RegistryError('qrels entries must be objects')
        nested = row.get('judgements')
        if isinstance(nested, list):
            judgements.extend(item for item in nested if isinstance(item, dict))
        else:
            judgements.append(row)
    return judgements


def _judgement_aliases(judgement):
    aliases = []
    doi = str(judgement.get('doi') or '').strip()
    if doi:
        aliases.append('doi:' + doi)
    for key in ('source_id', 'source_file', 'source_path'):
        value = str(judgement.get(key) or '').strip()
        if value:
            aliases.append('source:' + value)
    return aliases


# ---------------------------------------------------------------------------
# corpus adapter (real inputs -> registration records)
# ---------------------------------------------------------------------------


def load_inventory(path):
    """Load an inventory of ``(source, sha256)`` pairs in any accepted shape."""
    payload = _load_json(path)
    if isinstance(payload, dict):
        items = list(payload.items())
    else:
        items = []
        for entry in payload:
            if isinstance(entry, dict):
                source = entry.get('source') or entry.get('path') or entry.get('filename')
                sha = entry.get('sha256') or entry.get('content_sha256') or ''
                items.append((source, sha))
            else:
                items.append((entry[0], entry[1] if len(entry) > 1 else ''))
    pairs = []
    for source, sha in items:
        source = str(source or '').strip()
        if not source:
            raise RegistryError('inventory entry without a source')
        pairs.append((source, str(sha or '').strip().lower()))
    return pairs


def load_cards(path):
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise RegistryError('cards file must contain a list')
    cards = {}
    for card in payload:
        if not isinstance(card, dict) or not card.get('source'):
            raise RegistryError('card without a source')
        source = str(card['source'])
        if source in cards:
            raise RegistryError('duplicate card source: %s' % source)
        cards[source] = card
    return cards


def _manifest_entries(path):
    payload = _load_json(path)
    files = payload.get('files') if isinstance(payload, dict) else payload
    if isinstance(files, dict):
        return {str(source): entry for source, entry in files.items()}
    entries = {}
    for entry in files or ():
        if not isinstance(entry, dict):
            raise RegistryError('manifest entries must be objects')
        source = entry.get('source') or entry.get('path') or entry.get('filename')
        if not source:
            raise RegistryError('manifest entry without a source')
        entries[str(source)] = entry
    return entries


def _load_pmc_index(canonical_path):
    """PMCIDs are not part of ``document_metadata._CANONICAL_FIELDS``."""
    by_pmid = {}
    by_doi = {}
    with open(canonical_path, 'r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pmcid = normalize_pmcid(row.get('PMC'))
            if not pmcid:
                continue
            pmid = normalize_pmid(row.get('PMID'))
            doi = normalize_doi(row.get('DOI'))
            if pmid:
                by_pmid.setdefault(pmid, set()).add(pmcid)
            if doi:
                by_doi.setdefault(doi, set()).add(pmcid)
    return (
        {key: next(iter(value)) for key, value in by_pmid.items() if len(value) == 1},
        {key: next(iter(value)) for key, value in by_doi.items() if len(value) == 1},
    )


def _folder_doi_index(mapping_path):
    payload = _load_json(mapping_path)
    files = payload.get('files') if isinstance(payload, dict) else payload
    by_folder = {}
    for entry in files or ():
        folder = str(entry.get('folder') or '').strip()
        doi = normalize_doi(entry.get('doi'))
        if folder and doi and folder not in by_folder:
            by_folder[folder] = doi
    return by_folder


def _legacy_doi_index(mapping_path, mapping_pmid_path=None):
    """stem -> DOI as ``build_index`` saw it (the old B3 paper_id source)."""
    by_stem = {}
    for path in (mapping_path, mapping_pmid_path):
        if not path:
            continue
        payload = _load_json(path)
        files = payload.get('files') if isinstance(payload, dict) else payload
        for entry in files or ():
            filename = str(entry.get('filename') or '')
            doi = normalize_doi(entry.get('doi'))
            stem = Path(filename).stem
            if stem and doi and stem not in by_stem:
                by_stem[stem] = doi
    return by_stem


def build_corpus_records(cards_path, manifest_path, inventory_path, canonical_path,
                         mapping_path=None, mapping_pmid_path=None, supplement_map=None):
    """Turn the saved cards/manifest/inventory into registration records.

    ``supplement_map`` maps a source stem to ``{'kind': 'doi'|'folder',
    'value': ...}``; a ``folder`` target is resolved through ``mapping_path``.
    An entry may also state ``role``: only ``supplement`` (the default) turns
    the file into a supplement artifact of that paper; ``version`` merely
    supplies the bibliographic DOI as an identity alias, so the file joins the
    paper it belongs to as an alternate version. Identity lookup and artifact
    role are therefore decided separately.
    Nothing is fetched: only the saved metadata and identity files are read.
    """
    try:
        import document_metadata as dm
    except ImportError:  # pragma: no cover - import path is set by callers
        dm = None

    cards = load_cards(cards_path)
    entries = _manifest_entries(manifest_path)
    inventory = load_inventory(inventory_path)
    inventory_map = dict(inventory)
    if len(inventory_map) != len(inventory):
        raise RegistryError('inventory repeats a source')
    canonical = dm.load_canonical(canonical_path) if dm else {'by_pmid': {}, 'by_doi': {}}
    pmc_by_pmid, pmc_by_doi = _load_pmc_index(canonical_path)
    folder_doi = _folder_doi_index(mapping_path) if mapping_path else {}
    legacy_doi = _legacy_doi_index(mapping_path, mapping_pmid_path)
    supplements = {}
    for stem, reference in (supplement_map or {}).items():
        kind = str(reference.get('kind') or '').strip().lower()
        value = str(reference.get('value') or '').strip()
        role = str(reference.get('role') or 'supplement').strip().lower()
        if role not in ('supplement', 'version'):
            raise RegistryError('unsupported artifact role for %r: %r' % (stem, role))
        if kind == 'folder':
            doi = folder_doi.get(value)
            if not doi:
                raise RegistryError('supplement folder %r has no DOI in mapping' % value)
            supplements[stem] = {'kind': 'doi', 'value': doi, 'role': role}
        elif kind == 'doi':
            supplements[stem] = {
                'kind': 'doi',
                'value': normalize_doi(value),
                'role': role,
            }
        else:
            raise RegistryError('unsupported supplement target kind: %r' % kind)

    sources = sorted(set(inventory_map) | set(cards) | set(entries), key=source_sort_key)
    records = []
    for source in sources:
        card = cards.get(source) or {}
        entry = entries.get(source) or {}
        metadata_in = card.get('metadata') or {}
        identifiers = {}
        match = PMID_CARD.fullmatch(source)
        if match:
            identifiers['pmid'] = match.group(1)
        doi = normalize_doi(card.get('doi'))
        if doi:
            identifiers['doi'] = doi
        canonical_row = None
        if 'pmid' in identifiers:
            canonical_row = canonical['by_pmid'].get(identifiers['pmid'])
        if canonical_row is None and 'doi' in identifiers:
            canonical_row = canonical['by_doi'].get(identifiers['doi'])
        if canonical_row is not None:
            if 'pmid' not in identifiers:
                pmid = normalize_pmid(canonical_row.get('PMID'))
                if pmid:
                    identifiers['pmid'] = pmid
            if 'doi' not in identifiers:
                canonical_doi = normalize_doi(canonical_row.get('DOI'))
                if canonical_doi:
                    identifiers['doi'] = canonical_doi
        pmcid = ''
        if 'pmid' in identifiers:
            pmcid = pmc_by_pmid.get(identifiers['pmid'], '')
        if not pmcid and 'doi' in identifiers:
            pmcid = pmc_by_doi.get(identifiers['doi'], '')
        if pmcid:
            identifiers['pmcid'] = pmcid

        sha = inventory_map.get(source) or entry.get('sha256') or ''
        if sha and entry.get('sha256') and sha != str(entry['sha256']):
            raise RegistryError('sha256 disagreement for %s' % source)
        if not sha:
            raise RegistryError('no content_sha256 bound to %s' % source)

        metadata = {
            'title': card.get('title'),
            'authors': card.get('authors'),
            'year': card.get('year'),
            'journal': card.get('journal'),
            'publication_type': card.get('pub_type'),
            'license': card.get('license'),
        }
        provenance_in = metadata_in.get('provenance') or {}
        provenance = {}
        for field in METADATA_FIELDS:
            value = provenance_in.get(field)
            if isinstance(value, (list, tuple)):
                value = ', '.join(str(item) for item in value)
            if value:
                provenance[field] = str(value)

        stem = stem_of(source)
        record = {
            'source': source,
            'content_sha256': sha,
            'identifiers': identifiers,
            'metadata': {key: value for key, value in metadata.items() if value},
            'provenance': provenance,
        }
        if entry.get('dup_of'):
            record['duplicate_of'] = {'kind': 'source', 'value': str(entry['dup_of'])}
        supplement = supplements.get(stem)
        if supplement:
            if supplement['role'] == 'supplement':
                record['supplement_of'] = {
                    'kind': supplement['kind'],
                    'value': supplement['value'],
                }
            elif 'doi' not in identifiers:
                # Version role: the DOI only supplies identity, never a role.
                identifiers['doi'] = supplement['value']
        legacy = {}
        if card.get('paper_id'):
            # Only chunked sources ever received an old B3 paper_id.
            legacy['b3_paper_id'] = str(card['paper_id'])
        legacy['source_stem'] = stem
        record['legacy'] = legacy
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_init(args):
    with PaperRegistry(args.db, create=True) as registry:
        print(_dump(registry.stats()))
    return 0


def _cmd_status(args):
    with PaperRegistry(args.db) as registry:
        payload = {
            'stats': registry.stats(),
            'multi_source_papers': registry.multi_source_papers(),
            'pending_conflicts': registry.conflicts(status='pending'),
        }
    print(_dump(payload))
    return 0


def _cmd_resolve(args):
    with PaperRegistry(args.db) as registry:
        payload = []
        for alias in args.alias:
            paper_ids, unresolved = registry.resolve_many([alias])
            payload.append(
                {
                    'alias': alias,
                    'paper_ids': sorted(paper_ids),
                    'status': 'resolved' if len(paper_ids) == 1 else (
                        'ambiguous' if paper_ids else 'unresolved'
                    ),
                    'unresolved': unresolved,
                }
            )
    print(_dump(payload))
    return 0


def _cmd_import_jsonl(args):
    with open(args.records, 'r', encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    results = []
    with PaperRegistry(args.db, create=True) as registry:
        for record in records:
            result = registry.register(record)
            result['source'] = record.get('source')
            results.append(result)
            if result.get('issues'):
                result.setdefault('legacy_warnings', []).extend(
                    issue.get('kind') for issue in result['issues']
                )
    print(_dump({'registered': len(results), 'results': results}))
    return 0


def _cmd_import_corpus(args):
    supplement_map = _load_json(args.supplement_map) if args.supplement_map else None
    records = build_corpus_records(
        args.cards,
        args.manifest,
        args.inventory,
        args.canonical_csv,
        mapping_path=args.mapping_json,
        mapping_pmid_path=args.mapping_pmid_json,
        supplement_map=supplement_map,
    )
    order = sorted(
        records,
        key=lambda record: (
            bool(record.get('supplement_of')) or bool(record.get('duplicate_of')),
            source_sort_key(record['source']),
        ),
    )
    results = []
    with PaperRegistry(args.db, create=True) as registry:
        for record in order:
            result = registry.register(record)
            result['source'] = record['source']
            results.append(result)
            legacy = result.get('legacy') or {}
            ambiguous = sorted(
                kind for kind, outcome in legacy.items() if outcome.get('status') == 'ambiguous'
            )
            if ambiguous:
                result.setdefault('legacy_warnings', []).extend(ambiguous)
            if result.get('issues'):
                result.setdefault('legacy_warnings', []).extend(
                    issue.get('kind') for issue in result['issues']
                )
        payload = {
            'registered_records': len(results),
            'statuses': {},
            'stats': registry.stats(),
            'multi_source_papers': registry.multi_source_papers(),
            'pending_conflicts': registry.conflicts(status='pending'),
            'legacy_map': registry.legacy_rows(),
        }
        for result in results:
            payload['statuses'][result['status']] = payload['statuses'].get(result['status'], 0) + 1
        if args.report:
            Path(args.report).write_text(_dump(payload), encoding='utf-8')
        if args.legacy_out:
            Path(args.legacy_out).write_text(
                _dump(payload['legacy_map']), encoding='utf-8'
            )
    print(_dump({key: payload[key] for key in ('registered_records', 'statuses', 'stats')}))
    return 0


def _cmd_verify(args):
    inventory = load_inventory(args.inventory) if args.inventory else None
    qrels = []
    if args.qrels:
        with open(args.qrels, 'r', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                qrels.extend(row.get('judgements') or [])
    with PaperRegistry(args.db) as registry:
        report = registry.verify(
            inventory=inventory,
            qrels=qrels or None,
            b2_cards=args.b2_cards,
            b2_rankings=args.b2_rankings,
        )
        if args.out:
            Path(args.out).write_text(_dump(report), encoding='utf-8')
    print(_dump(report))
    return 0 if report['status'] == 'PASS' else 1


def _cmd_adjudicate_legacy(args):
    with PaperRegistry(args.db) as registry:
        outcome = registry.adjudicate_legacy(
            args.kind, args.legacy_id, args.paper_id, args.resolution
        )
    print(_dump(outcome))
    return 0


def _cmd_repair_legacy(args):
    with PaperRegistry(args.db) as registry:
        report = registry.repair_legacy_spellings(dry_run=args.dry_run)
    print(_dump(report))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog='paper_registry',
        description='Immutable paper ids with additive identifier aliases (P09/C1).',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    init_parser = subparsers.add_parser('init', help='create an empty registry')
    init_parser.add_argument('--db', required=True)
    init_parser.set_defaults(func=_cmd_init)

    status_parser = subparsers.add_parser('status', help='print registry statistics')
    status_parser.add_argument('--db', required=True)
    status_parser.set_defaults(func=_cmd_status)

    resolve_parser = subparsers.add_parser('resolve', help='resolve old/legacy aliases')
    resolve_parser.add_argument('--db', required=True)
    resolve_parser.add_argument('--alias', action='append', required=True)
    resolve_parser.set_defaults(func=_cmd_resolve)

    jsonl_parser = subparsers.add_parser('import-jsonl', help='register JSONL records')
    jsonl_parser.add_argument('--db', required=True)
    jsonl_parser.add_argument('--records', required=True)
    jsonl_parser.set_defaults(func=_cmd_import_jsonl)

    corpus_parser = subparsers.add_parser(
        'import-corpus', help='register the saved corpus cards/manifest/inventory'
    )
    corpus_parser.add_argument('--db', required=True)
    corpus_parser.add_argument('--cards', required=True)
    corpus_parser.add_argument('--manifest', required=True)
    corpus_parser.add_argument('--inventory', required=True)
    corpus_parser.add_argument('--canonical-csv', required=True)
    corpus_parser.add_argument('--mapping-json')
    corpus_parser.add_argument('--mapping-pmid-json')
    corpus_parser.add_argument('--supplement-map')
    corpus_parser.add_argument('--report')
    corpus_parser.add_argument('--legacy-out')
    corpus_parser.set_defaults(func=_cmd_import_corpus)

    verify_parser = subparsers.add_parser('verify', help='read-only coverage checks')
    verify_parser.add_argument('--db', required=True)
    verify_parser.add_argument('--inventory')
    verify_parser.add_argument('--qrels')
    verify_parser.add_argument('--b2-cards')
    verify_parser.add_argument('--b2-rankings')
    verify_parser.add_argument('--out')
    verify_parser.set_defaults(func=_cmd_verify)

    adjudicate_parser = subparsers.add_parser(
        'adjudicate-legacy', help='explicitly settle an ambiguous old id'
    )
    adjudicate_parser.add_argument('--db', required=True)
    adjudicate_parser.add_argument('--kind', required=True, choices=list(LEGACY_KINDS))
    adjudicate_parser.add_argument('--legacy-id', required=True)
    adjudicate_parser.add_argument('--paper-id', required=True)
    adjudicate_parser.add_argument('--resolution', required=True)
    adjudicate_parser.set_defaults(func=_cmd_adjudicate_legacy)

    repair_parser = subparsers.add_parser(
        'repair-legacy',
        help='restore legacy source spellings folded away by the v1 migration',
    )
    repair_parser.add_argument('--db', required=True)
    repair_parser.add_argument(
        '--dry-run', action='store_true', help='report without writing'
    )
    repair_parser.set_defaults(func=_cmd_repair_legacy)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
