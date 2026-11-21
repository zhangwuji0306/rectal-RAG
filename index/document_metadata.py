"""Deterministic metadata association for B5 document cards (F1 repair).

Card fields are filled only from project metadata, never from titles, full
text or passage contents:

* ``直肠癌文献爬取/索引信息.csv`` -- canonical rows keyed by exact PMID/DOI;
* ``.claude/tmp/mapping.json`` -- MAC sidecar keyed by exact DOI or file name;
* the card's own already-stored paired-index values (``card`` source).

Identity rules
--------------
1. ``PMID_<digits>.md`` is an exact PMID identity; PMID always wins.
2. Otherwise a non-empty normalized DOI is an exact DOI identity.
3. Otherwise an exact ``mapping.json`` file-name stem may supply fields, but
   never a canonical identity on its own.
4. Duplicate keys are merged field by field only while all non-empty values
   agree.  A conflicting field is left empty and reported; no value is
   guessed.
5. A DOI that maps to several different PMIDs is ambiguous and is not used as
   a canonical key.
6. A PMID card whose DOI resolves to another PMID keeps the PMID row and
   reports ``pmid_doi_mismatch``.  A DOI card whose card DOI and mapping DOI
   disagree is not resolved to a canonical row.

Field priority (see ``CARD_FIELD_PRIORITY``) is fixed per field.  ``keywords``
stays empty because no free-text keyword field exists in the canonical
metadata; MeSH terms are stored separately as ``mesh_terms`` and are never
written into ``keywords``.

Abstract integrity
------------------
``abstract_truncated`` and the provenance ``truncated`` flag describe only the
6,000-character cut applied by this module.  They say nothing about the
completeness of the upstream value.  In particular ``.claude/tmp/analyze.py``
stores each MAC sidecar abstract as ``(row["Abstract Note"] or "")[:200]``, so a
mapping abstract is a preview with an unknown original length.  The abstract
provenance therefore also records ``source_integrity``:

* ``canonical`` -- canonical CSV value used in full; ``complete`` means no
  upstream preview is known and this step did not cut it;
* ``mapping`` -- known 200-character preview, completeness unknown
  (``source_preview_limit`` is 200);
* ``card`` -- value stored on the old card; the carried ``truncated`` flag is
  the old stored flag and no completeness is claimed;
* missing abstracts have no integrity record.

The 199/200-character length of a mapping preview is not evidence that the
original abstract was exactly that long, nor that it was longer; no text is
inferred or fetched.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ABSTRACT_LIMIT = 6000
# ``.claude/tmp/analyze.py`` writes the MAC sidecar abstract as
# ``(row["Abstract Note"] or "")[:200]``.  The sidecar does not record that cap,
# so a mapping abstract is a 200-character preview of unknown completeness.
MAPPING_PREVIEW_LIMIT = 200
PMID_CARD = re.compile(r'^PMID_(\d+)\.md$', re.IGNORECASE)
_PMID_PREFIX = re.compile(r'^PMID[\s:_-]*', re.IGNORECASE)
_DOI_PREFIX = re.compile(r'^(?:doi\s*:\s*|https?://(?:dx\.)?doi\.org/)', re.IGNORECASE)

CARD_FIELDS = ('title', 'abstract', 'keywords', 'pub_type', 'year', 'journal')

# Ordered fallbacks per card field.  ``canonical`` means the CSV row, ``mapping``
# the MAC sidecar entry and ``card`` the value already stored with the card.
CARD_FIELD_PRIORITY = {
    'title': ('card', 'canonical', 'mapping'),
    'year': ('card', 'canonical', 'mapping'),
    'journal': ('card', 'canonical', 'mapping'),
    'abstract': ('canonical', 'mapping', 'card'),
    'pub_type': ('canonical', 'card'),
    'keywords': ('card',),
    'mesh_terms': ('canonical',),
}

_CANONICAL_FIELDS = ('PMID', 'Title', 'Abstract', 'Journal', 'Year', 'PubType', 'MeSH', 'DOI')
_MAPPING_FIELDS = ('title', 'abstract', 'journal', 'year', 'doi')


def _clean(value):
    return str(value or '').strip()


def _text(value):
    """Collapse metadata whitespace (CSV cells may contain embedded newlines)."""
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def normalize_pmid(value):
    """Return a bare PMID for an exact PMID value, else ``''``."""
    text = _clean(value)
    text = _PMID_PREFIX.sub('', text)
    return text if re.fullmatch(r'\d+', text) else ''


def normalize_doi(value):
    """Lower-case DOI with URL/``doi:`` prefixes and trailing punctuation removed."""
    text = _clean(value).strip('<>').strip()
    while True:
        stripped = _DOI_PREFIX.sub('', text).strip()
        if stripped == text:
            break
        text = stripped
    text = text.lower().rstrip('.;, ')
    return text


def _merge_rows(rows, fields, source, key):
    """Field-wise merge of duplicate rows; conflicting values become empty."""
    merged, conflicts = {}, []
    for field in fields:
        if field == 'PMID':
            clean = normalize_pmid
        elif field.lower() == 'doi':
            clean = normalize_doi
        else:
            clean = _text
        values = sorted({clean(row.get(field)) for row in rows if clean(row.get(field))})
        if len(values) <= 1:
            merged[field] = values[0] if values else ''
        else:
            merged[field] = ''
            conflicts.append({'source': source, 'key': key, 'field': field,
                              'reason': 'conflicting_values', 'n_values': len(values)})
    return merged, conflicts


def build_canonical_index(rows):
    """Index canonical CSV rows by exact PMID and exact normalized DOI."""
    rows = list(rows)
    pmid_groups, doi_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        pmid = normalize_pmid(row.get('PMID'))
        if pmid:
            pmid_groups[pmid].append(row)
        doi = normalize_doi(row.get('DOI'))
        if doi:
            doi_groups[doi].append(row)
    by_pmid, by_doi, conflicts = {}, {}, []
    ambiguous_dois = set()
    duplicate_pmid_rows = sum(len(group) - 1 for group in pmid_groups.values())
    for pmid, group in sorted(pmid_groups.items()):
        merged, found = _merge_rows(group, _CANONICAL_FIELDS, 'canonical-csv', 'pmid:' + pmid)
        merged['PMID'] = pmid
        by_pmid[pmid] = merged
        conflicts.extend(found)
    duplicate_doi_rows = 0
    for doi, group in sorted(doi_groups.items()):
        pmids = {normalize_pmid(row.get('PMID')) for row in group} - {''}
        if len(pmids) > 1:
            # The same DOI claiming different papers is mutually exclusive.
            ambiguous_dois.add(doi)
            conflicts.append({'source': 'canonical-csv', 'key': 'doi:' + doi, 'field': 'PMID',
                              'reason': 'doi_maps_to_multiple_pmids', 'n_values': len(pmids)})
            continue
        duplicate_doi_rows += len(group) - 1
        merged, found = _merge_rows(group, _CANONICAL_FIELDS, 'canonical-csv', 'doi:' + doi)
        merged['PMID'] = next(iter(pmids), '')
        by_doi[doi] = merged
        conflicts.extend(found)
    return {'by_pmid': by_pmid, 'by_doi': by_doi, 'conflicts': conflicts, 'rows': len(rows),
            'ambiguous_dois': ambiguous_dois,
            'duplicate_rows': {'pmid': duplicate_pmid_rows, 'doi': duplicate_doi_rows}}


def build_mapping_index(entries):
    """Index ``mapping.json`` entries by exact DOI and exact file-name stem."""
    entries = list(entries)
    doi_groups, stem_groups = defaultdict(list), defaultdict(list)
    for entry in entries:
        doi = normalize_doi(entry.get('doi'))
        if doi:
            doi_groups[doi].append(entry)
        stem = Path(str(entry.get('filename') or '')).stem
        if stem:
            stem_groups[stem].append(entry)
    by_doi, by_stem, conflicts = {}, {}, []
    duplicate_doi_rows = 0
    for doi, group in sorted(doi_groups.items()):
        duplicate_doi_rows += len(group) - 1
        merged, found = _merge_rows(group, _MAPPING_FIELDS, 'mapping-json', 'doi:' + doi)
        merged['doi'] = doi
        by_doi[doi] = merged
        conflicts.extend(found)
    for stem, group in sorted(stem_groups.items()):
        merged, found = _merge_rows(group, _MAPPING_FIELDS, 'mapping-json', 'file:' + stem)
        by_stem[stem] = merged
        conflicts.extend(found)
    return {'by_doi': by_doi, 'by_stem': by_stem, 'conflicts': conflicts, 'entries': len(entries),
            'duplicate_rows': {'doi': duplicate_doi_rows}}


def load_canonical(path):
    with open(path, encoding='utf-8-sig', newline='') as handle:
        index = build_canonical_index(csv.DictReader(handle))
    index['path'] = str(path)
    return index


def load_mapping(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    index = build_mapping_index(data.get('files', []))
    index['path'] = str(path)
    return index


def _mapping_entry(card, mapping, conflicts, use_doi=True):
    """Return ``(entry, key)`` for the exact card DOI or exact file-name stem."""
    if not mapping:
        return None, ''
    card_doi = normalize_doi(card.get('doi'))
    doi_entry = mapping['by_doi'].get(card_doi) if card_doi and use_doi else None
    stem = Path(str(card.get('source') or '')).stem
    stem_entry = mapping['by_stem'].get(stem)
    if doi_entry and stem_entry:
        card_doi_key = normalize_doi(doi_entry.get('doi'))
        stem_doi_key = normalize_doi(stem_entry.get('doi'))
        if card_doi_key and stem_doi_key and card_doi_key != stem_doi_key:
            conflicts.append({'kind': 'doi_mismatch', 'field': 'doi',
                              'detail': 'card-doi and mapping-file do not agree'})
            return None, ''
    entry = doi_entry or stem_entry
    if not entry:
        return None, ''
    if doi_entry:
        return entry, 'doi:' + card_doi
    return entry, 'file:' + stem


def _pmid_mapping_entry(card, mapping, canonical, pmid, conflicts):
    """Sidecar entry for a PMID card, rejecting DOI links to another PMID.

    A PMID card may use a mapping entry only when its DOI is absent or agrees
    with the PMID row's DOI (or the card DOI when the row has no DOI), and the
    DOI does not resolve to a different canonical PMID.  The exact file-name
    stem is preferred because it cannot drift to another DOI.
    """
    if not mapping:
        return None, ''
    stem = Path(str(card.get('source') or '')).stem
    card_doi = normalize_doi(card.get('doi'))
    row = canonical['by_pmid'].get(pmid)
    row_doi = normalize_doi(row.get('DOI')) if row else ''
    reference = row_doi or card_doi
    candidates = [(mapping['by_stem'].get(stem), 'file:' + stem)]
    if card_doi:
        candidates.append((mapping['by_doi'].get(card_doi), 'doi:' + card_doi))
    for entry, key in candidates:
        if not entry:
            continue
        entry_doi = normalize_doi(entry.get('doi'))
        problem = ''
        if entry_doi:
            owner_row = canonical['by_doi'].get(entry_doi)
            owner = normalize_pmid(owner_row.get('PMID')) if owner_row else ''
            if owner and owner != pmid:
                problem = 'mapping DOI belongs to another PMID'
            elif not reference:
                problem = 'mapping DOI has no PMID/card DOI to verify'
            elif entry_doi != reference:
                problem = 'mapping DOI disagrees with PMID row/card DOI'
        if problem:
            conflicts.append({'kind': 'pmid_mapping_mismatch', 'field': 'doi',
                              'detail': problem})
            continue
        return entry, key
    return None, ''


def _resolve(card, canonical, mapping):
    """Resolve one card to ``(canonical row or None, mapping entry or None, metadata)``."""
    conflicts = []
    card_doi = normalize_doi(card.get('doi'))
    pmid_match = PMID_CARD.fullmatch(str(card.get('source') or ''))
    if pmid_match:
        kind, value = 'pmid', pmid_match.group(1)
    elif card_doi:
        kind, value = 'doi', card_doi
    else:
        kind, value = 'unmatched', ''

    row, row_key = None, ''
    if kind == 'pmid':
        row = canonical['by_pmid'].get(value)
        if row:
            row_key = 'pmid:' + value
        row_doi = normalize_doi(row.get('DOI')) if row else ''
        if card_doi:
            owner_row = canonical['by_doi'].get(card_doi)
            owner = normalize_pmid(owner_row.get('PMID')) if owner_row else ''
            if (owner and owner != value) or (row_doi and row_doi != card_doi):
                conflicts.append({'kind': 'pmid_doi_mismatch', 'field': 'doi',
                                  'detail': 'PMID row kept; card DOI points elsewhere'})
        entry, mapping_key = _pmid_mapping_entry(card, mapping, canonical, value, conflicts)
    else:
        entry, mapping_key = _mapping_entry(
            card, mapping, conflicts,
            use_doi=not (kind == 'doi' and value in canonical.get('ambiguous_dois', ())))
        if kind == 'doi' and value in canonical.get('ambiguous_dois', ()):
            conflicts.append({'kind': 'ambiguous_doi', 'field': 'doi',
                              'detail': 'DOI maps to several canonical PMIDs; canonical row not used'})
        if kind == 'doi':
            entry_doi = normalize_doi(entry.get('doi')) if entry else ''
            if entry_doi and entry_doi != card_doi:
                # The exact card DOI still wins; the contradictory sidecar
                # entry cannot supply fields.
                conflicts.append({'kind': 'doi_mismatch', 'field': 'doi',
                                  'detail': 'card DOI and mapping entry do not agree'})
                entry, mapping_key = None, ''
            lookup = card_doi or entry_doi
            row = canonical['by_doi'].get(lookup) if lookup else None
            if row:
                row_key = 'doi:' + lookup
        else:
            # No PMID/DOI identity: an exact mapping file-name match may still
            # supply sidecar fields, but never a canonical row.
            kind = 'filename' if entry else 'unmatched'
            value = Path(str(card.get('source') or '')).stem if entry else ''

    metadata = {'identity': {'kind': kind, 'value': value,
                             'canonical_key': row_key, 'mapping_key': mapping_key},
                'provenance': {}, 'conflicts': conflicts}
    return row, entry, metadata


def _pick(card, field, row, entry, metadata):
    """Return the first non-empty value according to the field priority."""
    card_key = 'card:' + str(card.get('source') or '')
    providers = {
        'card': lambda: _clean(card.get(field)),
        'canonical': lambda: _text(row.get(_CANONICAL_FIELD.get(field, ''))) if row else '',
        'mapping': lambda: _text(entry.get(_MAPPING_FIELD.get(field, ''))) if entry else '',
    }
    keys = {'card': card_key,
            'canonical': metadata['identity']['canonical_key'],
            'mapping': metadata['identity']['mapping_key']}
    for source in CARD_FIELD_PRIORITY[field]:
        value = providers[source]()
        if value:
            return value, source, keys[source]
    return '', 'missing', ''


_CANONICAL_FIELD = {'title': 'Title', 'abstract': 'Abstract', 'pub_type': 'PubType',
                    'year': 'Year', 'journal': 'Journal', 'mesh_terms': 'MeSH'}
_MAPPING_FIELD = {'title': 'title', 'abstract': 'abstract', 'year': 'year', 'journal': 'journal'}


def _provenance(source, key, source_files, truncated=None):
    record = {'source': source,
              'file': source_files.get(source, '') if source != 'missing' else '',
              'key': key}
    if truncated is not None:
        record['truncated'] = truncated
    return record


def _abstract_provenance(source, key, source_files, truncated):
    """Abstract provenance with this-step truncation and source integrity split.

    ``truncated`` is the 6,000-character cut only.  ``source_integrity`` and
    ``source_preview_limit`` describe the upstream value: a mapping abstract is
    a 200-character preview whose original length is unknown.
    """
    record = _provenance(source, key, source_files, truncated=truncated)
    if source == 'canonical':
        record['source_integrity'] = 'complete'
        record['source_preview_limit'] = None
    elif source == 'mapping':
        record['source_integrity'] = 'preview_unknown'
        record['source_preview_limit'] = MAPPING_PREVIEW_LIMIT
    elif source == 'card':
        record['source_integrity'] = 'stored'
        record['source_preview_limit'] = None
    return record


def _enrich_one(card, canonical, mapping, abstract_limit, source_files):
    row, entry, metadata = _resolve(card, canonical, mapping)
    card = dict(card)
    provenance = metadata['provenance']
    for field in ('title', 'year', 'journal', 'abstract', 'pub_type', 'keywords'):
        value, source, key = _pick(card, field, row, entry, metadata)
        if field == 'abstract':
            old_ids = list(card.get('abstract_passage_ids') or [])
            if source == 'card':
                # Keep the historical fact that the stored abstract was cut.
                truncated = bool(card.get('abstract_truncated'))
                if len(value) > abstract_limit:
                    value = value[:abstract_limit]
                    truncated = True
                card['abstract_passage_ids'] = old_ids
            else:
                truncated = len(value) > abstract_limit
                if truncated:
                    value = value[:abstract_limit]
                card['abstract_passage_ids'] = []
                if old_ids:
                    card['abstract_passage_ids_historical'] = old_ids
            card['abstract_truncated'] = truncated
            provenance['abstract'] = _abstract_provenance(source, key, source_files,
                                                          truncated=truncated)
        else:
            provenance[field] = _provenance(source, key, source_files)
        if field == 'keywords':
            # Never substitute MeSH here; MeSH is recorded separately below.
            card['keywords'] = _clean(card.get('keywords'))
        else:
            card[field] = value
    mesh, mesh_source, mesh_key = _pick(card, 'mesh_terms', row, entry, metadata)
    card['mesh_terms'] = mesh
    provenance['mesh_terms'] = _provenance(mesh_source if mesh else 'missing',
                                           mesh_key if mesh else '', source_files)
    card['missing_fields'] = [field for field in CARD_FIELDS if not _clean(card.get(field))]
    card['metadata'] = metadata
    return card


def enrich_cards(cards, canonical, mapping=None, abstract_limit=ABSTRACT_LIMIT,
                 source_files=None):
    """Return enriched copies of ``cards`` with identical sources and order.

    ``source_files`` maps ``card``/``canonical``/``mapping`` to input paths for
    per-field provenance; paths recorded by ``load_canonical``/``load_mapping``
    are used as fallbacks.  The old card path cannot be inferred, so callers
    should pass ``{'card': ...}`` when a card is not already enriched.
    """
    if abstract_limit < 1:
        raise ValueError('abstract limit must be positive')
    cards = list(cards)
    sources = [str(card.get('source') or '') for card in cards]
    if any(not source for source in sources) or len(set(sources)) != len(sources):
        raise ValueError('duplicate/missing document identity')
    supplied = source_files or {}
    files = {'canonical': str(supplied.get('canonical') or canonical.get('path') or ''),
             'mapping': str(supplied.get('mapping') or (mapping or {}).get('path') or ''),
             'card': str(supplied.get('card') or '')}
    return [_enrich_one(card, canonical, mapping, abstract_limit, files) for card in cards]
