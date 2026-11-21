"""Local, isolated C3 bibliography graph. No network, models or production writes.

Fuzzy scores are review candidates only: no human calibration set is currently
part of this interface. Missing bibliographies never imply zero citations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import unicodedata

from paper_registry import normalize_identifier, normalize_title

SCHEMA = 'c3-v1'
RULES = 'c3-bibliography-v4-declaration-consumption-multifield-quarantine'
PARAMS = dict(damping=0.85, tolerance=1e-12, max_iterations=1000,
              initialization='uniform', stopping='L1 <= tolerance', precision='float64')
HEADING = re.compile(r'^\s*(?:#{1,6}\s*)?(?:\d+[. ]+)?(?:references?|referencesreferences|thiemereferences|references and recommended reading|bibliography|literature cited|参考文献)\s*[:：]?\s*#*\s*$', re.I)
NUMBER = re.compile(r'^\s*(?:\[(\d{1,4})\]|(\d{1,4})[.)．]|(\d{1,3})(?=[A-Z]))\s*(?=\S)')
DOI = re.compile(r'10\.\d{4,9}/[^\s<>"\[\]]+', re.I)
PMID = re.compile(r'(?:\b(?:PMID|PubMed)\s*[:：]?\s*|pubmed\.ncbi\.nlm\.nih\.gov/|ncbi\.nlm\.nih\.gov/pubmed/)(\d+)', re.I)
PMCID = re.compile(r'\bPMC\d+\b', re.I)


def declarations(raw):
    """Validate each marker independently, including nested label + URL forms.

    Offsets address the retained raw string, not an inferred document page.
    A valid marker never consumes or excuses a later broken declaration.
    """
    doi_url = r'(?:https?[ \t]*:[ \t]*//[ \t]*)?(?:dx[ \t]*\.[ \t]*)?doi[ \t]*\.[ \t]*org/'
    markers = [
        ('doi', 'label', r'\bdoi\s*[:：]|\b(?-i:DOI)[ \t]+(?![ \t]*(?-i:[A-Z]{1,4})[,.;])(?!\.[ \t]*org/)(?=\S)|\bdoi[ \t]+(?=10\.|(?:org|0rg)/)'),
        ('doi', 'url', doi_url),
        ('doi', 'bare', r'\b10\.\d{4,9}/'),
        ('pmid', 'label', r'\bPMID\b(?:\s*[:：]|[ \t]+(?=\S))|\bPubMed\s*[:：]|\bPubMed[ \t]+(?=\d)'),
        ('pmid', 'url', r'(?:https?://)?(?:pubmed\.ncbi\.nlm\.nih\.gov/|(?:www\.)?ncbi\.nlm\.nih\.gov/pubmed/)'),
        ('pmcid', 'label', r'\bPMCID\b(?:\s*[:：]|[ \t]+(?=\S))'),
        ('pmcid', 'url', r'(?:https?://)?(?:pmc\.ncbi\.nlm\.nih\.gov/articles/|(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/)'),
    ]
    values = {
        'doi': r'(?:'+doi_url+r'\s*)?(10\.\d{4,9}/[^\s<>"\[\]]+)',
        'pmid': r'(?:(?:https?://)?(?:pubmed\.ncbi\.nlm\.nih\.gov/|(?:www\.)?ncbi\.nlm\.nih\.gov/pubmed/)\s*)?(\d+)(?=$|[\s\].,;:)\/])',
        'pmcid': r'(?:(?:https?://)?(?:pmc\.ncbi\.nlm\.nih\.gov/articles/|(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/)\s*)?(PMC\d+)(?=$|[\s\].,;:)\/])',
    }
    found = []
    for scheme, kind, pattern in markers:
        for marker in re.finditer(pattern, raw, re.I):
            start = marker.start() if kind == 'bare' else marker.end()
            tail = raw[start:]
            matched = re.match(r'\s*'+values[scheme], tail, re.I)
            end = start+matched.end() if matched else marker.end()+len(re.match(r'\s*\S*',raw[marker.end():]).group())
            reason = 'consumed' if matched else 'unparsed_declaration'
            value = None
            if matched:
                value, _ = normalize_identifier(scheme, matched.group(1).rstrip('.,;:)'))
                if scheme == 'doi':
                    # A standalone lower-case continuation on the next line
                    # may be a lost DOI suffix. Do not concatenate or accept it.
                    remainder = raw[end:]
                    if re.match(r'[ \t]*\r?\n[ \t]*[a-z0-9][a-z0-9._/()\-]*[ \t]*(?:\r?\n|$)', remainder):
                        reason = 'uncertain_cross_line_doi_suffix'
            found.append(dict(scheme=scheme, kind=kind, start=marker.start(), marker_end=marker.end(), end=end,
                              raw=raw[marker.start():end], value=value,
                              status='valid' if reason == 'consumed' else 'unresolved', reason=reason))
    found.sort(key=lambda d:(d['start'],d['scheme'],d['kind']))
    # C1 DOI normalization explicitly permits repeated DOI prefixes. Validate
    # those inside out; PMID/PMCID labels do not have this normalization rule.
    for parent in reversed(found):
        if parent['scheme']!='doi' or parent['kind']!='label' or parent['status']=='valid':
            continue
        child = next((d for d in found if d['start']>=parent['marker_end']),None)
        if (child and child['scheme']==parent['scheme'] and child['kind']=='label' and child['status']=='valid'
                and not raw[parent['marker_end']:child['start']].strip()):
            parent.update(end=child['end'],raw=raw[parent['start']:child['end']],
                          value=child['value'],status='valid',reason='consumed_nested_marker')
    return found


def _words(value):
    text = ''.join(c for c in unicodedata.normalize('NFKD', str(value)) if not unicodedata.combining(c))
    return normalize_title(text).split()


def bibliographic_check(raw, target):
    """A conservative contradiction screen, never a title-based rebinding rule.

    Non-equivalent title + reliably parsed first-author and journal disagreement
    jointly trigger review. No numeric title threshold, year-only rejection or
    assumption that a failed parse proves a contradiction.
    """
    body = NUMBER.sub('',raw,count=1).strip()
    author = re.match(r"([A-ZÀ-Ž][\w’'\-]+),?[ \t]+[A-Z]{1,4}(?=[\s,.;(])",body)
    first = re.search(r'\.\s+',body)
    year_author = re.search(r'\((?:18|19|20)\d{2}\)',body)
    fields = dict(authors=None,title=None,journal=None)
    if author and first and not re.match(r'[A-Z]\.\s',body[first.end():]):
        cut = year_author.end() if year_author and year_author.start()<first.start() else first.end()
        fields['authors'] = body[:cut].strip()
        rest = re.sub(r'^\s*\((?:18|19|20)\d{2}\)\s*','',body[cut:])
        title_end = re.search(r'\.\s+',rest)
        if title_end and re.search(r'[A-Za-zÀ-Ž]',rest[:title_end.start()]):
            fields['title'] = rest[:title_end.start()].strip()
            journal_tail = rest[title_end.end():]
            journal = re.match(r'[A-Za-zÀ-Ž &().\-]+',journal_tail)
            if journal and journal_tail[journal.end():][:1].isdigit():
                fields['journal'] = journal.group().strip(' .(')
    target_fields = {k:str(target.get(k) or '') for k in ('title','authors','journal','year')}
    checks = dict(title='unknown',first_author='unknown',journal='unknown',year='unknown')
    if fields['title'] and target_fields['title']:
        checks['title'] = 'equivalent' if _words(fields['title']) == _words(target_fields['title']) else 'not_equivalent'
    canonical_author = target_fields['authors'].split(';')[0]
    if ',' in canonical_author:
        surname = canonical_author.split(',')[0]
    else:
        canonical_match = re.fullmatch(r'([\w’\-]+)\s+[A-Z]{1,4}',canonical_author)
        surname = canonical_match.group(1) if canonical_match else ''
    if author and fields['authors'] and surname:
        checks['first_author'] = 'agree' if ''.join(_words(author.group(1))) == ''.join(_words(surname)) else 'disagree'
    if fields['journal'] and target_fields['journal']:
        a,b = [[w for w in _words(x) if w not in ('of','the','and')] for x in (fields['journal'],target_fields['journal'])]
        if a and b:
            compatible = len(a)==len(b) and all(x.startswith(y) or y.startswith(x) for x,y in zip(a,b))
            checks['journal'] = 'compatible_abbreviation' if compatible else 'disagree'
    years = sorted(set(re.findall(r'\b(?:18|19|20)\d{2}\b',raw)))
    if years and target_fields['year']:
        checks['year'] = 'agree' if target_fields['year'] in years else 'disagree'
    quarantine = checks['title']=='not_equivalent' and checks['first_author']=='disagree' and checks['journal']=='disagree'
    return dict(rule='title_non_equivalence_AND_first_author_AND_journal_disagreement-v1',
                reference_fields=fields,reference_years=years,target_metadata=target_fields,
                checks=checks,decision='pending_review' if quarantine else 'no_multifield_contradiction_detected')


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def fingerprint(value):
    return hashlib.sha256(dump(value).encode('utf-8')).hexdigest()


def readonly(path):
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA query_only=ON')
    return conn


def registry(path):
    c = readonly(path)
    try:
        if dict(c.execute('SELECT key,value FROM registry_meta')).get('schema_version') != 'p09-c1-v2':
            raise ValueError('C1 v2 required; no migration allowed')
        return {r['paper_id']: dict(r) for r in c.execute('SELECT * FROM papers ORDER BY paper_id')}, [dict(r) for r in c.execute('SELECT * FROM artifacts ORDER BY artifact_id')], [dict(r) for r in c.execute('SELECT * FROM identifiers ORDER BY scheme,value')]
    finally:
        c.close()


def extract(lines):
    """Return source status, reason, and exact line spans (1-based inclusive).

    Only numbered entries beneath an explicit bibliography heading can become
    edges. Unnumbered paragraphs are retained for review, never silently joined
    to the previous citation across a blank line.
    """
    heads = [i for i, line in enumerate(lines) if HEADING.fullmatch(line)]
    if not heads:
        return 'references_missing', 'no_explicit_bibliography_heading', []
    refs = []
    for h in heads:
        end = next((i for i in range(h+1, len(lines)) if re.match(r'^\s*#{1,6}\s+', lines[i])), len(lines))
        # Some accepted two-column conversions insert a back-matter heading
        # between REFERENCES and its very first numbered bibliographic item.
        # Cross only an empty section and an explicitly observed heading; a
        # following prose paragraph or any other heading is still a boundary.
        if end < len(lines) and not any(x.strip() for x in lines[h+1:end]):
            next_line = next((i for i in range(end+1,len(lines)) if lines[i].strip()), None)
            if (re.fullmatch(r'\s*#{1,6}\s*(?:data availability statement|acknowledg[e]?ments)\s*', lines[end], re.I)
                    and next_line is not None and NUMBER.match(lines[next_line])
                    and re.search(r'\b(?:18|19|20)\d{2}\b', lines[next_line])):
                end = next((i for i in range(next_line+1,len(lines)) if re.match(r'^\s*#{1,6}\s+',lines[i])),len(lines))
                start = next_line
            else:
                start = h+1
        else:
            start = h+1
        section = lines[start:end]
        if len([x for x in section if x.strip()]) == 1 and re.fullmatch(r'\s*(?:none\.?|no references\.?)\s*', next(x for x in section if x.strip()), re.I):
            continue
        starts = [i for i in range(start, end) if NUMBER.match(lines[i])]
        # Preserve all nonempty paragraph spans, including parser gaps.
        boundaries = {start, end, *starts}
        for i in range(start, end):
            if not lines[i].strip():
                boundaries.add(i)
                boundaries.add(i+1)
        bounds = sorted(boundaries)
        for a, b in zip(bounds, bounds[1:]):
            while a < b and not lines[a].strip():
                a += 1
            while b > a and not lines[b-1].strip():
                b -= 1
            if a == b:
                continue
            raw = '\n'.join(lines[a:b])
            m = NUMBER.match(lines[a])
            number = next((x for x in m.groups() if x), '') if m else ''
            valid = bool(m and re.search(r'\b(?:18|19|20)\d{2}\b', raw) and not re.search(r'<table|!\[', raw, re.I))
            refs.append(dict(ordinal=len(refs)+1, number=number, raw=raw,
                             line_start=a+1, line_end=b, heading_line=h+1,
                             parse_status='parsed' if valid else 'boundary_unverified',
                             parse_reason='numbered_bibliographic_entry' if valid else 'requires_boundary_review'))
    if not refs:
        explicit = all(any(re.fullmatch(r'\s*(?:none\.?|no references\.?)\s*', x, re.I) for x in lines[h+1:h+4]) for h in heads)
        return ('confirmed_no_references', 'explicit_none', []) if explicit else ('parse_failed', 'empty_or_unreadable_bibliography', [])
    state = 'extracted' if all(r['parse_status'] == 'parsed' for r in refs) else 'partial'
    return state, 'numbered_only_edges; unverified_spans_retained', refs


class Matcher:
    def __init__(self, papers, identifiers):
        self.papers = papers
        self.owners = defaultdict(set)
        self.superseded = set()
        self.values = defaultdict(lambda: defaultdict(set))
        for i in identifiers:
            key = (i['scheme'], i['value'])
            if i['status'] == 'active':
                self.owners[key].add(i['paper_id'])
                self.values[i['paper_id']][i['scheme']].add(i['value'])
            else:
                self.superseded.add(key)
        self.titles = {p: normalize_title(r['title']) for p, r in papers.items() if r['title']}
        self.tokens = {p: set(t.split()) for p, t in self.titles.items()}
        self.inverted = defaultdict(set)
        for p, tokens in self.tokens.items():
            for t in tokens:
                if len(t) > 3:
                    self.inverted[t].add(p)

    def match(self, raw):
        ids = defaultdict(set)
        for scheme, pattern in [('doi', DOI), ('pmid', PMID), ('pmcid', PMCID)]:
            for value in pattern.findall(raw):
                try:
                    norm, _ = normalize_identifier(scheme, value.rstrip('.,;:)'))
                    ids[scheme].add(norm)
                except ValueError:
                    return dict(status='conflict', reason='invalid_strong_identifier', target=None, identifiers={k: sorted(v) for k,v in ids.items()}, candidates=[], confidence=None)
        declared = declarations(raw)
        for d in declared:
            if d['status']=='valid':
                ids[d['scheme']].add(d['value'])
        base = dict(target=None, identifiers={k: sorted(v) for k, v in ids.items()}, candidates=[], confidence=None,
                    declarations=declared, identifier_owner=None, binding_status='unresolved')
        def result(status, reason, **kwargs):
            return dict(base, status=status, reason=reason, **kwargs)
        # A broken identifier is not permission to fall back to title matching.
        # Keep whitespace-split DOI suffixes for source-level review, not repair.
        if (any(d['status']!='valid' for d in declared)
                or re.search(r'10\.\d{4,9}/\s', raw, re.I)
                or (not ids.get('doi') and re.search(r'\bdoi\s*:|doi\.org/|10\.\d{4,9}/', raw, re.I))
                or (not ids.get('pmid') and re.search(r'\b(?:PMID|PubMed)\s*:', raw, re.I))
                or (not ids.get('pmcid') and re.search(r'\bPMCID\s*:', raw, re.I))):
            return result('unresolved', 'unparsed_or_split_strong_identifier')
        if ids:
            keys = [(s, v) for s, vals in ids.items() for v in vals]
            owners = set().union(*(self.owners[k] for k in keys))
            if any(k in self.superseded for k in keys):
                return result('conflict', 'superseded_alias')
            if any(len(self.owners[k]) > 1 for k in keys) or len(owners) > 1:
                return result('ambiguous', 'multiple_owners')
            if any(len(v) > 1 for v in ids.values()):
                return result('conflict', 'multiple_values_same_scheme')
            if not owners:
                return result('external_or_unresolved', 'strong_identifiers_not_in_registry')
            target = next(iter(owners))
            if self.papers[target]['identity_status'] != 'verified':
                return result('conflict', 'unsafe_target_identity')
            for s, vals in ids.items():
                if self.values[target][s] and not vals.issubset(self.values[target][s]):
                    return result('conflict', 'identifier_disagrees_with_owner')
            # Unknown secondary IDs cannot be verified from C1 alone.
            if any(not self.owners[k] for k in keys):
                return result('unresolved', 'unowned_secondary_identifier')
            base['identifier_owner'] = target
            base['bibliographic_check'] = bibliographic_check(raw,self.papers[target])
            if base['bibliographic_check']['decision']=='pending_review':
                base['binding_status'] = 'quarantined'
                return result('bibliographic_conflict','unadjudicated_multifield_bibliographic_disagreement')
            base.update(target=target, confidence='identifier_exact_not_empirically_calibrated', binding_status='usable_exact')
            return result('exact', 'all_strong_identifiers_agree')
        normalized = normalize_title(raw)
        tokens = set(normalized.split())
        counts = Counter(p for t in tokens for p in self.inverted.get(t, ()))
        shortlist = sorted(counts, key=lambda p: (-counts[p], p))[:20]
        segments = [set(normalize_title(s).split()) for s in re.split(r'\.\s+', raw)] + [tokens]
        years = sorted(set(re.findall(r'\b(?:18|19|20)\d{2}\b', raw)))
        candidates = []
        for p in shortlist:
            shared = len(tokens & self.tokens[p])
            coverage = shared / len(self.tokens[p])
            score = max([2*len(self.tokens[p] & s)/(len(self.tokens[p])+len(s)) for s in segments] + [0])
            candidates.append(dict(paper_id=p, score=round(score, 6), title_token_coverage=round(coverage, 6), title=self.papers[p]['title'],
                                   year=self.papers[p]['year'], authors=self.papers[p]['authors'],
                                   reference_years=years, year_check='agree' if str(self.papers[p]['year']) in years else 'unknown_or_disagree',
                                   identity_status=self.papers[p]['identity_status'], confidence='uncalibrated', accepted=False,
                                   reason='human_calibration_and_heldout_validation_required'))
        candidates.sort(key=lambda r: (-r['score'], -r['title_token_coverage'], r['paper_id']))
        base['candidates'] = candidates[:3]
        return result('fuzzy_candidate' if candidates else 'unmatched', 'no_strong_identifier; fuzzy_edges_disabled')


def pagerank(nodes, edges, damping=0.85, tolerance=1e-12, max_iterations=1000):
    nodes = sorted(set(nodes))
    edges = sorted(set(edges))
    if not 0 < damping < 1 or tolerance <= 0 or max_iterations < 1:
        raise ValueError('invalid PageRank parameters')
    n = len(nodes)
    if not n:
        if edges:
            raise ValueError('edges outside graph')
        return {}, 0, 0.0
    outgoing = {p: [] for p in nodes}
    for a, b in edges:
        if a not in outgoing or b not in outgoing:
            raise ValueError('edges outside graph')
        outgoing[a].append(b)
    rank = {p: 1/n for p in nodes}
    for iteration in range(1, max_iterations+1):
        dangling = math.fsum(rank[p] for p in nodes if not outgoing[p])
        incoming = {p: [] for p in nodes}
        for a, b in edges:
            incoming[b].append(rank[a]/len(outgoing[a]))
        new = {p: (1-damping)/n + damping*(dangling/n + math.fsum(incoming[p])) for p in nodes}
        delta = math.fsum(abs(new[p]-rank[p]) for p in nodes)
        rank = new
        if delta <= tolerance:
            return rank, iteration, delta
    raise ValueError('PageRank did not converge; no successful graph published')


def graph(papers, references):
    edges = set()
    supports = []
    for r in references:
        a, b = r['paper_id'], r['match']['target']
        kind = 'excluded'
        if r['match']['status'] == 'exact' and r['parse_status'] == 'parsed':
            if papers[a]['identity_status'] != 'verified':
                kind = 'unsafe_citing_identity'
            elif r['role'] == 'supplement':
                kind = 'supplement'
            elif a == b:
                kind = 'self_loop'
            else:
                kind = 'main'
                edges.add((a, b))
        r['edge_class'] = kind
        if kind == 'main':
            supports.append((a, b, r['ref_id']))
    rank, iterations, delta = pagerank(papers, edges)
    indegree, outdegree = Counter(b for a, b in edges), Counter(a for a, b in edges)
    metrics = {p: dict(in_degree=indegree[p], out_degree=outdegree[p], pagerank=rank[p]) for p in sorted(papers)}
    return sorted(edges), sorted(supports), metrics, dict(iterations=iterations, residual_l1=delta, **PARAMS)


def protect_output(output, inputs):
    out = Path(output).resolve()
    for p in inputs:
        p = Path(p).resolve()
        if out == p or (out.exists() and p.exists() and os.path.samefile(out, p)):
            raise ValueError('output would overwrite input: ' + str(p))
    return out


def open_graph(path):
    c = readonly(path)
    try:
        meta = dict(c.execute('SELECT key,value FROM c3_meta'))
        if meta.get('schema') != SCHEMA or meta.get('rules') != RULES or not {'citation_graph_version','identifiers','summary','inputs'}.issubset(meta):
            raise ValueError('not a supported C3 database')
        for table, cols in {'nodes': {'paper_id', 'payload'}, 'sources': {'artifact_id', 'payload'}, 'refs': {'ref_id', 'payload'}, 'edges': {'citing', 'cited'}, 'supports': {'citing', 'cited', 'ref_id'}}.items():
            actual = {r['name'] for r in c.execute('PRAGMA table_info('+table+')')}
            if not cols.issubset(actual):
                raise ValueError('incomplete C3 schema')
        return c
    except Exception:
        c.close()
        raise


def build(c1, inventory, source_root, corrections, output):
    papers, artifacts, ids = registry(c1)
    inv = dict(json.loads(Path(inventory).read_text(encoding='utf-8')))
    correction = json.loads(Path(corrections).read_text(encoding='utf-8')) if corrections else None
    if len(inv) != len(artifacts) or any(inv.get(a['source']) != a['content_sha256'] for a in artifacts):
        raise ValueError('C1 artifact/inventory bindings differ')
    if correction and inv.get(correction['source']) != correction['new_sha256']:
        raise ValueError('overlay binding differs')
    paths = {a['artifact_id']: (Path(correction['corrected_path']) if correction and a['source'] == correction['source'] else Path(source_root)/a['source']).resolve() for a in artifacts}
    inputs = [c1, inventory, *paths.values()] + ([corrections] if corrections else [])
    out = protect_output(output, inputs)
    if out.exists():
        open_graph(out).close()  # reject C1/C2/ordinary SQLite before ANY DDL
    matcher = Matcher(papers, ids)
    sources, references = [], []
    for a in artifacts:
        path = paths[a['artifact_id']]
        source = dict(a, path=str(path), binding='accepted_overlay' if correction and a['source'] == correction['source'] else 'accepted_inventory', observed_sha256=None)
        try:
            raw = path.read_bytes()
            source['observed_sha256'] = hashlib.sha256(raw).hexdigest()
            if source['observed_sha256'] != a['content_sha256']:
                source.update(status='source_binding_mismatch', reason='actual_bytes_differ_from_accepted_artifact', reference_count=0)
                sources.append(source)
                continue
            lines = raw.decode('utf-8-sig').splitlines()
            state, reason, entries = extract(lines)
            source.update(status=state, reason=reason, reference_count=len(entries))
            for entry in entries:
                r = dict(entry, ref_id=a['artifact_id']+':'+str(entry['ordinal']), paper_id=a['paper_id'], artifact_id=a['artifact_id'], role=a['role'], source=a['source'], path=str(path), content_sha256=a['content_sha256'])
                r['match'] = matcher.match(r['raw'])
                references.append(r)
        except FileNotFoundError:
            source.update(status='source_missing', reason='accepted_path_unavailable', reference_count=0)
        except (UnicodeError, OSError) as e:
            source.update(status='parse_failed', reason=type(e).__name__+': '+str(e), reference_count=0)
        sources.append(source)
    references.sort(key=lambda r: r['ref_id'])
    edges, supports, metrics, numerical = graph(papers, references)
    # One semantic version binds identity, artifact observations and every retained
    # reference, without a redundant corpus-wide tracking or hash manifest.
    scope = dict(papers=papers, identifiers=ids, sources=sources, references=references, rules=RULES, params=PARAMS)
    version = 'c3-'+fingerprint(scope)[:24]
    summary = dict(nodes=len(papers), artifacts=len(artifacts), roles=dict(Counter(a['role'] for a in artifacts)),
                   extraction_artifacts=dict(Counter(s['status'] for s in sources)),
                   extraction_papers={state: len({s['paper_id'] for s in sources if s['status'] == state}) for state in sorted({s['status'] for s in sources})},
                   references=len(references), parse_states=dict(Counter(r['parse_status'] for r in references)),
                   match_states=dict(Counter(r['match']['status'] for r in references)), edge_classes=dict(Counter(r['edge_class'] for r in references)),
                   edges_before_dedup=len(supports), edges=len(edges), isolates=sum(m['in_degree'] == m['out_degree'] == 0 for m in metrics.values()),
                   metric_coverage=len(metrics), fuzzy_edges=0, human_calibration='missing; no fuzzy edges permitted',
                   citation_graph_version=version, numerical=numerical,
                   coverage_complete=all(s['status'] in ('extracted','confirmed_no_references') for s in sources))
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='c3-build-', suffix='.sqlite3', dir=out.parent)
    os.close(fd)
    try:
        c = sqlite3.connect(tmp)
        try:
            c.executescript('CREATE TABLE c3_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE nodes(paper_id TEXT PRIMARY KEY,payload TEXT NOT NULL); CREATE TABLE sources(artifact_id TEXT PRIMARY KEY,payload TEXT NOT NULL); CREATE TABLE refs(ref_id TEXT PRIMARY KEY,payload TEXT NOT NULL); CREATE TABLE edges(citing TEXT,cited TEXT,PRIMARY KEY(citing,cited)); CREATE TABLE supports(citing TEXT,cited TEXT,ref_id TEXT,PRIMARY KEY(citing,cited,ref_id));')
            meta = dict(schema=SCHEMA, rules=RULES, citation_graph_version=version, identifiers=dump(ids), summary=dump(summary), inputs=dump(dict(c1=str(Path(c1).resolve()), inventory=str(Path(inventory).resolve()), source_root=str(Path(source_root).resolve()), corrections=str(Path(corrections).resolve()) if corrections else None)))
            c.executemany('INSERT INTO c3_meta VALUES (?,?)', sorted(meta.items()))
            c.executemany('INSERT INTO nodes VALUES (?,?)', [(p, dump(dict(papers[p], **metrics[p]))) for p in sorted(papers)])
            c.executemany('INSERT INTO sources VALUES (?,?)', [(s['artifact_id'], dump(s)) for s in sources])
            c.executemany('INSERT INTO refs VALUES (?,?)', [(r['ref_id'], dump(r)) for r in references])
            c.executemany('INSERT INTO edges VALUES (?,?)', edges)
            c.executemany('INSERT INTO supports VALUES (?,?,?)', supports)
            c.commit()
        finally:
            c.close()
        os.replace(tmp, out)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()
    return summary


def snapshot(path):
    c = open_graph(path)
    try:
        return dict(meta=dict(c.execute('SELECT key,value FROM c3_meta ORDER BY key')),
                    **{t: [dict(r) for r in c.execute('SELECT * FROM '+t+' ORDER BY 1,2')] for t in ('nodes','sources','refs','edges','supports')})
    finally:
        c.close()


def verify(path, locators=False):
    data = snapshot(path)
    refs = [json.loads(r['payload']) for r in data['refs']]
    nodes = {r['paper_id']: json.loads(r['payload']) for r in data['nodes']}
    identity = {p: {k: v for k,v in n.items() if k not in ('in_degree','out_degree','pagerank')} for p,n in nodes.items()}
    identifiers = json.loads(data['meta']['identifiers'])
    matcher = Matcher(identity, identifiers)
    for r in refs:
        if matcher.match(r['raw']) != r['match']:
            raise ValueError('match recomputation differs: '+r['ref_id'])
    scope = dict(papers=identity, identifiers=identifiers, sources=[json.loads(r['payload']) for r in data['sources']], references=refs, rules=RULES, params=PARAMS)
    if 'c3-'+fingerprint(scope)[:24] != data['meta']['citation_graph_version']:
        raise ValueError('citation_graph_version differs from saved scope')
    edges, supports, metrics, _ = graph(nodes, refs)
    if edges != [(r['citing'],r['cited']) for r in data['edges']]:
        raise ValueError('edge recomputation differs')
    if supports != [(r['citing'],r['cited'],r['ref_id']) for r in data['supports']]:
        raise ValueError('support recomputation differs')
    for p, m in metrics.items():
        if any(nodes[p][k] != m[k] for k in ('in_degree','out_degree')) or abs(nodes[p]['pagerank']-m['pagerank']) > 1e-12:
            raise ValueError('metric recomputation differs')
    checked = 0
    if locators:
        grouped = defaultdict(list)
        for r in refs:
            grouped[r['path']].append(r)
        for source, entries in grouped.items():
            lines = Path(source).read_text(encoding='utf-8-sig').splitlines()
            for r in entries:
                if '\n'.join(lines[r['line_start']-1:r['line_end']]) != r['raw']:
                    raise ValueError('reference locator mismatch: '+r['ref_id'])
                checked += 1
    return dict(status='PASS', nodes=len(nodes), edges=len(edges), references=len(refs), locators_checked=checked)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    for flag in ('c1','inventory','source-root','output'):
        b.add_argument('--'+flag, required=True)
    b.add_argument('--corrections')
    for name in ('status','verify','export'):
        s = sub.add_parser(name)
        s.add_argument('--db', required=True)
        if name == 'verify':
            s.add_argument('--locators', action='store_true')
        if name == 'export':
            s.add_argument('--output', required=True)
    a = p.parse_args(argv)
    if a.command == 'build':
        result = build(a.c1, a.inventory, a.source_root, a.corrections, a.output)
    elif a.command == 'verify':
        result = verify(a.db, a.locators)
    elif a.command == 'status':
        c = open_graph(a.db)
        try:
            result = json.loads(dict(c.execute('SELECT key,value FROM c3_meta'))['summary'])
        finally:
            c.close()
    else:
        result = snapshot(a.db)
        meta_inputs = json.loads(result['meta']['inputs'])
        sources = [json.loads(s['payload'])['path'] for s in result['sources']]
        protected = [a.db, meta_inputs['c1'], meta_inputs['inventory'], *sources]
        if meta_inputs['corrections']:
            protected.append(meta_inputs['corrections'])
        output = protect_output(a.output, protected)
        # Exclusive creation also prevents overwriting a different C1/C2 DB.
        with output.open('x', encoding='utf-8') as f:
            f.write(dump(result)+'\n')
        result = dict(exported=str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
