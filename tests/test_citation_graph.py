"""C3 synthetic bibliography, identity, numerical and filesystem boundaries."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'index'))
import citation_graph as cg
import paper_registry as pr


def papers():
    return {p: dict(paper_id=p, identity_status='verified', title=t, year='2020', authors='Smith A')
            for p, t in [('a', 'Rectal cancer imaging'), ('b', 'Rectal cancer staging'), ('c', 'Other study')]}


def ids():
    return [dict(scheme=s, value=v, paper_id=p, status='active') for s,v,p in
            [('doi','10.1234/a','a'), ('pmid','1','a'), ('doi','10.1234/b','b')]]


def ref(rid='r1', a='a', b='b', role='main'):
    return dict(ref_id=rid, paper_id=a, role=role, parse_status='parsed', match=dict(status='exact', target=b))


class ExtractionTests(unittest.TestCase):
    def test_lines_and_raw_preserved(self):
        lines = ['# Text','1. Body 2020', '## REFERENCES','', '1. Smith. Title. 2020.', ' doi:10.1234/a', '', '2. Jones. Other. 2021.']
        state, _, refs = cg.extract(lines)
        self.assertEqual(state, 'extracted')
        self.assertEqual(len(refs),2)
        self.assertEqual((refs[0]['line_start'],refs[0]['line_end']), (5,6))
        self.assertEqual(refs[0]['raw'], '\n'.join(lines[4:6]))

    def test_missing_zero_failure_distinct(self):
        self.assertEqual(cg.extract(['# Results','1. text'])[0], 'references_missing')
        self.assertEqual(cg.extract(['## References','None.'])[0], 'confirmed_no_references')
        self.assertEqual(cg.extract(['## References',''])[0], 'parse_failed')

    def test_stop_at_next_section(self):
        state, _, refs = cg.extract(['## References','1. A. Title. 2020.', '## Supplementary tables','2. caption 2021'])
        self.assertEqual(len(refs),1)

    def test_unverified_spans_not_merged(self):
        state, _, refs = cg.extract(['## References','1. A. Title. 2020.','','Unnumbered. Another. 2021.'])
        self.assertEqual(state, 'partial')
        self.assertEqual(len(refs),2)
        self.assertEqual(refs[1]['parse_status'],'boundary_unverified')

    def test_table_not_bibliography(self):
        _,_,refs=cg.extract(['## References','1. <table>2020</table>'])
        self.assertEqual(refs[0]['parse_status'],'boundary_unverified')

    def test_observed_heading_variants_not_reference_standard(self):
        for heading in ('## REFERENCE','## ReferencesReferences','## ThiemeReferences','## References and Recommended Reading'):
            self.assertEqual(cg.extract([heading,'1. A. Title. 2020.'])[0],'extracted')
        for heading in ('## Reference standard','## eReferences','## Supplementary references'):
            self.assertEqual(cg.extract([heading,'1. A. Title. 2020.'])[0],'references_missing')

    def test_empty_bibliography_interposed_backmatter_heading(self):
        for heading in ('## DATA AVAILABILITY STATEMENT','## ACKNOWLEDGMENTS'):
            lines=['## REFERENCES','',heading,'','1. A. Title. 2020.']
            state,_,refs=cg.extract(lines)
            self.assertEqual(state,'extracted')
            self.assertEqual(refs[0]['line_start'],5)
            self.assertEqual(refs[0]['heading_line'],1)
        self.assertEqual(cg.extract(['## References','## Acknowledgments','Normal body. 2020.'])[0],'parse_failed')


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.m=cg.Matcher(papers(),ids())

    def test_exact_and_normalization(self):
        m=self.m.match('1. A. Title. 2020. https://doi.org/10.1234/A. PMID: 1')
        self.assertEqual((m['status'],m['target']),('exact','a'))

    def test_pubmed_label_checked_with_doi(self):
        self.assertEqual(self.m.match('10.1234/a [PubMed: 1]')['status'],'exact')
        self.assertEqual(self.m.match('10.1234/b [PubMed: 1]')['status'],'ambiguous')
        self.assertEqual(self.m.match('doi: https://doi.org/10.1234/a')['status'],'exact')

    def test_split_or_unparsed_identifier_never_fuzzy(self):
        for text in ('Rectal cancer imaging. 2020. doi: https://doi.org/10.1234/ suffix',
                     'Rectal cancer imaging. 2020. 10.1234/ suffix',
                     'Rectal cancer imaging. 2020. PMID: unreadable'):
            m=self.m.match(text)
            self.assertEqual(m['status'],'unresolved')
            self.assertEqual(m['candidates'],[])

    def test_doi_pmid_conflict(self):
        self.assertEqual(self.m.match('10.1234/b PMID:1')['status'],'ambiguous')

    def test_secondary_unknown_refused(self):
        self.assertEqual(self.m.match('10.1234/a PMID:999')['status'],'conflict')

    def test_unowned_secondary_refused(self):
        self.assertEqual(self.m.match('10.1234/b PMID:999')['status'],'unresolved')

    def test_multi_owner_refused(self):
        m=cg.Matcher(papers(),ids()+[dict(scheme='doi',value='10.1234/a',paper_id='b',status='active')])
        self.assertEqual(m.match('10.1234/a')['status'],'ambiguous')

    def test_superseded_not_fuzzy(self):
        m=cg.Matcher(papers(),ids()+[dict(scheme='doi',value='10.1234/old',paper_id='a',status='superseded')])
        self.assertEqual(m.match('Rectal cancer imaging 10.1234/old')['reason'],'superseded_alias')

    def test_external_identifier_never_title_fallback(self):
        m=self.m.match('Smith. Rectal cancer imaging. 2020. 10.1234/external')
        self.assertEqual(m['status'],'external_or_unresolved')
        self.assertEqual(m['candidates'],[])

    def test_fuzzy_candidates_never_edges(self):
        m=self.m.match('Smith A. Rectal cancer imaging. 2020.')
        self.assertEqual(m['status'],'fuzzy_candidate')
        self.assertIsNone(m['target'])
        self.assertTrue(all(not x['accepted'] for x in m['candidates']))
        self.assertEqual(m['candidates'][0]['year_check'],'agree')

    def test_low_confidence_and_tied_candidates(self):
        ps=papers(); ps['b']['title']=ps['a']['title']; ps['b']['year']='2021'
        m=cg.Matcher(ps,[]).match('Smith. Rectal cancer imaging. 2020.')
        self.assertEqual(m['candidates'][0]['score'],m['candidates'][1]['score'])
        self.assertIsNone(m['target'])
        self.assertEqual(m['candidates'][1]['year_check'],'unknown_or_disagree')

    def test_unsafe_identity(self):
        ps=papers();ps['a']['identity_status']='conflict'
        self.assertEqual(cg.Matcher(ps,ids()).match('10.1234/a')['status'],'conflict')


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.ps=papers()
        self.ps['a'].update(title='Magnetic resonance imaging in rectal tumors',authors='Smith, Alice',journal='Journal of Alpha Imaging',year='2020')
        self.identifiers=ids()+[dict(scheme=s,value=v,paper_id=p,status='active') for s,v,p in
                                [('pmcid','PMC123','a'),('pmid','2','b'),('pmcid','PMC456','b')]]
        self.m=cg.Matcher(self.ps,self.identifiers)

    def test_each_broken_declaration_order_is_rejected(self):
        for valid,bad in [('DOI:10.1234/a','DOI: unreadable'),('PMID:1','PMID: unreadable'),('PMCID:PMC123','PMCID: unreadable'),('PubMed:1','PubMed: unreadable'),('DOI: DOI:10.1234/a','DOI: unreadable')]:
            for tail in (valid+' '+bad,bad+' '+valid):
                r=self.m.match('1. Study A. 2020. '+tail)
                self.assertEqual(r['status'],'unresolved')
                self.assertIsNone(r['target'])
                self.assertTrue(any(d['status']=='unresolved' and 'unreadable' in d['raw'] for d in r['declarations']))

    def test_valid_same_declarations_and_nested_urls(self):
        for tail in ('DOI:10.1234/a DOI:10.1234/a',
                     'doi: doi:10.1234/a',
                     'DOI: https://doi.org/10.1234/a', 'DOI:doi.org/10.1234/a',
                     'doi: https:// doi.org/10.1234/a', 'https://doi .org/10.1234/a',
                     'doi: https://doi. org/10.1234/a', 'doi:https ://doi.org/10.1234/a',
                     'PMID:1 PMID:1', 'PMID: https://pubmed.ncbi.nlm.nih.gov/1/',
                     'PMCID:PMC123 PMCID:PMC123', 'PMCID: https://pmc.ncbi.nlm.nih.gov/articles/PMC123/'):
            r=self.m.match('1. Study A. 2020. '+tail)
            self.assertEqual(r['status'],'exact',tail)
            self.assertTrue(all(d['status']=='valid' for d in r['declarations']))

    def test_other_scheme_broken_and_conflicting_valid_identifiers(self):
        self.assertEqual(self.m.match('DOI:10.1234/a PMCID: unreadable')['status'],'unresolved')
        for text in ('DOI:10.1234/a DOI:10.1234/b','PMID:1 PMID:2','PMCID:PMC123 PMCID:PMC456'):
            self.assertEqual(self.m.match(text)['status'],'ambiguous')

    def test_valid_id_does_not_mask_broken_url_or_other_declaration(self):
        for good,bad in [('DOI:10.1234/a','https://doi. org/.'),
                         ('PMID:1','http://dx doi org/10 broken'),
                         ('PMCID:PMC123','PMCID:')]:
            for tail in (good+' '+bad,bad+' '+good):
                self.assertEqual(self.m.match('1. Study A. 2020. '+tail)['status'],'unresolved',tail)

    def test_numeric_prefix_not_entire_declaration(self):
        for tail in ('PMID:1abc','PMCID:PMC123abc'):
            self.assertEqual(self.m.match('DOI:10.1234/a '+tail)['status'],'unresolved')

    def test_cross_line_suffix_uncertain_not_all_newlines(self):
        self.assertEqual(self.m.match('DOI:10.1234/a\nbc')['status'],'unresolved')
        self.assertEqual(self.m.match('DOI:10.1234/a\nPMID:1')['status'],'exact')
        self.assertEqual(self.m.match('DOI:10.1234/a\n[CrossRef]')['status'],'exact')
        self.assertEqual(self.m.match('DOI:10.1234/\nsuffix')['status'],'unresolved')

    def test_bad_entry_does_not_pollute_next(self):
        raw='1. Study A. 2020. DOI:10.1234/a'
        good=self.m.match(raw)
        self.m.match(raw+' DOI: unreadable')
        self.assertEqual(self.m.match(raw),good)

    def test_doi_surname_is_not_an_identifier_declaration(self):
        for author in ('Doi H,','Doi SAR,','DOI H,','DOI  H,'):
            r=self.m.match('1. '+author+' Other A. MRI in rectal tumors. J Alpha Imaging. 2020;2:3.')
            self.assertEqual(r['status'],'fuzzy_candidate')
            self.assertEqual(r['declarations'],[])

    def test_declaration_offsets_are_raw_positions(self):
        raw='1. Study. 2020. DOI:10.1234/a DOI: unreadable'
        r=self.m.match(raw)
        for d in r['declarations']:
            self.assertEqual(raw[d['start']:d['end']],d['raw'])

    def test_multifield_conflict_preserves_owner_without_rebinding(self):
        raw='1. Jones B, White C. Surgical approaches for pelvic disease. J Beta Surg. 2021;3:1-4. DOI:10.1234/a'
        self.ps['b'].update(title='Surgical approaches for pelvic disease')
        r=self.m.match(raw)
        self.assertEqual(r['status'],'bibliographic_conflict')
        self.assertEqual(r['identifier_owner'],'a')
        self.assertIsNone(r['target'])
        self.assertIsNone(r['confidence'])
        self.assertEqual(r['candidates'],[])
        self.assertEqual(r['bibliographic_check']['decision'],'pending_review')
        self.assertEqual(r['bibliographic_check']['target_metadata']['title'],self.ps['a']['title'])

    def test_single_field_or_year_only_not_sufficient(self):
        for raw in (
            '1. Smith A. Magnetic resonance imaging in rectal tumors. J Alpha Imaging. 2021;2:3. DOI:10.1234/a',
            '1. Smith A. A different title entirely. J Alpha Imaging. 2020;2:3. DOI:10.1234/a',
            '1. Jones B. Magnetic resonance imaging in rectal tumors. J Beta Surg. 2020;2:3. DOI:10.1234/a',
            '1. Jones B. A different title entirely. J Alpha Imaging. 2020;2:3. DOI:10.1234/a'):
            self.assertEqual(self.m.match(raw)['status'],'exact',raw)

    def test_abbreviation_layout_and_missing_fields_not_conflicts(self):
        for raw in ('1. Smith A. MRI in rectal tumors. J. Alpha Imaging. 2020;2:3. DOI:10.1234/a',
                    '1. Smith A (2020) Magnetic resonance imaging in rectal tumors. J Alpha Imaging 2:3. DOI:10.1234/a',
                    '1. Smith. Title. 2020. DOI:10.1234/a',
                    '1. Other, J. E. V. et al. MRI in rectal tumors. J Alpha Imaging 2:3. DOI:10.1234/a',
                    'Endoscopy\nDOI 10.1234/a\nISSN 1234-5678\npublished online 2020. Publisher. All rights reserved.'):
            self.assertEqual(self.m.match(raw)['status'],'exact')

    def test_valid_support_survives_quarantined_support(self):
        good=ref('good',a='b',b='a');good['match']=self.m.match('1. Smith A. Magnetic resonance imaging in rectal tumors. J Alpha Imaging. 2020;2:3. DOI:10.1234/a')
        bad=ref('bad',a='b',b='a',role='version');bad['match']=self.m.match('1. Jones B. Surgery for pelvic disease. J Beta Surg. 2021;2:3. DOI:10.1234/a')
        e,s,m,_=cg.graph(self.ps,[good,bad])
        self.assertEqual(e,[('b','a')])
        self.assertEqual(s,[('b','a','good')])
        self.assertEqual(m['a']['in_degree'],1)


class GraphTests(unittest.TestCase):
    def test_direction_degrees_isolate(self):
        edges,_,m,_=cg.graph(papers(),[ref()])
        self.assertEqual(edges,[('a','b')])
        self.assertEqual((m['a']['in_degree'],m['a']['out_degree']),(0,1))
        self.assertEqual((m['b']['in_degree'],m['b']['out_degree']),(1,0))
        self.assertEqual(m['c']['in_degree']+m['c']['out_degree'],0)

    def test_duplicate_versions_supports_order(self):
        refs=[ref(str(i),role=role) for i,role in enumerate(['main','version','duplicate','main'])]
        e,s,m,_=cg.graph(papers(),refs)
        e2,s2,m2,_=cg.graph(dict(reversed(list(papers().items()))),list(reversed(copy.deepcopy(refs))))
        self.assertEqual((e,s,m),(e2,s2,m2))
        self.assertEqual(len(e),1);self.assertEqual(len(s),4)
        self.assertEqual(m,cg.graph(papers(),[ref()])[2])

    def test_self_supplement_unmatched_separate(self):
        refs=[ref('self',b='a'),ref('supp',role='supplement'),ref('bad')]
        refs[-1]['match']=dict(status='external_or_unresolved',target=None)
        self.assertEqual(cg.graph(papers(),refs)[0],[])
        self.assertEqual([r['edge_class'] for r in refs],['self_loop','supplement','excluded'])

    def test_unverified_boundary_and_citing_identity(self):
        refs=[ref()];refs[0]['parse_status']='boundary_unverified'
        self.assertEqual(cg.graph(papers(),refs)[0],[])
        ps=papers();ps['a']['identity_status']='conflict'
        self.assertEqual(cg.graph(ps,[ref()])[0],[])

    def test_known_pagerank_dangling(self):
        ranks,_,_=cg.pagerank(['a','b'],[('a','b')])
        self.assertAlmostEqual(ranks['a'],1/2.85,places=11)
        self.assertAlmostEqual(ranks['b'],1.85/2.85,places=11)
        self.assertAlmostEqual(sum(ranks.values()),1.0,places=12)

    def test_cycle_empty_no_edges(self):
        self.assertEqual(cg.pagerank([],[])[0],{})
        self.assertEqual(cg.pagerank(['a','b'],[])[0],{'a':.5,'b':.5})
        self.assertEqual(cg.pagerank(['b','a'],[('a','b'),('b','a')])[0],{'a':.5,'b':.5})

    def test_nonconvergence_rejected(self):
        with self.assertRaises(ValueError):cg.pagerank(['a','b'],[('a','b')],max_iterations=1)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.c1=self.root/'c1.sqlite3'; self.inv=self.root/'inventory.json';self.out=self.root/'graph.sqlite3'
        self.source=self.root/'a.md'; self.source.write_text('## References\n1. Smith. Rectal cancer staging. 2020. DOI:10.1234/b\n',encoding='utf-8')
        self.b=self.root/'b.md';self.b.write_text('## References\nNone.\n',encoding='utf-8')
        r=pr.PaperRegistry(self.c1,create=True)
        try:
            for path,doi in [(self.source,'10.1234/a'),(self.b,'10.1234/b')]:
                r.register(dict(source=path.name,content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),identifiers={'doi':doi},metadata={'title':'Rectal cancer staging','year':'2020'}))
        finally:r.close()
        self.inventory=[[p.name,hashlib.sha256(p.read_bytes()).hexdigest()] for p in (self.source,self.b)]
        self.inv.write_text(json.dumps(self.inventory),encoding='utf-8')

    def build(self,output=None):
        return cg.build(self.c1,self.inv,self.root,None,output or self.out)

    def test_rerun_readonly_locators_export(self):
        before=self.c1.read_bytes();s=self.build(); data=cg.snapshot(self.out)
        self.assertEqual(self.build(),s);self.assertEqual(cg.snapshot(self.out),data)
        self.assertEqual(cg.verify(self.out,True)['locators_checked'],1)
        c=cg.open_graph(self.out)
        try:
            with self.assertRaises(sqlite3.OperationalError):c.execute('CREATE TABLE bad(x)')
        finally:c.close()
        self.assertEqual(before,self.c1.read_bytes())
        with redirect_stdout(io.StringIO()):cg.main(['export','--db',str(self.out),'--output',str(self.root/'export.json')])
        self.assertTrue((self.root/'export.json').exists())

    def test_input_order_version_change(self):
        s=self.build();self.inv.write_text(json.dumps(list(reversed(self.inventory))),encoding='utf-8')
        self.assertEqual(self.build()['citation_graph_version'],s['citation_graph_version'])
        self.source.write_text('## References\n1. Smith. Another. 2021. DOI:10.1234/b\n',encoding='utf-8')
        # A mismatched source is a new, incomplete graph, never the old success.
        new=self.build();self.assertNotEqual(new['citation_graph_version'],s['citation_graph_version'])
        self.assertEqual(new['extraction_artifacts']['source_binding_mismatch'],1)
        self.assertFalse(new['coverage_complete'])

    def test_missing_source_not_zero(self):
        self.source.unlink();s=self.build()
        self.assertEqual(s['extraction_artifacts'],{'source_missing':1,'confirmed_no_references':1})

    def test_bad_db_no_writes_all_commands(self):
        for kind in ('c1','c2','ordinary','unknown_c3'):
            path=self.root/(kind+'.sqlite3')
            if kind!='c1':
                c=sqlite3.connect(path)
                try:
                    if kind=='c2':c.execute('CREATE TABLE meta(key,value)')
                    elif kind=='unknown_c3':c.execute('CREATE TABLE c3_meta(key,value)');c.execute("INSERT INTO c3_meta VALUES ('schema','unknown')")
                    else:c.execute('CREATE TABLE ordinary(x)')
                    c.commit()
                finally:c.close()
            before=path.read_bytes()
            for command in ('status','verify','export'):
                with self.assertRaises((ValueError,sqlite3.DatabaseError)):
                    args=[command,'--db',str(path)]
                    if command=='export':args+=['--output',str(self.root/'bad.json')]
                    cg.main(args)
            with self.assertRaises((ValueError,sqlite3.DatabaseError)):self.build(path)
            self.assertEqual(path.read_bytes(),before)
        self.assertFalse((self.root/'bad.json').exists())

    def test_output_protection(self):
        for target in (self.c1,self.inv,self.source,self.b):
            before=target.read_bytes()
            with self.assertRaises(ValueError):self.build(target)
            self.assertEqual(before,target.read_bytes())
        self.build()
        for target in (self.c1,self.source,self.out,self.inv):
            with self.assertRaises(ValueError):cg.main(['export','--db',str(self.out),'--output',str(target)])

    def test_missing_readonly_no_create(self):
        path=self.root/'absent.sqlite3'
        with self.assertRaises(sqlite3.OperationalError):cg.open_graph(path)
        self.assertFalse(path.exists())

    def test_locator_corruption_detected(self):
        self.build();self.source.write_text('changed',encoding='utf-8')
        with self.assertRaises(ValueError):cg.verify(self.out,True)


if __name__=='__main__':unittest.main()
