"""Synthetic B5 boundaries; no model or private corpus."""
import sys, unittest, tempfile, json
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'index'))
from document_index import bounded_retrieve, merge_candidates, card_text, build, DocumentIndex
import document_metadata as dm

def h(i,source=None,doi=''):
    return dict(id=str(i),source=source or str(i)+'.md',doi=doi,score=1-i/10000,section='s')
def canonical_row(**kw):
    row=dict(PMID='1',Title='Canonical title',Abstract='Canonical abstract',Journal='Canonical journal',
             Year='2020',PubType='Journal Article',MeSH='Humans',DOI='10.1000/main')
    row.update(kw); return row
def metadata_card(source,**kw):
    card=dict(source=source,doi='',title='',abstract='',keywords='',pub_type='',year='',journal='',
              abstract_passage_ids=[])
    card.update(kw); return card
class DocumentTests(unittest.TestCase):
    def run_case(self,fn,pc=100,dc=0,**kw):
        return bounded_retrieve(fn,lambda n:[],pc,dc,target_unique_papers=4,initial_budget=2,max_budget=20,**kw)[1]
    def test_cap(self):
        papers,t=merge_candidates([h(i,'same','DOI') for i in range(8)]+[h(9)],[h(10,'same','doi')])
        self.assertEqual(papers[0]['n_hits'],3)
        self.assertEqual(len(t['capped_passage_ids']),5)
        self.assertAlmostEqual(papers[0]['score'],2/61)
        self.assertEqual(t['passage_mean_chunks_per_paper'],4.5)
    def test_target(self): self.assertEqual(self.run_case(lambda n:[h(i) for i in range(n)])['stop'],'target_reached')
    def test_exhausted(self): self.assertEqual(self.run_case(lambda n:[h(0)],pc=1)['stop'],'corpus_exhausted')
    def test_no_new(self): self.assertEqual(self.run_case(lambda n:[h(0)])['stop'],'no_new_candidates')
    def test_fixed(self): self.assertEqual(self.run_case(lambda n:[h(0)],dynamic=False)['stop'],'fixed_budget')
    def test_budget_cap(self):
        _,t=bounded_retrieve(lambda n:[h(i,'same') for i in range(n)],lambda n:[],100,0,target_unique_papers=4,initial_budget=2,max_budget=4)
        self.assertEqual(t['stop'],'candidate_cap')
    def test_empty(self): self.assertEqual(self.run_case(lambda n:[],pc=0)['stop'],'corpus_exhausted')
    def test_invalid(self):
        with self.assertRaises(ValueError): bounded_retrieve(None,None,1,1,target_unique_papers=51)
        with self.assertRaises(ValueError): merge_candidates([h(0),h(0)],[])
        with self.assertRaises(ValueError): self.run_case(lambda n:[h(i) for i in range(n+1)])
    def test_missing(self): self.assertIn('abstract: [missing]',card_text({'title':'a'}))
    def test_tie(self):
        a,_=merge_candidates([h(1)],[h(2)])
        self.assertEqual([x['source'] for x in a],['1.md','2.md'])
    def test_cards_cosine_binding_missing(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cards'; binding={'pair':'synthetic'}
            build([h(1),h(2)],np.array([[1.,0.],[0.,1.]]),path,binding,'model')
            with patch('sparse_index.pair_binding',return_value=binding):
                idx=DocumentIndex(path,'db','man','model')
                self.assertEqual([x['id'] for x in idx.search_vector([0.,1.],2)],['card:2.md','card:1.md'])
                with self.assertRaises(ValueError): DocumentIndex(path,'db','man','other')
            with patch('sparse_index.pair_binding',return_value={'pair':'other'}):
                with self.assertRaises(ValueError): DocumentIndex(path,'db','man','model')
            with self.assertRaises(FileNotFoundError): DocumentIndex(path/'missing','db','man','model')
    def test_cli_document_route(self):
        import retrieve, document_index
        argv=['retrieve','query','-k','50','--document-path','cards','--db','db','--manifest','man','--model','model']
        with patch.object(sys,'argv',argv), patch.object(document_index,'retrieve_document_literature',return_value=[]) as call:
            retrieve.main()
            self.assertEqual(call.call_args.kwargs['max_chunks_per_paper'],3)
            self.assertEqual(call.call_args.kwargs['manifest_path'],'man')
    def test_missing_cards_never_open_client(self):
        import rag_core, document_index
        with tempfile.TemporaryDirectory() as tmp, patch.object(rag_core,'index_preflight') as check:
            with self.assertRaises(FileNotFoundError):
                document_index.retrieve_document_literature('q',Path(tmp)/'missing')
            check.assert_not_called()
    def test_foreign_cards_never_open_client(self):
        import rag_core, document_index
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); (path/'config.json').write_text(json.dumps({'binding':{'pair':'foreign'}}))
            with patch('sparse_index.pair_binding',return_value={'pair':'actual'}), patch.object(rag_core,'index_preflight') as check:
                with self.assertRaises(ValueError):
                    document_index.retrieve_document_literature('q',path,db_dir='synthetic',manifest_path='synthetic')
                check.assert_not_called()
class MetadataTests(unittest.TestCase):
    def test_pmid_exact_priority_and_doi_conflict(self):
        index=dm.build_canonical_index([canonical_row(PMID='1',DOI='10.1000/a',Title='PM1',Abstract='A1'),
                                        canonical_row(PMID='2',DOI='10.1000/b',Title='PM2',Abstract='A2')])
        out=dm.enrich_cards([metadata_card('PMID_1.md',doi='10.1000/b')],index)[0]
        self.assertEqual((out['title'],out['abstract']),('PM1','A1'))
        self.assertEqual(out['metadata']['identity']['kind'],'pmid')
        self.assertEqual(out['metadata']['identity']['canonical_key'],'pmid:1')
        self.assertEqual([c['kind'] for c in out['metadata']['conflicts']],['pmid_doi_mismatch'])
    def test_duplicate_key_same_value(self):
        index=dm.build_canonical_index([canonical_row(Abstract='same'),canonical_row(Abstract='same')])
        self.assertEqual(index['duplicate_rows']['pmid'],1)
        self.assertEqual(index['conflicts'],[])
        out=dm.enrich_cards([metadata_card('PMID_1.md')],index)[0]
        self.assertEqual(out['abstract'],'same')
        self.assertEqual(out['metadata']['provenance']['abstract'],
                         {'source':'canonical','file':'','key':'pmid:1','truncated':False,
                          'source_integrity':'complete','source_preview_limit':None})
    def test_duplicate_key_conflicting_field(self):
        index=dm.build_canonical_index([canonical_row(Abstract='one'),canonical_row(Abstract='two')])
        self.assertEqual(index['by_pmid']['1']['Abstract'],'')
        self.assertIn('Abstract',[c['field'] for c in index['conflicts']])
        out=dm.enrich_cards([metadata_card('PMID_1.md')],index)[0]
        self.assertEqual(out['abstract'],'')
        self.assertEqual(out['metadata']['provenance']['abstract']['source'],'missing')
        self.assertEqual(out['pub_type'],'Journal Article')
    def test_duplicate_doi_is_ambiguous(self):
        index=dm.build_canonical_index([canonical_row(PMID='1',DOI='10.1000/x',Title='A'),
                                        canonical_row(PMID='2',DOI='10.1000/x',Title='B')])
        self.assertNotIn('10.1000/x',index['by_doi'])
        self.assertEqual(index['conflicts'][0]['reason'],'doi_maps_to_multiple_pmids')
        mapping=dm.build_mapping_index([dict(filename='different.pdf',doi='10.1000/x',abstract='Map abstract')])
        out=dm.enrich_cards([metadata_card('other.md',doi='https://doi.org/10.1000/X')],index,mapping)[0]
        self.assertEqual(out['metadata']['identity']['kind'],'doi')
        self.assertEqual(out['metadata']['identity']['canonical_key'],'')
        self.assertEqual(out['metadata']['identity']['mapping_key'],'')
        self.assertEqual((out['title'],out['abstract'],out['pub_type']),('','',''))
        self.assertEqual([c['kind'] for c in out['metadata']['conflicts']],['ambiguous_doi'])
    def test_mapping_backfill_by_doi(self):
        index=dm.build_canonical_index([])
        mapping=dm.build_mapping_index([dict(filename='Paper 2020.pdf',doi='10.1000/m',title='Map title',
                                             abstract='Map abstract',year='2021',journal='Map journal')])
        out=dm.enrich_cards([metadata_card('Paper 2020.md',doi='10.1000/m')],index,mapping)[0]
        self.assertEqual((out['title'],out['abstract'],out['year'],out['journal']),
                         ('Map title','Map abstract','2021','Map journal'))
        self.assertEqual(out['pub_type'],'')
        self.assertEqual(out['metadata']['identity']['mapping_key'],'doi:10.1000/m')
        self.assertEqual(out['metadata']['provenance']['abstract']['source'],'mapping')
    def test_field_priority(self):
        index=dm.build_canonical_index([canonical_row(DOI='10.1000/p',Title='C title',Abstract='C abstract',
                                                      PubType='Review')])
        mapping=dm.build_mapping_index([dict(filename='p.pdf',doi='10.1000/p',title='M title',
                                             abstract='M abstract',journal='M journal')])
        card=metadata_card('p.md',doi='10.1000/p',title='Card title',abstract='Card abstract',
                           journal='Card journal',year='1998',pub_type='Card type')
        out=dm.enrich_cards([card],index,mapping)[0]
        self.assertEqual([out[k] for k in ('title','abstract','year','journal','pub_type')],
                         ['Card title','C abstract','1998','Card journal','Review'])
        prov=out['metadata']['provenance']
        self.assertEqual((prov['title']['source'],prov['abstract']['source'],prov['pub_type']['source']),
                         ('card','canonical','canonical'))
        empty=dm.build_canonical_index([canonical_row(DOI='10.1000/p',Abstract='',PubType='')])
        out=dm.enrich_cards([card],empty,mapping)[0]
        self.assertEqual(out['abstract'],'M abstract')
        self.assertEqual(out['metadata']['provenance']['abstract']['source'],'mapping')
        self.assertEqual(out['pub_type'],'Card type')
    def test_mesh_never_keywords(self):
        index=dm.build_canonical_index([canonical_row(MeSH='Humans; Rectal Neoplasms')])
        out=dm.enrich_cards([metadata_card('PMID_1.md')],index)[0]
        self.assertEqual(out['keywords'],'')
        self.assertEqual(out['mesh_terms'],'Humans; Rectal Neoplasms')
        self.assertEqual(out['metadata']['provenance']['keywords']['source'],'missing')
        self.assertEqual(out['metadata']['provenance']['mesh_terms']['source'],'canonical')
        text=card_text(out)
        self.assertIn('keywords: [missing]',text)
        self.assertIn('mesh_terms (MeSH): Humans; Rectal Neoplasms',text)
        self.assertNotIn('keywords: Humans',text)
    def test_doi_normalization(self):
        index=dm.build_canonical_index([canonical_row(DOI='10.1000/AbC.',Title='Normalized')])
        for doi in ('https://doi.org/10.1000/abc','DOI: 10.1000/ABC',' <http://dx.doi.org/10.1000/abc.> '):
            out=dm.enrich_cards([metadata_card('x.md',doi=doi)],index)[0]
            self.assertEqual(out['metadata']['identity']['canonical_key'],'doi:10.1000/abc')
            self.assertEqual(out['title'],'Normalized')
            self.assertEqual(out['metadata']['conflicts'],[])
    def test_empty_fields_stay_missing(self):
        index=dm.build_canonical_index([canonical_row(Abstract='',PubType='',MeSH='')])
        out=dm.enrich_cards([metadata_card('PMID_1.md')],index)[0]
        self.assertEqual((out['abstract'],out['pub_type'],out['mesh_terms']),('','',''))
        prov=out['metadata']['provenance']
        self.assertTrue(all(prov[f]['source']=='missing' for f in ('abstract','pub_type','mesh_terms')))
    def test_abstract_truncation(self):
        index=dm.build_canonical_index([canonical_row(Abstract='x'*6001)])
        out=dm.enrich_cards([metadata_card('PMID_1.md')],index)[0]
        self.assertEqual(len(out['abstract']),6000)
        self.assertTrue(out['abstract_truncated'])
        self.assertEqual(out['metadata']['provenance']['abstract'],
                         {'source':'canonical','file':'','key':'pmid:1','truncated':True,
                          'source_integrity':'complete','source_preview_limit':None})
    def test_no_title_guessing(self):
        index=dm.build_canonical_index([canonical_row(DOI='10.1000/x',Title='Same title')])
        out=dm.enrich_cards([metadata_card('other.md',doi='10.1000/y',title='Same title')],index)[0]
        self.assertEqual(out['title'],'Same title')
        self.assertEqual(out['metadata']['identity']['canonical_key'],'')
        self.assertEqual(out['abstract'],'')
        self.assertEqual(out['metadata']['provenance']['abstract']['source'],'missing')
    def test_identity_preserved_and_input_untouched(self):
        index=dm.build_canonical_index([canonical_row(DOI='10.1000/a')])
        cards=[metadata_card('a.md',doi='10.1000/a'),metadata_card('PMID_2.md'),metadata_card('c.md')]
        before=json.dumps(cards,sort_keys=True)
        out=dm.enrich_cards(cards,index)
        self.assertEqual([c['source'] for c in out],[c['source'] for c in cards])
        self.assertEqual(json.dumps(cards,sort_keys=True),before)
        with self.assertRaises(ValueError): dm.enrich_cards([metadata_card('a.md'),metadata_card('a.md')],index)
        with self.assertRaises(ValueError): dm.enrich_cards([],index,abstract_limit=0)
    def test_mapping_by_filename(self):
        index=dm.build_canonical_index([])
        mapping=dm.build_mapping_index([dict(filename='Paper 2020.pdf',doi='10.1000/m',abstract='Map abstract')])
        out=dm.enrich_cards([metadata_card('Paper 2020.md')],index,mapping)[0]
        self.assertEqual(out['metadata']['identity']['kind'],'filename')
        self.assertEqual(out['metadata']['identity']['mapping_key'],'file:Paper 2020')
        self.assertEqual(out['abstract'],'Map abstract')
        self.assertEqual(out['metadata']['identity']['canonical_key'],'')
    def test_card_doi_wins_over_conflicting_mapping(self):
        index=dm.build_canonical_index([canonical_row(DOI='10.1000/a',Title='A',Abstract='A abstract'),
                                        canonical_row(DOI='10.1000/b',Title='B')])
        mapping=dm.build_mapping_index([dict(filename='P.pdf',doi='10.1000/b',abstract='B abstract')])
        out=dm.enrich_cards([metadata_card('P.md',doi='10.1000/a')],index,mapping)[0]
        self.assertEqual((out['title'],out['abstract']),('A','A abstract'))
        self.assertEqual(out['metadata']['identity']['mapping_key'],'')
        self.assertEqual([c['kind'] for c in out['metadata']['conflicts']],['doi_mismatch'])
    def test_mapping_duplicate_conflict_protected(self):
        mapping=dm.build_mapping_index([dict(filename='P.pdf',doi='10.1000/m',abstract='one'),
                                        dict(filename='Q.pdf',doi='10.1000/m',abstract='two')])
        self.assertEqual(mapping['by_doi']['10.1000/m']['abstract'],'')
        self.assertIn('abstract',[c['field'] for c in mapping['conflicts']])
        out=dm.enrich_cards([metadata_card('P.md',doi='10.1000/m')],dm.build_canonical_index([]),mapping)[0]
        self.assertEqual(out['abstract'],'')
        self.assertEqual(out['metadata']['provenance']['abstract']['source'],'missing')


    def test_pmid_conflict_blocks_mapping_pollution(self):
        index=dm.build_canonical_index([canonical_row(PMID='1',DOI='10.1000/a',Title='PM1',Abstract=''),
                                        canonical_row(PMID='2',DOI='10.1000/b',Title='PM2',Abstract='OTHER_PAPER')])
        mapping=dm.build_mapping_index([dict(filename='wrong.pdf',doi='10.1000/b',
                                             title='OTHER_TITLE',abstract='OTHER_PAPER')])
        out=dm.enrich_cards([metadata_card('PMID_1.md',doi='10.1000/b')],index,mapping)[0]
        self.assertEqual((out['title'],out['abstract'],out['pub_type']),('PM1','','Journal Article'))
        self.assertEqual(out['metadata']['identity']['mapping_key'],'')
        self.assertIn('pmid_doi_mismatch',[c['kind'] for c in out['metadata']['conflicts']])
        self.assertEqual(out['metadata']['provenance']['abstract']['source'],'missing')
    def test_pmid_empty_canonical_doi_blocks_other_pmid_mapping(self):
        index=dm.build_canonical_index([canonical_row(PMID='1',DOI='',Title='PM1',Abstract=''),
                                        canonical_row(PMID='2',DOI='10.1000/b',Title='PM2',Abstract='OTHER_PAPER')])
        mapping=dm.build_mapping_index([dict(filename='wrong.pdf',doi='10.1000/b',abstract='OTHER_PAPER')])
        out=dm.enrich_cards([metadata_card('PMID_1.md',doi='10.1000/b')],index,mapping)[0]
        self.assertEqual((out['title'],out['abstract']),('PM1',''))
        self.assertEqual(out['metadata']['identity']['mapping_key'],'')
        kinds=[c['kind'] for c in out['metadata']['conflicts']]
        self.assertIn('pmid_doi_mismatch',kinds)
        self.assertIn('pmid_mapping_mismatch',kinds)
    def test_pmid_filename_mapping_with_foreign_doi_rejected(self):
        index=dm.build_canonical_index([canonical_row(PMID='1',DOI='10.1000/a',Title='PM1',Abstract=''),
                                        canonical_row(PMID='2',DOI='10.1000/b',Title='PM2',Abstract='OTHER_PAPER')])
        mapping=dm.build_mapping_index([dict(filename='PMID_1.pdf',doi='10.1000/b',abstract='OTHER_PAPER')])
        out=dm.enrich_cards([metadata_card('PMID_1.md')],index,mapping)[0]
        self.assertEqual(out['abstract'],'')
        self.assertEqual(out['metadata']['identity']['mapping_key'],'')
        self.assertEqual([c['kind'] for c in out['metadata']['conflicts']],['pmid_mapping_mismatch'])
    def test_pmid_filename_mapping_same_pmid_survives_card_doi_conflict(self):
        index=dm.build_canonical_index([canonical_row(PMID='1',DOI='10.1000/a',Title='PM1',Abstract=''),
                                        canonical_row(PMID='2',DOI='10.1000/b',Title='PM2',Abstract='OTHER_PAPER')])
        mapping=dm.build_mapping_index([dict(filename='PMID_1.pdf',doi='10.1000/a',abstract='CORRECT_ABSTRACT')])
        out=dm.enrich_cards([metadata_card('PMID_1.md',doi='10.1000/b')],index,mapping)[0]
        self.assertEqual(out['abstract'],'CORRECT_ABSTRACT')
        self.assertEqual(out['metadata']['identity']['mapping_key'],'file:PMID_1')
        self.assertIn('pmid_doi_mismatch',[c['kind'] for c in out['metadata']['conflicts']])
        self.assertNotIn('pmid_mapping_mismatch',[c['kind'] for c in out['metadata']['conflicts']])
    def test_stored_abstract_truncation_preserved(self):
        index=dm.build_canonical_index([canonical_row(PMID='1',Abstract='')])
        card=metadata_card('PMID_1.md',abstract='x'*6000,abstract_truncated=True,abstract_passage_ids=['p1'])
        out=dm.enrich_cards([card],index)[0]
        self.assertEqual(len(out['abstract']),6000)
        self.assertTrue(out['abstract_truncated'])
        self.assertEqual(out['abstract_passage_ids'],['p1'])
        self.assertEqual(out['metadata']['provenance']['abstract'],
                         {'source':'card','file':'','key':'card:PMID_1.md','truncated':True,
                          'source_integrity':'stored','source_preview_limit':None})
    def test_complete_abstract_recomputes_truncation_and_archives_ids(self):
        index=dm.build_canonical_index([canonical_row(PMID='1',Abstract='y'*100)])
        card=metadata_card('PMID_1.md',abstract='x'*6000,abstract_truncated=True,
                           abstract_passage_ids=['p1','p2'])
        out=dm.enrich_cards([card],index)[0]
        self.assertEqual(out['abstract'],'y'*100)
        self.assertFalse(out['abstract_truncated'])
        self.assertEqual(out['abstract_passage_ids'],[])
        self.assertEqual(out['abstract_passage_ids_historical'],['p1','p2'])
        self.assertEqual(out['metadata']['provenance']['abstract']['truncated'],False)
        self.assertEqual(out['metadata']['provenance']['abstract']['source'],'canonical')
    def test_provenance_source_files(self):
        index=dm.build_canonical_index([canonical_row(PMID='1',Abstract='')])
        mapping=dm.build_mapping_index([dict(filename='PMID_1.pdf',doi='10.1000/main',abstract='Map abstract')])
        files={'card':'synthetic/old-cards.json','canonical':'synthetic/canonical.csv',
               'mapping':'synthetic/mapping.json'}
        out=dm.enrich_cards([metadata_card('PMID_1.md',title='Card title')],index,mapping,
                            source_files=files)[0]
        prov=out['metadata']['provenance']
        self.assertEqual(prov['title'],{'source':'card','file':files['card'],'key':'card:PMID_1.md'})
        self.assertEqual(prov['abstract'],{'source':'mapping','file':files['mapping'],
                                           'key':'file:PMID_1','truncated':False,
                                           'source_integrity':'preview_unknown',
                                           'source_preview_limit':200})
        self.assertEqual(prov['pub_type']['file'],files['canonical'])
        self.assertEqual(prov['mesh_terms']['file'],files['canonical'])
        self.assertEqual(prov['keywords']['source'],'missing')
        self.assertEqual(prov['keywords']['file'],'')

    def test_mapping_preview_marks_unknown_integrity(self):
        index=dm.build_canonical_index([])
        preview='P'*200
        mapping=dm.build_mapping_index([dict(filename='P.pdf',doi='10.1000/p',abstract=preview)])
        out=dm.enrich_cards([metadata_card('P.md',doi='10.1000/p')],index,mapping)[0]
        self.assertEqual(out['abstract'],preview)
        self.assertEqual(len(out['abstract']),200)
        self.assertFalse(out['abstract_truncated'])
        prov=out['metadata']['provenance']['abstract']
        self.assertEqual(prov['source'],'mapping')
        self.assertEqual(prov['truncated'],False)
        self.assertEqual(prov['source_integrity'],'preview_unknown')
        self.assertEqual(prov['source_preview_limit'],200)
        self.assertNotEqual(prov['source_integrity'],'complete')
    def test_short_mapping_preview_still_unknown(self):
        index=dm.build_canonical_index([])
        preview='Q'*199
        mapping=dm.build_mapping_index([dict(filename='Q.pdf',doi='10.1000/q',abstract=preview)])
        out=dm.enrich_cards([metadata_card('Q.md',doi='10.1000/q')],index,mapping)[0]
        self.assertEqual(len(out['abstract']),199)
        self.assertFalse(out['abstract_truncated'])
        prov=out['metadata']['provenance']['abstract']
        self.assertEqual(prov['source_integrity'],'preview_unknown')
        self.assertEqual(prov['source_preview_limit'],200)
    def test_canonical_complete_replaces_mapping_preview(self):
        index=dm.build_canonical_index([canonical_row(DOI='10.1000/p',Abstract='C'*450)])
        mapping=dm.build_mapping_index([dict(filename='P.pdf',doi='10.1000/p',abstract='M'*200)])
        out=dm.enrich_cards([metadata_card('P.md',doi='10.1000/p')],index,mapping)[0]
        self.assertEqual(out['abstract'],'C'*450)
        self.assertFalse(out['abstract_truncated'])
        prov=out['metadata']['provenance']['abstract']
        self.assertEqual(prov['source'],'canonical')
        self.assertEqual(prov['source_integrity'],'complete')
        self.assertIsNone(prov['source_preview_limit'])
        self.assertNotIn('M'*200,out['abstract'])


if __name__=='__main__': unittest.main()
