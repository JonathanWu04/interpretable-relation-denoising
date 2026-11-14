import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from relden.conversion import convert_split
from relden.data import load_split
from relden.features import feature_dict, masked_context
from relden.evaluation import threshold_for, folds, metrics
from relden.model import fit_model, predict, export_model, predict_exported


def make_row(i=0,noise=False):
    text=f'Alice worked at Acme during year {i}.'
    return {'instance_id':str(i),'sentence_text_id':str(i),'relation':'/business/person/company',
            'text':text,'h':{'name':'Alice','id':'alice','pos':[0,5]},
            't':{'name':'Acme','id':'acme','pos':[16,20]},'is_noise':noise,'validation_status':'valid'}


class Contracts(unittest.TestCase):
    def test_feature_labels_and_identifiers_cannot_leak(self):
        a=make_row(); b={**a,'is_noise':True,'instance_id':'secret','sentence_text_id':'secret','source_split':'test'}
        for variant in ['manual','rule_lr','relation_only','rule_relation','context_only','context_lr']:
            self.assertEqual(feature_dict(a,variant),feature_dict(b,variant))

    def test_entities_are_masked(self):
        row=make_row(); features=feature_dict(row,'context_lr')
        self.assertFalse(any('alice' in k or 'acme' in k for k in features))
        self.assertIn('local:worked at',features)

    def test_retention_threshold_ties_and_boundaries(self):
        y=np.array([0,0,0,0,1,1]); p=np.array([.1,.2,.8,.8,.8,.9])
        t=threshold_for(y,p,'retain90',.9)
        self.assertEqual(t,.9)
        self.assertEqual(metrics(y,p>=t)['clean_retention'],1.)
        t=threshold_for(y,p,'recall90',.9)
        self.assertEqual(t,.8)

    def test_sentence_group_isolation(self):
        rows=[{**make_row(i, bool(i%2)), 'sentence_text_id':str(i//2)} for i in range(30)]
        seen=[]
        for train,val in folds(rows,3,123):
            self.assertFalse({rows[i]['sentence_text_id'] for i in train}&{rows[i]['sentence_text_id'] for i in val})
            seen.extend(val)
        self.assertEqual(sorted(seen),list(range(30)))

    def test_portable_export_matches_sklearn_and_ignores_unseen_features(self):
        rows=[make_row(i,bool(i%2)) for i in range(20)]
        for i,row in enumerate(rows):
            row['text']=row['text'].replace('worked','stayed' if i%2 else 'worked')
        for variant in ['rule_lr','context_lr']:
            xs=[feature_dict(r,variant) for r in rows]
            pipe=fit_model(xs,[r['is_noise'] for r in rows],variant,1.)
            model=export_model(pipe,variant,1.,{})
            np.testing.assert_allclose(predict(pipe,xs),predict_exported(model,rows),atol=1e-12)

    def test_unknown_ambiguous_duplicate_and_conflicting_labels(self):
        records=[{'sentText':'Alice works at Acme.', 'relationMentions':[{'em1Text':'Alice','em2Text':'Acme','label':'business/person/company','is_noise':False}]*2},
                 {'sentText':'Bob left Acme.', 'relationMentions':[{'em1Text':'Bob','em2Text':'Acme','label':'business/person/company','is_noise':None}]},
                 {'sentText':'Alice met Alice at Acme.', 'relationMentions':[{'em1Text':'Alice','em2Text':'Acme','label':'business/person/company','is_noise':False}]}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'dev.json'; path.write_text('\n'.join(json.dumps(r) for r in records))
            rows,a=load_split(path)
            self.assertEqual(len(rows),1)
            self.assertEqual(a['duplicate_annotations'],1)
            self.assertEqual(a['unknown_label'],1)
            self.assertEqual(a['invalid_or_ambiguous_span'],1)
            records[0]['relationMentions'][1]={**records[0]['relationMentions'][1],'is_noise':True}
            path.write_text('\n'.join(json.dumps(r) for r in records))
            rows,a=load_split(path); self.assertEqual(len(rows),0)
            self.assertEqual(a['conflicting_annotations'],2)

    def test_empty_denominators_are_not_reported_as_success(self):
        m=metrics([0,0],[False,False]); self.assertIsNone(m['noise_recall']); self.assertIsNone(m['noise_precision'])

if __name__=='__main__': unittest.main()
