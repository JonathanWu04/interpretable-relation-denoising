"""Predeclared nested, sentence-grouped experiments and diagnostic evaluation."""
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from .data import load_split, pair_key
from .features import prepare_features
from .model import fit_model, predict, export_model, predict_exported
from .evaluation import (labels, folds, choose_thresholds, decisions, summarize,
                         paired_bootstrap)


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def subset(xs, idx):
    return [xs[i] for i in idx]


def tune(rows, features, variant, config, seed):
    if variant=='manual':
        return {'C':None,'scores':np.array([x['raw_noise'] for x in features]),'grid':[]}
    grid=[]; best=None
    inner=folds(rows,config['inner_folds'],seed)
    y=labels(rows)
    for C in config['C_grid']:
        probs=np.full(len(rows),np.nan)
        for train,val in inner:
            pipe=fit_model(subset(features,train),y[train],variant,C)
            probs[val]=predict(pipe,subset(features,val))
        assert np.isfinite(probs).all()
        loss=float(log_loss(y,probs,labels=[0,1])); grid.append({'C':C,'log_loss':loss})
        if best is None or (loss,C)<best[0]: best=((loss,C),probs.copy())
    return {'C':best[0][1],'scores':best[1],'grid':grid}


def prediction_rows(rows, scores, drops, fold_ids=None):
    # Hashes and decisions suffice to audit metrics without republishing news text.
    return [{'id':hashlib.sha256('|'.join([r['sentence_text_id'],r['h']['id'],r['t']['id'],r['relation']]).encode()).hexdigest(),
             'group':r['sentence_text_id'],'relation':r['relation'],'is_noise':r['is_noise'],
             'noise_score':float(s),'drop':{k:bool(v[i]) for k,v in drops.items()},
             **({'fold':int(fold_ids[i])} if fold_ids is not None else {})}
            for i,(r,s) in enumerate(zip(rows,scores))]


def run(data_dir, output_dir, protocol_path):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(protocol_path).read_text())
    write_json(out/'protocol.json',config)
    write_json(out/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
                                    'numpy':np.__version__,'sklearn':sklearn.__version__,
                                    'thread_limit':1})
    data_dir=Path(data_dir); start=time.perf_counter()
    dev,audit=load_split(data_dir/'dev.json')
    write_json(out/'dev_audit.json',audit)
    variants=config['models']; fs=prepare_features(dev,variants); y=labels(dev)
    outer=folds(dev,config['outer_folds'],config['seed'])
    fold_ids=np.full(len(dev),-1)
    lineage=[]
    for f,(train,val) in enumerate(outer):
        fold_ids[val]=f
        lineage.append({'fold':f,'train_groups':sorted({dev[i]['sentence_text_id'] for i in train}),
                        'validation_groups':sorted({dev[i]['sentence_text_id'] for i in val})})
    write_json(out/'outer_groups.json',lineage)
    all_results={}; predictions={}; fitted={}; frozen={}; all_tuning={}
    for variant in variants:
        vs=time.perf_counter(); scores=np.full(len(dev),np.nan)
        drops={k:np.zeros(len(dev),bool) for k in config['policies']}; fold_details=[]
        for f,(train,val) in enumerate(outer):
            tr=subset(dev,train); va=subset(dev,val)
            selected=tune(tr,subset(fs[variant],train),variant,config,config['seed']+f+1)
            if variant=='manual': p=np.array([fs[variant][i]['raw_noise'] for i in val])
            else:
                pipe=fit_model(subset(fs[variant],train),y[train],variant,selected['C'])
                p=predict(pipe,subset(fs[variant],val))
            scores[val]=p
            thresholds={k:choose_thresholds(tr,selected['scores'],k,t) for k,t in config['policies'].items()}
            fd={'fold':f,'C':selected['C'],'grid':selected['grid'],'thresholds':thresholds,'metrics':{}}
            for policy,ts in thresholds.items():
                drops[policy][val]=decisions(va,p,ts)
                fd['metrics'][policy]=summarize(va,drops[policy][val],p)
            fold_details.append(fd)
            print(variant,'outer',f,'complete',round(time.perf_counter()-vs,2),'s',flush=True)
        all_results[variant]={'oof':{k:summarize(dev,d,scores) for k,d in drops.items()},'folds':fold_details}
        predictions[variant]={'scores':scores,'drops':drops}
        write_json(out/'predictions'/f'{variant}_oof.json',prediction_rows(dev,scores,drops,fold_ids))
        selected=tune(dev,fs[variant],variant,config,config['seed']+100)
        thresholds={k:choose_thresholds(dev,selected['scores'],k,t) for k,t in config['policies'].items()}
        all_tuning[variant]={'C':selected['C'],'grid':selected['grid'],'thresholds':thresholds}
        if variant=='manual':
            frozen[variant]={'variant':'manual','thresholds':thresholds}
        else:
            fit_start=time.perf_counter(); pipe=fit_model(fs[variant],y,variant,selected['C'])
            fitted[variant]=pipe
            frozen[variant]=export_model(pipe,variant,selected['C'],thresholds,
                {'train_split':'official dev, deduplicated valid labeled records','train_n':len(dev),
                 'final_fit_seconds':time.perf_counter()-fit_start,
                 'development_source_sha256':audit['source_sha256']})
            assert np.allclose(predict(pipe,fs[variant]),predict_exported(frozen[variant],dev),atol=1e-12)
        all_results[variant]['development_experiment_seconds']=time.perf_counter()-vs
    # Freeze every model and development-based decision BEFORE reading test labels.
    candidate=all_results['context_lr']['oof']['retain90']; baseline=all_results['rule_lr']['oof']['retain90']
    passed=(candidate['overall']['noise_recall']>baseline['overall']['noise_recall'] and
            candidate['overall']['clean_retention']>=baseline['overall']['clean_retention']-.02 and
            all(candidate[g]['clean_retention']>=.88 for g in ['NA','non-NA']))
    selected_variant='context_lr' if passed else 'rule_lr'
    decision={'selected_variant':selected_variant,'context_promotion_passed':passed,
              'selection_used':'development OOF only; protocol promotion gate',
              'policy':'retain90','no_pristine_holdout':True}
    write_json(out/'selection.json',decision); write_json(out/'tuning.json',all_tuning)
    for v,m in frozen.items(): write_json(out/'models'/f'{v}.json',m)
    write_json(out/'development_results.json',all_results)
    write_json(out/'frozen_manifest.json',{v:hashlib.sha256((out/'models'/f'{v}.json').read_bytes()).hexdigest() for v in variants})
    comparisons=[('manual','rule_lr'),('rule_lr','context_lr'),('relation_only','context_lr'),('rule_relation','context_lr'),('context_only','context_lr')]
    ci={}
    for a,b in comparisons:
        ci[f'{b}_minus_{a}']=paired_bootstrap(dev,predictions[a]['drops']['retain90'],predictions[b]['drops']['retain90'])
    write_json(out/'oof_paired_bootstrap.json',ci)
    seen_text={r['sentence_text_id'] for r in dev}; seen_pair={pair_key(r) for r in dev}
    test_audits={}; timings={}
    for split in ['test','test_noise']:
        rows,split_audit=load_split(data_dir/f'{split}.json'); test_audits[split]=split_audit
        diagnostic_sets={split:rows,
                         split+'_no_dev_text':[r for r in rows if r['sentence_text_id'] not in seen_text],
                         split+'_unseen_pairs':[r for r in rows if r['sentence_text_id'] not in seen_text and pair_key(r) not in seen_pair]}
        split_audit['without_dev_text']=len(diagnostic_sets[split+'_no_dev_text'])
        split_audit['without_dev_text_or_pair']=len(diagnostic_sets[split+'_unseen_pairs'])
        for name,rs in diagnostic_sets.items():
            feats=prepare_features(rs,variants)
            for variant in variants:
                p=np.array([x['raw_noise'] for x in feats[variant]]) if variant=='manual' else predict(fitted[variant],feats[variant])
                ds={k:decisions(rs,p,ts) for k,ts in frozen[variant]['thresholds'].items()}
                all_results[variant][name]={k:summarize(rs,d,p) for k,d in ds.items()}
                write_json(out/'predictions'/f'{variant}_{name}.json',prediction_rows(rs,p,ds))
        if split=='test':
            for variant in ['rule_lr','context_lr']:
                times=[]
                for rep in range(5):
                    t=time.perf_counter(); scores=predict_exported(frozen[variant],rows)
                    _=decisions(rows,scores,frozen[variant]['thresholds']['retain90'])
                    elapsed=time.perf_counter()-t
                    if rep: times.append(elapsed)
                timings[variant]={'n':len(rows),'seconds':times,'median_seconds':float(np.median(times)),
                                 'scope':'feature extraction + portable JSON scoring + decisions; excludes disk IO',
                                 'model_json_bytes':(out/'models'/f'{variant}.json').stat().st_size}
    write_json(out/'diagnostic_audits.json',test_audits)
    write_json(out/'timings.json',timings); write_json(out/'results.json',all_results)
    write_json(out/'run_summary.json',{'elapsed_seconds':time.perf_counter()-start,'selected':decision,
                                     'models':variants,'protocol_sha256':hashlib.sha256(Path(protocol_path).read_bytes()).hexdigest()})
    print('DONE',json.dumps(decision),flush=True)
    for v in variants:
        print(v,'OOF',all_results[v]['oof']['retain90']['overall'],flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path); p.add_argument('--protocol',type=Path,default=Path('configs/protocol.json'))
    a=p.parse_args()
    with threadpool_limits(limits=1): run(a.data_dir,a.out,a.protocol)

if __name__=='__main__': main()
