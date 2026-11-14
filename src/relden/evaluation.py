"""Evaluation contracts: all thresholds are learned from training-side OOF."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold


def labels(rows):
    if any(r['is_noise'] is None for r in rows):
        raise ValueError('Unknown labels cannot enter quality metrics or fitting')
    return np.array([int(r['is_noise']) for r in rows])


def folds(rows, n_splits, seed, group_field='sentence_text_id'):
    y = labels(rows)
    groups = np.array([r[group_field] for r in rows])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = list(splitter.split(np.zeros(len(rows)), y, groups))
    for train, val in result:
        assert set(groups[train]).isdisjoint(groups[val])
    return result


def metrics(y, drop, scores=None):
    y = np.asarray(y, dtype=bool); drop = np.asarray(drop, dtype=bool)
    tp = int(np.sum(y & drop)); fp = int(np.sum(~y & drop))
    fn = int(np.sum(y & ~drop)); tn = int(np.sum(~y & ~drop))
    div = lambda a, b: a / b if b else None
    out = dict(n=len(y), clean=tn+fp, noise=tp+fn, TP=tp, FP=fp, FN=fn, TN=tn,
               noise_precision=div(tp,tp+fp), noise_recall=div(tp,tp+fn),
               noise_f1=div(2*tp,2*tp+fp+fn), clean_retention=div(tn,tn+fp),
               retained_purity=div(tn,tn+fn), retained_fraction=div(tn+fn,len(y)))
    if scores is not None and len(set(y)) == 2:
        out['roc_auc'] = float(roc_auc_score(y, scores))
        out['average_precision'] = float(average_precision_score(y, scores))
    return out


def threshold_for(y, scores, policy, target):
    """Exact tied-score thresholds; strict drop >= t; keep-all always possible."""
    y = np.asarray(y, dtype=bool); scores = np.asarray(scores)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite score')
    order = np.argsort(-scores, kind='stable'); s=scores[order]; z=y[order]
    ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True]) if len(s) else np.array([],int)
    candidates = [(float(np.nextafter(s.max(), np.inf)) if len(s) else 1.0, 0, 0)]
    tp = np.cumsum(z); fp=np.cumsum(~z)
    candidates += [(float(s[i]), int(tp[i]), int(fp[i])) for i in ends]
    clean=int((~y).sum()); noise=int(y.sum()); chosen=None
    for t,a,b in candidates:
        cr=(clean-b)/clean if clean else 1.0; nr=a/noise if noise else 0.0
        if policy.startswith('retain'):
            feasible = cr >= target - 1e-12
            key=(nr,cr,t)
        else:
            feasible = nr >= target - 1e-12
            key=(cr,nr,t)
        if feasible and (chosen is None or key > chosen[0]):
            chosen=(key,t)
    if chosen is None:
        raise ValueError('No feasible threshold')
    return chosen[1]


def choose_thresholds(rows, scores, policy, target):
    y=labels(rows); scores=np.asarray(scores)
    result={}
    for group in ['NA','non-NA']:
        mask=np.array([(r['relation']=='NA')==(group=='NA') for r in rows])
        result[group]=threshold_for(y[mask],scores[mask],policy,target)
    return result


def decisions(rows, scores, thresholds):
    return np.array([s>=thresholds['NA' if r['relation']=='NA' else 'non-NA'] for r,s in zip(rows,scores)])


def summarize(rows, drop, scores=None):
    y=labels(rows); drop=np.asarray(drop)
    score_array=None if scores is None else np.asarray(scores)
    masks={'overall': np.ones(len(rows),bool),
           'NA': np.array([r['relation']=='NA' for r in rows]),
           'non-NA': np.array([r['relation']!='NA' for r in rows])}
    masks.update({'relation:'+rel: np.array([r['relation']==rel for r in rows]) for rel in sorted({r['relation'] for r in rows})})
    return {g: metrics(y[m],drop[m],None if score_array is None else score_array[m]) for g,m in masks.items()}


def paired_bootstrap(rows, left, right, n=1000, seed=42):
    groups={}
    for i,r in enumerate(rows): groups.setdefault(r['sentence_text_id'],[]).append(i)
    values=list(groups.values()); rng=np.random.default_rng(seed); y=labels(rows)
    samples={'noise_recall':[],'clean_retention':[],'retained_purity':[]}
    for _ in range(n):
        idx=np.concatenate([values[i] for i in rng.integers(len(values),size=len(values))])
        a=metrics(y[idx],left[idx]); b=metrics(y[idx],right[idx])
        for key in samples:
            if a[key] is not None and b[key] is not None: samples[key].append(b[key]-a[key])
    return {k: {'difference':metrics(y,right)[k]-metrics(y,left)[k],
                'ci95':np.quantile(v,[.025,.975]).tolist()} for k,v in samples.items()}
