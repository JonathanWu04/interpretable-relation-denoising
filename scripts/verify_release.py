"""Recompute every released metric from saved decisions and verify group isolation."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from relden.evaluation import summarize


def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('results/20261114')
    results=json.loads((root/'results.json').read_text())
    checked=0
    observed=set()
    expected_keys={(v,s,p) for v,values in results.items()
                   for s,block in values.items() if isinstance(block,dict) and 'retain90' in block
                   for p in ['retain90','retain95','recall90']}
    for path in sorted((root/'predictions').glob('*.json*')):
        if path.name.endswith('.gz'):
            with gzip.open(path,'rt') as f: records=json.load(f)
            stem=path.name[:-8]
        else: records=json.loads(path.read_text()); stem=path.stem
        variant=next(v for v in sorted(results,key=len,reverse=True) if stem.startswith(v+'_'))
        split=stem[len(variant)+1:]
        if split=='oof': split='oof'
        rows=[{'relation':r['relation'],'is_noise':r['is_noise']} for r in records]
        assert len({r['id'] for r in records})==len(records),path
        frozen=json.loads((root/'models'/f'{variant}.json').read_text())
        for policy in ['retain90','retain95','recall90']:
            key=(variant,split,policy)
            assert key not in observed,('Duplicate prediction artifact',key)
            observed.add(key)
            for record in records:
                thresholds=(results[variant]['folds'][record['fold']]['thresholds']
                            if split=='oof' else frozen['thresholds'])
                group='NA' if record['relation']=='NA' else 'non-NA'
                assert record['drop'][policy]==(record['noise_score']>=thresholds[policy][group]),(path,record['id'],policy)
            actual=summarize(rows,np.array([r['drop'][policy] for r in records]),[r['noise_score'] for r in records])
            expected=results[variant][split][policy]
            for group,values in actual.items():
                for key,value in values.items():
                    other=expected[group][key]
                    if value is None: assert other is None,(path,group,key)
                    else: assert abs(value-other)<1e-10,(path,group,key,value,other)
            checked+=1
    assert observed==expected_keys,('Missing or unexpected results',expected_keys-observed,observed-expected_keys)
    groups=json.loads((root/'outer_groups.json').read_text()); seen=set()
    for fold in groups:
        a=set(fold['train_groups']); b=set(fold['validation_groups'])
        assert not a&b; assert not seen&b; seen|=b
    manifest=json.loads((root/'frozen_manifest.json').read_text())
    for variant,digest in manifest.items():
        assert hashlib.sha256((root/'models'/f'{variant}.json').read_bytes()).hexdigest()==digest
    print(f'PASS: {checked} policy/split/model tables independently recomputed; group boundaries and frozen model hashes verified.')

if __name__=='__main__': main()
