"""Portable inference that never silently discards uncertain records."""
import argparse
import json
from pathlib import Path
from .data import read_prediction_input
from .model import predict_exported
from .evaluation import decisions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--policy',choices=['retain90','retain95','recall90'],default='retain90')
    args=parser.parse_args(); model=json.loads(args.model.read_text())
    if model.get('variant')=='manual': raise ValueError('CLI expects a learned model JSON')
    rows=read_prediction_input(args.input)
    with args.output.open('x') as f:
        for row in rows:
            if row['validation_status']!='valid':
                result={'decision':'abstain','reason':row['validation_status'],'noise_score':None}
            else:
                score=predict_exported(model,[row]); drop=decisions([row],score,model['thresholds'][args.policy])[0]
                result={'decision':'drop' if drop else 'keep','reason':args.policy,'noise_score':float(score[0])}
            f.write(json.dumps({'instance_id':row['instance_id'],**result})+'\n')

if __name__=='__main__': main()
