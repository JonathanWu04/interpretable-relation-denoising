"""Validate spans and labels, deduplicate annotations, and preserve provenance."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

from .conversion import convert_split, normalize_name, parse_noise


def annotation_key(row):
    return (row['sentence_text_id'], row['h']['id'], row['t']['id'], row['relation'])


def pair_key(row):
    return '|'.join(sorted([row['h']['id'], row['t']['id']]))


def load_split(path, require_labels=True):
    """Official ARNOR JSON/JSONL or this package's traceable converted JSONL."""
    path = Path(path)
    with path.open() as f:
        first = f.readline().strip()
    first_object = json.loads(first) if first and not first.startswith('[') else {}
    if 'validation_status' in first_object:
        rows = [json.loads(line) for line in path.open() if line.strip()]
    else:
        rows = convert_split(path)
    counts = Counter(input_records=len(rows))
    grouped = defaultdict(list)
    for row in rows:
        row['is_noise'] = parse_noise(row.get('is_noise'))
        if row.get('validation_status') != 'valid':
            counts['invalid_or_ambiguous_span'] += 1
            continue
        for entity in (row['h'], row['t']):
            start, end = entity['pos']
            if not 0 <= start < end <= len(row['text']):
                raise ValueError('Out-of-bounds entity span')
            if normalize_name(row['text'][start:end]) != normalize_name(entity['name']):
                raise ValueError('Entity span does not match entity surface')
        if require_labels and row['is_noise'] is None:
            counts['unknown_label'] += 1
            continue
        grouped[annotation_key(row)].append(row)
    kept = []
    for records in grouped.values():
        if len({r['is_noise'] for r in records}) > 1:
            counts['conflicting_annotations'] += len(records)
            continue
        counts['duplicate_annotations'] += len(records) - 1
        kept.append(records[0])
    kept.sort(key=lambda r: r['instance_id'])
    counts['eligible_records'] = len(kept)
    counts['sentence_groups'] = len({r['sentence_text_id'] for r in kept})
    counts['entity_pairs'] = len({pair_key(r) for r in kept})
    audit = dict(counts)
    audit['labels'] = dict(Counter(str(r['is_noise']) for r in kept))
    audit['source_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return kept, audit


def read_prediction_input(path):
    """Convert without dropping rows: uncertain spans are explicitly abstained on."""
    return convert_split(path)

