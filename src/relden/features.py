"""Fixed linguistic rules plus entity-masked local lexical features.

Only sentence text, entity spans, and the supplied relation enter features.
Neither gold validity nor identifiers enter the design matrix.
"""
from collections import Counter
import re
from .legacy_rules import HeuristicAttentionProvider, SentenceInstance

RULE_FEATURES = ['distance', 'between', 'keyword', 'order', 'boosts',
                 'uncertainty', 'clutter', 'overlap']


def components(row, provider=None):
    provider = provider or HeuristicAttentionProvider()
    instance = SentenceInstance(row['relation'], row['text'], row['text'].lower(),
                                row['h']['name'], row['t']['name'],
                                tuple(row['h']['pos']), tuple(row['t']['pos']), {})
    return provider.score_components(instance)


def masked_context(row):
    text = row['text'].lower()
    spans = sorted([(row['h']['pos'], 'h_ent'), (row['t']['pos'], 't_ent')])
    (a, h), (b, t) = spans
    if a[1] > b[0]:
        return ['overlapping_entities'], []
    before = re.findall(r"[a-z]+(?:'[a-z]+)?|[.,;!?]", text[:a[0]])[-4:]
    between = re.findall(r"[a-z]+(?:'[a-z]+)?|[.,;!?]", text[a[1]:b[0]])
    after = re.findall(r"[a-z]+(?:'[a-z]+)?|[.,;!?]", text[b[1]:])[:4]
    # Preserve both ends for distant entities without unbounded feature growth.
    if len(between) > 40:
        between = between[:20] + ['gap'] + between[-20:]
    return before + [h] + between + [t] + after, between


def feature_dict(row, variant, cached_components=None):
    c = cached_components if cached_components is not None else components(row)
    if variant == 'manual':
        return {'raw_noise': -c['raw_score']}
    if variant == 'rule_lr':
        return {**{k: c[k] for k in RULE_FEATURES}, 'is_NA': float(row['relation'] == 'NA')}
    relation = row['relation']
    result = {f'relation={relation}': 1.0}
    if variant == 'relation_only':
        return result
    if variant not in ('rule_relation', 'context_only', 'context_lr'):
        raise ValueError(f'Unknown variant {variant}')
    if variant != 'context_only':
        for name in RULE_FEATURES:
            result['rule:' + name] = c[name]
            result[f'{relation}|rule:{name}'] = c[name]
    if variant.startswith('context_'):
        local, between = masked_context(row)
        for zone, tokens in [('local', local), ('between', between)]:
            terms = set(tokens) | {a + ' ' + b for a, b in zip(tokens, tokens[1:])}
            for term in terms:
                result[f'{zone}:{term}'] = 1.0
                result[f'{relation}|{zone}:{term}'] = 1.0
        result['head_first'] = float(row['h']['pos'][0] < row['t']['pos'][0])
    return result


def prepare_features(rows, variants):
    provider = HeuristicAttentionProvider()
    cs = [components(row, provider) for row in rows]
    return {v: [feature_dict(r, v, c) for r, c in zip(rows, cs)] for v in variants}

