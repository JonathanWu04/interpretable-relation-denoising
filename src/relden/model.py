"""Small L2 logistic model with portable JSON inference (no pickle)."""
import math
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from .features import feature_dict


def fit_model(features, labels, variant, C):
    pipe = make_pipeline(DictVectorizer(sparse=True),
                         StandardScaler(with_mean=False) if variant == 'rule_lr' else MaxAbsScaler(),
                         LogisticRegression(C=C, solver='liblinear', penalty='l2',
                                            max_iter=2000, tol=1e-7, random_state=0))
    pipe.fit(features, labels)
    if pipe[-1].n_iter_.max() >= 2000:
        raise RuntimeError('Model failed to converge')
    return pipe


def predict(pipe, features):
    if list(pipe[-1].classes_) != [0, 1]:
        raise ValueError('Expected clean=0, noise=1')
    return pipe.predict_proba(features)[:, 1]


def export_model(pipe, variant, C, thresholds, metadata=None):
    vocabulary = pipe[0].get_feature_names_out()
    weights = pipe[-1].coef_[0] / pipe[1].scale_
    return {'schema_version': 1, 'variant': variant, 'C': C,
            'intercept': float(pipe[-1].intercept_[0]),
            'coefficients': {str(k): float(v) for k, v in zip(vocabulary, weights) if v != 0},
            'thresholds': thresholds, 'metadata': metadata or {},
            'score_note': 'Noise probability estimate; not externally calibrated.'}


def predict_exported(model, rows):
    scores = []
    for row in rows:
        x = feature_dict(row, model['variant'])
        z = model['intercept'] + sum(v * model['coefficients'].get(k, 0.0) for k, v in x.items())
        scores.append(1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z)))
    return np.asarray(scores)

