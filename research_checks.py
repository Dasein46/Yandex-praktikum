"""Evaluation utilities shared by the audited research notebooks."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline


def smape(target, prediction):
    """sMAPE in percent; two zeros contribute zero, nonfinite inputs fail."""
    target, prediction = np.asarray(target, dtype=float), np.asarray(prediction, dtype=float)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError('Expected nonempty arrays with identical shapes')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError('sMAPE requires finite targets and predictions')
    denominator = np.abs(target) + np.abs(prediction)
    ratios = np.divide(2 * np.abs(target - prediction), denominator,
                       out=np.zeros_like(denominator), where=denominator != 0)
    return 100 * ratios.mean(axis=0)


def weighted_smape(target, prediction):
    """Two targets in order: rougher recovery (25%), final recovery (75%)."""
    target = np.asarray(target)
    if target.ndim != 2 or target.shape[1] != 2:
        raise ValueError('Expected exactly two target columns')
    return float(np.dot(smape(target, prediction), [0.25, 0.75]))


def gold_search(model, parameters, features, target):
    pipeline = Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                         ('model', model)])
    search = GridSearchCV(pipeline, {'model__' + k: v for k, v in parameters.items()},
                          cv=TimeSeriesSplit(n_splits=5),
                          scoring=make_scorer(weighted_smape, greater_is_better=False),
                          error_score='raise')
    return search.fit(features, target)


def taxi_features(data, max_lag, rolling_mean_size):
    """Features for a one-hour-ahead forecast with observed past demand."""
    if not isinstance(data.index, pd.DatetimeIndex) or not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError('Expected a sorted unique DatetimeIndex')
    if len(data) < 2 or not (data.index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError('Expected a regular hourly series')
    if max_lag < 1 or rolling_mean_size < 1:
        raise ValueError('Lag and window sizes must be positive')
    if data['num_orders'].isna().any() or not np.isfinite(data['num_orders']).all():
        raise ValueError('Demand must be observed and finite')
    result = data.copy()
    for name in ['month', 'day', 'dayofweek', 'hour']:
        result[name] = getattr(data.index, name)
    for lag in range(1, max_lag + 1):
        result[f'lag_{lag}'] = data['num_orders'].shift(lag)
    result['rolling_mean'] = data['num_orders'].shift(1).rolling(rolling_mean_size).mean()
    return result


def text_search(model, parameters, features, target, stop_words=None):
    pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words=stop_words)), ('model', model)])
    search = GridSearchCV(pipeline, {'model__' + k: v for k, v in parameters.items()},
                          cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=12345),
                          scoring='f1', error_score='raise')
    return search.fit(features, target)
