import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import clone
from research_checks import smape, weighted_smape, taxi_features, gold_search, text_search


class ResearchChecks(unittest.TestCase):
    def test_smape_known_values(self):
        self.assertEqual(smape([0, 10], [0, 0]), 100)
        self.assertAlmostEqual(smape([10], [20]), 200/3)
        self.assertEqual(weighted_smape([[10, 10]], [[10, 0]]), 150)

    def test_smape_bad_input(self):
        for a, b in [([], []), ([1], [[1]]), ([np.nan], [0]), ([1], [np.inf])]:
            with self.assertRaises(ValueError): smape(a, b)
        with self.assertRaises(ValueError): weighted_smape([1, 2], [1, 2])

    def test_features_are_causal_and_do_not_mutate_input(self):
        data = pd.DataFrame({'num_orders': np.arange(100.)}, index=pd.date_range('2020-01-01', periods=100, freq='h'))
        before = data.copy(deep=True)
        first = taxi_features(data, 24, 3)
        changed = data.copy(); changed.iloc[50:] = 10000
        second = taxi_features(changed, 24, 3)
        pd.testing.assert_frame_equal(data, before)
        pd.testing.assert_frame_equal(first.drop(columns='num_orders').iloc[:51], second.drop(columns='num_orders').iloc[:51])
        self.assertEqual(first.iloc[50]['rolling_mean'], 48)
        self.assertEqual(first.iloc[50]['lag_24'], 26)

    def test_irregular_series_fails(self):
        data = pd.DataFrame({'num_orders': [1,2]}, index=pd.to_datetime(['2020-01-01', '2020-01-03']))
        with self.assertRaises(ValueError): taxi_features(data, 1, 1)

    def test_gold_cv_fit_and_fold_imputation(self):
        x = pd.DataFrame({'x': np.arange(60.)}); x.loc[3, 'x'] = np.nan
        y = np.column_stack([np.arange(60.)+10, np.arange(60.)+20])
        search = gold_search(LinearRegression(), {}, x, y)
        self.assertTrue(np.isfinite(search.best_score_))
        self.assertEqual(search.predict(x).shape, (60, 2))
        for train, valid in search.cv.split(x):
            self.assertLess(train.max(), valid.min())
            fitted = clone(search.estimator).fit(x.iloc[train], y[train])
            self.assertEqual(fitted['imputer'].statistics_[0], x.iloc[train]['x'].median())

    def test_text_vocabulary_is_fold_local(self):
        texts = pd.Series(['good shared unique'+str(i) if i%2 else 'bad shared unique'+str(i) for i in range(30)])
        target = np.arange(30)%2
        search = text_search(LogisticRegression(random_state=1), {'C': [1, 2]}, texts, target)
        self.assertTrue(np.isfinite(search.best_score_))
        for train, valid in search.cv.split(texts, target):
            fitted = clone(search.estimator).fit(texts.iloc[train], target[train])
            self.assertNotIn('unique'+str(valid[0]), fitted['tfidf'].vocabulary_)
        self.assertEqual(len(search.predict(texts)), 30)

if __name__ == '__main__': unittest.main()
