"""
Copyright © Enzo Busseti 2019.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from unittest import TestCase
from .baseline import *

import pandas as pd


class TestBaseline(TestCase):

    def test_featurize_index_for_baseline(self):
        X = featurize_index_for_baseline(np.array(range(10)),
                                         np.array([100, 20]),
                                         True)
        self.assertEqual(X[0, 0], 0.)
        self.assertEqual(X[1, 0], np.sin(2 * np.pi * 1 / 100))
        self.assertEqual(X[1, 1], np.cos(2 * np.pi * 1 / 100))
        self.assertEqual(X[1, -1], 1E-9)

    def test_make_periods(self):
        periods = make_periods(1, 1, 0)
        self.assertEqual(periods[0], 86400)
        self.assertEqual(periods[1], 86400 * 7)
        periods = make_periods(3, 1, 1)
        self.assertEqual(periods[2], 86400 / 3)
        self.assertEqual(periods[-1], 8766 * 3600)

    data = pd.read_pickle('data/wind_test_data.pickle')
    train = data[data.index.year.isin([2010, 2011])]
    test = data[data.index.year == 2012]

    def test_fit_baseline(self):
        print(self.train.head())

        std, daily_harmonics, weekly_harmonics, annual_harmonics, \
            trend, baseline_fit_results, test_rmse = \
            fit_baseline(
                self.train,
                self.test,
                daily_harmonics=None,
                weekly_harmonics=None,
                annual_harmonics=None,
                trend=None)

        train_baseline = residual_to_data(
            pd.Series(0., index=self.train.index),
            std,
            daily_harmonics,
            weekly_harmonics,
            annual_harmonics,
            trend,
            baseline_fit_results)
        print(train_baseline.head())
        self.assertTrue(np.isclose((self.train - train_baseline).mean(), 0))

        print('test rmse from grid search', test_rmse)

        train_rmse = (self.train - train_baseline).std()
        print('train rmse', train_rmse)
        self.assertTrue(train_rmse < 6)

        test_baseline = residual_to_data(pd.Series(0, index=self.test.index),
                                         std,
                                         daily_harmonics,
                                         weekly_harmonics,
                                         annual_harmonics,
                                         trend,
                                         baseline_fit_results)

        test_mean_error = (self.test - test_baseline).mean()
        print('test mean', test_mean_error)

        self.assertTrue(np.abs(test_mean_error) < .4)

        test_std = (self.test - test_baseline).std()
        print('test std', test_rmse)

        self.assertTrue(test_rmse > train_rmse)
