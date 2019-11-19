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
import multiprocessing

import pandas as pd
import numpy as np

from tsar.baseline import featurize_index_for_baseline, make_periods, \
    fit_scalar_baseline, normalized_residual_to_data, fit_many_baselines,\
    compute_baseline


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

    data = pd.read_pickle('tests/data/wind_test_data.pickle')
    train = data[data.index.year.isin([2010, 2011])]
    test = data[data.index.year == 2012]

    def test_fit_baseline(self):
        print(self.train.head())

        daily_harmonics, weekly_harmonics, annual_harmonics, \
            trend, baseline_fit_results, std = \
            fit_scalar_baseline(self.train,
                                K_day=None,
                                K_week=None,
                                K_year=None,
                                K_trend=None,
                                train_test_ratio=2/3,
                                gamma=1E-8, W=2)

        train_baseline = normalized_residual_to_data(
            pd.Series(0., index=self.train.index),
            std,
            daily_harmonics,
            weekly_harmonics,
            annual_harmonics,
            trend,
            baseline_fit_results)
        print(train_baseline.head())
        self.assertTrue(np.isclose((self.train - train_baseline).mean(), 0))

        # print('test rmse from grid search', test_rmse)

        train_rmse = (self.train - train_baseline).std()
        print('train rmse', train_rmse)
        self.assertTrue(train_rmse < 6)

        test_baseline = normalized_residual_to_data(
            pd.Series(0, index=self.test.index),
            std,
            daily_harmonics,
            weekly_harmonics,
            annual_harmonics,
            trend,
            baseline_fit_results)

        test_mean_error = (self.test - test_baseline).mean()
        print('test mean', test_mean_error)

        self.assertTrue(np.abs(test_mean_error) < .4)

        test_rmse = np.sqrt(((self.test - test_baseline)**2).mean())
        print('test std', test_rmse)

        self.assertTrue(test_rmse > train_rmse)

    def test_fit_many_baselines(self):

        mydata = pd.concat([self.train, self.test], axis=1)
        mydata = pd.concat([mydata, mydata], axis=1)

        mydata.columns = ['col%d' % i for i in range(mydata.shape[1])]

        import time
        s = time.time()
        all_baseline_fit_results, baseline_params_dict = \
            fit_many_baselines(mydata,
                               {col: {} for col in mydata.columns},
                               parallel=False)
        non_par_time = (time.time()-s)
        print('non parallel took %.2f seconds' % non_par_time)
        print(baseline_params_dict)

        bas1 = compute_baseline(self.test.index,
                                **baseline_params_dict['col1'],
                                baseline_fit_result=all_baseline_fit_results[
                                    'col1']['baseline_fit_result'])

        bas2 = compute_baseline(self.test.index,
                                **baseline_params_dict['col2'],
                                baseline_fit_result=all_baseline_fit_results[
                                    'col2']['baseline_fit_result'])

        print(np.mean(bas1 - bas2) / bas1.mean())
        self.assertTrue(np.mean(bas1 - bas2) / bas1.mean() < 0.05)
        print(np.mean((bas1 - bas2)**2) / (bas1**2).mean())
        self.assertTrue(np.mean((bas1 - bas2)**2) / (bas1**2).mean() < 0.05)

        s = time.time()
        all_baseline_fit_results_par, baseline_params_dict_par = \
            fit_many_baselines(mydata,
                               {col: {} for col in mydata.columns},
                               parallel=True)
        par_time = (time.time()-s)
        print('parallel took %.2f seconds' % par_time)

        if multiprocessing.cpu_count() >= 4:
            self.assertTrue(par_time < non_par_time)

        print(baseline_params_dict_par)

        self.assertEqual(baseline_params_dict_par, baseline_params_dict)
        self.assertTrue(np.all(all_baseline_fit_results_par['col1'][
            'baseline_fit_result'] ==
            all_baseline_fit_results['col1'][
            'baseline_fit_result']))
