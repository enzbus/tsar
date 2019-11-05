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
from .AR import *
from .baseline import *


class TestScalarAR(TestCase):

    def fit_scalar_ar(self, future_lag, max_past_lag, train, test):
        total_errors = []

        for past_lag in range(0, max_past_lag):

            lag = future_lag + past_lag

            lagged_covariances, Sigma = \
                fit_AR(train.values, [], lag)

            rmse = rmse_AR(V=np.empty((Sigma.shape[0], 0)),
                           S=np.empty((0, 0)),
                           S_inv=np.empty((0, 0)),
                           D_blocks=[Sigma],
                           D_matrix=sp.block_diag([Sigma]).tocsc(),
                           D_inv=np.linalg.inv(Sigma),
                           past_lag=past_lag, future_lag=future_lag,
                           test=pd.DataFrame(test),
                           available_data_lags_columns={0: 0})

            total_errors.append(rmse.sum().sum())
        return total_errors

    def test_scalar_ar(self):

        np.random.seed(0)

        T = 10000
        data = pd.DataFrame(
            index=pd.date_range(start=pd.datetime.now(),
                                periods=T, freq='D'),
            data=np.random.randn(T))

        data = data.rolling(5).max()

        for i in range(100):

            data.iloc[np.random.choice(range(len(data)))] = np.nan

        train = data.iloc[:-T // 2]
        mean = train.mean()
        train -= mean
        test = data.iloc[-T // 2:]
        test -= mean

        future_lag = 10
        total_errors = self.fit_scalar_ar(future_lag, 20, train, test)

        self.assertTrue(np.argmin(total_errors) == 17)

        past_lag, rank, V, S, S_inv, D_blocks, D_matrix, D_inv = \
            fit_low_rank_plus_block_diagonal_AR(train,
                                                test,
                                                future_lag,
                                                past_lag=None,
                                                rank=None,
                                                available_data_lags_columns={0: 0},
                                                ignore_prediction_columns=[])

        self.assertEqual(past_lag, 14)

    def test_scalar_ar_2(self):

        np.random.seed(0)

        T = 10000
        data = pd.DataFrame(
            index=pd.date_range(start=pd.datetime.now(),
                                periods=T, freq='D'),
            data=np.random.randn(T))

        data = data.rolling(5).mean()

        train = data.iloc[:-T // 2]
        mean = train.mean()
        train -= mean
        test = data.iloc[-T // 2:]
        test -= mean

        future_lag = 10

        total_errors = self.fit_scalar_ar(future_lag, 21, train, test)

        self.assertTrue(np.argmin(total_errors) == 20)

        past_lag, rank, V, S, S_inv, D_blocks, D_matrix, D_inv = \
            fit_low_rank_plus_block_diagonal_AR(train,
                                                test,
                                                future_lag,
                                                past_lag=None,
                                                rank=None,
                                                available_data_lags_columns={0: 0},
                                                ignore_prediction_columns=[])

        self.assertEqual(past_lag, 20)

    def test_scalar_ar_3(self):

        np.random.seed(0)

        T = 10000
        data = pd.DataFrame(
            index=pd.date_range(start=pd.datetime.now(),
                                periods=T, freq='D'),
            data=np.random.randn(T))

        data -= data.rolling(5).mean()

        train = data.iloc[:-T // 2]
        mean = train.mean()
        train -= mean
        test = data.iloc[-T // 2:]
        test -= mean

        future_lag = 5

        total_errors = self.fit_scalar_ar(future_lag, 11, train, test)

        self.assertTrue(np.argmin(total_errors) == 10)

        past_lag, rank, V, S, S_inv, D_blocks, D_matrix, D_inv = \
            fit_low_rank_plus_block_diagonal_AR(train,
                                                test,
                                                future_lag,
                                                past_lag=None,
                                                rank=None,
                                                available_data_lags_columns={0: 0},
                                                ignore_prediction_columns=[])

        self.assertEqual(past_lag, 10)


class TestWithScalarWindData(TestCase):
    data = pd.read_pickle('data/wind_test_data.pickle')
    train = data[data.index.year.isin([2010, 2011])]
    test = data[data.index.year == 2012]

    def test_fit_AR(self):

        std, daily_harmonics, weekly_harmonics, annual_harmonics, \
            trend, baseline_fit_results, test_rmse = \
            fit_baseline(
                self.train,
                self.test,
                daily_harmonics=None,
                weekly_harmonics=None,
                annual_harmonics=None,
                trend=None)

        train_residual = data_to_residual(self.train,
                                          std,
                                          daily_harmonics,
                                          weekly_harmonics,
                                          annual_harmonics,
                                          trend,
                                          baseline_fit_results)
        test_residual = data_to_residual(self.test,
                                         std,
                                         daily_harmonics,
                                         weekly_harmonics,
                                         annual_harmonics,
                                         trend,
                                         baseline_fit_results)

        future_lag = 4 * 24

        past_lag, rank, V, S, S_inv, D_blocks, D_matrix, D_inv = \
            fit_low_rank_plus_block_diagonal_AR(pd.DataFrame(train_residual),
                                                pd.DataFrame(test_residual),
                                                future_lag,
                                                past_lag=4 * 24 * 2,
                                                rank=None,
                                                available_data_lags_columns={0: 0},
                                                ignore_prediction_columns=[])

        real_RMSE = std * rmse_AR(V, S, S_inv, D_blocks,
                                  D_matrix, D_inv,
                                  past_lag, future_lag,
                                  pd.DataFrame(test_residual),
                                  available_data_lags_columns={0: 0})

        self.assertTrue(real_RMSE.iloc[0, 0] < 1.4)
        self.assertTrue(real_RMSE.iloc[0, 0] > 1.3)
        self.assertTrue(real_RMSE.iloc[-1, 0] > 5.5)
        self.assertTrue(real_RMSE.iloc[-1, 0] < 5.6)
