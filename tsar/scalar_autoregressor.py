"""
Copyright (C) Enzo Busseti 2019.

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


import numpy as np
import pandas as pd
import numba as nb
import logging
import scipy.sparse.linalg as spl
logger = logging.getLogger(__name__)
import numba as nb

from .utils import check_series
from .greedy_grid_search import greedy_grid_search
from .base_autoregressor import BaseAutoregressor


@nb.jit(nopython=True)
def make_Sigma_scalar_AR(lagged_covariances):
    lag = len(lagged_covariances)
    Sigma = np.empty((lag, lag))
    for i in range(lag):
        for k in range(-i, lag - i):
            Sigma[i, k + i] = lagged_covariances[
                k] if k > 0 else lagged_covariances[-k]
    # assert np.allclose((Sigma - Sigma.T), 0)
    return Sigma


@nb.jit(nopython=True)
def lag_covariance(array, lag):
    # shifted = np.shift(series, lag)
    multiplied = array[lag:] * array[:len(array) - lag]
    return np.nanmean(multiplied)  # [~np.isnan(multiplied)])
    # cov = pd.concat([series, series.shift(lag)], axis=1).cov()
    # mycov = cov.iloc[1, 0]
    # assert np.isclose(newcov, mycov)
    # return newcov


@nb.jit()  # nopython=True)
def update_covariance_Sigma(train_array, old_lag_covariances, lag):
    lagged_covariances = np.empty(lag)
    lagged_covariances[:len(old_lag_covariances)] = old_lag_covariances
    for i in range(len(old_lag_covariances), lag):
        print('computing covariance at lag', i)
        # cov = pd.concat([self.train,
        #                  self.train.shift(i)], axis=1).cov()
        # mycov = cov.iloc[1, 0]
        mycov = lag_covariance(train_array, lag=i)
        if np.isnan(mycov):
            logger.warning(
                'Covariance at lag %d for column %s is NaN.' %
                (i, self.train.name))
            mycov = 0.
        lagged_covariances[i] = mycov
    Sigma = make_Sigma_scalar_AR(lagged_covariances)
    return lagged_covariances, Sigma


class ScalarAutoregressor(BaseAutoregressor):

    def __init__(self,
                 train,
                 future_lag,
                 past_lag):

        check_series(train)
        self.train = train
        assert np.isclose(self.train.mean(), 0., atol=1e-6)
        self.future_lag = future_lag
        self.past_lag = past_lag
        self.lagged_covariances = np.empty(0)
        self.N = 1

        self._fit()

    # def _fit(self):
    #     self._fit_Sigma()
    #     self._make_Sigma()

    # @property
    # def lag(self):
    #     return self.future_lag + self.past_lag

    def _fit(self):
        self.lagged_covariances, self.Sigma = \
            update_covariance_Sigma(train_array=self.train.values,
                                    old_lag_covariances=self.lagged_covariances,
                                    lag=self.lag)
        # old_lag_covariances = self.lagged_covariances
        # self.lagged_covariances = np.empty(self.lag)
        # self.lagged_covariances[
        #     :len(old_lag_covariances)] = old_lag_covariances
        # for i in range(len(old_lag_covariances), self.lag):
        #     print('computing covariance lag %d' % i)
        #     # cov = pd.concat([self.train,
        #     #                  self.train.shift(i)], axis=1).cov()
        #     # mycov = cov.iloc[1, 0]
        #     mycov = lag_covariance(self.train.values, lag=i)
        #     if np.isnan(mycov):
        #         logger.warning(
        #             'Covariance at lag %d for column %s is NaN.' %
        #             (i, self.train.name))
        #         mycov = 0.
        #     self.lagged_covariances[i] = mycov
        # self.Sigma = make_Sigma_scalar_AR(self.lagged_covariances)

    # def _make_Sigma(self):
    #     self.Sigma = make_Sigma_scalar_AR(np.array(self.lagged_covariances))

    def test_predict(self, test):
        check_series(test)
        return super().test_predict(test)

    # def test_predict(self, test):
    #     check_series(test)

    #     test_concatenated = pd.concat([
    #         test.shift(-i)
    #         for i in range(self.lag)], axis=1)

    #     null_mask = pd.Series(False,
    #                           index=test_concatenated.columns)
    #     null_mask[self.past_lag:] = True

    #     to_guess = pd.DataFrame(test_concatenated, copy=True)
    #     to_guess.loc[:, null_mask] = np.nan
    #     guessed = guess_matrix(to_guess, self.Sigma).iloc[
    #         :, self.past_lag:]
    #     assert guessed.shape[1] == self.future_lag
    #     guessed_at_lag = []
    #     for i in range(self.future_lag):
    #         to_append = guessed.iloc[:, i:
    #                                  (i + 1)].shift(i + self.past_lag)
    #         to_append.columns = [el + '_lag_%d' % (i + 1) for el in
    #                              to_append.columns]
    #         guessed_at_lag.append(to_append)
    #     return guessed_at_lag

    # def test_RMSE(self, test):
    #     guessed_at_lags = self.test_predict(test)
    #     all_errors = np.zeros(0)
    #     for i, guessed in enumerate(guessed_at_lags):
    #         errors = (guessed_at_lags[i].iloc[:, 0] -
    #                   test).dropna().values
    #         print('RMSE at lag %d = %.2f' % (i + 1,
    #                                          np.sqrt(np.mean(errors**2))))
    #         all_errors = np.concatenate([all_errors, errors])
    #     return np.sqrt(np.mean(all_errors**2))


def autotune_scalar_autoregressor(train,
                                  test,
                                  future_lag,
                                  max_past_lag=100):

    print('autotuning scalar autoregressor on %d train and %d test points' %
          (len(train), len(test)))

    past_lag = np.arange(1, max_past_lag + 1)

    model = ScalarAutoregressor(train,
                                future_lag,
                                1)

    def test_RMSE(past_lag):
        model.past_lag = past_lag
        model._fit()
        return model.test_RMSE(test)

    res = greedy_grid_search(test_RMSE,
                             [past_lag],
                             num_steps=1)

    print('optimal params: %s' % res)
    print('test std. dev.: %.2f' % test.std())

    return res
