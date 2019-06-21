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
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numba as nb
logger = logging.getLogger(__name__)


# from .utils import check_series
from .greedy_grid_search_new import greedy_grid_search
from .linear_algebra_new import iterative_denoised_svd
# from .base_autoregressor import BaseAutoregressor


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
def fit_AR(train_array, cached_lag_covariances, lag):
    lagged_covariances = np.empty(lag)
    lagged_covariances[:len(cached_lag_covariances)] = cached_lag_covariances
    for i in range(len(cached_lag_covariances), lag):
        logger.debug('computing covariance at lag', i)
        # cov = pd.concat([self.train,
        #                  self.train.shift(i)], axis=1).cov()
        # mycov = cov.iloc[1, 0]
        mycov = lag_covariance(train_array, lag=i)
        if np.isnan(mycov):
            logger.warning(
                'Covariance at lag %dis NaN.' %
                (i))
            mycov = 0.
        lagged_covariances[i] = mycov
    Sigma = make_Sigma_scalar_AR(lagged_covariances)
    return lagged_covariances, Sigma


@nb.jit(nopython=True)
def make_sliced_flattened_matrix(data_table: np.ndarray, lag: int):
    T, N = data_table.shape
    result = np.empty((T - lag + 1, N * lag))
    for i in range(T - lag + 1):
        data_slice = data_table[i:i + lag]
        result[i, :] = np.ravel(data_slice.T)  # , order='F')
    return result


def fit_per_column_AR(train_table, cached_lag_covariances, lag):
    logger.info('Building AR models for %d columns, with %d samples each' %
                (train_table.shape[1], train_table.shape[0])
                )
    Sigmas = []
    # TODO parallelize
    for i in range(train_table.shape[1]):
        logger.debug('Building AR model for column %d' % i)
        cached_lag_covariances[i], Sigma = \
            fit_AR(train_table[:, i], cached_lag_covariances[i], lag)
        Sigmas.append(Sigma)
    return cached_lag_covariances, Sigmas


def make_V(v, s, lag):
    # TODO make it faster?
    logger.debug("Building V matrix.")
    k, N = v.shape
    V = sp.lil((N * lag, k * lag))
    for i in range(lag):
        V[i::lag, i::lag] = v.T
    return V


def _fit_low_rank_plus_block_diagonal_ar(train, lag, rank,
                                         cached_lag_covariances,
                                         cached_svd,
                                         cached_factor_lag_covariances):

    logger.debug('Fitting low rank plus diagonal model.')

    cached_lag_covariances, scalar_Sigmas = fit_per_column_AR(
        train, cached_lag_covariances, lag)

    if rank not in cached_svd:
        cached_svd[rank] = iterative_denoised_svd(train, rank)

    u, s, v = cached_svd[rank]

    if rank not in cached_factor_lag_covariances:
        cached_factor_lag_covariances[rank] = []

    cached_factor_lag_covariances[rank], factor_Sigmas = fit_per_column_AR(
        u, cached_factor_lag_covariances[rank], lag)

    V = make_V(v, s, lag)

    S = sp.block_diag(factor_Sigmas)
    S_inv = sp.block_diag([np.linalg.inv(block) for block in factor_Sigmas])

    D_blocks = [scalar_Sigmas[i] -
                V[lag * i: lag * (i + 1)] @ S @ V[lag * i: lag * (i + 1)].T
                for i in range(len(scalar_Sigmas))]
    D_matrix = sp.block_diag(D_blocks)
    D_inv = sp.block_diag([np.linalg.inv(block) for block in D_blocks])

    return V, S, S_inv, D_blocks, D_matrix, D_inv,\
        cached_lag_covariances, cached_svd, cached_factor_lag_covariances


def guess_matrix(matrix_with_na, V, S, S_inv,
                 D_blocks, D_matrix,
                 D_inv, min_rows=5, max_eval=5):
    print('guessing matrix')
    full_null_mask = matrix_with_na.isnull()
    ranked_masks = pd.Series([tuple(el) for el in
                              full_null_mask.values]).value_counts().index

    for i in range(len(ranked_masks))[:max_eval]:

        print('null mask %d' % i)
        mask_indexes = (full_null_mask ==
                        ranked_masks[i]).all(1)
        if mask_indexes.sum() <= min_rows:
            break
        print('there are %d rows' % mask_indexes.sum())

        # TODO fix
        matrix_with_na.loc[mask_indexes] = schur_complement_matrix(
            matrix_with_na.loc[mask_indexes].values,
            np.array(ranked_masks[i]),
            Sigma)
    return matrix_with_na


def make_prediction_mask(available_data_lags_columns, past_lag, future_lag):
    lag = past_lag + future_lag
    N = len(available_data_lags_columns)
    mask = np.zeros(N * (past_lag + future_lag), dtype=bool)
    for i in range(N):
        mask[lag * i + past_lag + available_data_lags_columns[i]:
             lag * (i + 1)] = True
    return mask


def make_rmse_mask(columns, ignore_prediction_columns, lag):
    N = len(columns)
    mask = np.ones(N * lag, dtype=bool)

    for i, col in enumerate(columns):
        if col in ignore_prediction_columns:
            mask[lag * i: lag * (i + 1)] = False
    return mask


def fit_low_rank_plus_block_diagonal_AR(train,
                                        test=None,
                                        future_lag,
                                        past_lag,
                                        rank,
                                        available_data_lags_columns,
                                        ignore_prediction_columns=[]):

    if test is not None:

        past_lag_range = pass
        rank_range = pass

        cached_lag_covariances = [[] for i in range(train.shape[1])]
        cached_svd = {}
        cached_factor_lag_covariances = {}

        def test_RMSE(past_lag, rank):

            lag = past_lag + future_lag

            V, S, S_inv, D_blocks, D_matrix, D_inv,\
                cached_lag_covariances, cached_svd,
                cached_factor_lag_covariances = \
                    _fit_low_rank_plus_block_diagonal_ar(
                        train,
                        lag,
                        rank,
                        cached_lag_covariances,
                        cached_svd,
                        cached_factor_lag_covariances)

            test_flattened = \
                make_sliced_flattened_matrix(test, lag)

            prediction_mask = make_prediction_mask(
                available_data_lags_columns, lag)

            rmse_mask = make_rmse_mask(train.columns,
                                       ignore_prediction_columns, lag)

            real_values_rmse = test_flattened[:, (prediction_mask & rmse_mask)]
            test_flattened[:, prediction_mask] = np.nan
            guessed = guess_matrix(test_flattened,
                                   V, S, S_inv,
                                   D_blocks, D_matrix,
                                   D_inv)

            pass
            # np.nanmean((guessed[:, (prediction_mask & rmse_mask)] -
            #             real_values_rmse)**2)

        optimal_rmse, (past_lag,
                       rank) = greedy_grid_search(test_RMSE,
                                                  [past_lag_range,
                                                   rank_range],
                                                  num_steps=2)

    # def dataframe_to_vector

    # def fit_AR(vector, cached_lag_covariances, lag):
    # 	cached_lag_covariances, Sigma = \
    # 	       update_covariance_Sigma(train_array=vector,
    # 	                               old_lag_covariances=cached_lag_covariances,
    # 	                               lag=lag)

    # class ScalarAutoregressor(BaseAutoregressor):

    #     def __init__(self,
    #                  train,
    #                  future_lag,
    #                  past_lag):

    #         check_series(train)
    #         self.train = train
    #         assert np.isclose(self.train.mean(), 0., atol=1e-6)
    #         self.future_lag = future_lag
    #         self.past_lag = past_lag
    #         self.lagged_covariances = np.empty(0)
    #         self.N = 1

    #         self._fit()

    #     # def _fit(self):
    #     #     self._fit_Sigma()
    #     #     self._make_Sigma()

    #     # @property
    #     # def lag(self):
    #     #     return self.future_lag + self.past_lag

    #     def _fit(self):
    #         self.lagged_covariances, self.Sigma = \
    #             update_covariance_Sigma(train_array=self.train.values,
    #                                     old_lag_covariances=self.lagged_covariances,
    #                                     lag=self.lag)
    #         # old_lag_covariances = self.lagged_covariances
    #         # self.lagged_covariances = np.empty(self.lag)
    #         # self.lagged_covariances[
    #         #     :len(old_lag_covariances)] = old_lag_covariances
    #         # for i in range(len(old_lag_covariances), self.lag):
    #         #     print('computing covariance lag %d' % i)
    #         #     # cov = pd.concat([self.train,
    #         #     #                  self.train.shift(i)], axis=1).cov()
    #         #     # mycov = cov.iloc[1, 0]
    #         #     mycov = lag_covariance(self.train.values, lag=i)
    #         #     if np.isnan(mycov):
    #         #         logger.warning(
    #         #             'Covariance at lag %d for column %s is NaN.' %
    #         #             (i, self.train.name))
    #         #         mycov = 0.
    #         #     self.lagged_covariances[i] = mycov
    #         # self.Sigma = make_Sigma_scalar_AR(self.lagged_covariances)

    #     # def _make_Sigma(self):
    #     #     self.Sigma = make_Sigma_scalar_AR(np.array(self.lagged_covariances))

    #     def test_predict(self, test):
    #         check_series(test)
    #         return super().test_predict(test)

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

    # def autotune_scalar_autoregressor(train,
    #                                   test,
    #                                   future_lag,
    #                                   max_past_lag=100):

    #     print('autotuning scalar autoregressor on %d train and %d test points' %
    #           (len(train), len(test)))

    #     past_lag = np.arange(1, max_past_lag + 1)

    #     model = ScalarAutoregressor(train,
    #                                 future_lag,
    #                                 1)

    #     def test_RMSE(past_lag):
    #         model.past_lag = past_lag
    #         model._fit()
    #         return model.test_RMSE(test)

    #     res = greedy_grid_search(test_RMSE,
    #                              [past_lag],
    #                              num_steps=1)

    #     print('optimal params: %s' % res)
    #     print('test std. dev.: %.2f' % test.std())

    #     return res
