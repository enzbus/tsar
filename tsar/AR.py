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


import numpy as np
import pandas as pd
import numba as nb
import logging
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import numba as nb
logger = logging.getLogger(__name__)
from typing import List, Any


# from .utils import check_series
from .greedy_grid_search import greedy_grid_search
from .linear_algebra import iterative_denoised_svd, \
    symm_low_rank_plus_block_diag_schur, make_block_indexes

from .utils import DataFrameRMSE
# from .base_autoregressor import BaseAutoregressor


# @nb.jit(nopython=True)
# def make_Sigma_scalar_AR(lagged_covariances: List[float]):
#     lag = len(lagged_covariances)
#     Sigma = np.empty((lag, lag))
#     for i in range(lag):
#         for k in range(-i, lag - i):
#             Sigma[i, k + i] = lagged_covariances[
#                 k] if k > 0 else lagged_covariances[-k]
#     # assert np.allclose((Sigma - Sigma.T), 0)
#     return Sigma
#
#
# @nb.jit(nopython=True)
# def lag_covariance(array: np.ndarray, lag: int):
#     # shifted = np.shift(series, lag)
#     multiplied = array[lag:] * array[:len(array) - lag]
#     return np.nanmean(multiplied)  # [~np.isnan(multiplied)])
#     # cov = pd.concat([series, series.shift(lag)], axis=1).cov()
#     # mycov = cov.iloc[1, 0]
#     # assert np.isclose(newcov, mycov)
#     # return newcov


# @nb.jit()  # nopython=True)
# def fit_AR(
#         train_array: np.ndarray,
#         cached_lag_covariances: List[float],
#         lag: int):
#     logger.info('Train array has mean %f and std %f' % (np.mean(train_array),
#                                                         np.std(train_array)))
#
#     lagged_covariances = np.empty(lag)
#     lagged_covariances[:len(cached_lag_covariances)] = \
#         cached_lag_covariances[:lag]
#     for i in range(len(cached_lag_covariances), lag):
#         logger.debug('computing covariance at lag %d' % i)
#         # cov = pd.concat([self.train,
#         #                  self.train.shift(i)], axis=1).cov()
#         # mycov = cov.iloc[1, 0]
#         mycov = lag_covariance(train_array, lag=i)
#         if np.isnan(mycov):
#             logger.warning(
#                 'Covariance at lag %dis NaN.' %
#                 (i))
#             mycov = 0.
#         logger.debug('result is %f' % mycov)
#         lagged_covariances[i] = mycov
#     Sigma = make_Sigma_scalar_AR(lagged_covariances)
#     return lagged_covariances, Sigma


@nb.jit(nopython=True)
def make_sliced_flattened_matrix(data_table: np.ndarray, lag: int):
    T, N = data_table.shape
    result = np.empty((T - lag + 1, N * lag))
    for i in range(T - lag + 1):
        data_slice = data_table[i:i + lag]
        result[i, :] = np.ravel(data_slice.T)  # , order='F')
    return result


# def fit_per_column_AR(train_table: pd.DataFrame,
#                       cached_lag_covariances: List[float], lag: int):
#     logger.info('Building AR models for %d columns, with %d samples each' %
#                 (train_table.shape[1], train_table.shape[0])
#                 )
#
#     Sigmas = []
#     # TODO parallelize
#     for i in range(train_table.shape[1]):
#         logger.debug(
#             'Building AR model for column %s' %
#             train_table.columns[i])
#         cached_lag_covariances[i], Sigma = fit_AR(
#             train_table.iloc[:, i].values, cached_lag_covariances[i], lag)
#         if not cached_lag_covariances[i][0]:
#             logger.warning(f'AR model for column {train_table.columns[i]} is null!')
#             Sigma = np.eye(lag)
#         Sigmas.append(Sigma)
#     return Sigmas


def make_V(v: np.ndarray, lag) -> sp.lil_matrix:
    # TODO make it faster?
    logger.debug("Building V matrix.")
    k, N = v.shape
    V = sp.lil_matrix((N * lag, k * lag))
    for i in range(lag):
        V[i::lag, i::lag] = v.T
    return V.tocsc()


@nb.jit(nopython=True)
def lag_covariance_asymm(array1: np.ndarray, array2: np.ndarray, lag: int):
    assert len(array1) == len(array2)
    multiplied = array1[lag:] * array2[:len(array2) - lag]
    return np.nanmean(multiplied)  # [~np.isnan(multiplied)])


@nb.jit(nopython=True)
def make_Sigma_scalar_AR_asymm(lagged_covariances_pos, lagged_covariances_neg):
    lag = len(lagged_covariances_pos)
    Sigma = np.empty((lag, lag))
    for i in range(lag):
        for k in range(-i, lag - i):
            Sigma[i, k + i] = lagged_covariances_pos[
                k] if k > 0 else lagged_covariances_neg[-k]
    return Sigma


@nb.jit(nopython=True)
def make_lag_covs(u, lag, n):
    lag_covs = np.zeros((n, n, lag))
    for i in range(n):
        for j in range(n):
            for t in range(lag):
                lag_covs[i, j, t] = lag_covariance_asymm(u[:, i], u[:, j], t)
    return lag_covs


def build_S(u, lag):
    n = u.shape[1]
    if not n:
        return np.empty((0, 0))
    lag_covs = make_lag_covs(u, lag, n)
    return np.bmat([[make_Sigma_scalar_AR_asymm(lag_covs[j, i, :],
                                                lag_covs[i, j, :])
                     for i in range(n)] for j in range(n)])


def _fit_low_rank_plus_block_diagonal_ar(
        train: pd.DataFrame,
        lag: int,
        rank: int,
        cached_lag_covariances: List[float],
        cached_svd: dict,
        cached_factor_lag_covariances: dict,
        full_covariance: bool,
        full_covariance_blocks: List):

    logger.debug('Fitting low rank plus diagonal model.')

    if full_covariance:
        logger.debug('Building full Sigma')
        Sigma = build_S(train.values, lag)
        np.fill_diagonal(Sigma, 1.)
        Sigma = np.nan_to_num(Sigma)
        return sp.csc_matrix((Sigma.shape[0], 0)), np.empty((0, 0)), \
                np.empty((0, 0)), [np.array(Sigma)], Sigma

    scalar_Sigmas = []
    for block in full_covariance_blocks:
        block_Sigma = build_S(train[block].values, lag)
        block_Sigma = np.nan_to_num(block_Sigma, copy=True)
        np.fill_diagonal(block_Sigma, 1.)
        scalar_Sigmas.append(block_Sigma)

    #
    # fit_per_column_AR(
    #     train, cached_lag_covariances, lag)

    if train.shape[1] <= 1:

        u, s, v = np.empty((train.shape[0], 0)), \
            np.empty((0, 0)), np.empty((0, train.shape[1]))

    else:

        if rank not in cached_svd:
            cached_svd[rank] = iterative_denoised_svd(train, rank)
        u, s, v = cached_svd[rank]

    if rank not in cached_factor_lag_covariances:
        cached_factor_lag_covariances[rank] = [[] for i in range(rank)]

    # factor_Sigmas = fit_per_column_AR(
    #     pd.DataFrame(u),
    #     cached_factor_lag_covariances[rank],
    #     lag)
    if v.shape[0] == 0:
        V = sp.csc_matrix((train.shape[1] * lag, 0))
    else:
        V = make_V(np.diag(s) @ v, lag)

    # S = (sp.block_diag(factor_Sigmas).todense() if len(
    #     factor_Sigmas) else np.empty((0, 0)))
    # S_inv = (sp.block_diag([np.linalg.inv(block) for block in factor_Sigmas]).todense(
    # ) if len(factor_Sigmas) else np.empty((0, 0)))

    logger.debug('Building S')
    S = build_S(u, lag)
    logger.debug('Building S^-1')
    S_inv = np.linalg.inv(S)

    #for i in range(len(scalar_Sigmas)):

    D_blocks = []
    cur = 0
    for i in range(len(scalar_Sigmas)):
        size = scalar_Sigmas[i].shape[0]
        logger.debug('correcting block of size %d' % size)
        D_blocks.append( scalar_Sigmas[i] -
                V[cur: cur + size] @ S @ V[cur: cur + size].T)
        cur += size

    D_matrix = sp.block_diag(D_blocks).todense()
    #D_inv = sp.block_diag([np.linalg.inv(block) for block in D_blocks])

    return V, S, S_inv, D_blocks, D_matrix


def guess_matrix(matrix_with_na: np.ndarray, V, S, S_inv,
                 D_blocks, D_matrix,
                 #D_inv,  # min_rows=5,
                 max_eval=1,
                 quadratic_regularization: float =0.):
    print('guessing matrix')
    # matrix_with_na = pd.DataFrame(matrix_with_na)
    full_null_mask = np.isnan(matrix_with_na)
    ranked_masks_counts = pd.Series([tuple(el) for el in
                                     full_null_mask]).value_counts()

    print('count values of NaN masks of test')
    print(ranked_masks_counts.values)

    ranked_masks = ranked_masks_counts.index

    for i in range(len(ranked_masks))[:max_eval]:

        print('null mask %d' % i)
        mask_indexes = (full_null_mask ==
                        ranked_masks[i]).all(1)
        # if mask_indexes.sum() <= min_rows:
        #     break
        print('there are %d rows' % mask_indexes.sum())

        # TODO fix
        D_blocks_indexes = make_block_indexes(D_blocks)
        known_mask = ~np.array(ranked_masks[i])
        known_matrix = matrix_with_na[mask_indexes].T[known_mask].T
        result = \
            symm_low_rank_plus_block_diag_schur(
                V,
                S,
                S_inv,
                D_blocks,
                D_blocks_indexes,
                D_matrix,
                known_mask=known_mask,
                known_matrix=known_matrix,
                return_conditional_covariance=False,
                quadratic_regularization = quadratic_regularization)

        logger.debug('Assigning conditional expectation to matrix.')
        mat_slice = matrix_with_na[mask_indexes]
        mat_slice[:, ~known_mask] = result.T
        matrix_with_na[mask_indexes] = mat_slice


def make_prediction_mask(
        available_data_lags_columns,
        columns,
        past_lag,
        future_lag):
    lag = past_lag + future_lag
    N = len(available_data_lags_columns)
    mask = np.zeros(N * (past_lag + future_lag), dtype=bool)
    for i in range(N):
        mask[lag * i + past_lag + available_data_lags_columns[columns[i]]:
             lag * (i + 1)] = True
    return mask


# def make_rmse_mask(columns, ignore_prediction_columns, lag):
#     N = len(columns)
#     mask = np.ones(N * lag, dtype=bool)

#     for i, col in enumerate(columns):
#         if col in ignore_prediction_columns:
#             mask[lag * i: lag * (i + 1)] = False
#     return mask


def rmse_AR(V, S, S_inv, D_blocks, D_matrix,
            past_lag, future_lag, test: pd.DataFrame,
            available_data_lags_columns: dict,
            quadratic_regularization:float):

    lag = past_lag + future_lag
    test_flattened = make_sliced_flattened_matrix(test.values, lag)
    prediction_mask = make_prediction_mask(
        available_data_lags_columns, test.columns, past_lag, future_lag)
    real_values = pd.DataFrame(test_flattened, copy=True)
    test_flattened[:, prediction_mask] = np.nan
    guess_matrix(test_flattened, V, S, S_inv, D_blocks, D_matrix,
    quadratic_regularization=quadratic_regularization)

    test_flattened = pd.DataFrame(test_flattened)

    total_test_obs = test_flattened.shape[0]

    non_nan_test_obs = (test_flattened.isnull().sum(1) == 0).sum()

    if non_nan_test_obs < total_test_obs:
        logger.warning(
            "There are %d test obs., but we are only testing on %d (because of NaNs)." %
            (total_test_obs, non_nan_test_obs))

    #assert (not test_flattened.isnull().sum().sum())

    rmses = DataFrameRMSE(real_values, test_flattened)
    print(rmses)
    #assert (not rmses.isnull().sum())
    #assert False

    my_RMSE = pd.DataFrame(columns=test.columns,
                           index=range(1, future_lag + 1))

    for i, column in enumerate(test.columns):
        my_RMSE[column] = rmses.iloc[lag * i + past_lag: lag * (i + 1)].values

    print(my_RMSE)

    return my_RMSE


# @dataclass
# class LowRankBlockDiagonal:
#     """Class for holding a low-rank plus block-diagonal."""
#
#     V: np.matrix
#     S: np.matrix
#     S_inv: np.matrix
#     D_blocks
#     D_matrix: np.matrix
#     D_inv: np. matrix


def fit_low_rank_plus_block_diagonal_AR(train: pd.DataFrame,
                                        test: pd.DataFrame,
                                        future_lag,
                                        past_lag,
                                        rank,
                                        available_data_lags_columns,
                                        ignore_prediction_columns,
                                        full_covariance,
                                        full_covariance_blocks,
                                        quadratic_regularization):

    logger.info('Fitting low-rank plus block diagonal AR')
    logger.info('Train table has shape (%d, %d)' % (train.shape))

    cached_lag_covariances = [[] for i in range(train.shape[1])]
    cached_svd = {}
    cached_factor_lag_covariances = {}

    if (past_lag is None) or (rank is None):
        logger.debug('Test table has shape (%d, %d)' % (test.shape))

        # TODO FIX
        past_lag_range = np.arange(1, future_lag * 2 + 1) \
            if past_lag is None else [past_lag]
        rank_range = np.arange(
            0, max(1, train.shape[1] - 2)) if rank is None else [rank]

        def test_RMSE(past_lag, rank, quadratic_regularization):

            lag = past_lag + future_lag

            V, S, S_inv, D_blocks, D_matrix = _fit_low_rank_plus_block_diagonal_ar(
                train, lag, rank, cached_lag_covariances, cached_svd,
                cached_factor_lag_covariances, full_covariance,
                full_covariance_blocks)

            RMSE_df = rmse_AR(V, S, S_inv, D_blocks,
                              D_matrix,
                              past_lag, future_lag, test,
                              available_data_lags_columns,
                              quadratic_regularization)

            return RMSE_df.loc[:, ~RMSE_df.columns.isin(
                ignore_prediction_columns)].sum().sum()

            # np.nanmean((guessed[:, (prediction_mask & rmse_mask)] -
            #             real_values_rmse)**2)

        optimal_rmse, (past_lag, rank, quadratic_regularization) = greedy_grid_search(test_RMSE,
                                                            [past_lag_range,
                                                             rank_range],
                                                            num_steps=2)

    lag = past_lag + future_lag
    V, S, S_inv, D_blocks, D_matrix = \
        _fit_low_rank_plus_block_diagonal_ar(
            train, lag, rank, cached_lag_covariances,
            cached_svd, cached_factor_lag_covariances,
            full_covariance,
            full_covariance_blocks)

    return past_lag, rank, quadratic_regularization, V, S, S_inv, D_blocks, D_matrix

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
