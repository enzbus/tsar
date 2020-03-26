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

from .greedy_grid_search import greedy_grid_search
from typing import List  # , Any
import logging

import numpy as np
import pandas as pd
import numba as nb
import scipy.sparse as sp
import lrbd

# import scipy.sparse.linalg as spl

from .utils import DataFrameRMSE
# from tsar.new_linear_algebra import symm_low_rank_plus_block_diag_schur
from .linear_algebra import iterative_denoised_svd

from .gaussian_model import \
    _fit_low_rank_plus_block_diagonal_ar_using_eigendecomp

logger = logging.getLogger(__name__)

HOW_MANY_QUAD_REG_RANGE_POINTS = 50


@nb.jit(nopython=True)
def make_sliced_flattened_matrix(data_table: np.ndarray, lag: int):
    T, N = data_table.shape
    result = np.empty((T - lag + 1, N * lag))
    for i in range(T - lag + 1):
        data_slice = data_table[i:i + lag]
        result[i, :] = np.ravel(data_slice.T)  # , order='F')
    return result


def make_V(v: np.ndarray, lag: int) -> sp.csc_matrix:
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
def make_lagged_covariances(u: np.ndarray, lag: int):
    n = u.shape[1]
    lag_covs = np.zeros((n, n, lag))
    for i in range(n):
        for j in range(n):
            for t in range(lag):
                # lag_covs[i, j, t] = lag_covariance_asymm(u[:, i], u[:, j], t)
                lag_covs[j, i, t] = lag_covariance_asymm(u[:, i], u[:, j], t)

    return lag_covs


def build_dense_covariance_matrix(lagged_covariances):
    _, n, lag = lagged_covariances.shape
    if not n:
        return np.empty((0, 0))
    # lag_covs = make_lag_covs(u, lag, n)
    return np.bmat([[make_Sigma_scalar_AR_asymm(lagged_covariances[j, i, :],
                                                lagged_covariances[i, j, :])
                     for i in range(n)] for j in range(n)])


@nb.njit()
def check_toeplitz(square_matrix):
    m, _ = square_matrix.shape
    for i in range(m - 1):
        for j in range(m - 1):
            assert square_matrix[i, j] == square_matrix[i + 1, j + 1]


@nb.njit()
def invert_build_dense_covariance_matrix(cov, lag):
    M = cov.shape[0] // lag
    assert np.all(cov == cov.T)
    lagged_covariances = np.empty((M, M, lag))
    for i in range(M):
        for j in range(M):
            toeplitz_block = cov[i * lag:(i + 1) * lag, j * lag:(j + 1) * lag]
            check_toeplitz(toeplitz_block)
            lagged_covariances[i, j, :] = toeplitz_block[0, :]
    return lagged_covariances


def build_matrix(s_times_v: np.ndarray,
                 S_lagged_covariances: np.ndarray,
                 block_lagged_covariances: np.ndarray):

    logger.info('Building matrices.')

    lag = S_lagged_covariances.shape[2]

    if s_times_v.shape[0] == 0:
        V = sp.csc_matrix((s_times_v.shape[1] * lag, 0))
    else:
        V = make_V(s_times_v, lag)

    S = build_dense_covariance_matrix(S_lagged_covariances)
    # S_inv = np.linalg.inv(S)

    D_blocks = []
    cur = 0

    for block in block_lagged_covariances:
        block_Sigma = build_dense_covariance_matrix(block)
        block_Sigma = np.nan_to_num(block_Sigma, copy=True)
        np.fill_diagonal(block_Sigma, 1.)
        size = block_Sigma.shape[0]
        D_blocks.append(block_Sigma -
                        V[cur: cur + size] @ S @ V[cur: cur + size].T)
        cur += size

    # D_matrix = sp.block_diag(D_blocks).todense()

    return lrbd.LowRankBlockDiag(V.todense().T, S,
                                 [np.array(D) for D in D_blocks])
    #    return V, S, S_inv, D_blocks, D_matrix


def _fit_low_rank_plus_block_diagonal_ar_using_svd(
        train: pd.DataFrame,
        lag: int,
        rank: int,
        full_covariance: bool,
        full_covariance_blocks: List,
        noise_correction: bool,
        variables_weight: np.array,
        workspace: dict):

    logger.debug('Fitting low rank plus diagonal model.')

    if full_covariance:
        logger.debug('Fitting full Sigma')
        return np.empty((0, train.shape[1])), \
            np.empty((0, 0, lag)), \
            [make_lagged_covariances(train.values, lag)]

    if 'ranks' not in workspace:
        workspace['ranks'] = {}
    if rank not in workspace['ranks']:
        workspace['ranks'][rank] = {}

    if train.shape[1] <= 1:
        u, s, v = np.empty((train.shape[0], 0)), \
            np.empty((0, 0)), np.empty((0, train.shape[1]))

    else:
        u, s, v = iterative_denoised_svd(
            train * variables_weight, rank, noise_correction)
        v /= variables_weight.values

    if 's_times_v' not in workspace['ranks'][rank]:
        workspace['ranks'][rank]['s_times_v'] = np.diag(s) @ v
    if 'factor_lagged_covs' not in workspace['ranks'][rank]:
        workspace['ranks'][rank]['factor_lagged_covs'] = \
            make_lagged_covariances(u, lag)

    if 'block_lagged_covs' not in workspace:
        workspace['block_lagged_covs'] = \
            [make_lagged_covariances(train[block].values, lag) for block in
             full_covariance_blocks]

    return workspace['ranks'][rank]['s_times_v'], \
        workspace['ranks'][rank]['factor_lagged_covs'], \
        workspace['block_lagged_covs']


def guess_matrix(matrix_with_na: np.ndarray, Sigma,
                 quadratic_regularization: float,
                 prediction_mask, real_values,
                 max_eval=3,
                 ):

    logger.info('guessing matrix')
    # matrix_with_na = pd.DataFrame(matrix_with_na)
    full_null_mask = np.isnan(matrix_with_na)
    ranked_masks_counts = pd.Series([tuple(el) for el in
                                     full_null_mask]).value_counts()

    logger.info('count values of NaN masks of test %s' %
                ranked_masks_counts.values)

    ranked_masks = ranked_masks_counts.index
    total_num_predictions_made = 0

    for i in range(len(ranked_masks))[:]:

        logger.info('null mask %d' % i)
        mask_indexes = (full_null_mask == ranked_masks[i]).all(1)

        logger.info('there are %d rows' % mask_indexes.sum())
        total_num_predictions_made += mask_indexes.sum()

        known_mask = ~np.array(ranked_masks[i])

        prediction_indexes = np.arange(matrix_with_na.shape[0])[mask_indexes]
        for i in prediction_indexes:
            logger.debug('solving reg. schur')
            matrix_with_na[i, prediction_mask] = \
                Sigma.regularized_schur(
                left_mask=prediction_mask,
                right_mask=known_mask,
                value=matrix_with_na[i, known_mask],
                lambd=quadratic_regularization)

        logger.debug('Assigning conditional expectation to matrix.')

    return total_num_predictions_made


def make_prediction_mask(
        available_data_lags_columns,
        ignore_prediction_col_mask,
        columns,
        past_lag,
        future_lag):
    lag = past_lag + future_lag
    N = len(available_data_lags_columns)
    unknown_mask = np.zeros(N * (past_lag + future_lag), dtype=bool)
    prediction_mask = np.zeros(N * (past_lag + future_lag), dtype=bool)
    for i in range(N):
        unknown_mask[lag * i + past_lag + available_data_lags_columns[columns[i]]:
                     lag * (i + 1)] = True
        if not ignore_prediction_col_mask[i]:
            prediction_mask[lag * i + past_lag + available_data_lags_columns[columns[i]]:
                            lag * (i + 1)] = True
    return prediction_mask, unknown_mask


def rmse_AR(Sigma,
            past_lag, future_lag, test: pd.DataFrame,
            available_data_lags_columns: dict,
            ignore_prediction_columns: list,
            quadratic_regularization: float,
            # do_gradients=False
            ):

    lag = past_lag + future_lag
    test_flattened = make_sliced_flattened_matrix(test.values, lag)
    ignore_prediction_col_mask = test.columns.isin(ignore_prediction_columns)
    prediction_mask, unknown_mask = make_prediction_mask(
        available_data_lags_columns, ignore_prediction_col_mask,
        test.columns, past_lag, future_lag)
    real_values = pd.DataFrame(test_flattened, copy=True)
    test_flattened[:, unknown_mask] = np.nan
    total_num_predictions_made = guess_matrix(
        test_flattened, Sigma,
        quadratic_regularization=quadratic_regularization,
        prediction_mask=prediction_mask, real_values=real_values,
        # do_gradients=do_gradients
    )

    test_flattened = pd.DataFrame(test_flattened)

    total_test_obs = test_flattened.shape[0]

    if total_num_predictions_made < total_test_obs:
        logger.warning(
            "There are %d test obs but we only test on %d because of NaNs." %
            (total_test_obs, total_num_predictions_made))

    estimate_total_loss_entries = total_num_predictions_made * \
        sum(prediction_mask)

    # assert (not test_flattened.isnull().sum().sum())

    rmses = DataFrameRMSE(real_values, test_flattened)
    # print(rmses)

    my_RMSE = pd.DataFrame(columns=test.columns,
                           index=range(1, future_lag + 1))

    for i, column in enumerate(test.columns):
        my_RMSE[column] = rmses.iloc[lag * i + past_lag: lag * (i + 1)].values

    return my_RMSE


def fit_low_rank_plus_block_diagonal_AR(data,
                                        rank: int,
                                        quadratic_regularization: float,
                                        future_lag: int,
                                        past_lag: int,
                                        available_data_lags_columns,
                                        ignore_prediction_columns,
                                        full_covariance: bool,
                                        full_covariance_blocks,
                                        noise_correction: bool,
                                        variables_weight: np.array,
                                        use_svd_fit: bool,
                                        train_test_ratio: float,
                                        alpha=np.cbrt(10),
                                        W=2):

    logger.info('Fitting Gaussian model with data (%d, %d)' % (data.shape))

    if use_svd_fit:
        fitter = _fit_low_rank_plus_block_diagonal_ar_using_svd
    else:
        fitter = _fit_low_rank_plus_block_diagonal_ar_using_eigendecomp

    # cached_lag_covariances = [[] for i in range(train.shape[1])]
    # cached_svd = {}
    # cached_factor_lag_covariances = {}

    train = data.iloc[:int(len(data) * train_test_ratio)]
    test = data.iloc[int(len(data) * train_test_ratio):]

    from functools import lru_cache

    @lru_cache()
    def myfitter(rank):

        lag = past_lag + future_lag

        workspace = {}

        s_times_v, S_lagged_covariances, block_lagged_covariances = fitter(
            train, lag, rank,  # cached_lag_covariances, cached_svd,
            # cached_factor_lag_covariances,
            full_covariance,
            full_covariance_blocks,
            noise_correction,
            variables_weight,
            workspace=workspace)

        # return build_matrix(
        return s_times_v,\
            S_lagged_covariances,\
            block_lagged_covariances

    def test_RMSE(rank, quadratic_regularization):

        RMSE_df = rmse_AR(build_matrix(*myfitter(rank)),
                          past_lag, future_lag, test,
                          available_data_lags_columns,
                          ignore_prediction_columns,
                          quadratic_regularization,
                          # do_gradients=False
                          )

        return RMSE_df.loc[:, ~RMSE_df.columns.isin(
            ignore_prediction_columns)]

    if (quadratic_regularization is None) or (rank is None):

        logger.info(f"Tuning hyper-parameters with {len(train)} train and "
                    f"{len(test)} test points")

        if not len(train):
            raise ValueError("There is not enough train data.")

        if not len(test):
            raise ValueError("There is not enough test data.")

        def ggs_test_RMSE(rank, quadratic_regularization):
            return test_RMSE(rank, quadratic_regularization).sum().sum()

        M = data.shape[1]
        rank_range = np.arange(0, M - noise_correction)\
            if rank is None else [rank]

        max_lambda = M * (past_lag + future_lag)
        quad_reg_range = max_lambda / alpha**np.arange(50)\
            if quadratic_regularization is None else [quadratic_regularization]

        # np.nanmean((guessed[:, (prediction_mask & rmse_mask)] -
        #             real_values_rmse)**2)

        # optimal_rmse, (past_lag, rank, quadratic_regularization) = \
        #     greedy_grid_search(test_RMSE,
        #                        [past_lag_range,
        #                         rank_range],
        #                        num_steps=2)
        _, (rank, quadratic_regularization) = greedy_grid_search(
            ggs_test_RMSE, [rank_range, quad_reg_range],
            num_steps=W,
            logger_info=True)

    logger.info(f"Fitting Gaussian model with rank = {rank},")
    logger.info(f"chosen λ = {quadratic_regularization}")

    internal_RMSE = test_RMSE(rank, quadratic_regularization)

    workspace = {}
    lag = past_lag + future_lag
    s_times_v, S_lagged_covariances, block_lagged_covariances = fitter(
        data, lag, rank,  # cached_lag_covariances,
        # cached_svd, cached_factor_lag_covariances,
        full_covariance,
        full_covariance_blocks,
        noise_correction,
        variables_weight,
        workspace=workspace)

    return internal_RMSE, past_lag, rank, quadratic_regularization, \
        s_times_v, S_lagged_covariances, block_lagged_covariances
