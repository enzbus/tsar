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
logger = logging.getLogger(__name__)
import scipy.sparse as sp
import scipy.sparse.linalg as spl

import time

from .greedy_grid_search import greedy_grid_search

FEATURES_PERIODS_AND_INDEXERS = {
    'hour_of_day': (24, lambda index: index.hour),
    'day_of_week': (7, lambda index: index.dayofweek),
    'month_of_year': (12, lambda index: index.month - 1),
    'day_of_year': (366, lambda index: index.dayofyear - 1),
    'week_of_year': (53, lambda index: index.week - 1)
}


def cumulative_periods(used_features):
    n_periods = [FEATURES_PERIODS_AND_INDEXERS[fea][0]
                 for fea in used_features]
    cum_periods = np.concatenate([[1], np.cumprod(n_periods)])
    return cum_periods.astype(int)


def pandas_index_to_θ_index(index, used_features):
    """From pandas index to index of θ array."""
    cum_periods = cumulative_periods(used_features)
    result = np.zeros(len(index), dtype=int)
    for (i, fea) in enumerate(used_features):
        indexer = FEATURES_PERIODS_AND_INDEXERS[fea][1]
        result += indexer(index) * cum_periods[i]
    return result

#@numba.jit


def build_block_diag_diff(N, diff, period):
    num_blocks = N // period
    assert(N / period == N // period)
    logger.debug(f'Building block diag. with {num_blocks} blocks.')
    # TODO refactor, make a numba-friendly for-loop
    return sp.block_diag([build_cyclic_diff(period, diff)] * num_blocks)

#@numba.jit


def build_cyclic_diff(N, diff):
    """Build cyclic diff matrices for regularization."""
    logger.debug(f'Building matrix of size {N} with diff {diff}.')
    return sp.eye(N) - sp.eye(N, k=diff) - sp.eye(N, k=diff - N)


def make_cyclic_regularization_mats(used_features):
    """Build cyclic regularization matrices."""
    QTQ = []
    logger.debug('building reg matrices')
    cum_periods = cumulative_periods(used_features)
    for i in range(len(used_features)):
        diff = cum_periods[i]
        period = cum_periods[i + 1]
        Q = build_block_diag_diff(cum_periods[-1], diff, period)
        QTQ.append(Q.T@Q)  # note this!
    return QTQ


def make_least_squares_cost(data, used_features):
    """Build LS penalty part of the system."""
    NDATA = len(data)
    N = cumulative_periods(used_features)[-1]
    logger.debug(f'Making quadratic loss term for {NDATA} obs.')
    P = sp.coo_matrix((
        np.ones(NDATA),
        (range(NDATA), pandas_index_to_θ_index(data.index, used_features))),
        shape=(NDATA, N))
    x = data.values
    return P, x


def build_least_squares_system(P, x, QTQ, λ):
    M = len(x)
    # build matrix and right hand side
    matrix = P.T @ P / M
    for k in range(len(QTQ)):
        matrix += QTQ[k] * λ[k]
    rhs = P.T @ x / M
    return matrix, rhs


def cg_solve(matrix, vector, x0=None):
    # print([matrix[i, i] for i in range(5)])
    s = time.time()
    result = spl.cg(matrix, vector, x0=x0)
    logger.debug(f'CG took {time.time()-s} seconds.')
    # print(result)
    # if status != 0:
    #    raise Exception("CG failed.")
    return result[0]
    #cache[:, column] = result[:, column]


def compute_residual_and_RMSE(theta, P, x):
    # val_pred = self.P_2@theta
    # val_res = val_pred - self.x_2.reshape(val_pred.shape)
    M = len(x)
    if M == 0:
        return np.zeros(x.shape), 0.
    residual = ((P@theta) - x)
    RMSE = np.sqrt(np.mean(residual**2))
    return residual, RMSE


def compute_baseline(index,
                     used_features,
                     theta):

    return theta[pandas_index_to_θ_index(index, used_features)]


def data_to_residual(data: pd.Series,
                     std: float,
                     used_features,
                     theta: np.ndarray, **kwargs) -> pd.Series:
    return (data - compute_baseline(data.index,
                                    used_features,
                                    theta)) / std


def residual_to_data(residual: pd.Series, std: float,
                     used_features, theta: np.ndarray,  **kwargs) -> pd.Series:
    return residual * std + compute_baseline(residual.index,
                                             used_features,
                                             theta)


def _fit_baseline(data,
                  used_features,
                  lambdas,
                  theta_0=None):

    QTQ = make_cyclic_regularization_mats(used_features)
    P, x = make_least_squares_cost(data, used_features)
    matrix, rhs = build_least_squares_system(P, x, QTQ, λ=lambdas)
    return cg_solve(matrix, rhs, x0=theta_0)


def fit_baseline(train, test,
                 used_features,
                 lambdas, **kwargs):

    train = train.dropna()

    if not len(train):
        # , returning null baseline.')
        logger.warning(f'Train column {train.name} is all NaNs')
        # return 1., 0, 0, 0, False, np.array([0.]), \
        #     np.sqrt((test.dropna()**2).mean()) if test is not None else None

    if test is not None:
        test = test.dropna()

        logger.debug('Autotuning baseline on %d train and %d test points' %
                     (len(train), len(test)))

        lambda_ranges = [np.logspace(-6, 3, 20)[::-1]
                         if λ is None else [λ] for λ in lambdas]

        QTQ = make_cyclic_regularization_mats(used_features)
        P, x = make_least_squares_cost(train, used_features)
        theta_cache = np.zeros(QTQ[0].shape[0])

        def test_RMSE(*lambdas):

            matrix, rhs = build_least_squares_system(P, x, QTQ, λ=lambdas)
            theta = cg_solve(matrix, rhs, x0=theta_cache)
            theta_cache[:] = theta

            return np.sqrt(
                ((test - compute_baseline(
                    test.index,
                    used_features,
                    theta))**2).mean())

        optimal_rmse, lambdas = greedy_grid_search(test_RMSE,
                                                   lambda_ranges,
                                                   num_steps=2)

    theta = _fit_baseline(train,
                          used_features,
                          lambdas)

    std = np.std(data_to_residual(train,
                                  std=1.,
                                  used_features=used_features,
                                  theta=theta))

    return std, lambdas, theta, optimal_rmse if test is not None else None
