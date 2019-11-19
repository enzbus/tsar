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

from typing import Optional
import logging
import functools

import numpy as np
# import pandas as pd
import numba as nb
import scipy.sparse as sp
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)


@nb.jit(nopython=True)
def lag_covariance_asymm(array1: np.ndarray, array2: np.ndarray, lag: int):
    "Compute c^{i,j}_tau where the i-th and j-th columns are provided."
    assert len(array1) == len(array2)
    multiplied = array1[lag:] * array2[:len(array2) - lag]
    return np.nanmean(multiplied)  # [~np.isnan(multiplied)])


@nb.jit(nopython=True)
def make_lagged_covariances(u: np.ndarray, lag: int):
    "Compute all c^{i,j}_tau, as an M x M x (P+F) array."
    logger.info('Computing correlation coefficients.')
    n = u.shape[1]
    lag_covs = np.zeros((n, n, lag))
    for i in range(n):
        for j in range(n):
            for t in range(lag):
                # lag_covs[i, j, t] = lag_covariance_asymm(u[:, i], u[:, j], t)
                lag_covs[j, i, t] = lag_covariance_asymm(u[:, i], u[:, j], t)

    return lag_covs


@nb.jit(nopython=True)
def make_Sigma_scalar_AR_asymm(lagged_covariances_pos, lagged_covariances_neg):
    """Utility function to build the block-toeplitz matrix Σ^."""
    logger.info('Assembling Σ^ matrix.')
    lag = len(lagged_covariances_pos)
    Sigma = np.empty((lag, lag))
    for i in range(lag):
        for k in range(-i, lag - i):
            Sigma[i, k + i] = lagged_covariances_pos[
                k] if k > 0 else lagged_covariances_neg[-k]
    return Sigma


def build_dense_covariance_matrix(lagged_covariances):
    """Assemble Σ^ from the result of make_lagged_covariances."""
    _, n, lag = lagged_covariances.shape
    if not n:
        return np.empty((0, 0))
    return np.bmat([[make_Sigma_scalar_AR_asymm(lagged_covariances[j, i, :],
                                                lagged_covariances[i, j, :])
                     for i in range(n)] for j in range(n)])


def build_matrix_for_eigendecomposition(lagged_covariances):
    "Build the matrix to use for the eigendecomposition."
    return lagged_covariances[:, :, 0]


def compute_principal_directions(R, lagged_covariances):
    "Compute the R principal directions for the low-rank approximation."
    logger.debug(f'Computing {R} principal directions.')
    u, s, v = svds(build_matrix_for_eigendecomposition(
        lagged_covariances), k=R)
    return v


def make_V(v: np.ndarray, lag: int) -> sp.csc_matrix:
    "Make the V sparse matrix using the principal directions."
    logger.debug("Building V matrix.")
    R, M = v.shape
    V = sp.lil_matrix((M * lag, R * lag))
    for i in range(lag):
        V[i::lag, i::lag] = v.T
    return V.tocsc().T


# def fit_Sigma_hat(normalized_residual, lag):
#     "Fit the Σ^ matrix"


# def fit_low_rank_block_diagonal(Sigma_hat, lagged_covs, R, lag)


def fit_low_rank_block_diagonal(lagged_covs, Sigma_hat, R, P, F):
    # lagged_covs = make_lagged_covariances(normalized_residual, lag=P+F)
    # Sigma_hat = build_dense_covariance_matrix(lagged_covs)
    v = compute_principal_directions(R, lagged_covs)
    V = make_V(v, lag=P+F)
    logger.info('Computing Sigma_lr')
    Sigma_lr = V @ Sigma_hat @ V.T
    logger.info('Computing D.')
    blocks = []
    for m in range(lagged_covs.shape[0]):
        only_low_rank = V[:, m*(P+F):(m+1)*(P+F)].T @ Sigma_lr \
            @ V[:, m*(P+F):(m+1)*(P+F)]
        original = Sigma_hat[m*(P+F):(m+1)*(P+F), m*(P+F):(m+1)*(P+F)]
        blocks.append(original-only_low_rank)
    D = sp.bmat(blocks)
    return Sigma_lr, V, D


def compute_sigmas(residual):
    logger.info('Computing the sigmas.')
    sigmas = np.sqrt(np.nanmean(residual**2, axis=0))
    sigmas[np.isnan(sigmas)] = 1.
    sigmas[sigmas == 0.] = 1.
    return sigmas


def fit_gaussian_process(residual,
                         P: int,
                         F: int,
                         R: Optional[int] = None,
                         λ: Optional[float] = None,
                         train_test_ratio=2/3,
                         W=2):

    # assert type(residual) is pd.DataFrame
    T, M = residual.shape

    if (R is None) or (λ is None):

        train = residual[:int(T * train_test_ratio)]
        test = residual[int(T * train_test_ratio):]

        logger.info(f"Tuning hyper-parameters with {len(train)} train and "
                    f"{len(test)} test points")

        sigmas = compute_sigmas(train)
        normalized_train = train/sigmas
        normalized_test = test/sigmas
        lagged_covs = make_lagged_covariances(normalized_train, lag=P+F)
        Sigma_hat = build_dense_covariance_matrix(lagged_covs)

        @functools.lru_cache()
        def _fit(R):
            return fit_low_rank_block_diagonal(lagged_covs, Sigma_hat, R, P, F)

        def _evaluate(model, test, λ):
            pass

        # TODO finish

    sigmas = compute_sigmas(residual)
    normalized_residual = residual/sigmas

    lagged_covs = make_lagged_covariances(normalized_residual, lag=P+F)
    Sigma_hat = build_dense_covariance_matrix(lagged_covs)
    Sigma_lr, V, D = fit_low_rank_block_diagonal(lagged_covs,
                                                 Sigma_hat,
                                                 R, P, F)

    return R, λ, sigmas, Sigma_lr, V, D
