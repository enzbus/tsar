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
import scipy.sparse.linalg as spl
logger = logging.getLogger(__name__)
import scipy.sparse as sp


def iterative_denoised_svd(dataframe, P, T=10):

    if not dataframe.isnull().sum().sum():
        T = 1

    y = pd.DataFrame(0., index=dataframe.index,
                     columns=dataframe.columns)
    for t in range(T):
        u, s, v = spl.svds(dataframe.fillna(y), P + 1)
        dn_u, dn_s, dn_v = u[:, 1:], s[1:] - s[0], v[1:]
        new_y = dn_u @ np.diag(dn_s) @ dn_v
        logger.debug('Iterative svd, MSE(y_%d - y_{%d}) = %.2e' % (
            t + 1, t, ((new_y - y)**2).mean().mean()))
        y.iloc[:, :] = dn_u @ np.diag(dn_s) @ dn_v
    return dn_u, dn_s, dn_v


def make_block_indexes(blocks):
    logger.debug('Computing indexes for block matrix.')
    block_indexes = np.zeros((sum([len(b) for b in blocks]),
                              len(blocks)),
                             dtype=bool)
    cur = 0
    for i, block in enumerate(blocks):
        block_indexes[cur:cur + len(block), i] = True
        cur += len(block)

    assert np.all(np.sum(block_indexes, 1) == 1)

    return block_indexes


def woodbury_inverse(V, S_inv, D_inv):
    """ https://en.wikipedia.org/wiki/Woodbury_matrix_identity

    Compute (V @ S @ V.T + D)^-1.
    """
    logger.debug('Solving Woodbury inverse.')
    internal = S_inv + V.T @ D_inv @ V
    return D_inv - D_inv @ (V @ np.linalg.inv(internal) @ V.T) @ D_inv


def symm_slice_blocks(blocks, block_indexes, mask):
    # TODO jit
    logger.debug('Slicing block matrix.')

    new_block_indexes = np.zeros(block_indexes.shape, dtype=bool)
    new_block_indexes[mask] = block_indexes[mask]

    return [block[new_block_indexes[block_indexes[:, i], i]].T[
        new_block_indexes[block_indexes[:, i], i]].T
        for i, block in enumerate(blocks)]


def symm_low_rank_plus_block_diag_schur(V, S, S_inv,
                                        D_blocks, D_blocks_indexes, D_matrix,
                                        known_mask, known_matrix,
                                        return_conditional_covariance=False):
    """Let Sigma = V @ S @ V^T + D,
    where D is a block diagonal matrix.

    We solve the Schur complement for the conditional
    expectation, with mean zero, and optionally
    return the conditional covariance.
    """
    logger.debug('Solving Schur complement of low-rank plus block diagonal.')

    sliced_V = V[known_mask, :]
    sliced_D_blocks = symm_slice_blocks(D_blocks, D_blocks_indexes, known_mask)
    sliced_D_inv = sp.block_diag([np.linalg.inv(block) for
                                  block in sliced_D_blocks])

    C_inv = woodbury_inverse(sliced_V, S_inv, sliced_D_inv)

    B = V[~known_mask, :] @ S @ sliced_V.T + \
        D_matrix[~known_mask].T[known_mask].T

    BC_inv = B @ C_inv

    conditional_expect = BC_inv @ known_matrix.T

    if return_conditional_covariance:
        return conditional_expect, BC_inv @ B.T

    return conditional_expect
