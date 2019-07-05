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

    if P == 0:
        return np.zeros((dataframe.shape[0], 0)), np.zeros(0), \
            np.zeros((0, dataframe.shape[1])),

    if not dataframe.isnull().sum().sum():
        T = 1

    y = pd.DataFrame(0., index=dataframe.index,
                     columns=dataframe.columns)
    for t in range(T):
        #if P < dataframe.shape[1]:
        #u, s, v = spl.svds(dataframe.fillna(y), P)
        # else:
        #     u, s, v = np.linalg.svd(dataframe.fillna(y), full_matrices=False)
        if P == (dataframe.shape[1]-1):
            u, s, v = np.linalg.svd(dataframe.fillna(y), full_matrices=False)
            u = u[:, ::-1]
            s = s[::-1]
            v = v[::-1]
        else:
            u, s, v = spl.svds(dataframe.fillna(y), P + 1)
        dn_u, dn_s, dn_v = u[:, 1:], s[1:] - s[0], v[1:]
        new_y = dn_u @ np.diag(dn_s) @ dn_v
        #new_y = u @ np.diag(s) @ v
        logger.debug('Iterative svd, MSE(y_%d - y_{%d}) = %.2e' % (
            t + 1, t, ((new_y - y)**2).mean().mean()))
        y.iloc[:, :] = dn_u @ np.diag(dn_s) @ dn_v
        #y.iloc[:, :] = u @ np.diag(s) @ v
    return dn_u, dn_s, dn_v
    #return u, s, v


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


def woodbury_inverse(V: np.matrix,  # sp.csc.csc_matrix,
                     S_inv: np.matrix,
                     D_inv: np.matrix):
    """ https://en.wikipedia.org/wiki/Woodbury_matrix_identity

    Compute (V @ S @ V.T + D)^-1.
    """
    #assert V.__class__ is sp.csc.csc_matrix
    assert (S_inv.__class__ is np.matrix) or (S_inv.__class__ is np.ndarray)
    assert D_inv.__class__ is np.matrix

    #V = V.todense()

    logger.debug('Solving Woodbury inverse.')
    logger.debug('Building internal matrix.')
    internal = S_inv + V.T @ D_inv @ V
    logger.debug('Inverting internal matrix.')
    intinv = np.linalg.inv(
        internal.todense() if hasattr(
            internal,
            'todense') else internal)
    logger.debug('Building inverse.')
    # D_invV = (D_inv @ V)
    # return D_inv - D_invV @ intinv @ D_invV.T

    return D_inv - (D_inv @ (V @ intinv)) @ (D_inv @ V).T


def symm_slice_blocks(blocks, block_indexes, mask):
    # TODO jit
    logger.debug('Slicing block matrix.')

    new_block_indexes = np.zeros(block_indexes.shape, dtype=bool)
    new_block_indexes[mask] = block_indexes[mask]

    return [block[new_block_indexes[block_indexes[:, i], i]].T[
        new_block_indexes[block_indexes[:, i], i]].T
        for i, block in enumerate(blocks)]

def dense_schur(Sigma, known_mask, known_vector,
                return_conditional_covariance=False,
                quadratic_regularization=0.):
    logger.debug('Solving dense Schur complement.')
    B = Sigma[~known_mask].T[known_mask].T
    C = Sigma[known_mask].T[known_mask].T
    res = B @ np.linalg.solve(C + quadratic_regularization * np.eye(C.shape[0]),
     known_vector)

    if return_conditional_covariance:
        logger.debug('Building conditional covariance')
        return res, Sigma[~known_mask].T[~known_mask].T - B @ np.linalg.inv(C + quadratic_regularization * np.eye(C.shape[0])) @ B.T

    return res

def alternative_symm_low_rank_plus_block_diag_schur(V: sp.csc.csc_matrix,
                                        S: np.matrix,
                                        S_inv: np.matrix,
                                        D_blocks,
                                        D_blocks_indexes,
                                        D_matrix: np.matrix,
                                        known_mask, known_matrix,
                                        return_conditional_covariance=False,
                                        quadratic_regularization: float = 0.):

    logger.debug('Making Sigma')
    Sigma = V @ S @ V.T + D_matrix

    B = Sigma[~known_mask].T[known_mask].T
    C = Sigma[known_mask].T[known_mask].T
    logger.debug('Inverting C')
    Cinv = np.linalg.inv(C + quadratic_regularization * np.eye(C.shape[0]))
    res = B @ Cinv @ known_matrix.T

    if return_conditional_covariance:
        logger.debug('Building conditional covariance')
        return res, Sigma[~known_mask].T[~known_mask].T - B @ Cinv @ B.T

    return res

def symm_low_rank_plus_block_diag_schur(V: sp.csc.csc_matrix,
                                        S: np.matrix,
                                        S_inv: np.matrix,
                                        D_blocks,
                                        D_blocks_indexes,
                                        D_matrix: np.matrix,
                                        known_mask, known_matrix,
                                        return_conditional_covariance=False,
                                        quadratic_regularization: float = 0.):
    """Let Sigma = V @ S @ V^T + D,
    where D is a block diagonal matrix.

    We solve the Schur complement for the conditional
    expectation, with mean zero, and optionally
    return the conditional covariance.
    """
    logger.debug('Solving Schur complement of low-rank plus block diagonal.')

    # TODO fix upstream
    # V = sp.csc_matrix(V)
    assert V.__class__ is sp.csc.csc_matrix
    V = V.todense()
    assert (S.__class__ is np.matrix) or (S.__class__ is np.ndarray)
    # S = S.todense() if hasattr(S, 'todense') else S
    # TODO this is not needed I guess, rethink everything here!
    assert (S_inv.__class__ is np.matrix) or (S_inv.__class__ is np.ndarray)
    assert D_matrix.__class__ is np.matrix

    # D_matrix = sp.csc_matrix(D_matrix)

    sliced_V = V[known_mask, :]
    sliced_D_blocks = symm_slice_blocks(D_blocks, D_blocks_indexes, known_mask)
    sliced_D_inv = sp.block_diag([np.linalg.inv(block +
                    quadratic_regularization * np.eye(block.shape[0]))
                            for block in sliced_D_blocks]).todense()

    C_inv = woodbury_inverse(V=sliced_V,
                             S_inv=S_inv,
                             D_inv=sliced_D_inv)

    logger.debug('Building B matrix')
    B = V[~known_mask, :] @ S @ sliced_V.T + \
        D_matrix[~known_mask].T[known_mask].T

    assert C_inv.__class__ is np.matrix
    assert B.__class__ is np.matrix
    # logger.debug('Building B @ C^-1')
    # BC_inv = B @ C_inv

    logger.debug('Computing conditional expectation')
    conditional_expect = B @ (C_inv @ known_matrix.T)

    if return_conditional_covariance:
        # TODO optimize
        logger.debug('Computing conditional covariance')
        Vsliced = V[~known_mask, :]
        Dsliced = D_matrix[~known_mask, ~known_mask]
        return conditional_expect, (Vsliced @ S) @ Vsliced.T + \
            Dsliced - (B @ C_inv) @ B.T

    return conditional_expect
