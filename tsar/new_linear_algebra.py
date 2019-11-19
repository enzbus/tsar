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


import scipy.sparse as sp
import numpy as np
# import pandas as pd
# import numba as nb
import logging
# import scipy.sparse.linalg as spl
logger = logging.getLogger(__name__)


def symm_slice_blocks(blocks, block_indexes, mask):
    # TODO jit
    logger.debug('Slicing block matrix.')

    new_block_indexes = np.zeros(block_indexes.shape, dtype=bool)
    new_block_indexes[mask] = block_indexes[mask]

    return [block[new_block_indexes[block_indexes[:, i], i]].T[
        new_block_indexes[block_indexes[:, i], i]].T
        for i, block in enumerate(blocks)]


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
    # assert V.__class__ is sp.csc.csc_matrix
    assert (S_inv.__class__ is np.matrix) or (S_inv.__class__ is np.ndarray)
    assert D_inv.__class__ is np.matrix

    # V = V.todense()

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


def symm_low_rank_plus_block_diag_schur(V: sp.csc.csc_matrix,
                                        S: np.matrix,
                                        S_inv: np.matrix,
                                        D_blocks,
                                        # D_blocks_indexes,
                                        # D_matrix: np.matrix,
                                        known_mask,
                                        known_matrix,
                                        prediction_mask,
                                        real_result,
                                        quadratic_regularization: float,
                                        return_conditional_covariance=False,
                                        do_anomaly_score=False,
                                        return_gradient=False
                                        ):
    """Let Sigma = V @ S @ V^T + D,
    where D is a block diagonal matrix.

    We solve the Schur complement for the conditional
    expectation, with mean zero, and optionally
    return the conditional covariance.
    """

    # prediction_mask = ~known_mask

    logger.debug('Solving Schur complement of low-rank plus block diagonal.')

    D_blocks_indexes = make_block_indexes(D_blocks)
    D_matrix = sp.block_diag(D_blocks).todense()

    # V = sp.csc_matrix(V)
    assert V.__class__ is sp.csc.csc_matrix
    V = V.todense()
    assert (S.__class__ is np.matrix) or (S.__class__ is np.ndarray)
    # S = S.todense() if hasattr(S, 'todense') else S
    assert (S_inv.__class__ is np.matrix) or (S_inv.__class__ is np.ndarray)
    assert D_matrix.__class__ is np.matrix

    # D_matrix = sp.csc_matrix(D_matrix)

    sliced_V = V[known_mask, :]
    sliced_D_blocks = symm_slice_blocks(D_blocks, D_blocks_indexes, known_mask)
    inverse_blocks = []
    count = 0
    for block in sliced_D_blocks:
        # how_many_var_in_bloc = block.shape[0] // lag
        inverse_blocks.append(np.linalg.inv(
            block + np.eye(block.shape[0]) * quadratic_regularization
            # quadratic_regularization * np.eye(block.shape[0])
        ))
        count += block.shape[0]

    sliced_D_inv = sp.block_diag(inverse_blocks).todense()

    C_inv = woodbury_inverse(V=sliced_V,
                             S_inv=S_inv,
                             D_inv=sliced_D_inv)

    if do_anomaly_score:
        # TODO should adjust by diag of C
        return (C_inv @ known_matrix.T)

    logger.debug('Building B matrix')
    B = V[prediction_mask, :] @ S @ sliced_V.T + \
        D_matrix[prediction_mask].T[known_mask].T

    assert C_inv.__class__ is np.matrix
    assert B.__class__ is np.matrix
    # logger.debug('Building B @ C^-1')
    # BC_inv = B @ C_inv

    logger.debug('Computing conditional expectation')
    partial = (C_inv @ known_matrix.T)
    conditional_expect = B @ partial

    # #print(conditional_expect - real_result.T)
    if return_gradient:
        error = conditional_expect - real_result.T
        error = np.nan_to_num(error)

        logger.info('Making left side of derivative')
        left = 2 * (error.T @ B) @ C_inv
        # print(left.shape)
        logger.info('Making right side of derivative')
        right = partial
        # print(right.shape)
        logger.info('Computing gradient')
        gradient = [(left[:, i].T @ right[i].T)[0, 0]
                    for i in range(left.shape[1])]
        # print(gradient)
        return conditional_expect, gradient

    if return_conditional_covariance:
        # TODO optimize
        logger.debug('Computing conditional covariance')
        Vsliced = V[prediction_mask, :]
        Dsliced = D_matrix[prediction_mask, prediction_mask]
        return conditional_expect, (Vsliced @ S) @ Vsliced.T + \
            Dsliced - (B @ C_inv) @ B.T

    return conditional_expect
