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


def symm_low_rank_plus_block_diag_schur(V, S, blocks,
                                        known_mask, known_matrix,
                                        return_conditional_covariance=False):
    """Let Sigma = V @ S @ V^T + D,
    where D is a block diagonal matrix.

    We solve the Schur complement for the conditional
    expectation, with mean zero, and optionally
    return the conditional covariance.
    """


class SymmetricLowRankPlusBlockDiagonal:
    """We represent the matrix A = V @ S @ V^T + D
    Where:
    U has dimensions (n, k),
    S has dimensions (k, k),
    V has dimensions (k, n)
    D has dimensions (n, n) and is block diagonal.

    We provide methods for:
    - slicing, which returns a LowRankPlusSparse
    - (A @ x) matrix multiplication
    - schur complement
    """

    def __init__(self, U, S, V, D):
        self.U = U if not sp.issparse(U) else U.tocsc()
        self.S = S if not sp.issparse(S) else S.tocsc()
        self.V = V if not sp.issparse(V) else V.tocsc()
        self.D = D.tocsc()

        assert U.shape[1] == S.shape[0]
        assert S.shape[1] == V.shape[0]
        assert D.shape[0] == U.shape[0]
        assert D.shape[1] == V.shape[1]

    @property
    def shape(self):
        return self.D.shape

    def __repr__(self):
        return "Low-rank plus sparse of shape (%d, %d)" % (self.shape)

    @property
    def T(self):
        return LowRankPlusSparse(self.V.T, self.S.T,
                                 self.U.T, self.D.T)

    def __getitem__(self, slices_or_indexes):
        if isinstance(slices_or_indexes, tuple):
            # this does not cover all corner cases
            slice_1, slice_2 = slices_or_indexes
        else:
            slice_1 = slices_or_indexes
            slice_2 = slice(None)

        #print(slice_1, slice_2)

        return LowRankPlusSparse(
            self.U.__getitem__(slice_1),
            self.S,
            self.V.__getitem__((slice(None), slice_2)),
            self.D.__getitem__(slice_1).T.__getitem__(slice_2).T
        )

    def __matmul__(self, other):
        return self.D @ other + self.U @ (self.S @ (self.V @ other))

    def __rmatmul__(self, other):
        return other @ self.D + ((other @ self.U) @ self.S) @ self.V

    def todense(self):
        return self.D.todense() + self.U @ self.S @ self.V
