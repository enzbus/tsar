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
import scipy.sparse as sp

from .block_diagonal import BlockDiagonal


class LowRankPlusSparse:
    """We represent the matrix A = U @ S @ V + D
    Where:
    U has dimensions (n, k),
    S has dimensions (k, k),
    V has dimensions (k, m)
    D has dimensions (n, m) and is sparse.

    We provide methods for:
    - slicing, which returns a LowRankPlusSparse
    - (A @ x) matrix multiplication
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


class SquareLowRankPlusDiagonal(LowRankPlusSparse):
    """We represent the matrix A = U @ S @ V + diag(d)

    Where:
    U has dimensions (n, k)
    S has dimensions (k, k)
    V has dimensions (k, n)
    d has dimension (n)

    We provide methods for:
    - slicing, which returns a LowRankPlusSparse
    - left and right matrix multiplications
    - inverse as dense or abstract operator
    """

    def __init__(self, U, S, V, d):
        self.d = d

        super().__init__(U, S, V, sp.diags(d).tocsc())

    def __repr__(self):
        return "Square low-rank plus diagonal of shape (%d, %d)" % (self.shape)

    def __getitem__(self, slices_or_indexes):
        if isinstance(slices_or_indexes, tuple):
            slice_1, slice_2 = slices_or_indexes
            #print(slice_1 == slice_2)
            try:
                if (isinstance(slice_1, np.ndarray)
                        and np.all(slice_1 == slice_2)) \
                        or slice_1 == slice_2:
                    print('square slicing')
                    return SquareLowRankPlusDiagonal(
                        self.U.__getitem__(slice_1),
                        self.S,
                        self.V.__getitem__((slice(None), slice_1)),
                        self.d.__getitem__(slice_1)
                    )
            except ValueError as e:
                # assert
                # print(slice_1, slice_2)
                # print(e)
                pass
        return super().__getitem__(slices_or_indexes)

    @property
    def T(self):
        return SquareLowRankPlusSparse(self.V.T, self.S.T,
                                       self.U.T, self.d)

    def inv(self):
        return self**(-1)

    def __pow__(self, exponent):
        """ https://en.wikipedia.org/wiki/Woodbury_matrix_identity """
        if not exponent == -1:
            raise ValueError('Can only compute inverse.')
        Dinv = sp.diags(self.d**-1)
        Sinv = np.linalg.inv(self.S)
        internal = (Sinv + self.V @ Dinv @ self.U)
        return Dinv - Dinv @ (self.U @ np.linalg.inv(internal) @ self.V) @ Dinv


class SquareLowRankPlusBlockDiagonal(LowRankPlusSparse):
    """We represent the matrix A = U @ S @ V + diag(d)

    Where:
    U has dimensions (n, k)
    S has dimensions (k, k)
    V has dimensions (k, n)
    d has dimension (n)

    We provide methods for:
    - slicing, which returns a LowRankPlusSparse
    - left and right matrix multiplications
    - inverse as dense or abstract operator
    """

    def __init__(self, U, S, V, blocks):
        self.blocks = blocks

        self.block_D = BlockDiagonal(self.blocks)

        super().__init__(U, S, V, self.block_D._matrix)

    def __repr__(self):
        return "Square low-rank plus block-diagonal of shape (%d, %d)" % (
            self.shape)

    def __getitem__(self, slices_or_indexes):
        if isinstance(slices_or_indexes, tuple):
            slice_1, slice_2 = slices_or_indexes
            #print(slice_1 == slice_2)
            try:
                if (isinstance(slice_1, np.ndarray)
                        and np.all(slice_1 == slice_2)) \
                        or slice_1 == slice_2:
                    print('square slicing')
                    return SquareLowRankPlusBlockDiagonal(
                        self.U.__getitem__(slice_1), self.S, self.V.__getitem__(
                            (slice(None), slice_1)), self.block_D.__getitem__(
                            (slice_1, slice_1)).diagonal_blocks)
            except ValueError as e:
                # assert
                # print(slice_1, slice_2)
                # print(e)
                pass
        return super().__getitem__(slices_or_indexes)

    @property
    def T(self):
        return SquareLowRankPlusBlockDiagonal(
            self.V.T, self.S.T, self.U.T, [
                block.T for block in self.blocks])

    def inv(self):
        return self**(-1)

    def __pow__(self, exponent):
        """ https://en.wikipedia.org/wiki/Woodbury_matrix_identity """
        if not exponent == -1:
            raise ValueError('Can only compute inverse.')
        Dinv = self.block_D.inv()._matrix
        Sinv = np.linalg.inv(self.S) if not hasattr(self.S, 'inv') \
            else self.S.inv()
        internal = (Sinv + self.V @ Dinv @ self.U)
        return Dinv - Dinv @ (self.U @ np.linalg.inv(internal) @ self.V) @ Dinv
