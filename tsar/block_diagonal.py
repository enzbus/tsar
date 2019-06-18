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


class BlockDiagonal:

    def __init__(self, diagonal_blocks):

        if not np.all([b.shape[0] == b.shape[1]
                       for b in diagonal_blocks]):
            raise ValueError('All blocks must be square.')

        self.diagonal_blocks = diagonal_blocks
        self._matrix = sp.block_diag(diagonal_blocks).tocsc()
        self.block_indexes = np.zeros((self._matrix.shape[0],
                                       len(self.diagonal_blocks)),
                                      dtype=bool)
        cur = 0
        for i, block in enumerate(self.diagonal_blocks):
            self.block_indexes[cur:cur + block.shape[0], i] = True
            cur += block.shape[0]

        assert np.all(np.sum(self.block_indexes, 1) == 1)

    def square_slice(self, slicer):
        print('square slicing')
        new_block_indexes = np.zeros(self.block_indexes.shape, dtype=bool)
        new_block_indexes.__setitem__(
            (slicer, slice(None)), self.block_indexes.__getitem__(
                (slicer, slice(None))))
        new_blocks = [
            block[
                new_block_indexes[self.block_indexes[:, i], i]].T[
                new_block_indexes[self.block_indexes[:, i], i]].T
            for i, block in enumerate(self.diagonal_blocks)]
        return BlockDiagonal(new_blocks)

    def __getitem__(self, slices_or_indexes):
        if isinstance(slices_or_indexes, tuple):
            slice_1, slice_2 = slices_or_indexes
            try:
                if (isinstance(slice_1, np.ndarray)
                        and np.all(slice_1 == slice_2)) \
                        or slice_1 == slice_2:
                    return self.square_slice(slice_1)
            except ValueError as e:
                pass
        return self._matrix.__getitem__(slices_or_indexes)

    def __matmul__(self, other):
        return self._matrix.__matmul__(other)

    def __rmatmul__(self, other):
        return self._matrix.__rmatmul__(other)

    def todense(self):
        return self._matrix.todense()

    @property
    def T(self):
        return BlockDiagonal([block.T for block in self.diagonal_blocks])

    def inv(self):
        def inverse(block):
            return np.linalg.inv(block
                                 if isinstance(block, np.ndarray)
                                 else block.todense())
        return BlockDiagonal([inverse(block)
                              for block in self.diagonal_blocks])
