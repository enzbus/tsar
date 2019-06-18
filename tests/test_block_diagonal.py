import unittest
import pandas as pd
import numpy as np
import scipy.sparse as sp

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from tsar.block_diagonal import BlockDiagonal


class TestBlockDiagonal(unittest.TestCase):

    def test_block_diagonal(self):

        blocks = (np.eye(3), np.ones([3, 3]) + np.eye(3))
        mat = BlockDiagonal(blocks)

        def check_inv(mat):
            self.assertTrue(np.allclose(mat.inv().todense(),
                                        np.linalg.inv(mat.todense())))

        check_inv(mat)
        check_inv(mat[:2, :2])
        check_inv(mat[:, :])
        check_inv(mat[1:-1, 1:-1])
        check_inv(mat[-1, -1])
        check_inv(mat[:0, :0])

        self.assertTrue(np.allclose(mat[:2, :0].todense(),
                                    mat.todense()[:2, :0]))
