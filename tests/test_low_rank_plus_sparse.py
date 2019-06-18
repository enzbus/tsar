import unittest
import pandas as pd
import numpy as np
import scipy.sparse as sp

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from tsar.low_rank_plus_sparse import LowRankPlusSparse, \
    SquareLowRankPlusDiagonal, SquareLowRankPlusBlockDiagonal


class TestLowRankPlusSparse(unittest.TestCase):

    def test_rectangular(self):

        for (n, m, k) in [(10, 20, 5),
                          (10, 20, 0),
                          (1, 3, 5),
                          (1, 1, 0)]:

            U = np.random.randn(n, k)
            S = np.random.randn(k, k)
            V = np.random.randn(k, m)
            D = sp.eye(m).tocsc()[:n, :m]

            A = LowRankPlusSparse(U, S, V, D)
            self.assertTrue(np.allclose(A.todense()
                                        - U@S@V - D.todense(), 0))

            self.assertTrue(np.allclose(A @ np.ones(m)
                                        - U@S@V@ np.ones(m) -
                                        D@ np.ones(m), 0))

            with self.assertRaises(AssertionError):
                D = sp.eye(m).tocsc()[:n, :m - 1]
                LowRankPlusSparse(U, S, V, D)

    def test_slice(self):

        for (n, m, k) in [(10, 20, 5),
                          (10, 20, 0),
                          (1, 1, 1)]:

            U = np.random.randn(n, k)
            S = np.random.randn(k, k)
            V = np.random.randn(k, m)
            D = sp.eye(m).tocsc()[:n, :m]

            A = LowRankPlusSparse(U, S, V, D)

            self.assertTrue(np.allclose(A[:3, 3:].todense(),
                                        A.todense()[:3, 3:]))

            self.assertTrue(np.allclose(A[3:].todense(),
                                        A.todense()[3:]))

            mask = np.random.uniform(size=n) > .5

            self.assertTrue(np.allclose(A[mask].todense(),
                                        A.todense()[mask]))

            self.assertTrue(np.allclose(A[mask, 1:].todense(),
                                        A.todense()[mask, 1:]))

            self.assertTrue(np.allclose(A[~mask, 1:].todense(),
                                        A.todense()[~mask, 1:]))

    def test_slice_square(self):

        for (n, m, k) in [(20, 20, 5),
                          (10, 10, 0),
                          (1, 1, 10),
                          (1, 1, 0)]:

            U = np.random.randn(n, k)
            S = np.random.randn(k, k)
            V = np.random.randn(k, m)
            d = np.random.randn(n)

            A = SquareLowRankPlusDiagonal(U, S, V, d)

            self.assertTrue(np.allclose(A[:3, 3:].todense(),
                                        A.todense()[:3, 3:]))

            self.assertTrue(np.allclose(A[3:].todense(),
                                        A.todense()[3:]))

            mask = np.random.uniform(size=n) > .5

            self.assertTrue(isinstance(A[mask, mask],
                                       SquareLowRankPlusDiagonal))

            self.assertTrue(isinstance(A[:3, :3],
                                       SquareLowRankPlusDiagonal))

            self.assertTrue(isinstance(A[1:2, 1:2],
                                       SquareLowRankPlusDiagonal))

            self.assertTrue(np.allclose(A[mask, 1:].todense(),
                                        A.todense()[mask, 1:]))

            self.assertTrue(np.allclose(A[~mask, 1:].todense(),
                                        A.todense()[~mask, 1:]))

    def test_square(self):

        for (n, m, k) in [(20, 20, 5),
                          (10, 10, 0),
                          (1, 1, 10),
                          (1, 1, 0)]:

            U = np.random.randn(n, k)
            S = np.random.randn(k, k)
            V = np.random.randn(k, m)
            d = np.random.randn(n)

            A = SquareLowRankPlusDiagonal(U, S, V, d)
            self.assertTrue(np.allclose(A.todense()
                                        - U@S@V - np.diag(d), 0))

            self.assertTrue(np.allclose(A @ np.ones(m)
                                        - U@S@V@ np.ones(m) -
                                        np.diag(d)@ np.ones(m), 0))

            with self.assertRaises(AssertionError):
                SquareLowRankPlusDiagonal(U, S, V, d[:-1])

            self.assertTrue(np.allclose(A.inv()
                                        - np.linalg.inv(A.todense()), 0))

    def test_block_diagonal(self):

        for (n, m, k) in [(20, 20, 5),
                          (10, 10, 0),
                          (1, 1, 10),
                          (1, 1, 0)]:

            U = np.random.randn(n, k)
            S = np.random.randn(k, k)
            V = np.random.randn(k, m)
            d1 = np.random.randn(n // 2, n // 2)
            d2 = np.random.randn(n - n // 2, n - n // 2)

            A = SquareLowRankPlusBlockDiagonal(U, S, V, [d1, d2])

            self.assertTrue(np.allclose(A.todense().T, A.T.todense()))

            self.assertTrue(np.allclose(np.linalg.inv(A.todense()),
                                        A.inv()))

            self.assertTrue(np.allclose(A @ np.ones(n),
                                        A.todense() @ np.ones(n)))

            self.assertTrue(np.allclose(A[:n // 3, :n // 3].todense(),
                                        A.todense()[:n // 3, :n // 3]))
