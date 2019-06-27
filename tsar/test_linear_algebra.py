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


from unittest import TestCase
from .linear_algebra import *

import numpy as np


class TestSVD(TestCase):

    def test_iterative_svd(self):

        np.random.seed(0)

        for (m, n, nan_ratio, noise_level) in  \
                [[20, 10, 10, 0.01],
                 [200, 100, 10, 0.01],
                 [200, 100, 5, 0.01],
                 [200, 100, 20, 0.001],
                 [10, 5, 5, 0.01],
                 [10, 50, 5, 0.01]]:

            a = np.random.randn(m, 1)
            a -= np.mean(a)
            a /= np.sqrt(a.T@a)
            b = np.random.randn(1, n)
            b -= np.mean(b)
            b /= np.sqrt(b@b.T)

            # print(a.T@a, b@b.T)

            X = a @ b + np.random.randn(m, n) * noise_level

            for i in range(n * m // nan_ratio):
                X[np.random.choice(m), np.random.choice(n)] = np.nan

            X = pd.DataFrame(X)

            print('fraction_null', np.mean(X.isnull().values.flatten()))

            u, s, v = iterative_denoised_svd(X, P=1, T=10)
            # print(u.T@u, v@v.T)

            u_ratio = u.flatten() / a.flatten()
            v_ratio = v.flatten() / b.flatten()

            for ratio in [u_ratio, v_ratio]:
                ratio_mean = ratio.mean()
                print('ratio mean', ratio_mean)
                self.assertTrue(np.abs(ratio_mean) > 0.5)
                self.assertTrue(np.abs(ratio_mean) < 1.5)

                ratio_std = ratio.std()
                print('ratio_std', ratio_std)
                self.assertTrue(ratio_std < 3)


class TestInverse(TestCase):

    def test_woodbury_inverse(self):

        np.random.seed(0)

        for m, k in [(10, 3), (20, 0), (10, 1), (0, 10)]:

            V = sp.csc_matrix(np.random.randn(m, k))
            S = np.random.randn(k, k)
            S = sp.csc_matrix(S@S.T)
            D = np.random.randn(m, m)
            D = sp.csc_matrix(D@D.T)

            true_inv = np.linalg.inv((V @ S @ V.T + D).todense())
            my_inv = woodbury_inverse(V, np.linalg.inv(S.todense()),
                                      np.linalg.inv(D.todense()))

            self.assertTrue(np.allclose(true_inv, my_inv))


class TestBlockMatrix(TestCase):

    def test_make_block_indexes(self):

        blocks = [np.eye(2), np.eye(3)]
        indexes = make_block_indexes(blocks)
        print(indexes)

        self.assertTrue(indexes[0, 0])
        self.assertFalse(indexes[0, 1])
        self.assertTrue(indexes[2, 1])
        self.assertFalse(indexes[2, 0])

        blocks = [np.eye(2), []]
        indexes = make_block_indexes(blocks)
        print(indexes)

        self.assertTrue(indexes[-1, 0])
        self.assertFalse(indexes[-1, 1])

    def test_symm_slice_blocks(self):

        blocks = [np.eye(2), np.eye(3)]
        blocks_indexes = make_block_indexes(blocks)

        mask = [True, False, True, False, True]

        new_blocks = symm_slice_blocks(blocks, blocks_indexes, mask)
        self.assertEqual(new_blocks[0].shape[0], 1)
        self.assertEqual(new_blocks[0].shape[1], 1)

        self.assertEqual(new_blocks[1].shape[0], 2)
        self.assertEqual(new_blocks[1].shape[1], 2)

        mask = [False, False, True, False, True]

        new_blocks = symm_slice_blocks(blocks, blocks_indexes, mask)
        self.assertEqual(new_blocks[0].shape[0], 0)
        self.assertEqual(new_blocks[0].shape[1], 0)

        self.assertEqual(new_blocks[1].shape[0], 2)
        self.assertEqual(new_blocks[1].shape[1], 2)

        mask = [False, False, False, False, True]

        new_blocks = symm_slice_blocks(blocks, blocks_indexes, mask)
        self.assertEqual(new_blocks[0].shape[0], 0)
        self.assertEqual(new_blocks[0].shape[1], 0)

        self.assertEqual(new_blocks[1].shape[0], 1)
        self.assertEqual(new_blocks[1].shape[1], 1)

    def test_symm_low_rank_plus_block_diag_schur(self):

        for m, k in [(10, 3), (20, 0), (10, 1), (0, 10)]:

            V = np.random.randn(m, k)
            S = np.random.randn(k, k)
            S = S@S.T
            S_inv = np.linalg.inv(S)

            D_blocks = [np.random.randn(m // 5, m // 5) for i in range(5)]
            D_blocks = [block@block.T for block in D_blocks]
            D_blocks_indexes = make_block_indexes(D_blocks)
            D_matrix = sp.block_diag(D_blocks).tocsc()

            known_mask = np.random.uniform(size=m) > .5
            known_matrix = np.random.randn(100, sum(known_mask))

            my_conditional_expect = symm_low_rank_plus_block_diag_schur(
                V,
                S,
                S_inv,
                D_blocks,
                D_blocks_indexes,
                D_matrix,
                known_mask,
                known_matrix)

            Sigma = V @ S @ V.T + D_matrix.todense()
            C = Sigma[known_mask].T[known_mask].T
            B = Sigma[~known_mask].T[known_mask].T
            conditional_expect = B @ np.linalg.inv(C) @ known_matrix.T

            self.assertTrue(
                np.allclose(
                    conditional_expect,
                    my_conditional_expect))
