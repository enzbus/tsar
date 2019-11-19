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

import numpy as np

from tsar.data_reshape import make_sliced_flattened_matrix, one_slice
from tsar.AR import build_dense_covariance_matrix, \
    invert_build_dense_covariance_matrix


def generate_lagged_covs(M, lag):
    lagged_covariances = np.random.randn(M, M, lag)
    lagged_covariances[:, :, 0] += lagged_covariances[:, :, 0].T
    return lagged_covariances


class TestBLockToeplitz(TestCase):

    def test_basic(self):
        M = 2
        lag = 3
        lagged_covariances = np.zeros((M, M, lag))
        lagged_covariances[0, 0] = [1, 2, 3]
        lagged_covariances[0, 1] = [4, 5, 6]
        lagged_covariances[1, 0] = [4, 8, 9]
        lagged_covariances[1, 1] = [10, 11, 12]
        # _, n, lag = lagged_covariances.shape
        cov = build_dense_covariance_matrix(lagged_covariances)
        self.assertTrue(cov.shape == (M*lag, M*lag))
        target = [[1.,  2.,  3.,  4.,  5.,  6., ],
                  [2.,  1.,  2.,  8.,  4.,  5., ],
                  [3.,  2.,  1.,  9.,  8.,  4., ],
                  [4.,  8.,  9., 10., 11., 12., ],
                  [5.,  4.,  8., 11., 10., 11., ],
                  [6.,  5.,  4., 12., 11., 10., ]]
        self.assertTrue(np.all(target == cov))

    def test_many(self):
        for M, lag in [(1, 1), (1, 2), (2, 1), (2, 3), (4, 5)]:
            lagged_covariances = generate_lagged_covs(M, lag)
            print('lagged_covs', lagged_covariances)
            cov = build_dense_covariance_matrix(lagged_covariances)
            reconstructed = invert_build_dense_covariance_matrix(cov, lag)
            print('reconstructed', reconstructed)
            self.assertTrue(np.all(reconstructed == lagged_covariances))


class TestDataReshape(TestCase):

    # data = pd.DataFrame(pd.read_pickle('tests/data/wind_test_data.pickle'))
    # train = data[data.index.year.isin([2010, 2011])]
    # test = data[data.index.year == 2012]

    def test_one_slice_no_nans(self):

        M, T = 5, 100
        P, F = 3, 3
        data = np.random.randn(T, M)
        print(data[: 6])
        result = np.empty(M*(P+F))
        one_slice(data, P, F, P-1, result)
        print(result)

        self.assertEqual(list(result[: (P+F)]), list(data[: (P+F), 0]))
        self.assertEqual(
            list(result[(M-1)*(P+F):M*(P+F)]), list(data[:(P+F), M-1]))

    def test_one_slice_nans_past(self):

        M, T = 3, 100
        P, F = 2, 2
        data = np.random.randn(T, M)
        print(data[:P+F])
        result = np.empty(M*(P+F))
        one_slice(data, P, F, P-1-1, result)
        print(result)

        self.assertTrue(np.isnan(result[0]))
        self.assertEqual(list(result[1:(P+F)]), list(data[:(P+F)-1, 0]))

    def test_one_slice_nans_future(self):

        M, T = 3, 100
        P, F = 2, 2
        data = np.random.randn(T, M)
        print(data[-P-F:])
        result = np.empty(M*(P+F))
        one_slice(data, P, F, T, result)
        print(result)

        self.assertEqual(result[0], data[-1, 0])
        self.assertTrue(np.isnan(result[1]))
        self.assertEqual(result[-P-F], data[-1, -1])
        # self.assertEqual(
        #     list(result[(M-1)*(P+F)+1:M*(P+F)]), list(data[1:(P+F), M-1]))

        # sliced = make_sliced_flattened_matrix(data, P, F,
    #                                        prediction_times=np.array([10])))
