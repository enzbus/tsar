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
import scipy.sparse as sp
logger = logging.getLogger(__name__)

from .linear_algebra import iterative_denoised_svd
from .greedy_grid_search import greedy_grid_search
from .base_autoregressor import BaseAutoregressor
from .low_rank_plus_sparse import SquareLowRankPlusDiagonal, SquareLowRankPlusBlockDiagonal
from .utils import check_timeseries
from .scalar_autoregressor import make_Sigma_scalar_AR
from .block_diagonal import BlockDiagonal


def lagged_covariance(train, lag):
    N = train.shape[1]
    raw = pd.concat((train,
                     train.shift(lag)),
                    axis=1).cov().iloc[:N, N:]
    return raw


class LowRankPlusBlockDiagonalAR(BaseAutoregressor):

    def __init__(self,
                 train,
                 scalar_sigma_arrays,
                 P,
                 future_lag,
                 past_lag,
                 off_diagonal_covariance=False):

        check_timeseries(train)

        if P > train.shape[1] - 2:
            raise ValueError('P too large')

        self.train = train
        self.future_lag = future_lag
        self.past_lag = past_lag
        self.N = self.train.shape[1]
        assert len(scalar_sigma_arrays) == self.N
        self.scalar_sigma_arrays = scalar_sigma_arrays
        self.P = P
        self.off_diagonal_covariance = off_diagonal_covariance

        self.svd_results = {}
        self.embedding_covariances = {}
        self.train_rmses = np.sqrt((self.train**2).mean()).values

        self._fit()

    def _fit(self):
        self._make_block_diagonal()
        self._fit_low_rank_covariances()
        self._assemble_Sigma()

    @property
    def lag(self):
        return self.future_lag + self.past_lag

    def _fit_low_rank_covariances(self):
        if self.P not in self.svd_results:
            print('computing rank %d svd of train data' % self.P)
            self.svd_results[self.P] = \
                iterative_denoised_svd(self.train,
                                       P=self.P)
            self.embedding_covariances[self.P] = {}
        u, s, v = self.svd_results[self.P]
        embedding = pd.DataFrame(u, index=self.train.index)

        for i in range(0, self.lag):
            if i not in self.embedding_covariances[self.P]:
                print('computing covariance at lag %d' % i)
                self.embedding_covariances[self.P][i] = \
                    lagged_covariance(embedding, i)
                # v.T @ np.diag(s) @ \
                # lagged_covariance(embedding, i)  # \
                #@ np.diag(s) @ v

    @property
    def fraction_variance_explained(self):
        return pd.Series(self.orig_diag / self.train_rmses,
                         index=self.train.columns)

    @property
    def factors(self):
        u, s, v = self.svd_results[self.P]
        return pd.DataFrame(v, columns=self.train.columns)

    def _make_block_diagonal(self):
        # TODO rethink strategy, might make sense to cache
        print('making block diagonal matrix')
        self.blocks = []
        for scalar_sigma_array in self.scalar_sigma_arrays:
            lagged_covariance = np.zeros(self.lag)
            lagged_covariance[:len(scalar_sigma_array)] = \
                scalar_sigma_array[:self.lag]
            self.blocks.append(
                make_Sigma_scalar_AR(
                    lagged_covariance,
                    self.lag))
        #self.block_diagonal = BlockDiagonal(self.blocks)

    def _assemble_Sigma(self):
        # TODO this should be low-rank plus diag
        print('assembling covariance matrix')
        S = np.block(
            [[(self.embedding_covariances[self.P][i].values.T
               if self.off_diagonal_covariance
               else np.diag(np.diag(self.embedding_covariances[self.P][i].values.T)))
              if i > 0 else
              (self.embedding_covariances[self.P][-i].values
                if self.off_diagonal_covariance
               else np.diag(np.diag(self.embedding_covariances[self.P][-i].values)))
                for i in range(-j, self.lag - j)]
                for j in range(self.lag)])

        assert np.allclose((S - S.T), 0)

        _, s, v = self.svd_results[self.P]

        # V = np.block([np.diag(s) @ v for i in range(self.lag)])
        base = np.diag(s) @ v
        V = sp.lil_matrix((self.P * self.lag, self.N * self.lag))
        if self.P:
            for i in range(self.N):
                V[:self.P, i * self.lag] = np.matrix(base[:, i]).T
            for i in range(1, self.lag):
                V[i * self.P:(i + 1) * self.P, i:] = V[:self.P, :-i]
        V = V.tocsc()
        # print(V)

        # V = sp.bmat([[None] * i +
        #              [np.diag(s) @ v] +
        #              [None] * (self.lag - i - 1)
        #              for i in range(self.lag)])

        self.orig_diag = np.diag(v.T @ np.diag(s) @
                                 (self.embedding_covariances[self.P][0]
                                  if self.off_diagonal_covariance else
                                  np.diag(np.diag(self.embedding_covariances[self.P][0]))) @
                                 np.diag(s) @ v)

        # d = (1. - np.concatenate([self.orig_diag] * self.lag))

        #d = (np.repeat(self.train_rmses - self.orig_diag, self.lag))

        # self.orig_diag = np.diag(self.Sigma)

        self.low_rank_Sigma = SquareLowRankPlusDiagonal(
            V.T, S, V, np.zeros(self.N * self.lag))

        # assert np.allclose(np.diag(self.Sigma.todense())
        #                    - np.repeat(self.train_rmses, self.lag), 0.)

        # self.Sigma += sp.diags(1 - self.orig_diag)

        print('adding block diagonal part')
        self.effective_blocks = []
        for i, block in enumerate(self.blocks):
            low_rank_block = \
                self.low_rank_Sigma[i * self.lag:(i + 1) * self.lag,
                                    i * self.lag:(i + 1) * self.lag]
            self.effective_blocks.append(block - low_rank_block.todense())

        self.Sigma = SquareLowRankPlusBlockDiagonal(
            V.T, S, V, self.effective_blocks)


def autotune_low_rank_plus_block_diag_ar(train,
                                         test,
                                         scalar_sigma_arrays,
                                         future_lag,
                                         past_lag=None,
                                         P=None
                                         ):

    print('autotuning low-rank autoregressor on %d train and %d test points' %
          (len(train), len(test)))

    past_lag = np.arange(1, 100) if past_lag is None else [past_lag]
    P = np.arange(0, train.shape[1] - 2) if P is None else [P]

    model = LowRankPlusBlockDiagonalAR(train,
                                       scalar_sigma_arrays,
                                       0,
                                       future_lag,
                                       1)

    def test_RMSE(P, past_lag):
        model.past_lag = past_lag
        model.P = P
        model._fit()
        return model.test_RMSE(test)

    res = greedy_grid_search(test_RMSE,
                             [P, past_lag],
                             num_steps=2)

    print('optimal params: %s' % res)
    # print('test std. dev.: %.2f' % np.sqrt((test**2).mean().mean()))

    return res
