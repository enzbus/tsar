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
from .low_rank_plus_sparse import SquareLowRankPlusDiagonal
from .utils import check_timeseries


def lagged_covariance(train, lag):
    N = train.shape[1]
    raw = pd.concat((train,
                     train.shift(lag)),
                    axis=1).cov().iloc[:N, N:]
    return raw


class LowRankAR(BaseAutoregressor):

    def __init__(self,
                 train,
                 P,
                 future_lag,
                 past_lag):

        check_timeseries(train)

        if P > train.shape[1] - 2:
            raise ValueError('P too large')

        self.train = train
        self.future_lag = future_lag
        self.past_lag = past_lag
        self.N = self.train.shape[1]
        self.P = P

        self.svd_results = {}
        self.embedding_covariances = {}

        self._fit()

    def _fit(self):
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

    # @property
    # def fraction_explained_variance(self):
    #     return pd.Series(self.orig_diag,
    #                      index=self.train.columns)

    def _assemble_Sigma(self):
        # TODO this should be low-rank plus diag
        print('assembling covariance matrix')
        S = np.block(
            [[self.embedding_covariances[self.P][i].values.T
              if i > 0 else
              self.embedding_covariances[self.P][-i].values
                for i in range(-j, self.lag - j)]
                for j in range(self.lag)])

        assert np.allclose((S - S.T), 0)

        _, s, v = self.svd_results[self.P]

        # V = np.block([np.diag(s) @ v for i in range(self.lag)])
        V = sp.bmat([[None] * i +
                     [np.diag(s) @ v] +
                     [None] * (self.lag - i - 1)
                     for i in range(self.lag)])

        self.orig_diag = np.diag(v.T @ np.diag(s) @
                                 self.embedding_covariances[self.P][0] @
                                 np.diag(s) @ v)

        d = (1. - np.concatenate([self.orig_diag] * self.lag))

        # self.orig_diag = np.diag(self.Sigma)

        self.Sigma = SquareLowRankPlusDiagonal(V.T, S, V, d)

        # self.Sigma += sp.diags(1 - self.orig_diag)


def autotune_low_rank_autoregressor(train,
                                    test,
                                    future_lag,
                                    past_lag=None,
                                    P=None
                                    ):

    print('autotuning low-rank autoregressor on %d train and %d test points' %
          (len(train), len(test)))

    past_lag = np.arange(1, 100) if past_lag is None else [past_lag]
    P = np.arange(0, train.shape[1] - 2) if P is None else [P]

    model = LowRankAR(train,
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
    #print('test std. dev.: %.2f' % np.sqrt((test**2).mean().mean()))

    return res
