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

__all__ = ['AutoRegressive', 'Autotune_AutoRegressive']


def lagged_covariance(train_normalized, lag):
    N = train_normalized.shape[1]
    raw = pd.concat((train_normalized,
                     train_normalized.shift(lag)),
                    axis=1).cov().iloc[:N, N:]
    return raw


def iterative_denoised_svd(dataframe, P, T=10):
    y = pd.DataFrame(0., index=dataframe.index,
                     columns=dataframe.columns)
    for t in range(T):
        u, s, v = spl.svds(dataframe.fillna(y), P + 1)
        dn_u, dn_s, dn_v = u[:, 1:], s[1:] - s[0], v[1:]
        new_y = dn_u @ np.diag(dn_s) @ dn_v
        print('MSE(y_%d - y_{%d}) = %.2e' % (
            t + 1, t, ((new_y - y)**2).mean().mean()))
        y.iloc[:, :] = dn_u @ np.diag(dn_s) @ dn_v
    return dn_u, dn_s, dn_v


# def lagged_covariance_svd(train_normalized, lag, P=1):
#     N = train_normalized.shape[1]
#     if lag > 0:
#         concat = pd.concat((train_normalized,
#                             train_normalized.shift(lag)),
#                            axis=1)
#     else:
#         concat = train_normalized
#         if P > N:
#             print('capping P to %d' % (N - 2))
#             P = N - 2
#     u, s, v = iterative_denoised_svd(concat, P)
#     # reconstructed = pd.DataFrame(
#     #     u @ np.diag(s) @ v,
#     #     index=concat.index,
#     #     columns=concat.columns)
#     # filled_concat = concat.fillna(reconstructed)
#     if lag > 0:
#         orig_cov = concat.cov().iloc[:N, N:]
#         low_rank_cov = ((v.T @ np.diag(s**2) @ v) /
#                         len(concat))[:N, N:]
#     else:
#         orig_cov = concat.cov()
#         low_rank_cov = ((v.T @ np.diag(s**2) @ v) /
#                         len(concat))
#     low_rank_plus_diag_cov = low_rank_cov +\
#         np.diag(np.diag(orig_cov - low_rank_cov))
#     print('MSE between original and lrpd covs',
#           ((orig_cov - low_rank_plus_diag_cov)**2).mean().mean())
#     return pd.DataFrame(
#         low_rank_plus_diag_cov,
#         index=orig_cov.index,
#         columns=orig_cov.columns)  # , orig_cov


# def low_rank_plus_diagonal(covariance, noise_level):
#     u, s, v = np.linalg.svd(covariance)
#     P = len(np.nonzero(s > noise_level)[0])
#     print('keeping %d  factors' % P)
#     low_rank = u[:, :P] @ np.diag(s[:P] - noise_level) @ v[:P, :]
#     idyo = np.diag(covariance.values - low_rank)
#     return pd.DataFrame(low_rank + np.diag(idyo),
#                         index=covariance.index,
#                         columns=covariance.columns)


@nb.jit(nopython=True)
def schur_complement(array_with_na, Sigma):
    null_mask = np.isnan(array_with_na)
    y = array_with_na[~null_mask]

    # A = Sigma[null_mask].T[null_mask]
    B = Sigma[null_mask].T[~null_mask].T
    C = Sigma[~null_mask].T[~null_mask]

    expected_x = B @ np.linalg.solve(C, y)
    array_with_na[null_mask] = expected_x
    return array_with_na


#@nb.jit(nopython=True)
def schur_complement_matrix(matrix_with_na,
                            null_mask,
                            Sigma):
    # null_mask = np.isnan(array_with_na)
    Y = matrix_with_na[:, ~null_mask]

    # A = Sigma[null_mask].T[null_mask]
    B = Sigma[null_mask].T[~null_mask].T
    C = Sigma[~null_mask].T[~null_mask]
    inv_C = np.linalg.inv(C)

    expected_X = B @ inv_C @ Y.T
    matrix_with_na[:, null_mask] = expected_X.T
    return matrix_with_na


# @nb.jit(nopython=True)
# def guess_each_row_matrix(matrix_with_na, Sigma):
#     result = np.empty(matrix_with_na.shape)
#     for i in range(len(matrix_with_na)):
#         if ((i % 100) == 0):
#             print((100 * i //
#                    len(matrix_with_na)), ' percent')
#         result[i] = schur_complement(matrix_with_na[i],
#                                      Sigma)
#     return result


def guess_matrix(matrix_with_na, Sigma, min_rows=5):
    print('guessing matrix')
    full_null_mask = matrix_with_na.isnull()
    ranked_masks = pd.Series([tuple(el) for el in
                              full_null_mask.values]).value_counts().index

    for i in range(len(ranked_masks)):
        print('null mask %d' % i)
        mask_indexes = (full_null_mask ==
                        ranked_masks[i]).all(1)
        if mask_indexes.sum() <= min_rows:
            break
        print('there are %d rows' % mask_indexes.sum())
        matrix_with_na.loc[mask_indexes] = schur_complement_matrix(
            matrix_with_na.loc[mask_indexes].values,
            np.array(ranked_masks[i]),
            Sigma)
        # print(matrix_with_na)
    return matrix_with_na


# def soft_threshold_off_diag(Sigma, threshold):
#     clip = np.clip(Sigma, -threshold, threshold)
#     new_Sigma = Sigma - clip
#     new_Sigma += np.diag(np.diag(Sigma) -
#                          np.diag(new_Sigma))
#     return new_Sigma


class AutoRegressive:

    def __init__(self, train_normalized,
                 test_normalized,
                 future_lag,
                 past_lag, P):
        self.train_normalized = train_normalized
        self.test_normalized = test_normalized
        self.future_lag = future_lag
        self.past_lag = past_lag
        self.N = self.train_normalized.shape[1]
        self.P = P
        self.svd_results = {}
        self.embedding_covariances = {}
        self.full_covariances = {}
        # self.fit_AR_low_rank()
        self.fit_AR()
        # if plot:
        #     self.train_sq_err = self.test_model(self.train_normalized)
        #     self.plot_RMSE(np.sqrt(self.train_sq_err.mean()), train=True)
        #     self.test_normalized.std().plot(color='r',
        #                                     style='--')
        #     self.plot_RMSE(np.sqrt(self.test_sq_err.mean()), train=False)
        # self.train_normalized.std().plot(color='r',
        #                                  style='--')

    def plot(self):
        self.train_sq_err = self.test_model(self.train_normalized)
        self.plot_RMSE(np.sqrt(self.train_sq_err.mean()), train=True)
        self.test_normalized.std().plot(color='r',
                                        style='--')
        self.plot_RMSE(np.sqrt(self.test_sq_err.mean()), train=False)

    @property
    def test_RMSE(self):
        return np.sqrt(self.test_sq_err.mean().mean())

    @property
    def lag(self):
        return self.future_lag + self.past_lag

    def fit_AR(self):
        self._fit_low_rank_covariances()
        self._fit_full_AR()
        self._assemble_Sigma()
        self.test_sq_err = self.test_model(self.test_normalized)

    def _fit_low_rank_covariances(self):
        if self.P not in self.svd_results:
            print('computing rank %d svd of train data' % self.P)
            self.svd_results[self.P] = \
                iterative_denoised_svd(self.train_normalized,
                                       P=self.P)
            self.embedding_covariances[self.P] = {}
        u, s, v = self.svd_results[self.P]
        embedding = pd.DataFrame(u,
                                 index=self.train_normalized.index)
        print('computing lagged covariances')
        # self.lagged_covariances = {}
        # self.lagged_covariances[0] = \
        #     lagged_covariance(
        #         self.train_normalized, 0)
        for i in range(0, self.lag):
            # self.lagged_covariances[i] = lagged_covariance_svd(
            #    self.train_normalized, i, self.P)
            if i not in self.embedding_covariances[self.P]:
                self.embedding_covariances[self.P][i] = \
                    v.T @ np.diag(s) @ \
                    lagged_covariance(embedding, i)\
                    @ np.diag(s) @ v

            # self.lagged_covariances[i] = \
            #     v.T @ np.diag(s) @ lagged_covariance(
            #     embedding, i) @ np.diag(s) @ v

    def _fit_full_AR(self):
        for i in range(self.lag):  # self.lag):
            # self.lagged_covariances[i] = lagged_covariance_svd(
            #    self.train_normalized, i, self.P)
            if i not in self.full_covariances:
                print('computing raw covariance lag %d' % i)
                self.full_covariances[i] = lagged_covariance(
                    self.train_normalized, i)

    def _assemble_Sigma(self):
        print('adding diagonal to covariance')
        # self.raw_covariances = {}
        for i in range(self.lag):  # self.lag):
            # self.lagged_covariances[i] = lagged_covariance_svd(
            #    self.train_normalized, i, self.P)
            # self.raw_covariances[i] = lagged_covariance(
            #     self.train_normalized, i)

            self.embedding_covariances[self.P][i] += \
                np.diag(np.diag(self.full_covariances[i]) -
                        np.diag(self.embedding_covariances[self.P][i]))
            assert np.allclose(
                np.diag(self.embedding_covariances[self.P][i]) -
                np.diag(self.full_covariances[i]), 0)

        print('assembling covariance matrix')
        self.Sigma = pd.np.block(
            [[self.embedding_covariances[self.P][i].values.T
              if i > 0 else
              self.embedding_covariances[self.P][-i].values
                for i in range(-j, self.lag - j)]
                for j in range(self.lag)])
        assert np.allclose((self.Sigma - self.Sigma.T), 0)

    # def fit_AR_low_rank(self):
        # print('computing svd of train data')
        # u, s, v = \
        #     iterative_denoised_svd(self.train_normalized,
        #                            P=self.P)
        # embedding = pd.DataFrame(u,
        #                          index=self.train_normalized.index)
        # print('computing lagged covariances')
        # self.lagged_covariances = {}
        # # self.lagged_covariances[0] = \
        # #     lagged_covariance(
        # #         self.train_normalized, 0)
        # for i in range(0, self.lag):
        #     # self.lagged_covariances[i] = lagged_covariance_svd(
        #     #    self.train_normalized, i, self.P)

        #     self.lagged_covariances[i] = \
        #         v.T @ np.diag(s) @ lagged_covariance(
        #         embedding, i) @ np.diag(s) @ v
        # self.noise_level)

        # print('computing raw covariances')
        # self.raw_covariances = {}
        # for i in range(self.lag):  # self.lag):
        #     # self.lagged_covariances[i] = lagged_covariance_svd(
        #     #    self.train_normalized, i, self.P)
        #     self.raw_covariances[i] = lagged_covariance(
        #         self.train_normalized, i)

        #     self.lagged_covariances[i] += \
        #         np.diag(np.diag(self.raw_covariances[i]) -
        #                 np.diag(self.lagged_covariances[i]))
        #     assert np.allclose(
        #         np.diag(self.lagged_covariances[i]) -
        #         np.diag(self.raw_covariances[i]), 0)

        # print('assembling covariance matrix')
        # self.Sigma = pd.np.block(
        #     [[self.lagged_covariances[i].values.T
        #       if i > 0 else
        #       self.lagged_covariances[-i].values
        #         for i in range(-j, self.lag - j)]
        #         for j in range(self.lag)])
        # assert np.allclose((self.Sigma - self.Sigma.T), 0)

        # def fit_AR(self):
        #     print('computing lagged covariances')
        #     self.lagged_covariances = {}
        #     for i in range(self.lag):
        #         # self.lagged_covariances[i] = lagged_covariance_svd(
        #         #    self.train_normalized, i, self.P)
        #         self.lagged_covariances[i] = lagged_covariance(
        #             self.train_normalized, i)
        #         # self.noise_level)

        #     print('assembling covariance matrix')
        #     self.Sigma = pd.np.block(
        #         [[self.lagged_covariances[i].values.T
        #           if i > 0 else
        #           self.lagged_covariances[-i].values
        #             for i in range(-j, self.lag - j)]
        #             for j in range(self.lag)])
        #     assert np.allclose((self.Sigma - self.Sigma.T), 0)

    def plot_RMSE(self, RMSE, train):
        N = self.train_normalized.shape[1]
        for i in range(self.future_lag):
            RMSE[self.N * i:self.N * (i + 1)].plot(color='b' if train
                                                   else 'r',
                                                   alpha=1. / (i + 1))

    def test_model(self, data):
        test_concatenated = pd.concat([
            data.shift(-i)
            for i in range(self.lag)], axis=1)

        null_mask = pd.Series(False,
                              index=test_concatenated.columns)
        null_mask[self.past_lag * self.N:] = True

        to_guess = pd.DataFrame(test_concatenated, copy=True)
        to_guess.loc[:, null_mask] = np.nan
        # pd.DataFrame(data=
        guessed = guess_matrix(to_guess, self.Sigma)
        # index=to_guess.index,
        # columns=to_guess.columns)

        squared_error = (test_concatenated.loc[:, null_mask]
                         - guessed.loc[:, null_mask])**2
        return squared_error

        # costs = pd.Series(0., index=test_concatenated.index)
        # non_null_vals = pd.Series(0., index=test_concatenated.index)
        # for i, t in enumerate(test_concatenated.index):
        #     if i % 100 == 0:
        #         print('%.2f%%' % (100 * i /
        #                           len(test_concatenated)))
        #     original = test_concatenated.loc[t]
        #     test_row = pd.Series(original, copy=True)
        #     test_row[null_mask] = np.nan
        #     # self.schur_complement(test_row)
        #     guessed = \
        #         schur_complement(test_row.values, self.Sigma)
        #     costs.loc[t] = (((guessed[null_mask] -
        #                       original[null_mask])**2).sum())
        #     non_null_vals.loc[t] = ((~original[null_mask].isnull(
        #     )).sum())
        # time_RMSE = np.sqrt(costs / non_null_vals)
        # print('test RMSE of AR model %.2f' %
        #       np.sqrt((time_RMSE**2).mean()))
        # return time_RMSE

        # print('test residuals std. dev. %.2f' %
        #       np.sqrt((self.test_normalized**2).sum().sum() /
        #               (~self.test_normalized.isnull()).sum().sum()))

    # def schur_complement(self, concatenated_row):
    #     null_mask = concatenated_row.isnull().values
    #     y = concatenated_row[~null_mask].values

    #     A = self.Sigma[null_mask].T[null_mask]
    #     B = self.Sigma[null_mask].T[~null_mask].T
    #     C = self.Sigma[~null_mask].T[~null_mask]

    #     expected_x = B @ np.linalg.solve(C, y)
    #     concatenated_row[null_mask] = expected_x
    #     return concatenated_row


def Autotune_AutoRegressive(train_residuals,
                            test_residuals,
                            future_lag):
    ar_model = AutoRegressive(train_residuals,
                              test_residuals,
                              past_lag=1,
                              future_lag=future_lag,
                              P=0)
    P_range = train_residuals.shape[1] - 1
    past_lag_range = future_lag * 2

    Ps = np.arange(0, P_range)
    past_lags = np.arange(1, past_lag_range)
    result_RMSE = pd.DataFrame(index=past_lags,
                               columns=Ps)
    for past_lag in range(1, past_lag_range):
        ar_model.past_lag = past_lag
        for P in range(0, P_range):
            ar_model.P = P
            # if not np.isnan(result.loc[past_lag, P]):
            #     continue
            ar_model.fit_AR()

            print()
            print()
            print('past_lag = %d, P = %d' % (
                ar_model.past_lag, ar_model.P))
            print('test RMSE:', ar_model.test_RMSE)
            print()
            print()

            result_RMSE.loc[past_lag, P] = ar_model.test_RMSE
            if (P > 0) and result_RMSE.loc[past_lag, P] > \
                    result_RMSE.loc[past_lag, P - 1]:
                break
        if past_lag > 1 and result_RMSE.min(1)[past_lag] > \
                result_RMSE.min(1)[past_lag - 1]:
            break
    print('converged!')
    best_P = result_RMSE.min().idxmin()
    best_past_lag = result_RMSE.min(1).idxmin()
    return AutoRegressive(train_residuals,
                          test_residuals,
                          past_lag=best_past_lag,
                          future_lag=future_lag,
                          P=best_P)
