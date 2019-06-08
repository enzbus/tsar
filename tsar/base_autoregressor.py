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

from .greedy_grid_search import greedy_grid_search
from .linear_algebra import schur_complement_matrix


def guess_matrix(matrix_with_na, Sigma, min_rows=5, max_eval=5):
    print('guessing matrix')
    full_null_mask = matrix_with_na.isnull()
    ranked_masks = pd.Series([tuple(el) for el in
                              full_null_mask.values]).value_counts().index

    for i in range(len(ranked_masks))[:max_eval]:
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


class BaseAutoregressor:

    @property
    def lag(self):
        return self.future_lag + self.past_lag

    def test_predict(self, test):

        # test_concatenated = pd.concat([
        #     test.shift(-i)
        #     for i in range(self.lag)], axis=1)

        # TODO packet into own function
        if self.N > 1:
            test_concatenated = pd.concat([
                pd.concat([test.iloc[:, j].shift(-i)
                           for i in range(self.lag)], axis=1)
                for j in range(self.N)], axis=1)
        else:
            test_concatenated = pd.concat([
                test.shift(-i)
                for i in range(self.lag)], axis=1)

        future = pd.Series(False,
                           index=test_concatenated.columns)
        for j in range(self.N):
            future[j * self.lag + self.past_lag:
                   (j + 1) * self.lag] = True

        #null_mask[self.past_lag * self.N:] = True

        to_guess = pd.DataFrame(test_concatenated, copy=True)
        to_guess.loc[:, future] = np.nan
        guessed = guess_matrix(to_guess, self.Sigma).loc[
            :, future]
        assert guessed.shape[1] == self.future_lag * self.N
        guessed_at_lag = []
        masker = np.arange(self.future_lag * self.N)
        for i in range(self.future_lag):
            # to_append = guessed.iloc[:, self.N * i:
            # self.N * (i + 1)].shift(i + self.past_lag)
            to_append = guessed.iloc[
                :, (masker % self.future_lag) == i].shift(i + self.past_lag)
            # to_append.columns = [el + '_lag_%d' % (i + 1) for el in
            #                      to_append.columns]
            guessed_at_lag.append(to_append)
        return guessed_at_lag

    def test_RMSE(self, test):  # , prediction_columns=None):
        guessed_at_lags = self.test_predict(test)
        all_errors = np.zeros(0)
        for i, guessed in enumerate(guessed_at_lags):
            # if prediction_columns is None:
            errors = (guessed_at_lags[i].values -
                      pd.DataFrame(test).values).flatten()
            # else:
            #     errors = (
            #         guessed_at_lags[i][prediction_columns] -
            #         pd.DataFrame(test)[prediction_columns]).values.flatten()
            errors = errors[~np.isnan(errors)]
            print('RMSE at lag %d = %.2f' % (i + 1,
                                             np.sqrt(np.mean(errors**2))))
            all_errors = np.concatenate([all_errors, errors])
        return np.sqrt(np.mean(all_errors**2))
