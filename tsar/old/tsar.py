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

from .gaussianize import *
from .baseline import HarmonicBaseline, AutotunedBaseline
from .autotune import Autotune_AutoRegressive
from .autoregressive import AutoRegressive  # , Autotune_AutoRegressive
from .utils import RMSE, check_timeseries
from .scalar_autoregressor import autotune_scalar_autoregressor, ScalarAutoregressor
from .low_rank_autoregressor import *
from .low_rank_plus_block_diagonal_AR import LowRankPlusBlockDiagonalAR, \
    autotune_low_rank_plus_block_diag_ar

logger = logging.getLogger(__name__)

__all__ = ['Model']


class Model:

    def __init__(
            self,
            data,
            future_lag,
            baseline_per_column_options={},
            P=None,
            past_lag=None):

        check_timeseries(data)

        self.frequency = data.index.freq
        self._columns = data.columns
        self.future_lag = future_lag
        # TODO fix
        self.max_past_lag = future_lag
        self.data = data

        self.train = data.iloc[:-len(data) // 3]
        self.test = data.iloc[-len(data) // 3:]
        # prilen(train), len(test)

        self.baseline_per_column_options =\
            baseline_per_column_options
        self.P = P
        self.past_lag = past_lag

        self._fit_ranges()
        # self._fit_gaussianization(self.train)

        # self.gaussianized_train = self._gaussianize(self.train)
        # self.gaussianized_test = self._gaussianize(self.test)

        # self._fit_baselines(self.gaussianized_train,
        #                     self.gaussianized_test)

        self._fit_baselines(self.train,
                            self.test)

        self.train_residual = self.residuals(self.train)
        self.test_residual = self.residuals(self.test)

        bad_columns = self.test_residual.columns[
            np.sqrt((self.test_residual**2).mean()) > 3]

        if len(bad_columns):
            logger.warning(
                'Columns %s have very large test residual, you should drop them.' %
                list(bad_columns))
        # self._residuals_stds = self._train_residuals.std()
        # self._train_normalized_residuals = self._train_residuals / self._residuals_stds
        # self._test_normalized_residuals = self._test_residuals / self._residuals_stds
        # self.train_residuals =

        self._fit_scalar_AR(self.train_residual,
                            self.test_residual)
        pass
        # self._fit_AR(self.train_residual, self.test_residual)
        self._fit_low_rank_plus_block_diag_AR(self.train_residual,
                                              self.test_residual)
        pass

        # TODO refit with full data

    # def _fit_gaussianization(self, train):
    #     self.gaussanization_params = {}
    #     for col in self._columns:
    #         self.gaussanization_params[col] = \
    #             compute_gaussian_interpolation(train[col])

    # def _gaussianize(self, data):
    #     return pd.DataFrame(
    #         {
    #             k: gaussianize(
    #                 data[k],
    #                 *
    #                 self.gaussanization_params[k]) for k in self._columns},
    #         index=data.index)[
    #             self._columns]

    # def _invert_gaussianize(self, data):
    #     return pd.DataFrame(
    #         {
    #             k: invert_gaussianize(
    #                 data[k],
    #                 *self.gaussanization_params[k]) for k in self._columns},
    #         index=data.index)[
    #             self._columns]

    def _fit_ranges(self):
        self._min = self.train.min()
        self._max = self.train.max()

    def _clip_prediction(self, prediction):
        return prediction.clip(self._min, self._max, axis=1)

    # @property
    # def _train_baseline(self):
    #     return pd.concat(
    #         [pd.Series(self._baselines[col]._baseline,
    #             index=for col in self._columns], axis=1)

    # @property
    # def _test_baseline(self):
    #     return pd.concat(
    #         [self._baselines[col]._predict_baseline(self.test.index)
    #          for col in self._columns], axis=1)

    # @property
    # def _train_residuals(self):
    # return self.gaussianized_train - self.predict_baseline(self.data.index)

    # @property
    # def _test_residuals(self):
    # return self.gaussianized_test - self.predict_baseline(self.test.index)

    def predict(self, last_data, number_scenarios=0):
        print('last_data', last_data.shape)
        # print(last_data.index)
        if len(last_data) > self.lag:
            raise ValueError('Only provide the last data.')
        if not last_data.index.freq == self.frequency:
            raise ValueError('Provided data has wrong frequency.')
        len_chunk = len(last_data)
        for i in range(1, 1 + self.lag - len(last_data)):
            # print('i = ', i)
            t = last_data.index[len_chunk - 1] + i * self.frequency
            # print('adding row', t)
            last_data.loc[t] = np.nan
        print('last_data', last_data.shape)
        baseline = self.predict_baseline(last_data.index)
        print('baseline', baseline.shape)
        residuals = self._gaussianize(last_data) - baseline
        normalized_residuals = residuals / self._residuals_stds
        normalized_residuals_list = self._predict_normalized_residual_AR(
            normalized_residuals, number_scenarios)
        all_results = []
        for normalized_residuals in normalized_residuals_list:
            residuals = normalized_residuals * self._residuals_stds
            all_results.append(
                self._clip_prediction(self._invert_gaussianize(
                    residuals + baseline)))
            if not number_scenarios:
                return all_results[-1]
        return all_results

    def baseline(self, index):
        return pd.concat(
            [self._baselines[col].baseline(index)
             for col in self._columns], axis=1)

    def residuals(self, data):
        return pd.concat(
            [self._baselines[col].residuals(data[col])
             for col in self._columns], axis=1)

    def invert_residuals(self, data):
        return pd.concat(
            [self._baselines[col].invert_residuals(data[col])
             for col in self._columns], axis=1)

    def _fit_baselines(self, train, test):
        self._baselines = {}
        for column in self._columns:
            if column in self.baseline_per_column_options:
                self._baselines[column] = AutotunedBaseline(
                    train[column],
                    test[column],
                    **self.baseline_per_column_options[column]
                )
            else:
                self._baselines[column] = AutotunedBaseline(
                    train[column],
                    test[column]
                )

    def _fit_scalar_AR(self, train, test):
        self._scalar_ARs = {}
        for column in self._columns:
            # if column in self.baseline_per_column_options:
            #     self._baselines[column] = AutotunedBaseline(
            #         train[column],
            #         test[column],
            #         **self.baseline_per_column_options[column]
            #     )
            # else:
            print('fitting scalar AR for column %s' % column)
            # past_lag, = autotune_scalar_autoregressor(
            #     train[column], test[column],
            #     future_lag=self.future_lag,
            #     past_lag = self.past_lag
            #     max_past_lag=self.max_past_lag)
            self._scalar_ARs[column] = ScalarAutoregressor(
                train[column],
                future_lag=self.future_lag,
                past_lag=self.past_lag)

    def _fit_low_rank_plus_block_diag_AR(self, train, test):

        scalar_sigma_arrays = [self._scalar_ARs[col].lagged_covariances
                               for col in self._columns]

        P, past_lag = autotune_low_rank_plus_block_diag_ar(
            train,
            test,
            scalar_sigma_arrays,
            self.future_lag,
            self.past_lag,
            self.P)

        self.ar_model = LowRankPlusBlockDiagonalAR(train, scalar_sigma_arrays,
                                                   P, self.future_lag,
                                                   past_lag)

    @property
    def scalar_RMSEs(self):
        RMSEs = pd.DataFrame(
            index=['RMSE_prediction_lag_%d' %
                   (i + 1) for i in range(self.future_lag)] +
            ['RMSE_baseline'])

        for col in self._columns:
            guessed_at_lags = \
                self._scalar_ARs[col].test_predict(self.test_residual[col])

            real_guessed_at_lags = \
                [self._baselines[col].invert_residuals(guessed.iloc[:, 0])
                 for guessed in guessed_at_lags]

            RMSEs[col] = [
                RMSE(real_guessed_at_lags[i].values,
                     self.test[col].values) for i in range(
                    self.future_lag)] + [
                RMSE(self._baselines[col].baseline(
                    self.test.index).values,
                    self.test[col].values)]
        return RMSEs

    @property
    def Sigma(self):
        return self.ar_model.Sigma

    @property
    def lag(self):
        return self.ar_model.lag

    # def _fit_AR(self, train, test):
    #     self.ar_model = Autotune_AutoRegressive(
    #         train,
    #         test,
    #         self.future_lag,
    #         self.P,
    #         self.past_lag)

    def _fit_low_rank_AR(self, train, test):

        P, past_lag = autotune_low_rank_autoregressor(
            train,
            test,
            self.future_lag,
            self.past_lag,
            self.P)

        self.ar_model = LowRankAR(train, P, self.future_lag,
                                  past_lag)

    @property
    def low_rank_RMSEs(self):
        # TODO alloc here is inefficient
        RMSEs = pd.DataFrame(
            index=['RMSE_prediction_lag_%d' %
                   (i + 1) for i in range(self.future_lag)] +
            ['RMSE_baseline'], columns=self._columns)

        guessed_at_lags = \
            self.ar_model.test_predict(self.test_residual)

        real_guessed_at_lags = \
            [self.invert_residuals(guessed)
             for guessed in guessed_at_lags]

        for i in range(self.future_lag):
            RMSEs.iloc[i, :] = np.sqrt(((real_guessed_at_lags[i] -
                                         self.test)**2).mean())
        RMSEs.iloc[-1, :] = np.sqrt(((self.baseline(self.test.index) -
                                      self.test)**2).mean())

        return RMSEs
        # print('computing lagged covariances')
        # self.lagged_covariances = {}
        # for i in range(self.lag):
        #     self.lagged_covariances[i] = \
        #         pd.concat((self._normalized_residuals,
        #                    self._normalized_residuals.shift(i)),
        #                   axis=1).corr().iloc[:len(self._columns),
        #                                       len(self._columns):]
        # print('assembling covariance matrix')
        # self.Sigma = pd.np.block(
        #     [[self.lagged_covariances[np.abs(i)].values
        #         for i in range(-j, self.lag - j)]
        #         for j in range(self.lag)]
        # )

    def _predict_concatenated_AR(self,
                                 concatenated,
                                 number_scenarios=0):

        # https://en.wikipedia.org/wiki/Schur_complement
        # (Applications_to_probability_theory_and_statistics)

        null_mask = concatenated.isnull().values
        y = concatenated[~null_mask].values

        A = self.Sigma[null_mask].T[null_mask]
        B = self.Sigma[null_mask].T[~null_mask].T
        C = self.Sigma[~null_mask].T[~null_mask]

        expected_x = B @ np.linalg.solve(C, y)
        concatenated[null_mask] = expected_x

        if number_scenarios:
            print('computing conditional covariance')
            Sigma_x = A - B @ np.linalg.inv(C) @ B.T
            samples = np.random.multivariate_normal(
                expected_x, Sigma_x, number_scenarios)
            sample_concatenations = []
            for sample in samples:
                concatenated[null_mask] = sample
                sample_concatenations.append(
                    pd.Series(concatenated, copy=True))
            return sample_concatenations

        return [concatenated]

    def plot_test_RMSEs(self):
        import matplotlib.pyplot as plt

        all_residuals = self.ar_model.test_model_NEW(
            self.ar_model.test_normalized)
        all_results = []
        baseline = self.baseline(self.test.index)

        for residuals in all_residuals:
            # residuals = el * self._residuals_stds
            all_results.append(
                self._clip_prediction(
                    self.invert_residuals(residuals)))
        # inverted_baseline = self._invert_gaussianize(baseline)
        for column in self._columns:
            plt.figure()
            plt.plot([pd.np.sqrt((all_results[i][column] - self.test[column])**2).mean()
                      for i in range(self.future_lag)], 'k.-', label='AR prediction')
            plt.plot([pd.np.sqrt((baseline[column] - self.test[column])**2).mean()
                      for i in range(self.future_lag)], 'k--', label='baseline')
            plt.title(column)
            plt.legend()
            plt.xlabel('prediction lag')
            plt.ylabel('RMSE')

    def _predict_normalized_residual_AR(self, chunk,
                                        number_scenarios=0):
        # chunk = model._normalized_residuals.iloc[-10:]
        assert len(chunk) == self.lag
        chunk_index = chunk.index

        concatenated = pd.concat(
            [
                chunk.iloc[i]
                for i in range(self.lag)
            ])

        filled_list = self._predict_concatenated_AR(concatenated,
                                                    number_scenarios)
        chunk_filled_list = []

        for filled in filled_list:
            chunk_filled = pd.concat(
                [filled.iloc[len(self._columns) * i:len(self._columns) * (i + 1)]
                    for i in range(self.lag)], axis=1).T
            chunk_filled.index = chunk_index
            chunk_filled_list.append(chunk_filled)

        return chunk_filled_list
