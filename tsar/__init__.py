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

import pandas as pd
import logging
from typing import Optional, List, Any
logger = logging.getLogger(__name__)


from .baseline import fit_baseline, data_to_residual, residual_to_data
from .AR import fit_low_rank_plus_block_diagonal_AR, rmse_AR
from .utils import DataFrameRMSE, check_multidimensional_time_series
from .linear_algebra import *


# TODOs
# - cache results of vector autoregression (using hash(array.tostring()))
# - same for results of matrix Schur


class TSAR:

    def __init__(self, data: pd.DataFrame,
                 future_lag: int,
                 past_lag: Optional[int] = None,
                 rank: Optional[int] = None,
                 train_test_split: float = 2 / 3,
                 baseline_params_columns: dict = {},
                 available_data_lags_columns: dict = {},
                 ignore_prediction_columns: List[Any] = []):

        check_multidimensional_time_series(data)

        self.data = data
        self.frequency = data.index.freq
        self.future_lag = future_lag
        self.past_lag = past_lag
        self.rank = rank
        self.train_test_split = train_test_split
        self.baseline_params_columns = baseline_params_columns
        self.baseline_results_columns = {}
        self.available_data_lags_columns = available_data_lags_columns
        self.ignore_prediction_columns = ignore_prediction_columns

        self.columns = self.data.columns

        for col in self.columns:
            self.baseline_results_columns[col] = {}
            if col not in self.baseline_params_columns:
                self.baseline_params_columns[col] = {}
            if col not in self.available_data_lags_columns:
                self.available_data_lags_columns[col] = 0

        self.fit(refit_with_whole_data=True)
        # del self.data

    def fit(self, refit_with_whole_data=True):
        logger.debug('Fitting model on train and test data.')
        self._fit_ranges(self.train)
        self._fit_baselines(self.train, self.test)
        self._fit_low_rank_plus_block_diagonal_AR(self.train, self.test)

        if refit_with_whole_data:
            logger.debug('Fitting model on whole data.')
            self._fit_ranges(self.data)
            self._fit_baselines(self.data)
            self._fit_low_rank_plus_block_diagonal_AR(self.data)

    @property
    def train(self):
        return self.data.iloc[:int(len(self.data) * self.train_test_split)]

    @property
    def test(self):
        return self.data.iloc[
            int(len(self.data) * self.train_test_split):]

    def _fit_ranges(self, data):
        logger.info('Fitting ranges.')
        self._min = data.min()
        self._max = data.max()

    def _clip_prediction(self, prediction: pd.DataFrame) -> pd.DataFrame:
        return prediction.clip(self._min, self._max, axis=1)

    def _fit_baselines(self,
                       train: pd.DataFrame,
                       test: Optional[pd.DataFrame] = None):

        logger.info('Fitting baselines.')

        if test is not None:
            logger.debug('Computing baseline RMSE.')
            self.baseline_RMSE = pd.Series(index=self.columns)

        # TODO parallelize
        for col in self.columns:
            logger.debug('Fitting baseline on column %s.' % col)

            self.baseline_results_columns[col]['std'], \
                self.baseline_params_columns[col]['daily_harmonics'], \
                self.baseline_params_columns[col]['weekly_harmonics'], \
                self.baseline_params_columns[col]['annual_harmonics'], \
                self.baseline_params_columns[col]['trend'],\
                self.baseline_results_columns[col]['baseline_fit_result'], \
                optimal_rmse = fit_baseline(
                train[col],
                test[col] if test is not None else None,
                **self.baseline_params_columns[col])

            if test is not None:
                self.baseline_RMSE[col] = optimal_rmse

    def _fit_low_rank_plus_block_diagonal_AR(
            self, train: pd.DataFrame,
            test: Optional[pd.DataFrame] = None):

        logger.debug('Fitting low-rank plus block diagonal.')

        # self.Sigma, self.past_lag, self.rank, \
        #     predicted_residuals_at_lags

        self.past_lag, self.rank, self.V, self.S, self.S_inv, \
            self.D_blocks, self.D_matrix, self.D_inv =\
            fit_low_rank_plus_block_diagonal_AR(self._residual(train),
                                                self._residual(
                                                    test) if test is not None else None,
                                                self.future_lag,
                                                self.past_lag,
                                                self.rank,
                                                self.available_data_lags_columns,
                                                self.ignore_prediction_columns)

        if test is not None:

            self.AR_RMSE = rmse_AR(self.V, self.S, self.S_inv,
                                   self.D_blocks,
                                   self.D_matrix, self.D_inv,
                                   self.past_lag, self.future_lag,
                                   self._residual(test),
                                   self.available_data_lags_columns)

            for col in self.AR_RMSE.columns:
                self.AR_RMSE[col] *= self.baseline_results_columns[col]['std']

            # logger.debug('Computing autoregression RMSE.')
            # self.AR_RMSE = pd.DataFrame(columns=self.columns)
            # for lag in range(self.future_lag):
            #     self.AR_RMSE.loc[i] = DataFrameRMSE(
            #         self.test, self._invert_residual(
            #             predicted_residuals_at_lags[i]))

    def _residual(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.apply(self._column_residual)

    def _column_residual(self, column: pd.Series) -> pd.Series:
        return data_to_residual(column,
                                **self.baseline_results_columns[column.name],
                                **self.baseline_params_columns[column.name])

    def _invert_residual(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._clip_prediction(
            data.apply(self._column_invert_residual))

    def _column_invert_residual(self, column: pd.Series) -> pd.Series:
        return residual_to_data(column,
                                **self.baseline_results_columns[column.name],
                                **self.baseline_params_columns[column.name])

    def predict(self,
                data: pd.DataFrame,
                prediction_time:
                Optional[pd.Timestamp] = None,
                return_sigmas=False) -> pd.DataFrame:
        check_multidimensional_time_series(data, self.frequency, self.columns)

        if prediction_time is None:
            prediction_time = data.index[-1] + self.frequency

        logger.debug('Predicting at time %s.' % prediction_time)

        prediction_index = \
            pd.date_range(
                start=prediction_time - self.frequency * self.past_lag,
                end=prediction_time + self.frequency * (self.future_lag - 1),
                freq=self.frequency)

        prediction_slice = data.reindex(prediction_index)
        residual_slice = self._residual(prediction_slice)
        residual_vectorized = residual_slice.values.flatten(order='F')

        # TODO move up
        self.D_blocks_indexes = make_block_indexes(self.D_blocks)
        known_mask = ~np.isnan(residual_vectorized)
        res = \
            symm_low_rank_plus_block_diag_schur(
                V=self.V,
                S=self.S,
                S_inv=self.S_inv,
                D_blocks=self.D_blocks,
                D_blocks_indexes=self.D_blocks_indexes,
                D_matrix=self.D_matrix,
                known_mask=known_mask,
                known_matrix=residual_vectorized[known_mask],
                return_conditional_covariance=return_sigmas)
        if return_sigmas:
            predicted, Sigma = res
            sigval = np.zeros(len(residual_vectorized))
            sigval[~known_mask] = np.diag(Sigma)
            sigma = pd.DataFrame(sigval.reshape(residual_slice.shape,
                                                order='F'),
                                 index=residual_slice.index,
                                 columns=residual_slice.columns)
            for col in sigma.columns:
                sigma[col] *= self.baseline_results_columns[col]['std']

        else:
            predicted = res

        # TODO fix
        residual_vectorized[~known_mask] = np.array(predicted).flatten()
        # residual_vectorized[~known_mask]

        # schur_complement_solve(
        #     residual_vectorized, self.Sigma)
        predicted_residuals = pd.DataFrame(
            residual_vectorized.reshape(residual_slice.shape, order='F'),
            index=residual_slice.index,
            columns=residual_slice.columns)

        if return_sigmas:
            return self._invert_residual(predicted_residuals), sigma
        else:
            return self._invert_residual(predicted_residuals)

    def baseline(self, prediction_window: pd.DatetimeIndex) -> pd.DataFrame:
        return self._invert_residual(pd.DataFrame(0., index=prediction_window,
                                                  columns=self.columns))

    def save_model(self, filename):
        pass
