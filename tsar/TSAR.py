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
from .linear_algebra import dense_schur


# TODOs
# - cache results of vector autoregression (using hash(array.tostring()))
# - same for results of matrix Schur


class TSAR:

    def __init__(self, data: pd.DataFrame,
                 future_lag: int,
                 baseline_params_columns: dict = {},
                 past_lag: Optional[int] = None,
                 rank: Optional[int] = None,
                 return_performance_statistics=True,
                 train_test_split: float = 2 / 3,
                 available_data_lags_columns: dict = {},
                 ignore_prediction_columns: List[Any] = [],
                 full_covariance_blocks: List[Any] = [],
                 full_covariance = False,
                 quadratic_regularization = None):

        # TODO REMOVE NULL COLUMNS OR REFUSE THEM

        # TODO TELL USER WHEN INTERNAL TRAIN DATA IS ALL MISSING!

        check_multidimensional_time_series(data)

        self.data = data
        self.frequency = data.index.freq
        self.future_lag = future_lag
        self.past_lag = past_lag
        self.rank = rank
        self.train_test_split = train_test_split
        self.baseline_params_columns = baseline_params_columns
        self.return_performance_statistics = return_performance_statistics
        self.baseline_results_columns = {}
        self.available_data_lags_columns = available_data_lags_columns
        self.ignore_prediction_columns = ignore_prediction_columns
        self.full_covariance = full_covariance
        self.quadratic_regularization = quadratic_regularization

        self.full_covariance_blocks = full_covariance_blocks

        # TODO tell why
        assert len(sum(full_covariance_blocks, [])) == \
            len(set(sum(full_covariance_blocks, [])))

        self.not_in_blocks = set(self.data.columns).difference(
            sum(full_covariance_blocks, []))

        self.full_covariance_blocks += [[el] for el in self.not_in_blocks]

        # order columns by blocks
        self.columns = pd.Index(sum(self.full_covariance_blocks, []))

        self.data = self.data[self.columns]

        #self.columns = self.data.columns

        for col in self.columns:
            self.baseline_results_columns[col] = {}
            if col not in self.baseline_params_columns:
                self.baseline_params_columns[col] = {}
            if col not in self.available_data_lags_columns:
                self.available_data_lags_columns[col] = 0

        self.fit_train_test()
        self.fit()
        # del self.data

    def fit_train_test(self):
        logger.debug('Fitting model on train and test data.')
        self._fit_ranges(self.train)
        self._fit_baselines(self.train, self.test)
        self._fit_low_rank_plus_block_diagonal_AR(self.train, self.test)
        self.AR_RMSE = self.test_AR(self.test)

    def fit(self):
        logger.debug('Fitting model on whole data.')
        self._fit_ranges(self.data)
        self._fit_baselines(self.data, None)
        self._fit_low_rank_plus_block_diagonal_AR(self.data, None)

    @property
    def Sigma(self):
        return self.V @ self.S @ self.V.T + self.D_matrix

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

        if (test is not None) and self.return_performance_statistics:
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

            if (test is not None) and self.return_performance_statistics:
                self.baseline_RMSE[col] = optimal_rmse

    def _fit_low_rank_plus_block_diagonal_AR(
            self, train: pd.DataFrame,
            test: Optional[pd.DataFrame] = None):

        logger.debug('Fitting low-rank plus block diagonal.')

        # self.Sigma, self.past_lag, self.rank, \
        #     predicted_residuals_at_lags

        self.past_lag, self.rank, self.quadratic_regularization, \
            self.V, self.S, self.S_inv, \
            self.D_blocks, self.D_matrix = \
            fit_low_rank_plus_block_diagonal_AR(self._residual(train),
                                                self._residual(
                                                    test) if test is not None else None,
                                                self.future_lag,
                                                self.past_lag,
                                                self.rank,
                                                self.available_data_lags_columns,
                                                self.ignore_prediction_columns,
                                                self.full_covariance,
                                                self.full_covariance_blocks,
                                                self.quadratic_regularization)

    def test_AR(self, test):

        test = test[self.columns]

        AR_RMSE = rmse_AR(self.V, self.S, self.S_inv,
                               self.D_blocks,
                               self.D_matrix,
                               self.past_lag, self.future_lag,
                               self._residual(test),
                               self.available_data_lags_columns,
                               self.quadratic_regularization)

        for col in AR_RMSE.columns:
            AR_RMSE[col] *= self.baseline_results_columns[col]['std']

        return AR_RMSE

            # logger.debug('Computing autoregression RMSE.')
            # self.AR_RMSE = pd.DataFrame(columns=self.columns)
            # for lag in range(self.future_lag):
            #     self.AR_RMSE.loc[i] = DataFrameRMSE(
            #         self.test, self._invert_residual(
            #             predicted_residuals_at_lags[i]))

    def test_model(self, test):
        residual = self._residual(test)
        baseline = self.baseline(test.index)
        baseline_RMSE = DataFrameRMSE(test, baseline)
        AR_RMSE = self.test_AR(residual)
        return baseline_RMSE, AR_RMSE

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
                Optional[pd.Timestamp]=None,
                return_sigmas=False,
                quadratic_regularization=None) -> pd.DataFrame:
        check_multidimensional_time_series(data, self.frequency, self.columns)

        data = data[self.columns]

        if prediction_time is None:
            prediction_time = data.index[-1] + self.frequency

        logger.debug('Predicting at time %s.' % prediction_time)

        prediction_index = pd.date_range(
            start=prediction_time - self.frequency * self.past_lag,
            end=prediction_time + self.frequency * (self.future_lag - 1),
            freq=self.frequency)

        prediction_slice = data.reindex(prediction_index)
        residual_slice = self._residual(prediction_slice)
        residual_vectorized = residual_slice.values.flatten(order='F')

        # TODO move up
        self.D_blocks_indexes = make_block_indexes(self.D_blocks)
        known_mask = ~np.isnan(residual_vectorized)

        res = dense_schur(self.Sigma, known_mask=known_mask,
                          known_vector = residual_vectorized[known_mask],
                        return_conditional_covariance=return_sigmas,
                        quadratic_regularization=quadratic_regularization if\
                        quadratic_regularization is not None else
                        self.quadratic_regularization)
        # res = symm_low_rank_plus_block_diag_schur(
        #     V=self.V,
        #     S=self.S,
        #     S_inv=self.S_inv,
        #     D_blocks=self.D_blocks,
        #     D_blocks_indexes=self.D_blocks_indexes,
        #     D_matrix=self.D_matrix,
        #     known_mask=known_mask,
        #     known_matrix=np.matrix(residual_vectorized[known_mask]),
        #     return_conditional_covariance=return_sigmas)
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

    def plot_RMSE(self, col):
        import matplotlib.pyplot as plt
        ax = self.AR_RMSE[col].plot(style='k.-')
        ax.axhline(self.baseline_RMSE[col], color='k', linestyle='--')
        plt.xlabel('lag')
        plt.xlim([0, self.future_lag + 1])
        plt.ylim([None, self.baseline_RMSE[col] * 1.05])
        plt.title(f'Prediction RMSE {col}')
        return ax

    def plot_all_RMSEs(self):
        import matplotlib.pyplot as plt
        for col in self.columns:
            plt.figure()
            self.plot_RMSE(col)

    def save_model(self, filename):
        pass
