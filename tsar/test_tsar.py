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
from . import *


class TestTsar(TestCase):

    data = pd.DataFrame(pd.read_pickle('data/wind_test_data.pickle'))
    #train = data[data.index.year.isin([2010, 2011])]
    #test = data[data.index.year == 2012]

    def test_scalar(self):

        model = TSAR(self.data, future_lag=4 * 6)

    # def __init__(self, data: pd.DataFrame,
    #              future_lag: int,
    #              past_lag: Optional[int] = None,
    #              rank: Optional[int] = None,
    #              train_test_split: float = 2 / 3,
    #              baseline_params_columns: dict = {},
    #              available_data_lags_columns: dict = {},
    #              ignore_prediction_columns: List[Any] = []):

    #     check_multidimensional_time_series(data)

    #     self.data = data
    #     self.frequency = data.freq
    #     self.future_lag = future_lag
    #     self.past_lag = past_lag
    #     self.rank = rank
    #     self.train_test_split = train_test_split
    #     self.baseline_params_columns = baseline_params_columns
    #     self.available_data_lags_columns = available_data_lags_columns
    #     self.ignore_prediction_columns = ignore_prediction_columns

    #     self.columns = self.data.columns

    #     for col in self.columns:
    #         if col not in self.baseline_params_columns:
    #             self.baseline_params_columns[col] = {}
    #         if col not in self.available_data_lags_columns:
    #             self.available_data_lags_columns[col] = 0

    #     logger.debug('Fitting model on train and test data.')
    #     self._fit_ranges(self.train)
    #     self._fit_baselines(self.train, self.test)
    #     self._fit_low_rank_plus_block_diagonal_AR(self.train, self.test)

    #     logger.debug('Fitting model on whole data.')
    #     self._fit_ranges(self.data)
    #     self._fit_baselines(self.data)
    #     self._fit_low_rank_plus_block_diagonal_AR(self.data)

    #     # del self.data

    # @property
    # def train(self):
    #     return self.data.iloc[:len(data) * self.train_test_split]

    # @property
    # def test(self):
    #     self.test = self.data.iloc[len(data) * self.train_test_split:]

    # def _fit_ranges(self, data):
    #     logger.debug('Fitting ranges.')
    #     self._min = data.min()
    #     self._max = data.max()

    # def _clip_prediction(self, prediction: pd.DataFrame) -> pd.DataFrame:
    #     return prediction.clip(self._min, self._max, axis=1)

    # def _fit_baselines(self,
    #                    train: pd.DataFrame,
    #                    test: Optional[pd.DataFrame] = None):

    #     if test is not None:
    #         logger.debug('Computing baseline RMSE.')
    #         self.baseline_RMSE = pd.Series(index=self.columns)

    #     # TODO parallelize
    #     for col in self.columns:
    #         logger.debug('Fitting baseline on column %s.' % col)
    #         test_rmse = fit_baseline(
    #             train[col],
    #             test[col] if test is not None else None,
    #             self.baseline_params_columns[col])
    #         if test is not None:
    #             self.baseline_RMSE[col] = test_rmse

    # def _fit_low_rank_plus_block_diagonal_AR(
    #         self, train: pd.DataFrame,
    #         test: Optional[pd.DataFrame] = None):

    #     logger.debug('Fitting low-rank plus block diagonal.')

    #     self.Sigma, self.past_lag, self.rank, \
    #         predicted_residuals_at_lags = \
    #         fit_low_rank_plus_block_diagonal_AR(self._residual(train),
    #                                             self._residual(
    #                                                 test) if test is not None else None,
    #                                             self.future_lag,
    #                                             self.past_lag,
    #                                             self.rank,
    #                                             self.available_data_lags_columns,
    #                                             self.ignore_prediction_columns)

    #     if test is not None:
    #         logger.debug('Computing autoregression RMSE.')
    #         self.AR_RMSE = pd.DataFrame(columns=self.columns)
    #         for lag in range(self.future_lag):
    #             self.AR_RMSE.loc[i] = DataFrameRMSE(
    #                 self.test, self._invert_residual(
    #                     predicted_residuals_at_lags[i]))

    # def _residual(self, data: pd.DataFrame) -> pd.DataFrame:
    #     return data.apply(self._column_residual)

    # def _column_residual(self, column: pd.Series) -> pd.Series:
    #     return data_to_residual(column,
    #                             self.baseline_params_columns[column.name])

    # def _invert_residual(self, data: pd.DataFrame) -> pd.DataFrame:
    #     return self._clip_prediction(
    #         data.apply(self._column_invert_residual))

    # def _column_invert_residual(self, column: pd.Series) -> pd.Series:
    #     return residual_to_data(column,
    #                             self.baseline_params_columns[column.name])

    # def predict(self,
    #             data: pd.DataFrame,
    #             prediction_time:
    #             Optional[pd.Timestamp] = None) -> pd.DataFrame:
    #     check_multidimensional_time_series(data, self.frequency, self.columns)

    #     if prediction_time is None:
    #         prediction_time = data.index[-1] + self.frequency

    #     logger.debug('Predicting at time %s.' % prediction_time)

    #     prediction_index = \
    #         pd.date_range(
    #             start=prediction_time - self.frequency * self.past_lag,
    #             end=prediction_time + self.frequency * (self.future_lag - 1),
    #             freq=self.frequency)

    #     prediction_slice = data.reindex(prediction_index)
    #     residual_slice = self._residual(prediction_slice)
    #     residual_vectorized = residual_slice.values.flatten(order='F')
    #     predicted_vector = schur_complement_solve(
    #         residual_vectorized, self.Sigma)
    #     predicted_residuals = pd.DataFrame(
    #         predicted_vector.reshape(residual_slice.shape, order='F'),
    #         index=residual_slice.index,
    #         columns=residual_slice.columns)

    #     return self._invert_residual(predicted_residuals)

    # def baseline(self, prediction_window: pd.DatetimeIndex) -> pd.DataFrame:
    #     return self._invert_residual(pd.DataFrame(0., index=prediction_window,
    #                                               columns=self.columns))

    # def save_model(self, filename):
    #     pass
