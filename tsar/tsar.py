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

from .non_par_baseline import fit_baseline as non_par_fit_baseline, \
    data_to_residual as non_par_data_to_residual, \
    residual_to_data as non_par_residual_to_data
from tsar.linear_algebra import symm_low_rank_plus_block_diag_schur, \
    make_block_indexes
from .utils import DataFrameRMSE, check_multidimensional_time_series,\
    sanitize_baseline_params
from .AR import fit_low_rank_plus_block_diagonal_AR, \
    rmse_AR, anomaly_score, build_matrices, \
    make_sliced_flattened_matrix, make_prediction_mask, guess_matrix
from .baseline import fit_many_baselines, data_to_normalized_residual, \
    normalized_residual_to_data
import pandas as pd
import numpy as np
import pickle
import gzip

import logging
from typing import Optional, List, Any
logger = logging.getLogger(__name__)


# from .linear_algebra import dense_schur


# TODOs
# - cache results of vector autoregression (using hash(array.tostring()))
# - same for results of matrix Schur

def split_train(data, train_test_ratio):
    return data.iloc[:int(len(data) * train_test_ratio)]


def split_test(data, train_test_ratio):
    return data.iloc[int(len(data) * train_test_ratio):]


class tsar:

    def __init__(self, data: pd.DataFrame,
                 F: int,
                 baseline_params_columns: dict = None,
                 P: Optional[int] = None,
                 R: Optional[int] = None,
                 return_performance_statistics=True,
                 train_test_split: float = 2 / 3,
                 available_data_lags_columns: dict = None,
                 ignore_prediction_columns: Optional[List[Any]] = None,
                 full_covariance_blocks: Optional[List[Any]] = None,
                 full_covariance: bool = False,
                 quadratic_regularization: float = None,
                 noise_correction: bool = False,
                 prediction_variables_weight: Optional[float] = None,
                 use_svd_fit: bool = False,
                 compute_gradients: bool = False,
                 parallel_fit=True):

        # TODO REMOVE NULL COLUMNS OR REFUSE THEM

        # TODO TELL USER WHEN INTERNAL TRAIN DATA IS ALL MISSING!

        check_multidimensional_time_series(data)

        # self.data = data
        self.frequency = data.index.freq
        self.future_lag = F
        self.past_lag = P
        self.rank = R
        self.train_test_split = train_test_split

        self.baseline_params_columns = {} if baseline_params_columns is None \
            else baseline_params_columns

        self.return_performance_statistics = return_performance_statistics
        self.baseline_results_columns = {}

        self.available_data_lags_columns = {} if available_data_lags_columns \
            is None else available_data_lags_columns

        self.ignore_prediction_columns = [] if ignore_prediction_columns \
            is None else ignore_prediction_columns

        self.full_covariance = full_covariance
        self.quadratic_regularization = quadratic_regularization

        self.full_covariance_blocks = [] if full_covariance_blocks \
            is None else full_covariance_blocks

        self.noise_correction = noise_correction
        self.prediction_variables_weight = prediction_variables_weight

        # TODO tell why
        assert len(sum(self.full_covariance_blocks, [])) == \
            len(set(sum(self.full_covariance_blocks, [])))

        self.not_in_blocks = set(data.columns).difference(
            sum(self.full_covariance_blocks, []))

        self.full_covariance_blocks += [[el] for el in self.not_in_blocks]

        # order columns by blocks
        self.columns = pd.Index(sum(self.full_covariance_blocks, []))

        # self.data = self.data[self.columns]
        reordered_data = data[self.columns]

        self.use_svd_fit = use_svd_fit
        self.compute_gradients = compute_gradients

        self.parallel_fit = parallel_fit

        self.prediction_cache = {}

        # self.columns = self.data.columns

        # FIT
        # ranges TODO put in "if" block
        self._fit_ranges(reordered_data)

        # baselines
        self._prepare_baseline_params_and_results()
        self._fit_baselines(reordered_data)

        # gaussian
        self._fit_low_rank_plus_block_diagonal_AR(reordered_data)

        # # this block will be rewritten
        # self._fit_low_rank_plus_block_diagonal_AR(
        #     split_train(reordered_data, self.train_test_split),
        #     split_test(reordered_data, self.train_test_split))
        # self.compute_gradients = False
        # self.AR_RMSE = self.test_AR(
        #     split_test(reordered_data, self.train_test_split))
        # self.compute_gradients = compute_gradients
        # self._fit_low_rank_plus_block_diagonal_AR(reordered_data, None)

        # self.
        # self.fit_train_test()
        # self.fit()
        # del self.data

    def _prepare_baseline_params_and_results(self):

        self.baseline_params_columns = \
            sanitize_baseline_params(self.baseline_params_columns,
                                     self.columns)

        for col in self.columns:
            self.baseline_results_columns[col] = {}

            # if col not in self.baseline_params_columns:
            #     self.baseline_params_columns[col] = {}
            #
            # # non parametric baseline
            # if 'non_par_baseline' not in self.baseline_params_columns[col]:
            #     self.baseline_params_columns[col]['non_par_baseline'] = False
            #
            # if self.baseline_params_columns[col]['non_par_baseline'] and not\
            #    ('used_features' in self.baseline_params_columns[col]):
            #     self.baseline_params_columns[col]['used_features'] = \
            #         ['hour_of_day', 'day_of_week', 'month_of_year']
            # if self.baseline_params_columns[col]['non_par_baseline'] and not\
            #    ('lambdas' in self.baseline_params_columns[col]):
            #     self.baseline_params_columns[col]['lambdas'] = \
            #         [None] * len(self.baseline_params_columns[col]
            #                      ['used_features'])
            if col not in self.available_data_lags_columns:
                self.available_data_lags_columns[col] = 0

    @property
    def variables_weight(self):

        weights = pd.Series(1., index=self.columns)

        if (self.prediction_variables_weight is None) or \
                (not len(self.ignore_prediction_columns)):
            return weights

        tot_weight = len(self.columns)
        prediction_weight = self.prediction_variables_weight * tot_weight
        predictive_weight = tot_weight - prediction_weight

        weights[self.prediction_columns] = np.sqrt(
            prediction_weight / len(self.prediction_columns))

        weights[self.ignore_prediction_columns] = np.sqrt(
            predictive_weight / len(self.ignore_prediction_columns))

        assert np.isclose((weights**2).sum(), tot_weight)
        assert np.isclose((weights[self.prediction_columns]**2).sum(),
                          (weights[self.ignore_prediction_columns]**2).sum())

        return weights

    @property
    def prediction_columns(self):
        return set(self.columns).difference(self.ignore_prediction_columns)

    @property
    def lag(self):
        return self.past_lag + self.future_lag

    @property
    def factors(self):
        return pd.DataFrame(self.V.todense()[::self.lag, ::self.lag],
                            index=self.columns).T

    @property
    def normalized_factors(self):
        return (self.factors.T / np.sqrt((self.factors**2).mean(1))).T

    @property
    def factors_coverage(self):
        return np.sqrt((self.normalized_factors**2).mean())

    # def fit_train_test(self):
    #     logger.debug('Fitting model on train and test data.')
    #     self._fit_ranges(self.train)
    #     self._fit_baselines(self.data, traintest=True)
    #     self._fit_low_rank_plus_block_diagonal_AR(self.train, self.test)
    #     self.AR_RMSE, self.gradients = self.test_AR(self.test)
    #
    # def fit(self):
    #     logger.debug('Fitting model on whole data.')
    #     self._fit_ranges(self.data)
    #     self._fit_baselines(self.data, traintest=False)
    #     self._fit_low_rank_plus_block_diagonal_AR(self.data, None)

    @property
    def Sigma(self):
        return self.V @ self.S @ self.V.T + self.D_matrix

    @property
    def large_matrix_multiindex(self):
        return pd.MultiIndex.from_arrays(
            (np.repeat(self.columns, self.lag),
             np.concatenate(
             [np.arange(-self.past_lag + 1,
                        self.future_lag + 1)] * len(self.columns))))

    @property
    def Sigma_df(self):
        return pd.DataFrame(self.Sigma,
                            index=self.large_matrix_multiindex,
                            columns=self.large_matrix_multiindex)

    @property
    def single_variable_lag_covariances(self):
        return pd.DataFrame([el.flatten()
                             for el in self.block_lagged_covariances],
                            index=self.columns)

    # @property
    # def train(self):
    #     return self.data.iloc[:int(len(self.data) * self.train_test_split)]
    #
    # @property
    # def test(self):
    #     return self.data.iloc[
    #         int(len(self.data) * self.train_test_split):]

    def _fit_ranges(self, data):
        logger.info('Fitting ranges.')
        self._min = data.min()
        self._max = data.max()

    def _clip_prediction(self, prediction: pd.DataFrame) -> pd.DataFrame:
        return prediction.clip(self._min, self._max, axis=1)

    def _fit_baselines(self,
                       data: pd.DataFrame):

        # TODO pass W and gamma
        self.baseline_results_columns, self.baseline_params_columns = \
            fit_many_baselines(
                data,
                baseline_params_dict=self.baseline_params_columns,
                train_test_ratio=self.train_test_split,
                gamma=1E-8, W=2,
                parallel=self.parallel_fit)

    # def _fit_baselines(self,
    #                    data: pd.DataFrame,
    #                    traintest: bool):
    #
    #     logger.info('Fitting baselines.')
    #
    #     if traintest and self.return_performance_statistics:
    #         #logger.debug('Computing baseline RMSE.')
    #         self.baseline_RMSE = pd.Series(index=self.columns)
    #
    #     # TODO parallelize
    #     for col in self.columns:
    #         logger.info('Fitting baseline on column %s.' % col)
    #
    #         col_data = data[col].dropna()
    #
    #         if traintest:
    #             train = col_data.iloc[
    #                 :int(len(col_data) * self.train_test_split)]
    #             test = col_data.iloc[
    #                 int(len(col_data) * self.train_test_split):]
    #
    #         else:
    #             train, test = col_data, None
    #
    #         logger.info(f'\ttraining on {len(train)} points from ' +
    #                     f'{train.index[0]} to {train.index[-1]}')
    #
    #         if test is not None:
    #             logger.info(f'\ttesting on {len(test)} points from ' +
    #                         f'{test.index[0]} to {test.index[-1]}')
    #
    #         if not self.baseline_params_columns[col]['non_par_baseline']:
    #
    #             self.baseline_results_columns[col]['std'], \
    #                 self.baseline_params_columns[col]['daily_harmonics'], \
    #                 self.baseline_params_columns[col]['weekly_harmonics'], \
    #                 self.baseline_params_columns[col]['annual_harmonics'], \
    #                 self.baseline_params_columns[col]['trend'],\
    #        self.baseline_results_columns[col]['baseline_fit_result'], \
    #                 optimal_rmse = fit_baseline(
    #                 train, test,
    #                 **self.baseline_params_columns[col])
    #         else:
    #
    #             self.baseline_results_columns[col]['std'], \
    #                 self.baseline_params_columns[col]['lambdas'],\
    #                 self.baseline_results_columns[col]['theta'], \
    #                 optimal_rmse = non_par_fit_baseline(
    #                 train, test,
    #                 **self.baseline_params_columns[col])
    #
    #         if traintest and self.return_performance_statistics:
    #             logger.info(f'baseline prediction RMSE: {optimal_rmse}')
    #             logger.info(f'test data std.dev.: {test.std()}')
    #             self.baseline_RMSE[col] = optimal_rmse

    def _build_matrices(self):

        self.V, self.S, self.S_inv, \
            self.D_blocks, self.D_matrix = build_matrices(
                self.s_times_v, self.S_lagged_covariances,
                self.block_lagged_covariances)

    def _fit_low_rank_plus_block_diagonal_AR(
            self, data: pd.DataFrame):
        # train: pd.DataFrame,
        # test: Optional[pd.DataFrame] = None):

        logger.debug('Fitting low-rank plus block diagonal.')

        # self.Sigma, self.past_lag, self.rank, \
        #     predicted_residuals_at_lags

        self.past_lag, self.rank, self.quadratic_regularization, \
            self.s_times_v, self.S_lagged_covariances, \
            self.block_lagged_covariances = \
            fit_low_rank_plus_block_diagonal_AR(
                self._residual(data),
                rank=self.rank,
                quadratic_regularization=self.quadratic_regularization,
                future_lag=self.future_lag,
                past_lag=self.past_lag,
                available_data_lags_columns=self.available_data_lags_columns,
                ignore_prediction_columns=self.ignore_prediction_columns,
                full_covariance=self.full_covariance,
                full_covariance_blocks=self.full_covariance_blocks,
                noise_correction=self.noise_correction,
                variables_weight=self.variables_weight,
                use_svd_fit=self.use_svd_fit,
                train_test_ratio=self.train_test_split,
                alpha=np.cbrt(10),
                W=2)

        # (self._residual(train),
        #                                     self._residual(
        #     test) if test is not None else None,
        #     self.future_lag,
        #     self.past_lag,
        #     self.rank,
        #     self.available_data_lags_columns,
        #     self.ignore_prediction_columns,
        #     self.full_covariance,
        #     self.full_covariance_blocks,
        #     self.quadratic_regularization,
        #     self.noise_correction,
        #     self.variables_weight,
        #     use_svd_fit=self.use_svd_fit)

        self._build_matrices()

    def test_AR(self, test):

        test = test[self.columns]

        _ = rmse_AR(self.V, self.S, self.S_inv,
                    self.D_blocks,
                    self.D_matrix,
                    self.past_lag, self.future_lag,
                    self._residual(test),
                    self.available_data_lags_columns,
                    self.ignore_prediction_columns,
                    self.quadratic_regularization,
                    do_gradients=self.compute_gradients)

        if self.compute_gradients:
            AR_RMSE, gradients = _
        else:
            AR_RMSE = _

        for col in AR_RMSE.columns:
            AR_RMSE[col] *= self.baseline_results_columns[col]['std']

        return (AR_RMSE, gradients) if self.compute_gradients else AR_RMSE

    def anomaly_score(self, test):
        test = test[self.columns]
        return anomaly_score(self.V, self.S, self.S_inv,
                             self.D_blocks,
                             self.D_matrix,
                             self.past_lag, self.future_lag,
                             self._residual(test))

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
        # TODO merge with next function
        return data.apply(self._column_residual)

    def normalized_residual(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._residual(data)

    def _column_residual(self, column: pd.Series) -> pd.Series:
        return (non_par_data_to_residual if
                (('non_par_baseline' in
                  self.baseline_params_columns[column.name]) and
                 self.baseline_params_columns[column.name]['non_par_baseline'])
                else data_to_normalized_residual)(
            column,
            **self.baseline_params_columns[column.name],
            **self.baseline_results_columns[column.name])

    def _invert_residual(self, data: pd.DataFrame) -> pd.DataFrame:
        return self._clip_prediction(
            data.apply(self._column_invert_residual))

    def _column_invert_residual(self, column: pd.Series) -> pd.Series:
        return (non_par_residual_to_data if
                (('non_par_baseline' in
                  self.baseline_params_columns[column.name]) and
                 self.baseline_params_columns[column.name]['non_par_baseline'])
                else normalized_residual_to_data)(
            column,
            **self.baseline_results_columns[column.name],
            **self.baseline_params_columns[column.name])

    # def predict_NEW(self, data: pd.DataFrame,
    #                 prediction_time,
    #                 return_sigmas=False):
    #     check_multidimensional_time_series(data,
    # self.frequency, self.columns)
    #     # reorder columns
    #     data = data[self.columns]
    #
    #     pass

    def predict_many(self,
                     data: pd.DataFrame):

        check_multidimensional_time_series(data, self.frequency, self.columns)

        data = data[self.columns]
        residuals = self._residual(data)
        residuals_flattened = make_sliced_flattened_matrix(
            residuals.values, self.lag)

        ignore_prediction_col_mask = residuals.columns.isin(
            self.ignore_prediction_columns)

        prediction_mask, unknown_mask = make_prediction_mask(
            self.available_data_lags_columns, ignore_prediction_col_mask,
            residuals.columns, self.past_lag, self.future_lag)
        real_values = pd.DataFrame(residuals_flattened, copy=True)
        residuals_flattened[:, unknown_mask] = np.nan
        # gradients, total_num_predictions_made = \
        total_num_predictions_made = \
            guess_matrix(
                residuals_flattened, self.V, self.S,
                self.S_inv, self.D_blocks, self.D_matrix,
                quadratic_regularization=self.quadratic_regularization,
                prediction_mask=prediction_mask,
                real_values=real_values,
                do_gradients=False)

        res = pd.DataFrame(data=residuals_flattened,
                           columns=self.large_matrix_multiindex.reorder_levels(
                               (1, 0)),
                           index=data.index[self.past_lag:len(data)
                                            - self.future_lag + 1])

        return {(fut_lag + 1): self._invert_residual(res.loc[:, fut_lag + 1])
                for fut_lag in range(self.future_lag)}

    def predict(self,
                data: pd.DataFrame,
                prediction_time:
                Optional[pd.Timestamp] = None,
                return_sigmas=False) -> pd.DataFrame:
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

        # res = dense_schur(self.Sigma, known_mask=known_mask,
        #                   known_vector=residual_vectorized[known_mask],
        #                   return_conditional_covariance=return_sigmas,
        #            quadratic_regularization=quadratic_regularization if
        #                   quadratic_regularization is not None else
        #                   self.quadratic_regularization)

        res = symm_low_rank_plus_block_diag_schur(
            self.V,
            self.S,
            self.S_inv,
            self.D_blocks,
            self.D_blocks_indexes,
            self.D_matrix,
            known_mask=known_mask,
            known_matrix=np.matrix(
                residual_vectorized[known_mask]),
            prediction_mask=~known_mask,
            real_result=None,
            return_conditional_covariance=return_sigmas,
            quadratic_regularization=self.quadratic_regularization,
            prediction_cache=self.prediction_cache)

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

    # def robust_predict(self, data: pd.DataFrame,
    #                    prediction_time:
    #                    Optional[pd.Timestamp]=None,
    #                    return_sigmas=False):

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

    def save_model(self):
        # TODO test model is pickable!
        model_dict = {'frequency': self.frequency,
                      'past_lag': self.past_lag,
                      'future_lag': self.future_lag,
                      'columns': self.columns,
                      'baseline_params_columns': self.baseline_params_columns,
                      'baseline_results_columns': self.baseline_results_columns,
                      '_min': self._min,
                      '_max': self._max,
                      'available_data_lags_columns': self.available_data_lags_columns,
                      'rank': self.rank,
                      'quadratic_regularization': self.quadratic_regularization,
                      's_times_v': self.s_times_v,
                      'S_lagged_covariances': self.S_lagged_covariances,
                      'block_lagged_covariances': self.block_lagged_covariances,
                      'ignore_prediction_columns': self.ignore_prediction_columns}

        return gzip.compress(pickle.dumps(model_dict,
                                          protocol=pickle.HIGHEST_PROTOCOL))


def load_model(model_compressed):
    model = tsar.__new__(tsar)
    model.__dict__.update(pickle.loads(gzip.decompress(model_compressed)))
    model._build_matrices()
    return model
