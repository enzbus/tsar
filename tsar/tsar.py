"""
   Copyright 2019 Enzo Busseti

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import numpy as np
import pandas as pd
import numba as nb
import logging
logger = logging.getLogger(__name__)

__all__ = ['Model']


@nb.jit(nopython=True)
def featurize_index_for_baseline(seconds, periods):
    X = np.zeros((len(seconds), 1 + 2 * len(periods)))
    for i, period in enumerate(periods):  # in seconds
        X[:, 2 * i] = np.sin(2 * np.pi * seconds / period)
        X[:, 2 * i + 1] = np.cos(2 * np.pi * seconds / period)
    X[:, -1] = np.ones(len(seconds))
    return X


@nb.jit(nopython=True)
def fit_seasonal_baseline(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


@nb.jit(nopython=True)
def predict_with_baseline(X, parameters):
    return X @ parameters


def index_to_seconds(index):
    return np.array(index.astype(np.int64) / 1E9)


@nb.jit(nopython=True)
def make_periods(daily, weekly, annual, harmonics):
    print(daily, weekly, annual)
    PERIODS = np.empty(harmonics * (annual + daily + weekly))
    base_periods = (24 * 3600.,  # daily
                    24 * 7 * 3600,  # weekly
                    8766 * 3600)  # annual
    i = 0
    if daily:
        PERIODS[i * harmonics : (i + 1) * harmonics] = \
            base_periods[0] / np.arange(1, harmonics + 1)
        i += 1
    if weekly:
        PERIODS[i * harmonics : (i + 1) * harmonics] = \
            base_periods[1] / np.arange(1, harmonics + 1)
        i += 1
    if annual:
        PERIODS[i * harmonics : (i + 1) * harmonics] = \
            base_periods[2] / np.arange(1, harmonics + 1)
        i += 1

    return PERIODS


@nb.jit()
def featurize_residual(obs, M, L):
    X = np.zeros((len(obs) - M - L + 1, M))
    for i in range(M):
        X[:, i] = obs[M - i - 1:-L - i]

    y = np.zeros((len(obs) - M - L + 1, L))

    for i in range(L):
        y[:, i] = obs[M + i:len(obs) + 1 - L + i]

    return X, y


def fit_residual(X, y):
    M, L = X.shape[1], y.shape[1]
    pinv = np.linalg.inv(X.T @ X) @ X.T
    params = np.zeros((M, L))
    params = pinv @ y
    return params


class HarmonicBaseline:

    def __init__(self, data,
                 daily=True,
                 weekly=False,
                 annual=True,
                 harmonics=4):
        if not isinstance(data, pd.Series):
            raise ValueError(
                'Train data must be a pandas Series')
        self.daily = daily
        self.weekly = weekly
        self.annual = annual
        self.harmonics = harmonics
        self.periods = np.array(make_periods(self.daily,
                                             self.weekly,
                                             self.annual,
                                             self.harmonics))
        print(self.periods)
        self._train_baseline(data.dropna())
        self.name = data.name
        self._baseline = self._predict_baseline(data.index)
        #self._baseline.name = data.name

    def _train_baseline(self, train):

        Xtr = featurize_index_for_baseline(index_to_seconds(train.index),
                                           self.periods)
        ytr = train.values
        baseline_params = fit_seasonal_baseline(Xtr, ytr)
        print(baseline_params)
        self.baseline_params = baseline_params

    def _predict_baseline(self, index):
        Xte = featurize_index_for_baseline(index_to_seconds(index),
                                           self.periods)
        return pd.Series(data=predict_with_baseline(Xte, self.baseline_params),
                         index=index, name=self.name)


class Model:

    def __init__(
            self,
            data,
            baseline_per_column_options={},
            lag=10):

        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                'Train data must be a pandas DataFrame')
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                'Train data must be indexed by a pandas DatetimeIndex.')
        if data.index.freq is None:
            raise ValueError('Train data index must have a frequency. ' +
                             'Try using the pandas.DataFrame.asfreq method.')
        self.frequency = data.index.freq
        self._columns = data.columns
        self.lag = lag
        self.data = data
        self.baseline_per_column_options =\
            baseline_per_column_options
        self._fit_baselines()
        self._fit_ranges()
        self._residuals_stds = self.residuals.std()
        self._normalized_residuals = self.residuals / self._residuals_stds
        self._fit_AR()

    def _fit_ranges(self):
        self._min = self.data.min()
        self._max = self.data.max()

    def _clip_prediction(self, prediction):
        return prediction.clip(self._min, self._max, axis=1)

    @property
    def baseline(self):
        return pd.concat(
            [self._baselines[col]._baseline
             for col in self._columns], axis=1)

    @property
    def residuals(self):
        return self.data - self.baseline

    def predict(self, last_data, number_scenarios=0):
        print('last_data', last_data.shape)
        print(last_data.index)
        if len(last_data) > self.lag:
            raise ValueError('Only provide the last data.')
        if not last_data.index.freq == self.frequency:
            raise ValueError('Provided data has wrong frequency.')
        len_chunk = len(last_data)
        for i in range(1, 1 + self.lag - len(last_data)):
            #print('i = ', i)
            t = last_data.index[len_chunk - 1] + i * self.frequency
            #print('adding row', t)
            last_data.loc[t] = np.nan
        print('last_data', last_data.shape)
        baseline = self.predict_baseline(last_data)
        print('baseline', baseline.shape)
        residuals = last_data - baseline
        normalized_residuals = residuals / self._residuals_stds
        normalized_residuals_list = self._predict_normalized_residual_AR(
            normalized_residuals, number_scenarios)
        all_results = []
        for normalized_residuals in normalized_residuals_list:
            residuals = normalized_residuals * self._residuals_stds
            all_results.append(
                self._clip_prediction(residuals + baseline))
            if not number_scenarios:
                return all_results[-1]
        return all_results

    def predict_baseline(self, data):
        return pd.concat(
            [self._baselines[col]._predict_baseline(data.index)
             for col in self._columns], axis=1)

    def _fit_baselines(self):
        self._baselines = {}
        for column in self._columns:
            if column in self.baseline_per_column_options:
                self._baselines[column] = HarmonicBaseline(
                    self.data[column],
                    **self.baseline_per_column_options[column])
            else:
                self._baselines[column] = HarmonicBaseline(
                    self.data[column])

    def _fit_AR(self):
        print('computing lagged covariances')
        self.lagged_covariances = {}
        for i in range(self.lag):
            self.lagged_covariances[i] = \
                pd.concat((self._normalized_residuals,
                           self._normalized_residuals.shift(i)),
                          axis=1).corr().iloc[:len(self._columns),
                                              len(self._columns):]
        print('assembling covariance matrix')
        self.Sigma = pd.np.block(
            [[self.lagged_covariances[np.abs(i)].values
                for i in range(-j, self.lag - j)]
                for j in range(self.lag)]
        )

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

    def _predict_normalized_residual_AR(self, chunk,
                                        number_scenarios=0):
        #chunk = model._normalized_residuals.iloc[-10:]
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
