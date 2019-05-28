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
        self._baseline = self._predict_baseline(data.index)
        self._baseline.name = data.name

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
                         index=index)


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
        self._residuals_stds = self.residuals.std()
        self._normalized_residuals = self.residuals / self._residuals_stds
        self._fit_AR()

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
                                 return_sigma=False):

        # https://en.wikipedia.org/wiki/Schur_complement
        # (Applications_to_probability_theory_and_statistics)

        null_mask = concatenated.isnull().values
        y = concatenated[~null_mask].values

        A = self.Sigma[null_mask].T[null_mask]
        B = self.Sigma[null_mask].T[~null_mask].T
        C = self.Sigma[~null_mask].T[~null_mask]

        expected_x = B @ np.linalg.solve(C, y)
        concatenated[null_mask] = expected_x

        if return_sigma:
            Sigma_x = A - B @ np.linalg.inv(C) @ B.T
            return concatenated, null_mask, Sigma_x

        return concatenated

    def _predict_normalized_residual_AR(self, chunk):
        #chunk = model._normalized_residuals.iloc[-10:]
        assert len(chunk) == self.lag
        chunk_index = chunk.index

        concatenated = pd.concat(
            [
                chunk.iloc[i]
                for i in range(self.lag)
            ])

        filled = self._predict_concatenated_AR(concatenated)
        chunk_filled = pd.concat(
            [filled.iloc[len(self._columns) * i:len(self._columns) * (i + 1)]
                for i in range(self.lag)], axis=1).T
        chunk_filled.index = chunk_index
        return chunk_filled

    @property
    def baseline(self):
        return pd.concat(
            [self._baselines[col]._baseline
             for col in self._columns], axis=1)

    @property
    def residuals(self):
        return self.data - self.baseline

    def _train_matrix_ar(self, train):
        train_residual = train - self._predict_baseline(train.index)
        Xtr, ytr = featurize_residual(train_residual, self.M, self.T)
        self.residuals_params = fit_residual(Xtr, ytr)

    # def train(self, train):
    #     if not isinstance(train, pd.DataFrame):
    #         raise ValueError(
    #             'Train data must be a pandas DataFrame')
    #     if not isinstance(dati.index, pd.DatetimeIndex):
    #         raise ValueError(
    #             'Train data must be indexed by a pandas DatetimeIndex.')
    #     if train.index.freq is None:
    #         raise ValueError('Train data index must have a frequency. ' +
    #                          'Try using the pandas.DataFrame.asfreq method.')
    #     self.frequency = train.index.freq
    #     if train.isnull().sum().sum():
    #         raise ValueError('Train data must not have NaNs')
    #     self._train_baseline(train)
    #     self._train_matrix_ar(train)

    # def _predict_baseline(self, index):
    #     Xte = featurize_index_for_baseline(index_to_seconds(index),
    #                                        self.periods)
    #     return pd.Series(data=predict_with_baseline(Xte, self.baseline_params),
    #                      index=index)

    # def predict(self, data):
