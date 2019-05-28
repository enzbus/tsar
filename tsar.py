import numpy as np
import pandas as pd
import numba as nb


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


class Model:

    def __init__(
            self,
            daily=True,
            weekly=False,
            annual=True,
            harmonics=4,
            M=0,
            T=0):
        self.daily = daily
        self.weekly = weekly
        self.annual = annual
        self.harmonics = harmonics
        self.M = M
        self.T = T
        self.periods = np.array(make_periods(self.daily,
                                             self.weekly,
                                             self.annual,
                                             self.harmonics))
        print(self.periods)

    def _train_baseline(self, train):

        Xtr = featurize_index_for_baseline(index_to_seconds(train.index),
                                           self.periods)
        ytr = train.values
        baseline_params = fit_seasonal_baseline(Xtr, ytr)
        print(baseline_params)
        self.baseline_params = baseline_params

    def _train_matrix_ar(self, train):
        train_residual = train - self._predict_baseline(train.index)
        Xtr, ytr = featurize_residual(train_residual, self.M, self.T)
        self.residuals_params = fit_residual(Xtr, ytr)

    def train(self, train):
        self._train_baseline(train)
        self._train_matrix_ar(train)

    def _predict_baseline(self, index):
        Xte = featurize_index_for_baseline(index_to_seconds(index),
                                           self.periods)
        return pd.Series(data=predict_with_baseline(Xte, self.baseline_params),
                         index=index)
