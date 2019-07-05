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

import numpy as np
import pandas as pd
import numba as nb
import logging
logger = logging.getLogger(__name__)

from .greedy_grid_search import greedy_grid_search


@nb.jit(nopython=True)
def featurize_index_for_baseline(seconds: np.ndarray,
                                 periods: np.ndarray,
                                 trend: bool =False) -> np.ndarray:
    X = np.zeros((len(seconds), 1 + 2 * len(periods) + trend))
    for i, period in enumerate(periods):  # in seconds
        X[:, 2 * i] = np.sin(2 * np.pi * seconds / period)
        X[:, 2 * i + 1] = np.cos(2 * np.pi * seconds / period)
    X[:, -1 - trend] = np.ones(len(seconds))
    if trend:
        X[:, -1] = seconds / 1E9  # roughly around 1
    return X


@nb.jit(nopython=True)
def fit_seasonal_baseline(X, y):
    return np.linalg.solve(X.T @ X + 1E-5 * np.eye(X.shape[1]),
                           X.T @ y)


@nb.jit(nopython=True)
def predict_with_baseline(X, parameters):
    return X @ parameters


def index_to_seconds(index):
    return np.array(index.astype(np.int64) / 1E9)


@nb.jit(nopython=True)
def make_periods(daily_harmonics,
                 weekly_harmonics,
                 annual_harmonics):
    periods = np.empty(daily_harmonics + weekly_harmonics + annual_harmonics)
    base_periods = (24 * 3600.,  # daily
                    24 * 7 * 3600,  # weekly
                    8766 * 3600)  # annual
    i = 0
    for j in range(daily_harmonics):
        periods[i] = base_periods[0] / (j + 1)
        i += 1
    for j in range(weekly_harmonics):
        periods[i] = base_periods[1] / (j + 1)
        i += 1
    for j in range(annual_harmonics):
        periods[i] = base_periods[2] / (j + 1)
        i += 1
    assert i == len(periods)

    return periods


def compute_baseline(index,
                     daily_harmonics,
                     weekly_harmonics,
                     annual_harmonics,
                     trend,
                     baseline_fit_result):

    periods = make_periods(daily_harmonics,
                           weekly_harmonics,
                           annual_harmonics)

    X = featurize_index_for_baseline(index_to_seconds(index),
                                     periods, trend=trend)
    return predict_with_baseline(X, baseline_fit_result)


def data_to_residual(data: pd.Series, std: float,
                     daily_harmonics: int,
                     weekly_harmonics: int,
                     annual_harmonics: int,
                     trend: bool,
                     baseline_fit_result: np.ndarray) -> pd.Series:
    return (data - compute_baseline(data.index,
                                    daily_harmonics,
                                    weekly_harmonics,
                                    annual_harmonics,
                                    trend,
                                    baseline_fit_result)) / std


def residual_to_data(residual: pd.Series,
                     std: float,
                     daily_harmonics: int,
                     weekly_harmonics: int,
                     annual_harmonics: int,
                     trend: bool,
                     baseline_fit_result: np.ndarray) -> pd.Series:
    return residual * std + compute_baseline(residual.index,
                                             daily_harmonics,
                                             weekly_harmonics,
                                             annual_harmonics,
                                             trend,
                                             baseline_fit_result)


def _fit_baseline(data,
                  daily_harmonics,
                  weekly_harmonics,
                  annual_harmonics,
                  trend):

    periods = make_periods(daily_harmonics,
                           weekly_harmonics,
                           annual_harmonics)

    X = featurize_index_for_baseline(index_to_seconds(data.index),
                                     periods, trend=trend)

    return fit_seasonal_baseline(X, data.values)


def fit_baseline(train, test,
                 daily_harmonics=None,
                 weekly_harmonics=None,
                 annual_harmonics=None,
                 trend=None):

    train = train.dropna()

    if not len(train):
        logger.warning(f'Train column {train.name} is all NaNs, returning null baseline.')
        return 1., 0, 0, 0, False, np.array([0.]), \
            np.sqrt((test.dropna()**2).mean()) if test is not None else None

    if test is not None:
        test = test.dropna()

        logger.debug('Autotuning baseline on %d train and %d test points' %
                     (len(train), len(test)))

        daily_harmonics_range = np.arange(24) if daily_harmonics \
            is None else [daily_harmonics]
        weekly_harmonics_range = np.arange(6) if weekly_harmonics \
            is None else [weekly_harmonics]
        annual_harmonics_range = np.arange(50) if annual_harmonics \
            is None else [annual_harmonics]
        trend_range = [False, True] if trend is None else [trend]

        def test_RMSE(
                daily_harmonics,
                weekly_harmonics,
                annual_harmonics,
                trend):
            baseline_fit_result = _fit_baseline(train,
                                                daily_harmonics,
                                                weekly_harmonics,
                                                annual_harmonics,
                                                trend)

            return np.sqrt(
                ((test - compute_baseline(
                    test.index,
                    daily_harmonics,
                    weekly_harmonics,
                    annual_harmonics,
                    trend,
                    baseline_fit_result))**2).mean())

        optimal_rmse, (daily_harmonics,
                       weekly_harmonics,
                       annual_harmonics,
                       trend) = greedy_grid_search(test_RMSE,
                                                   [daily_harmonics_range,
                                                    weekly_harmonics_range,
                                                    annual_harmonics_range,
                                                    trend_range],
                                                   num_steps=2)

    baseline_fit_result = _fit_baseline(train, daily_harmonics,
                                        weekly_harmonics,
                                        annual_harmonics,
                                        trend)

    std = np.std(data_to_residual(train,
                                  std=1.,
                                  daily_harmonics=daily_harmonics,
                                  weekly_harmonics=weekly_harmonics,
                                  annual_harmonics=annual_harmonics,
                                  trend=trend,
                                  baseline_fit_result=baseline_fit_result))

    return std, daily_harmonics, weekly_harmonics, \
        annual_harmonics, trend, baseline_fit_result, \
        optimal_rmse if test is not None else None
