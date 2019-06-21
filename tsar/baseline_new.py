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

from .greedy_grid_search_new import greedy_grid_search


@nb.jit(nopython=True)
def featurize_index_for_baseline(seconds, periods, trend=False):
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


def data_to_residual(data: pd.Series, params: dict) ->: pd.Series:
    return (data - compute_baseline(data.index,
                                    params['daily_harmonics'],
                                    params['weekly_harmonics'],
                                    params['annual_harmonics'],
                                    params['trend'],
                                    params['baseline_fit_result'])) / params['std']


def residual_to_data(residual: pd.Series, params: dict) ->: pd.Series:
    return data * params['std'] + compute_baseline(
        data.index,
        params['daily_harmonics'],
        params['weekly_harmonics'],
        params['annual_harmonics'],
        params['trend'],
        params['baseline_fit_result'])


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


def fit_baseline(train, test=None, params):

    train = train.dropna()

    if test is not None:
        test = test.dropna()

        logger.debug('Autotuning baseline on %d train and %d test points' %
                     (len(train), len(test)))

        daily_harmonics_range = np.arange(24) if 'daily_harmonics' \
            not in params else [params['daily_harmonics']]
        weekly_harmonics_range = np.arange(6) if 'weekly_harmonics'
            not in params else [params['weekly_harmonics']]
        annual_harmonics_range = np.arange(50) if 'annual_harmonics'
            not in params else [params['annual_harmonics']]
        trend = [False, True] if 'trend' not in params else [params['trend']]

        def test_RMSE(
                daily_harmonics,
                weekly_harmonics,
                annual_harmonics,
                trend):
            baseline_fit_result = _fit_baseline(data,
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

        optimal_rmse, (params['daily_harmonics'],
                       params['weekly_harmonics'],
                       params['annual_harmonics'],
                       params['trend']) = greedy_grid_search(test_RMSE,
                                                             [daily_harmonics,
                                                              weekly_harmonics,
                                                              annual_harmonics,
                                                              trend],
                                                             num_steps=2)

    params['baseline_fit_result'] =\
        _fit_baseline(train, params['daily_harmonics'],
                      params['weekly_harmonics'],
                      params['annual_harmonics'],
                      params['trend'])

    params['std'] = 1.
    params['std'] = np.std(data_to_residual(train, params))

    if test is not None:
        return optimal_rmse
