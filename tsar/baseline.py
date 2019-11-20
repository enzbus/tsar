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

import logging
from multiprocessing import Pool


import numpy as np
import pandas as pd
import numba as nb

from .greedy_grid_search import greedy_grid_search

logger = logging.getLogger(__name__)

MAX_DAILY_HARMONICS = 24*4

POOL_MAPPER = Pool().map


@nb.jit(nopython=True)
def featurize_index_for_baseline(seconds: np.ndarray,
                                 periods: np.ndarray,
                                 K_trend: bool = False) -> np.ndarray:
    X = np.zeros((len(seconds), 1 + 2 * len(periods) + K_trend))
    for i, period in enumerate(periods):  # in seconds
        X[:, 2 * i] = np.sin(2 * np.pi * seconds / period)
        X[:, 2 * i + 1] = np.cos(2 * np.pi * seconds / period)
    X[:, -1 - K_trend] = np.ones(len(seconds))
    if K_trend:
        X[:, -1] = seconds / 1E9  # roughly around 1
    return X


@nb.jit(nopython=True)
def fit_seasonal_baseline(X, y, gamma=1E-8):
    num_coeff = X.shape[1]
    regularizer = np.eye(num_coeff)
    if num_coeff % 2:
        # if odd, then beta_0 is last element
        regularizer[num_coeff-1, num_coeff-1] = 0.
    else:
        # otherwise it is second to last element
        regularizer[num_coeff-2, num_coeff-2] = 0.
    return np.linalg.solve(X.T @ X + gamma * regularizer,
                           X.T @ y)


@nb.jit(nopython=True)
def predict_with_baseline(X, parameters):
    return X @ parameters


def index_to_seconds(index):
    return np.array(index.astype(np.int64) / 1E9)


@nb.jit(nopython=True)
def make_periods(K_day,
                 K_week,
                 K_year):
    periods = np.empty(K_day + K_week + K_year)
    base_periods = (24 * 3600.,  # daily
                    24 * 7 * 3600,  # weekly
                    8766 * 3600)  # annual
    i = 0
    for j in range(K_day):
        periods[i] = base_periods[0] / (j + 1)
        i += 1
    for j in range(K_week):
        periods[i] = base_periods[1] / (j + 1)
        i += 1
    for j in range(K_year):
        periods[i] = base_periods[2] / (j + 1)
        i += 1
    assert i == len(periods)

    return periods


def compute_baseline(index,
                     K_day,
                     K_week,
                     K_year,
                     K_trend,
                     baseline_fit_result):

    periods = make_periods(K_day,
                           K_week,
                           K_year)

    X = featurize_index_for_baseline(index_to_seconds(index),
                                     periods, K_trend=K_trend)
    return predict_with_baseline(X, baseline_fit_result)


def data_to_normalized_residual(
        data: pd.Series,
        std: float,
        K_day: int,
        K_week: int,
        K_year: int,
        K_trend: bool,
        baseline_fit_result: np.ndarray, **kwargs) -> pd.Series:
    return (data - compute_baseline(data.index,
                                    K_day,
                                    K_week,
                                    K_year,
                                    K_trend,
                                    baseline_fit_result)) / std


def normalized_residual_to_data(
        normalized_residual: pd.Series,
        std: float,
        K_day: int,
        K_week: int,
        K_year: int,
        K_trend: bool,
        baseline_fit_result: np.ndarray, **kwargs) -> pd.Series:
    return normalized_residual * std + compute_baseline(
        normalized_residual.index,
        K_day,
        K_week,
        K_year,
        K_trend,
        baseline_fit_result)


def _fit_baseline(data, K_day, K_week, K_year, K_trend, gamma=1E-8):
    periods = make_periods(K_day, K_week, K_year)
    X = featurize_index_for_baseline(index_to_seconds(data.index),
                                     periods, K_trend=K_trend)

    return fit_seasonal_baseline(X, data.values, gamma=gamma)


def return_null_baseline(colname):
    logger.warning(
        f'Column {colname} has not enough data, fitting zero baseline.')
    return 0, 0, 0, False, np.array([0.])


def fit_scalar_baseline(data,
                        K_day=None,
                        K_week=None,
                        K_year=None,
                        K_trend=None,
                        train_test_ratio=2/3,
                        gamma=1E-8, W=2):

    assert type(data) is pd.Series

    data = data.dropna()
    logger.info(f"Fitting baseline for {data.name} with {len(data)} values.")

    if not len(data):
        return_null_baseline(data.name)

    if (K_day is None) or (K_week is None) or (K_year is None) or \
            (K_trend is None):
        train = data.iloc[:int(len(data) * train_test_ratio)]
        test = data.iloc[int(len(data) * train_test_ratio):]

        logger.info(f"Tuning hyper-parameters with {len(train)} train and "
                    f"{len(test)} test points")

        if not len(train):
            return_null_baseline(train.name)
        if not len(test):
            return_null_baseline(test.name)

        def test_RMSE(
                K_day,
                K_week,
                K_year,
                K_trend):
            baseline_fit_result = _fit_baseline(train,
                                                K_day,
                                                K_week,
                                                K_year,
                                                K_trend,
                                                gamma=gamma)

            return np.sqrt(
                ((test - compute_baseline(
                    test.index,
                    K_day,
                    K_week,
                    K_year,
                    K_trend,
                    baseline_fit_result))**2).mean())

        K_day_range = np.arange(MAX_DAILY_HARMONICS) if K_day \
            is None else [K_day]
        K_week_range = np.arange(6) if K_week \
            is None else [K_week]
        K_year_range = np.arange(50) if K_year \
            is None else [K_year]
        K_trend_range = [False, True] if K_trend is None else [K_trend]

        _, (K_day, K_week, K_year, K_trend) = greedy_grid_search(
            test_RMSE, [K_day_range, K_week_range,
                        K_year_range, K_trend_range],
            num_steps=W)

    logger.info(f"Fitting baseline with K_trend = {K_trend}, "
                f"K_day = {K_day}, "
                f"K_week = {K_week}, "
                f"K_year = {K_year}")

    baseline_fit_result = _fit_baseline(data,
                                        K_day=K_day,
                                        K_week=K_week,
                                        K_year=K_year,
                                        K_trend=K_trend)

    residual = data_to_normalized_residual(
        data,
        std=1.,
        K_day=K_day,
        K_week=K_week,
        K_year=K_year,
        K_trend=K_trend,
        baseline_fit_result=baseline_fit_result)

    assert np.isclose(np.nanmean(residual), 0.)
    std = np.sqrt(np.nanmean(residual**2))
    if np.isnan(std) or std == 0.:
        std = 1.

    return K_day, K_week, K_year, K_trend, baseline_fit_result, std


def fit_scalar_wrapper(args):

    data_series, params, train_test_ratio, gamma, W = args

    params = {}
    K_day, K_week, K_year, K_trend, baseline_fit_result, std =\
        fit_scalar_baseline(
            data_series,
            **params,
            train_test_ratio=train_test_ratio,
            gamma=gamma, W=W)
    params['K_day'] = K_day
    params['K_week'] = K_week
    params['K_year'] = K_year
    params['K_trend'] = K_trend
    return params, {'baseline_fit_result': baseline_fit_result, 'std': std}


def fit_many_baselines(data,
                       baseline_params_dict,
                       train_test_ratio=2/3,
                       gamma=1E-8, W=2,
                       parallel=True):

    all_baseline_fit_results = {}

    arguments = [(data[col].dropna(),
                  baseline_params_dict[col],
                  train_test_ratio, gamma, W)
                 for col in data.columns]

    if parallel:
        with Pool() as p:
            results = list(p.map(fit_scalar_wrapper, arguments))
    else:
        results = list(map(fit_scalar_wrapper, arguments))

    # results = list(
    #     (map if not parallel else Pool().map)
    #     (fit_scalar_wrapper, arguments))

    for i, col in enumerate(data.columns):
        baseline_params_dict[col], all_baseline_fit_results[col] = \
            results[i]

    return all_baseline_fit_results, baseline_params_dict


# def fit_many_baselines_parallel(data,
#                                 baseline_params_dict,
#                                 train_test_ratio=2/3,
#                                 gamma=1E-8, W=2):
#
#     p = Pool()
#     all_baseline_fit_results = {}
#
#     p.map(f, data.columns)
#
#
# if __name__ == '__main__':
#     p = Pool()
#     print(p.map(f, [1, 2, 3]))


# RANGES = {'K_day': np.arange(MAX_DAILY_HARMONICS),
#           'K_week': np.arange(6),
#           'K_year': np.arange(50),
#           'K_trend': [False, True]}


# def fit_with_greedy_grid_search(range_dict, func, W, **kwargs):
#     range_dict = dict(range_dict)
#     for k in kwargs:
#         if kwargs[k] is not None:
#             range_dict[k] = [kwargs[k]]

    # todo
    # (1) accept whole data and train/test ratio,
    # split internally after dropping nans
    # only if there are hyperparameters to fit otherwise
    # go with full fit!
    # (2) use memoization for fit function
    # so it is cheap to refit
    # (3) factor ggs wrapper into new function using
    # **kwargs and put it in ggs module

# def fit_baseline():
#     pass
# def fit_baseline(train, test,
#                  K_day=None,
#                  K_week=None,
#                  K_year=None,
#                  K_trend=None,
#                  gamma=1E-8,
#                  **kwargs):
#
#     train = train.dropna()
#
#     if not len(train):
#         logger.warning(
#       f'Train column {train.name} is all NaNs, returning null baseline.')
#         return 1., 0, 0, 0, False, np.array([0.]), \
#             np.sqrt((test.dropna()**2).mean()) if test is not None else None
#
#     if test is not None:
#         test = test.dropna()
#
#         logger.debug('Autotuning baseline on %d train and %d test points' %
#                      (len(train), len(test)))
#
#         K_day_range = np.arange(24) if K_day \
#             is None else [K_day]
#         K_week_range = np.arange(6) if K_week \
#             is None else [K_week]
#         K_year_range = np.arange(50) if K_year \
#             is None else [K_year]
#         K_trend_range = [False, True] if K_trend is None else [K_trend]
#
#         def test_RMSE(
#                 K_day,
#                 K_week,
#                 K_year,
#                 K_trend):
#             baseline_fit_result = _fit_baseline(train,
#                                                 K_day,
#                                                 K_week,
#                                                 K_year,
#                                                 K_trend,
#                                                 gamma)
#
#             return np.sqrt(
#                 ((test - compute_baseline(
#                     test.index,
#                     K_day,
#                     K_week,
#                     K_year,
#                     K_trend,
#                     baseline_fit_result))**2).mean())
#
#         optimal_rmse, (K_day,
#                        K_week,
#                        K_year,
#                        K_trend) = greedy_grid_search(test_RMSE,
#                                                      [K_day_range,
#                                                       K_week_range,
#                                                       K_year_range,
#                                                       K_trend_range],
#                                                      num_steps=2)
#
#     baseline_fit_result = _fit_baseline(train, K_day,
#                                         K_week,
#                                         K_year,
#                                         K_trend)
#
#     std = np.std(data_to_residual(train,
#                                   std=1.,
#                                   K_day=K_day,
#                                   K_week=K_week,
#                                   K_year=K_year,
#                                   K_trend=K_trend,
#                                   baseline_fit_result=baseline_fit_result))
#
#     return std, K_day, K_week, \
#         K_year, K_trend, baseline_fit_result, \
#         optimal_rmse if test is not None else None
