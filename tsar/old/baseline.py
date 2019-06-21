"""
Copyright (C) Enzo Busseti 2019.

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

from .gaussianize import *
from .utils import check_series
from .greedy_grid_search import greedy_grid_search


__all__ = ['HarmonicBaseline', 'baseline_autotune']  # , 'AutotunedBaseline']


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
    # print(daily_harmonics, weekly_harmonics, annual_harmonics)
    PERIODS = np.empty(daily_harmonics + weekly_harmonics + annual_harmonics)
    base_periods = (24 * 3600.,  # daily
                    24 * 7 * 3600,  # weekly
                    8766 * 3600)  # annual
    i = 0
    for j in range(daily_harmonics):
        PERIODS[i] = base_periods[0] / (j + 1)
        i += 1
    for j in range(weekly_harmonics):
        PERIODS[i] = base_periods[1] / (j + 1)
        i += 1
    for j in range(annual_harmonics):
        PERIODS[i] = base_periods[2] / (j + 1)
        i += 1
    assert i == len(PERIODS)

    return PERIODS


class HarmonicBaseline:

    def __init__(self,
                 train,
                 daily_harmonics=4,
                 weekly_harmonics=0,
                 annual_harmonics=4,
                 trend=False,
                 pre_gaussianize=False):  # ,
                 # post_gaussianize=False):
        check_series(train)
        self.train = train

        self.daily_harmonics = daily_harmonics
        self.weekly_harmonics = weekly_harmonics
        self.annual_harmonics = annual_harmonics
        self.trend = trend
        self.pre_gaussianize = pre_gaussianize
        #self.post_gaussianize = post_gaussianize

        self.name = train.name
        self.pre_gaussianization_params = None
        self.train_index_seconds = index_to_seconds(self.train.index)

        self._fit_baseline()
        # self._prepare_residuals()

        # self._baseline = self._predict_baseline(train.index)

    def _prepare_residuals(self):

        # if self.post_gaussianize:
        #     self.post_gaussianization_params = \
        #         compute_gaussian_interpolator(
        #             self._residuals(self.train))

        self.rmse = 1.
        data_std = np.sqrt((self.residuals(self.train)**2).mean())
        if data_std > 0:
            self.rmse = data_std
            #assert np.isclose(self.residuals(self.train).std(), 1.)
        # assert np.isclose(self.residuals(self.train).mean(), 0.)

    def _make_periods(self):
        self.periods = make_periods(self.daily_harmonics,
                                    self.weekly_harmonics,
                                    self.annual_harmonics)

    def _fit_baseline(self):

        self._make_periods()

        if self.pre_gaussianize:
            if self.pre_gaussianization_params is None:
                self.pre_gaussianization_params = \
                    compute_gaussian_interpolator(self.train)
                self.gaussianized_train = gaussianize(
                    self.train, *self.pre_gaussianization_params)
            self._train_baseline(self.gaussianized_train.values)

        else:
            self._train_baseline(self.train.values)

        self.rmse = 1.
        self._prepare_residuals()

    # def residuals(self, data):
    #     return (gaussianize(self._residuals(data),
    #                         *self.post_gaussianization_params)
    # if self.post_gaussianize else self._residuals(data)) / self.rmse

    def residuals(self, data):
        check_series(data)

        my_data = gaussianize(data,
                              *self.pre_gaussianization_params) \
            if self.pre_gaussianize else data

        return (my_data - self._predict_baseline(data.index)) / self.rmse

    def invert_residuals(self, data):
        check_series(data)

        plus_baseline = data * self.rmse + self._predict_baseline(data.index)

        return invert_gaussianize(plus_baseline,
                                  *self.pre_gaussianization_params) \
            if self.pre_gaussianize else plus_baseline

    # def invert_residuals(self, data):
    #     return self._invert_residuals(
    #         invert_gaussianize(data, *self.post_gaussianization_params)) \
    #         if self.post_gaussianize else self._invert_residuals(data)

    def baseline(self, index):
        # TODO this should be property, or indexable?
        return self.invert_residuals(pd.Series(0., index=index,
                                               name=self.name))

    def _train_baseline(self, train_values):

        Xtr = featurize_index_for_baseline(self.train_index_seconds,
                                           self.periods,
                                           trend=self.trend)
        baseline_params = fit_seasonal_baseline(Xtr, train_values)
        # print('fitted parameters: ', baseline_params)
        self.baseline_params = baseline_params

    def _predict_baseline(self, index):
        Xte = featurize_index_for_baseline(index_to_seconds(index),
                                           self.periods,
                                           trend=self.trend)
        return predict_with_baseline(Xte, self.baseline_params)


def baseline_autotune(train, test,
                      daily_harmonics=None,
                      weekly_harmonics=None,
                      annual_harmonics=None,
                      trend=None):  # ,
                      # pre_gaussianize=None):

    train = train.dropna()
    test = test.dropna()

    print('autotuning baseline on %d train and %d test points' %
          (len(train), len(test)))

    daily_harmonics = np.arange(
        24) if daily_harmonics is None else [daily_harmonics]
    weekly_harmonics = np.arange(
        6) if weekly_harmonics is None else [daily_harmonics]
    annual_harmonics = np.arange(
        50) if annual_harmonics is None else [annual_harmonics]
    trend = [False, True] if trend is None else [trend]
    # pre_gaussianize = [
    #     False, True] if pre_gaussianize is None else [pre_gaussianize]

    baseline = HarmonicBaseline(train, 0, 0, 0, 0, False)

    def test_RMSE(
            daily_harmonics,
            weekly_harmonics,
            annual_harmonics,
            trend):  # ,
        # pre_gaussianize):
        baseline.daily_harmonics = daily_harmonics
        baseline.weekly_harmonics = weekly_harmonics
        baseline.annual_harmonics = annual_harmonics
        baseline.trend = trend
        #baseline.pre_gaussianize = pre_gaussianize
        baseline._fit_baseline()

        return np.sqrt(((test - baseline.baseline(
            test.index))**2).mean())

    res = greedy_grid_search(test_RMSE,
                             [daily_harmonics,
                              weekly_harmonics,
                              annual_harmonics,
                              trend],  # ,
                             # pre_gaussianize],
                             num_steps=2)

    print('optimal params: %s' % res)
    print('test std. dev.: %.2f' % test.std())

    return res


def AutotunedBaseline(train, test, **kwargs):
    print('autotuning baseline for column %s' % train.name)
    # if len(train.dropna().value_counts()) == 1:
    #     kwargs['pre_gaussianize'] = False
    params = baseline_autotune(train.dropna(), test.dropna(), **kwargs)
    return HarmonicBaseline(train.dropna(), *params)
