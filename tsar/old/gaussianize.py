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
import scipy.stats

logger = logging.getLogger(__name__)

__all__ = ['compute_gaussian_interpolator',
           'gaussianize', 'invert_gaussianize']


def compute_gaussian_interpolator(column,
                                  min_N=10,
                                  corrector=1E-5):

    N = max(min_N, int(len(column) / 100))

    xs = np.array([scipy.stats.norm.ppf(i) for i in np.concatenate(
        [[1. / len(column)], (np.arange(1, N) / N), [1 - 1. / len(column)]])])
    quantiles = np.array([column.quantile(i) for i in (np.arange(N + 1) / N)])

    if not np.all(np.diff(quantiles) > 0):

        new_quantiles = (quantiles * (1 - corrector) + quantiles[0] * corrector +
                         np.arange(len(quantiles)) * corrector * (quantiles[-1] - quantiles[0])
                         / (len(quantiles) - 1))

        assert np.all(np.diff(new_quantiles) > 0)
        assert np.isclose((new_quantiles - quantiles)[0], 0)
        assert np.isclose((new_quantiles - quantiles)[-1], 0)
        return xs, new_quantiles

    return xs, quantiles


def gaussianize(column, xs, quantiles):
    return pd.Series(np.interp(column, quantiles, xs),
                     column.index, name=column.name)


def invert_gaussianize(column, xs, quantiles):
    return pd.Series(np.interp(column, xs, quantiles),
                     column.index, name=column.name)
