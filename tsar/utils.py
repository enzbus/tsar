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


def sanitize_baseline_params(params, columns):
    for col in columns:
        if col not in params:
            params[col] = {}
        # non parametric baseline
        if 'non_par_baseline' not in params[col]:
            params[col]['non_par_baseline'] = False

        if params[col]['non_par_baseline'] and not \
           ('used_features' in params[col]):
            params[col]['used_features'] = \
                ['hour_of_day', 'day_of_week', 'month_of_year']
        if params[col]['non_par_baseline'] and not \
           ('lambdas' in params[col]):
            params[col]['lambdas'] = \
                [None] * len(params[col]['used_features'])
    return params


def check_multidimensional_time_series(data, frequency=None, columns=None):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            'Data must be a pandas DataFrame')
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(
            'Data must be indexed by a pandas DatetimeIndex.')
    if data.index.freq is None:
        raise ValueError('Data index must have a frequency. ' +
                         'Try using the pandas.DataFrame.asfreq method.')
    if not frequency is None and not (data.index.freq == frequency):
        raise ValueError('Data index frequency is not correct.')
    if not columns is None and not np.all(data.columns == columns):
        raise ValueError('Data columns are not correct.')


def DataFrameRMSE_old(df1, df2):
    return np.sqrt(((df1 - df2)**2).mean())


def DataFrameRMSE(df1, df2):
    result = matrix_rmse(df1.values, df2.values)
    return pd.Series(result, index=df1.columns)


def matrix_rmse(mat1, mat2):
    assert (mat1.shape == mat2.shape)
    m, n = mat1.shape

    result = np.zeros(n)
    for i in range(n):
        result[i] = np.sqrt(np.nanmean((mat1[:, i] - mat2[:, i])**2))
    return result
