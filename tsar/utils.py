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


def check_series(data):
    if not isinstance(data, pd.Series):
        raise ValueError(
            'Data must be a pandas Series')


def check_timeseries(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            'Train data must be a pandas DataFrame')
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError(
            'Train data must be indexed by a pandas DatetimeIndex.')
    if data.index.freq is None:
        raise ValueError('Train data index must have a frequency. ' +
                         'Try using the pandas.DataFrame.asfreq method.')


def RMSE(a, b):
    diff = a - b
    diff = diff[~np.isnan(diff)]
    return np.sqrt(np.mean(diff**2))
