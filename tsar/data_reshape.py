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

import numba as nb
import numpy as np


@nb.njit()
def one_slice(data_table: np.ndarray,
              P: int, F: int,
              t: np.array,
              result: np.array):
    """Given data as 2-dimensional array, reshape as concatenated vector."""

    T, M = data_table.shape
    lag = P+F

    # assert t >= 0
    # assert t < T

    start = t - P + 1
    missing_at_the_start = -min(start, 0)
    end = t + F + 1
    missing_at_the_end = max(end, T) - T

    # result[:missing_at_the_start*M] = np.nan

    data_slice = data_table[max(start, 0):min(end, T)]

    if missing_at_the_start:
        data_slice = np.concatenate(
            (np.ones((missing_at_the_start, M)) * np.nan, data_slice))

    if missing_at_the_end:
        data_slice = np.concatenate(
            (data_slice, np.ones((missing_at_the_end, M)) * np.nan))

    # result[missing_at_the_start*M:(M*lag)-missing_at_the_end*M] = \
    #    np.ravel(data_slice.T)
    result[:] = np.ravel(data_slice.T)

    # result[(M*lag)-missing_at_the_end*M:] = np.nan


def make_sliced_flattened_matrix(data_table: np.ndarray,
                                 P: int, F: int,
                                 prediction_times: np.array):
    """Given data as 2-dimensional array, reshape as concatenated vectors."""
    lag = P + F
    T, M = data_table.shape
    # result = np.empty((T - lag + 1, M * lag))
    result = np.empty((len(prediction_times), M * lag))
    for i, t in enumerate(prediction_times):
        start = t - P
        end = t + F
        data_slice = data_table[max(start, 0):min(end, T)]
    # for i in range(T - lag + 1):
    #    data_slice = data_table[i:i + lag]
        result[i, :] = np.ravel(data_slice.T)  # , order='F')
    return result
