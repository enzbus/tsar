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

import pandas as pd
import numpy as np


# def smooth(vector):
#     return (vector +
#             np.concatenate([vector[1:], vector[-1:]]) +
#             np.concatenate([vector[:1], vector[:-1]])) / 3
#
# def generate_baseline(index, daily=True,
#  weekly=True, annual=True, trend=True):
#
#     result = pd.Series(data=np.random.randn(), index=index)
#     if daily:
#         result += smooth(np.random.randn(24))[index.hour]
#     if weekly:
#         result += smooth(np.random.randn(7))[index.weekday]
#     if annual:
#         result += smooth(np.random.randn(12))[index.month-1]
#     if trend:
#         result += np.arange(len(index)) * np.random.randn()
#
#     return result

def make_harmonic_function(length):
    result = np.zeros(length)
    for k in range(1, length):
        result += np.random.randn() * \
            np.cos(np.arange(length)/(length*2*np.pi*k))
        result += np.random.randn() * \
            np.sin(np.arange(length)/(length*2*np.pi*k))
    return result


def generate_baseline(index, daily, weekly, annual, trend):

    result = pd.Series(data=np.random.randn(), index=index)
    if daily:
        result += make_harmonic_function(24)[index.hour]
    if weekly:
        result += make_harmonic_function(7)[index.weekday]
    if annual:
        result += make_harmonic_function(12)[index.month-1]
    if trend:
        result += np.arange(len(index)) * np.random.randn()

    return result


def generate_data(M, T=100,
                  noise_level=1.,
                  baseline_level=1.,
                  autoreg_level=1.,
                  R=1,
                  factor_autoreg_level=1.,
                  daily=True, weekly=True,
                  annual=True, trend=True, freq='1H'):

    # initialize data with noise
    noise = noise_level * np.random.randn(T, M)
    dataframe = pd.DataFrame(index=pd.date_range(start='2019-01-01',
                                                 freq=freq, periods=T),
                             data=noise)
    # add baselines
    for column in dataframe.columns:
        dataframe[column] += baseline_level * generate_baseline(
            dataframe.index, daily, weekly, annual, trend)

    # add scalar auto-regressive part
    auto_reg_noise = np.random.randn(T, M)
    dataframe += autoreg_level * \
        (auto_reg_noise.cumsum(0).T/np.sqrt(range(1, T+1))).T

    # add factor auto-regressive part
    factors = np.random.randn(R, M)
    factor_reg_noise = np.random.randn(T, R)
    dataframe += factor_autoreg_level * \
        ((factor_reg_noise.cumsum(0).T/np.sqrt(range(1, T+1))).T @ factors)

    return dataframe
