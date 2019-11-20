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

from unittest import TestCase
from tsar.utils import DataFrameRMSE, DataFrameRMSE_old
import pandas as pd


class TestUtils(TestCase):

    def test_dataframermse(self):
        df1 = pd.DataFrame(index=[1, 2, 3], columns=['a', 'b', 'c'],
                           data=pd.np.random.randn(3, 3))
        df2 = pd.DataFrame(index=df1.index, columns=df1.columns,
                           data=pd.np.random.randn(3, 3))

        rmse1 = DataFrameRMSE(df1, df2)
        rmse2 = DataFrameRMSE_old(df1, df2)

        assert pd.np.all(rmse1.index == rmse2.index)
        assert pd.np.all(rmse1.values == rmse2.values)
