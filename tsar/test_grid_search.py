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
import pandas as pd
import numpy as np

from .greedy_grid_search import greedy_grid_search


class GreedyGridSearchTest(TestCase):

    def test_grid_search(self):

        def func(a, b, bool):
            return (a + b - 10)**2 if bool else 200

        value, res = greedy_grid_search(func, [np.arange(0, 10),
                                               np.arange(0, 10),
                                               [True, False]])

        self.assertTrue(tuple(res) == (9, 1, True))
        self.assertEqual(value, 0)

        def func(a, b, bool):
            return (a + b * 1.0001 - 10)**2 if bool else 200

        value, res = greedy_grid_search(func,
                                        [np.arange(0, 10),
                                         np.arange(0, 10),
                                         [True, False]])

        self.assertTrue(tuple(res) == (1, 9, True))
