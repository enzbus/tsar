import unittest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from tsar.greedy_grid_search import greedy_grid_search


class GreedyGridSearchTest(unittest.TestCase):

    def test_grid_search(self):

        def func(a, b, bool):
            return (a + b - 10)**2 if bool else 200

        res = greedy_grid_search(func, [np.arange(0, 10),
                                        np.arange(0, 10),
                                        [True, False]])

        self.assertTrue(tuple(res) == (9, 1, True))

        def func(a, b, bool):
            return (a + b * 1.0001 - 10)**2 if bool else 200

        res = greedy_grid_search(func,
                                 [np.arange(0, 10),
                                     np.arange(0, 10),
                                     [True, False]])

        self.assertTrue(tuple(res) == (1, 9, True))
