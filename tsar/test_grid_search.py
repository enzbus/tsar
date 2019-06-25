import unittest
import pandas as pd
import numpy as np

from .greedy_grid_search import greedy_grid_search


class GreedyGridSearchTest(unittest.TestCase):

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
