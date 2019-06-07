import unittest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from tsar import Model


class BaselineTest(unittest.TestCase):

    data = pd.read_pickle('data/wind_test_data.pickle')
    train = data[data.index.year.isin([2010, 2011])]
    test = data[data.index.year == 2012]

    def test_fit_baseline(self):
        print(self.train.head())

        model = Model(harmonics=4)
        model._train_baseline(self.train)

        train_baseline = model._predict_baseline(self.train.index)
        print(train_baseline.head())
        self.assertTrue(np.isclose((self.train - train_baseline).mean(), 0))

        train_rmse = (self.train - train_baseline).std()
        print('train rmse', train_rmse)
        self.assertTrue(train_rmse < 6)

        test_baseline = model._predict_baseline(self.test.index)

        test_mean_error = (self.test - test_baseline).mean()
        print('test me', test_mean_error)

        self.assertTrue(np.abs(test_mean_error) < .4)

        test_rmse = (self.test - test_baseline).std()
        print('test rmse', test_rmse)

        self.assertTrue(test_rmse > train_rmse)
