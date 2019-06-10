import unittest
import pandas as pd
import numpy as np
import scipy.sparse as sp

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from tsar.low_rank_autoregressor import *


class TestLowRankAR(unittest.TestCase):

    def test_low_rank_ar(self):

        np.random.seed(0)

        T, N = 10000, 10
        data = pd.DataFrame(
            index=pd.date_range(start=pd.datetime.now(),
                                periods=T, freq='D'),
            data=np.random.randn(T, N))

        # A = np.random.randn(N, 2)

        # signals = np.concatenate([
        #     [np.sin(np.arange(T) / 100)],
        #     [np.cos(np.arange(T) / 100)]], axis=0
        # )

        # data += 10 * (A @ signals).T

        data.iloc[:, 7] = data.iloc[:, 9].shift(-1)
        data.iloc[:, 8] = data.iloc[:, 0].shift(-1)
        data.iloc[:, 3] = data.iloc[:, 5].shift(-1) + data.iloc[:, 8]
        data.iloc[:, 4] = data.iloc[:, 6].shift(-1) + data.iloc[:, 7]
        data.iloc[:, 2] = data.iloc[:, 3] - data.iloc[:, 4]
        data.iloc[:, 1] = data.iloc[:, 2] - data.iloc[:, 4]

        data /= data.std()

        train = data.iloc[:-T // 2]
        test = data.iloc[-T // 2:]

        model = LowRankAR(train,
                          P=1,
                          future_lag=5,
                          past_lag=1)

        self.assertTrue(np.all(model.orig_diag > 0))
        self.assertTrue(np.all(model.orig_diag < 1))

        opt_P, opt_lag = \
            autotune_low_rank_autoregressor(train, test, 5)

        self.assertTrue(opt_P == 7)
        self.assertTrue(opt_lag == 1)

    def test_low_rank_ar_2(self):

        np.random.seed(0)

        T, N = 10000, 10
        data = pd.DataFrame(
            index=pd.date_range(start=pd.datetime.now(),
                                periods=T, freq='D'),
            data=np.random.randn(T, N))

        data.iloc[:, 7] = data.iloc[:, 9].shift(-2)
        data.iloc[:, 8] = data.iloc[:, 0].shift(-2) + data.iloc[:, 4].shift(-1)
        data.iloc[:, 3] = data.iloc[:, 5].shift(-2) + data.iloc[:, 8]
        data.iloc[:, 4] = data.iloc[:, 6].shift(-2) + data.iloc[:, 7]
        data.iloc[:, 2] = data.iloc[:, 3] - data.iloc[:, 4]
        data.iloc[:, 1] = data.iloc[:, 2] - data.iloc[:, 4]

        data /= data.std()

        train = data.iloc[:-T // 2]
        test = data.iloc[-T // 2:]

        opt_P, opt_lag = \
            autotune_low_rank_autoregressor(train, test, 5)

        self.assertTrue(opt_P == 7)
        self.assertTrue(opt_lag == 2)

    def test_low_rank_ar_3(self):

        np.random.seed(0)

        T, N = 10000, 10
        data = pd.DataFrame(
            index=pd.date_range(start=pd.datetime.now(),
                                periods=T, freq='D'),
            data=np.random.randn(T, N))

        A = np.random.randn(N, 2)

        signals = np.concatenate([
            [np.sin(np.arange(T))],
            [np.cos(np.arange(T) * np.sqrt(2))]], axis=0
        )

        data += 20 * (A @ signals).T

        data /= data.std()

        train = data.iloc[:-T // 2]
        test = data.iloc[-T // 2:]

        model = LowRankAR(train,
                          P=2,
                          future_lag=5,
                          past_lag=5)

        self.assertTrue(np.all(model.orig_diag > 0))
        self.assertTrue(np.all(model.orig_diag < 1))

        P_2_RMSE = model.test_RMSE(test)

        opt_P, opt_lag = \
            autotune_low_rank_autoregressor(train, test, 5, past_lag=5)

        self.assertTrue(opt_P >= 2)

        model = LowRankAR(train,
                          P=opt_P,
                          future_lag=5,
                          past_lag=5)

        # print(model.test_RMSE(test) / P_2_RMSE)
        self.assertTrue(
            (model.test_RMSE(test) / P_2_RMSE) > 0.95)
