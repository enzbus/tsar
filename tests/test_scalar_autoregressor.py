import unittest
import pandas as pd
import numpy as np
import scipy.sparse as sp

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from tsar.scalar_autoregressor import *


class TestScalarAR(unittest.TestCase):

    def test_scalar_ar(self):

        np.random.seed(0)

        T = 10000
        data = pd.Series(
            index=pd.date_range(start=pd.datetime.now(),
                                periods=T, freq='D'),
            data=np.random.randn(T))

        data = data.rolling(5).max()

        train = data.iloc[:-T // 2]
        mean = train.mean()
        train -= mean
        test = data.iloc[-T // 2:]
        test -= mean

        opt_lag, = \
            autotune_scalar_autoregressor(train, test, 5)

        self.assertTrue(opt_lag == 13)

    def test_scalar_ar_2(self):

        np.random.seed(0)

        T = 10000
        data = pd.Series(
            index=pd.date_range(start=pd.datetime.now(),
                                periods=T, freq='D'),
            data=np.random.randn(T))

        data = data.rolling(5).mean()

        train = data.iloc[:-T // 2]
        mean = train.mean()
        train -= mean
        test = data.iloc[-T // 2:]
        test -= mean

        opt_lag, = \
            autotune_scalar_autoregressor(train, test, 5, max_past_lag=10)

        self.assertTrue(opt_lag > 5)

    def test_scalar_ar_3(self):

        np.random.seed(0)

        T = 10000
        data = pd.Series(
            index=pd.date_range(start=pd.datetime.now(),
                                periods=T, freq='D'),
            data=np.random.randn(T))

        data -= data.rolling(5).mean()

        train = data.iloc[:-T // 2]
        mean = train.mean()
        train -= mean
        test = data.iloc[-T // 2:]
        test -= mean

        opt_lag, = \
            autotune_scalar_autoregressor(train, test, 5, max_past_lag=10)

        self.assertTrue(opt_lag > 5)
