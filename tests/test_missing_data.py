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
from tsar import tsar

import pandas as pd
import numpy as np
from scipy.sparse.linalg import eigs


from tsar.generate_data import generate_data


def evaluate_model(model, test, t):

    F = model.future_lag
    pred = model.predict(test.iloc[:t],
                         prediction_time=test.index[t])

    real = test.loc[test.index[t:t+F]]
    mypred = pred.loc[test.index[t:t+F]]

    return mypred, real


def random_mask(shape, density):
    from scipy.sparse import rand
    x = rand(shape[0], shape[1], density=density, format='csr')
    x.data[:] = True
    return x.todense().astype(bool)


def MSE(df):
    return (df**2).mean().mean()


class TestMissingData(TestCase):

    def check_smaller(self, df1, df2):
        pred_error = MSE(df1)
        zero_pred_error = MSE(df2)

        print('MSE df1, MSE df2', pred_error, zero_pred_error)
        self.assertTrue(pred_error < zero_pred_error)

    def check_equal(self, df1, df2):
        pred_error = (df1**2).mean().mean()
        zero_pred_error = (df2**2).mean().mean()

        print('MSE df1, MSE df2', pred_error, zero_pred_error)
        self.assertTrue(pred_error == zero_pred_error)

    def test_rank_zero(self):

        P = 12
        F = 12

        np.random.seed(1)
        data = generate_data(M=20, T=2000, freq='6H',
                             factor_autoreg_level=10.,
                             R=5, trend=False)
        train, test = data.iloc[:1000], data.iloc[1000:]

        train[random_mask(train.shape, density=0.3)] = np.nan

        print(train)
        rank_uncorrected = 0
        quad_reg_uncorrected = 0.5
        rank_corrected = rank_uncorrected
        quad_reg_corrected = 0.5

        model_1 = tsar(train, P=P, F=F, R=rank_uncorrected,
                       quadratic_regularization=quad_reg_uncorrected)
        model_2 = tsar(train, P=P, F=F, R=rank_corrected, use_svd_fit=True,
                       noise_correction=True,
                       quadratic_regularization=quad_reg_corrected)
        model_3 = tsar(train, P=P, F=F, R=rank_uncorrected, use_svd_fit=True,
                       noise_correction=False,
                       quadratic_regularization=quad_reg_uncorrected)

        zero_err = 0.
        mod_1_err = 0.
        mod_2_err = 0.
        mod_3_err = 0.
        for t in [100, 150, 200, 250, 300]:
            baseline_pred = model_1.predict(test.iloc[:1],
                                            prediction_time=test.index[t])
            mypred_1, real = evaluate_model(model_1, test, t=t)
            mypred_2, real = evaluate_model(model_2, test, t=t)
            mypred_3, real = evaluate_model(model_3, test, t=t)
            # print(mypred, mypred-real)

            zero_err += MSE(baseline_pred-real)
            mod_1_err += MSE(mypred_1 - real)
            mod_2_err += MSE(mypred_2 - real)
            mod_3_err += MSE(mypred_3 - real)

        self.check_smaller(mod_1_err, zero_err)
        self.check_equal(mod_2_err, mod_3_err)
        self.check_equal(mod_1_err, mod_3_err)

    def test_rank_one(self):

        P = 12
        F = 12

        np.random.seed(1)
        data = generate_data(M=20, T=2000, freq='6H',
                             factor_autoreg_level=10.,
                             R=5, trend=False)
        train, test = data.iloc[:1000], data.iloc[1000:]

        train[random_mask(train.shape, density=0.15)] = np.nan

        print(train)
        rank_uncorrected = 1
        quad_reg_uncorrected = .1
        rank_corrected = rank_uncorrected
        quad_reg_corrected = quad_reg_uncorrected

        model_1 = tsar(train, P=P, F=F, R=rank_uncorrected,
                       quadratic_regularization=quad_reg_uncorrected)
        model_2 = tsar(train, P=P, F=F, R=rank_corrected, use_svd_fit=True,
                       noise_correction=True,
                       quadratic_regularization=quad_reg_corrected)
        model_3 = tsar(train, P=P, F=F, R=rank_uncorrected, use_svd_fit=True,
                       noise_correction=False,
                       quadratic_regularization=quad_reg_uncorrected)

        zero_err = 0.
        mod_1_err = 0.
        mod_2_err = 0.
        mod_3_err = 0.

        for t in [100, 150, 200, 250, 300]:
            baseline_pred = model_1.predict(test.iloc[:1],
                                            prediction_time=test.index[t])
            mypred_1, real = evaluate_model(model_1, test, t=t)
            mypred_2, real = evaluate_model(model_2, test, t=t)
            mypred_3, real = evaluate_model(model_3, test, t=t)
            # print(mypred, mypred-real)

            zero_err += MSE(baseline_pred-real)
            mod_1_err += MSE(mypred_1 - real)
            mod_2_err += MSE(mypred_2 - real)
            mod_3_err += MSE(mypred_3 - real)

        self.check_smaller(mod_1_err, zero_err)
        self.check_smaller(mod_2_err, mod_3_err)

    def test_rank_two(self):

        P = 12
        F = 12

        np.random.seed(1)
        data = generate_data(M=20, T=2000, freq='6H',
                             factor_autoreg_level=10.,
                             R=5, trend=False)
        train, test = data.iloc[:1000], data.iloc[1000:]

        train[random_mask(train.shape, density=0.05)] = np.nan

        print(train)
        rank_uncorrected = 2
        quad_reg_uncorrected = .1
        rank_corrected = 2
        quad_reg_corrected = .1

        model_1 = tsar(train, P=P, F=F, R=rank_uncorrected,
                       quadratic_regularization=quad_reg_uncorrected)
        model_2 = tsar(train, P=P, F=F, R=rank_corrected, use_svd_fit=True,
                       noise_correction=True,
                       quadratic_regularization=quad_reg_corrected)
        model_3 = tsar(train, P=P, F=F, R=rank_uncorrected, use_svd_fit=True,
                       noise_correction=False,
                       quadratic_regularization=quad_reg_uncorrected)

        zero_err = 0.
        mod_1_err = 0.
        mod_2_err = 0.
        mod_3_err = 0.

        for t in [100, 150, 200, 250, 300]:

            baseline_pred = model_1.predict(test.iloc[:1],
                                            prediction_time=test.index[t])
            mypred_1, real = evaluate_model(model_1, test, t=t)
            mypred_2, real = evaluate_model(model_2, test, t=t)
            mypred_3, real = evaluate_model(model_3, test, t=t)
            # print(mypred, mypred-real)

            zero_err += MSE(baseline_pred - real)
            mod_1_err += MSE(mypred_1 - real)
            mod_2_err += MSE(mypred_2 - real)
            mod_3_err += MSE(mypred_3 - real)

        self.check_smaller(mod_1_err, zero_err)
        self.check_smaller(mod_2_err, mod_3_err)
        # self.check_smaller(mod_1_err, mod_3_err)

    def test_rank_five(self):

        P = 12
        F = 12

        np.random.seed(1)
        data = generate_data(M=20, T=2000, freq='6H',
                             factor_autoreg_level=10.,
                             R=5, trend=False)
        train, test = data.iloc[:1000], data.iloc[1000:]

        #train[random_mask(train.shape, density=0.05)] = np.nan

        print(train)
        rank_uncorrected = 5
        quad_reg_uncorrected = .1
        rank_corrected = 5
        quad_reg_corrected = .1

        model_1 = tsar(train, P=P, F=F, R=rank_uncorrected,
                       quadratic_regularization=quad_reg_uncorrected)
        model_2 = tsar(train, P=P, F=F, R=rank_corrected, use_svd_fit=True,
                       noise_correction=True,
                       quadratic_regularization=quad_reg_corrected)
        model_3 = tsar(train, P=P, F=F, R=rank_uncorrected, use_svd_fit=True,
                       noise_correction=False,
                       quadratic_regularization=quad_reg_uncorrected)

        zero_err = 0.
        mod_1_err = 0.
        mod_2_err = 0.
        mod_3_err = 0.

        for t in [100, 150, 200, 250, 300]:

            baseline_pred = model_1.predict(test.iloc[:1],
                                            prediction_time=test.index[t])
            mypred_1, real = evaluate_model(model_1, test, t=t)
            mypred_2, real = evaluate_model(model_2, test, t=t)
            mypred_3, real = evaluate_model(model_3, test, t=t)
            # print(mypred, mypred-real)

            zero_err += MSE(baseline_pred - real)
            mod_1_err += MSE(mypred_1 - real)
            mod_2_err += MSE(mypred_2 - real)
            mod_3_err += MSE(mypred_3 - real)

        self.check_smaller(mod_1_err, zero_err)
        self.check_smaller(mod_2_err, mod_3_err)
        self.assertTrue(np.isclose(mod_1_err, mod_3_err))
