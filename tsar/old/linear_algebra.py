"""
Copyright (C) Enzo Busseti 2019.

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


import numpy as np
import pandas as pd
import numba as nb
import logging
import scipy.sparse.linalg as spl
logger = logging.getLogger(__name__)

from .low_rank_plus_sparse import SquareLowRankPlusDiagonal

__all__ = ['iterative_denoised_svd']


def iterative_denoised_svd(dataframe, P, T=10):

    if not dataframe.isnull().sum().sum():
        T = 1

    y = pd.DataFrame(0., index=dataframe.index,
                     columns=dataframe.columns)
    for t in range(T):
        u, s, v = spl.svds(dataframe.fillna(y), P + 1)
        dn_u, dn_s, dn_v = u[:, 1:], s[1:] - s[0], v[1:]
        new_y = dn_u @ np.diag(dn_s) @ dn_v
        print('Iterative svd, MSE(y_%d - y_{%d}) = %.2e' % (
            t + 1, t, ((new_y - y)**2).mean().mean()))
        y.iloc[:, :] = dn_u @ np.diag(dn_s) @ dn_v
    return dn_u, dn_s, dn_v


#@nb.jit(nopython=True)
def schur_complement_matrix(matrix_with_na,
                            null_mask,
                            Sigma):
    # null_mask = np.isnan(array_with_na)
    Y = matrix_with_na[:, ~null_mask]

    # A = Sigma[null_mask].T[null_mask]

    if isinstance(Sigma, SquareLowRankPlusDiagonal):
        B = Sigma[null_mask, ~null_mask]
        C = Sigma[~null_mask, ~null_mask]
        inv_C = C.inv()

    else:
        B = Sigma[null_mask].T[~null_mask].T
        C = Sigma[~null_mask].T[~null_mask]
        if hasattr(C, 'todense'):
            print('converting C to dense, probably bug!!')
            C = C.todense()
        inv_C = np.linalg.inv(C)

    expected_X = B @ inv_C @ Y.T
    matrix_with_na[:, null_mask] = expected_X.T
    return matrix_with_na
