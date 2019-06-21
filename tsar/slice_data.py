import numpy as np
import numba as nb

# big_matrix = np.array(model.train_residual)
# lag = 48
# i = 10


@nb.jit(nopython=True)
def make_sliced_flattened_matrix(data_table: np.ndarray, lag: int):
    T, N = data_table.shape
    result = np.empty((T - lag + 1, N * lag))
    for i in range(T - lag + 1):
        data_slice = data_table[i:i + lag]
        result[i, :] = np.ravel(data_slice.T)  # , order='F')
    return result
# def mask_matrix(matrix:np.ndarray, lag:int)
