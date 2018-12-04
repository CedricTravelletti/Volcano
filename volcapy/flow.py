from volcapy import loading
import volcapy.grid as gd

import numpy as np


# Globals
dx = 50
dy = 50
dz = 50
spacings = (dx, dy, dz)

data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"

data = loading.load_niklas(data_path)
reg_grid, reg_dims = gd.regularize_grid(data['coords'], spacings)

centered_reg_grid, grid_dims = gd.regularize_grid_centered(
        data['coords'], spacings)

sigma_2 = 200.0
lambda_2 = 200.0**2

covariance_hash = gd.buil_hash_grid(centered_reg_grid, sigma_2, lambda_2)


def implicit_mat_mult(f, B, shape):
    """ Performs the matrix multiplication A*B,
    where A is defined implicitly via the function:
    f(i, j) = A[i, j].

    This is useful when the matrix A is g to fit in memory.

    Parameters
    ----------
    shape: (int, int)
        Shape of the matrix A.

    """
    out = np.zeros((shape[0], B.shape[1]))

    for i in range(shape[0]):
        for j in range(B.shape[1]):
            temp = 0
            for k in range(B.shape[0]):
                temp += f(i, k) * B[k, j]

            out[i, j] = temp
        print(i)

    return out

grid = data['coords']
F = data['F'][:3, :]

f = lambda x,y: gd.covariance_matrix(x, y, grid, grid_dims, spacings,
        covariance_hash)

# Sub problem.
n_model = 179171
C = implicit_mat_mult(f, F.T, (n_model, 3))
