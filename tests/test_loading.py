""" Test the loading module.

"""
from volcapy import loading as ld
from volcapy import grid as gd
from volcapy import matrix as mt

import numpy as np

from numpy.testing import assert_array_equal

data_path = "/home/cedric/PHD/Dev/Volcano/data/Cedric.mat"


def test_loading():
    data = ld.load_niklas(data_path)

def test_regularize_grid():
    unreg_grid = [
            [0, 0, 0],
            [1, 1, 1]]
    dx = 1.0
    dy = 1.0
    dz = 0.25

    spacings = (dx, dy, dz)

    # How the result should look.
    reg_grid = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 0.25],
            [1, 0, 0.25],
            [0, 1, 0.25],
            [1, 1, 0.25],
            [0, 0, 0.5],
            [1, 0, 0.5],
            [0, 1, 0.5],
            [1, 1, 0.5],
            [0, 0, 0.75],
            [1, 0, 0.75],
            [0, 1, 0.75],
            [1, 1, 0.75],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]])

    result, dims = gd.regularize_grid(unreg_grid, spacings)

    assert_array_equal(result, reg_grid)

def test_partial_mult():
    # Function that builds element i, j of matrix A.
    def f_A(i, j):
        return 3*i - j

    # Build partial rows of A, we say A has 8 columns.
    def f_A_part(row_begin, row_end):
        cols_A = 8
        n_rows = row_end - row_begin + 1

        out = np.zeros((n_rows, cols_A))

        for i in range(n_rows):
            for j in range(cols_A):
                out[i, j] = f_A(row_begin + i, j)
        return out

    # Test matrix.
    B = np.array([[1,2,3,4,5],
            [9,10,11,31,5],
            [3, -9, -1, 19, 0],
            [11,9,11,-1,4],
            [33,-3,2,2,1],
            [1,1,0,7,5],
            [1,1,1,1,1],
            [-22,6,7,8,9]])

    row_begin_1 = 0
    row_end_1 = 3
    row_begin_2 = 3
    row_end_2 = 6

    true_result_1 = np.dot(f_A_part(0, 7), B)[row_begin_1:row_end_1 + 1, :]
    true_result_2 = np.dot(f_A_part(0, 7), B)[row_begin_2:row_end_2 + 1, :]

    assert_array_equal(mt.partial_mult(f_A_part, B, row_begin_1, row_end_1),
            true_result_1)
    assert_array_equal(mt.partial_mult(f_A_part, B, row_begin_2, row_end_2),
            true_result_2)
