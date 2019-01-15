""" Provides matrix multiplication capabilities for big matrices found in
inversion problems.
"""
import numpy as np


def partial_mult(f, B, row_begin, row_end):
    """ Computes the rows row_begin to row_end of the matrix product A*B, where
    A is defined by a function f(i, j) which returns rows i to j of A.

    """
    # Fetch the rows of A we need.
    A_part = f(row_begin, row_end)

    return np.dot(A_part, B)

def implicit_mat_mult(f, B, shape):
    """ Performs the matrix multiplication A*B,
    where A is defined implicitly via the function:
    f(i, j) = A[i, j].

    This is useful when the matrix A is too big to fit in memory.

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
