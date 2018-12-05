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

    # Dimensions of output.
    n_rows_out = row_end - row_begin + 1
    n_cols_out = B.shape[1]
    # Allocate memory.
    out = np.zeros((n_rows_out, n_cols_out))

    # Fill the output matrix.
    # Warning: we only compute a group of rows of the output matrix,
    # hence the row index 0 doesn't correspond to the first row of the output
    # matrix, but to the first row we are computing, i.e. to row row_begin.
    #
    # We have to take this into account when accessing B. The partial version
    # of A that we have already has this built-in.
    for i in range(n_rows_out):
        for j in range(n_cols_out):
            t = 0.0
            for k in range(B.shape[0]):
                t += A_part[i, k] * B[k, j]
            out[i, j] = t

    return out

