#cython: boundscheck=False, wraparound=False, nonecheck=False
from libc.math cimport exp


def build_cov(double[:, :] coords, double[:, :] out, int row_begin, int row_end):
    """ Builds the covariance matrix from row_begin to row_end, both included..

    Parameters
    ----------
    coords
    out
    row_begin
    row_end

    """
    cdef int dim_j = coords.shape[0]
    cdef int D = coords.shape[1]

    cdef double sigma_2 = 200.0
    cdef double lambda_2 = 200.0**2

    cdef double dist = 0.0

    for i in range(row_begin, row_end + 1):
        for j in range(dim_j):
            dist = 0.0
            for d in range(D):
                dist += (coords[i, d] - coords[j, d])**2
            out[i, j] = sigma_2 * exp(- dist / lambda_2)

    return out
