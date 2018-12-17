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

    cdef double sigma_2 = 100.0
    cdef double lambda_2 = 200.0**2

    cdef double dist = 0.0

    # Number of rows we will need to generate.
    cdef int n_rows = row_end - row_begin + 1

    cdef int row_ind = 0
    cdef int i, j, d

    for i in range(n_rows):
        # Where we are in the big matrix.
        row_ind = row_begin + i

        for j in range(dim_j):
            dist = 0.0
            for d in range(D):
                dist += (coords[row_ind, d] - coords[j, d])**2
            out[i, j] = sigma_2 * exp(- dist / lambda_2)

    return out


def get_cov(double[:, :] coords, int i, int j):
    """ Gets the covariance between two points in model space.
    Can be used to build the covariance matrix.

    Parameters
    ----------
    coords

    """
    cdef int D = coords.shape[1]

    cdef double sigma_2 = 100.0
    cdef double lambda_2 = 200.0**2
    cdef double dist = 0.0
    cdef float out = 0.0

    for d in range(D):
        dist += (coords[i, d] - coords[j, d])**2

    out = sigma_2 * exp(- dist / lambda_2)
    return out
