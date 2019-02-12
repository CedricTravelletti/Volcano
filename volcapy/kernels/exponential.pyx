""" Exponential (non-squared) kernel.

"""
from libc.math cimport exp, sqrt
import cython
import numpy as np
cimport numpy as np
from cython.parallel import prange
from libc.stdio cimport printf


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def build_cov(float[:, ::1] coords, int row_begin, int row_end,
        float sigma_2, float lambda_2):
    """ Builds the covariance matrix from row_begin to row_end, both included..

    Parameters
    ----------
    coords
    row_begin
    row_end

    """
    cdef int dim_j = coords.shape[0]
    cdef int D = coords.shape[1]

    cdef float dist = 0.0

    # Number of rows we will need to generate.
    cdef int n_rows = row_end - row_begin + 1

    cdef int row_ind = 0
    cdef int i, j, d

    # Allocate memory.
    cdef np.float32_t[:,::1] out
    out = np.zeros((n_rows, dim_j), dtype=np.float32, order='C')

    with nogil:
        for i in range(n_rows):
            # Where we are in the big matrix.
            row_ind = row_begin + i

            for j in range(dim_j):
                dist = 0.0
                for d in range(D):
                    dist = dist + (coords[row_ind, d] - coords[j, d])**2
                out[i, j] = sigma_2 * exp(- sqrt(dist) / lambda_2)

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def build_and_mult(float[:, ::1] coords,
        float sigma_2, float lambda_2,
        float[:, ::1] GT):
    """ Refactor of the above, we build and multiply with GT at the same time.

    Parameters
    ----------
    coords
    row_begin
    row_end

    """
    cdef int dim_j = coords.shape[0]
    cdef int n_cols = GT.shape[1]
    cdef int n_rows = dim_j

    cdef int D = coords.shape[1]

    cdef float dist = 0.0

    cdef int row_ind = 0
    cdef int i, j, d, k

    cdef float cov_elem

    # Allocate memory.
    cdef np.float32_t[:,::1] out
    out = np.zeros((n_rows, n_cols), dtype=np.float32, order='C')

    with nogil:
        for i in range(n_rows):
            printf("%d\n", i)
            for j in prange(dim_j):
                dist = 0.0
                for d in range(D):
                    dist = dist + (coords[row_ind, d] - coords[j, d])**2

                    # Element i,j of the covariance matrix.
                    cov_elem = sigma_2 * exp(- sqrt(dist) / lambda_2)

                    # Loop over the columns of second matrix.
                    for k in range(n_cols):
                        out[i, k] = out[i, k] + cov_elem * GT[j, k]

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def get_cov(double[:, :] coords, int i, int j,
        double sigma_2, double lambda_2):
    """ Gets the covariance between two points in model space.
    Can be used to build the covariance matrix.

    Parameters
    ----------
    coords

    """
    cdef int D = coords.shape[1]

    cdef double dist = 0.0
    cdef float out = 0.0

    for d in range(D):
        dist += (coords[i, d] - coords[j, d])**2

    out = sigma_2 * exp(- dist / lambda_2)
    return out
