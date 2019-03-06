""" Goal is to benchmark perf of cache friedly implementation, i.e. pure
looping on 1D array.

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
def build_cov(float[::1] coords_x, float[::1] coords_x2):
    cdef int dim_1 = coords_x.shape[0]
    cdef int dim_2 = coords_x2.shape[0]

    # Allocate memory.
    cdef np.float32_t[:,::1] out
    out = np.zeros((dim_1, dim_2), dtype=np.float32, order='C')

    cdef int i, j

    for i in range(dim_1):
        for j in range(dim_2):
            out[i, j] = (coords_x[i] - coords_x2[j])**2

    return out
