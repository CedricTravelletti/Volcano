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
def compute_mesh_squared_diff(float[::1] coords1, float[::1] coords2, out):
    """ Given two arrays, computed the squared difference between all couples.

    Parameters
    ----------
    coords1
    coords2
    out: 2D array
        Array to store the output. Should have first dimension equal to size of
        first array and second equal to size of second array.

    Returns
    -------
    2D array
        Element i,j contains squared difference between element i of first
        array and 2 of second array.

    """
    cdef int dim_1 = coords1.shape[0]
    cdef int dim_2 = coords2.shape[0]

    cdef int i, j

    for i in range(dim_1):
        for j in range(dim_2):
            out[i, j] = (coords1[i] - coords2[j])**2

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def compute_mesh_squared_euclidean_distance(
        float[::1] coords_x1, float[::1] coords_x2,
        float[::1] coords_y1, float[::1] coords_y2,
        float[::1] coords_z1, float[::1] coords_z2):
    """ Given two arrays, computed the squared difference between all couples.

    Parameters
    ----------
    coords_x1
    coords_x2
    coords_y1
    coords_y2
    coords_z1
    coords_z2

    Returns
    -------
    2D array
        Element i,j contains squared difference between element i of first
        array and 2 of second array.

    """
    cdef int dim_1 = coords_x1.shape[0]
    cdef int dim_2 = coords_x2.shape[0]

    # Allocate memory.
    cdef np.float32_t[:,::1] out
    out = np.zeros((dim_1, dim_2), dtype=np.float32, order='C')

    cdef int i, j

    for i in range(dim_1):
        for j in range(dim_2):
            out[i, j] = (coords_x1[i] - coords_x2[j])**2

    # Now do y and z coordinates.
    for i in range(dim_1):
        for j in range(dim_2):
            out[i, j] += (coords_y1[i] - coords_y2[j])**2

    for i in range(dim_1):
        for j in range(dim_2):
            out[i, j] += (coords_z1[i] - coords_z2[j])**2

    # Have to wrap a numpy array around the memoryview. This spares later
    # trouble, since later cast to numpy array might trigger copy under the
    # hood.
    return np.asarray(out)
