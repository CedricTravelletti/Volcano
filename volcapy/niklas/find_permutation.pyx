""" Find the permuation that links the indices of cell in our version to the
ones in Niklas.
"""
import numpy as np
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from libc.math cimport abs

cimport cython


my_F_path = "./latest_F.npy"
niklas_F_path = "./F_niklas_raw.npy"

my_F = np.load(my_F_path)
niklas_F = np.load(niklas_F_path)


@cython.boundscheck(False)
@cython.wraparound(False)
def main(np.ndarray[np.float64_t, ndim=2] my_F_c,
        np.ndarray[np.float64_t, ndim=2] niklas_F_c):

    cdef float tol = 10e-7

    cdef int N = my_F.shape[1]
    cdef int M = niklas_F.shape[1]
    cdef int i, j

    ind_map = []
    for i in range(N):
        # Loop over Niklas and see if corresponds.
        print("i: " + str(i))
        for j in range(M):
            if abs(my_F_c[10, i] - niklas_F_c[10, j]) < tol:
                if abs(my_F_c[200, i] - niklas_F_c[200, j]) < tol:
                    ind_map.append((i, j))
                    print("Success.")
                    print(my_F_c[10, i])
                    print(niklas_F_c[10, i])
    return ind_map
