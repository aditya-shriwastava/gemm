cimport cython
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def gemm_pyx(double[:, :] A, double[:, :] B):
    assert A.shape[1] == B.shape[0]

    cdef int rows = A.shape[0]
    cdef int inner = A.shape[1]
    cdef int cols = B.shape[1]

    C = np.zeros((rows, cols), dtype=np.float64)
    cdef double[:, :] C_view = C

    cdef double acc = 0.0
    for i in range(rows):
        for j in range(cols):
            acc = 0.0
            for k in range(inner):
                acc += A[i, k] * B[k, j]
            C_view[i, j] = acc

    return C
