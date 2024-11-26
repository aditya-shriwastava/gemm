#!/usr/bin/env python3

import os
import time
import ctypes

import numpy as np
import numba as nb
import cupy as cp
import dask.array as da
import torch

from gemm_pyx import gemm_pyx

def gemm_da(A, B):
    A_da = da.from_array(A)
    B_da = da.from_array(B)
    C_da = da.dot(A_da, B_da)
    return C_da.compute()

def gemm_cp(A, B):
    A_cp = cp.array(A)
    B_cp = cp.array(B)
    return cp.asnumpy(A_cp @ B_cp)

def gemm_torch(A, B):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    A_pt = torch.from_numpy(A).float().to(device)
    B_pt = torch.from_numpy(B).float().to(device)
    C_pt = A_pt @ B_pt
    return C_pt.detach().cpu().numpy()

def gemm_np(A, B):
    return A @ B

@nb.njit(fastmath=True)
def gemm_numba(A, B):
    assert A.shape[1] == B.shape[0]

    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            acc = 0.0
            for k in range(A.shape[1]):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc

    return C

def gemm_py(A, B):
    assert A.shape[1] == B.shape[0]

    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            acc = 0.0
            for k in range(A.shape[1]):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc

    return C

def comp(C1, C2):
    assert C1.shape == C2.shape
    print((C1-C2).max())

def main():
    N = 2000
    A = np.random.random((N, N)).astype('float64')
    B = np.random.random((N, N)).astype('float64')
    libgemm = ctypes.CDLL('./libgemm.so')
    
    ts = time.monotonic()
    C0 = gemm_np(A, B)
    dt = time.monotonic() - ts
    print(f'gemm_np | dt: {dt}')

    ts = time.monotonic()
    C1 = gemm_numba(A, B)
    dt = time.monotonic() - ts
    print(f'gemm_numba | dt: {dt}')

    # ts = time.monotonic()
    # C2 = gemm_py(A, B)
    # dt = time.monotonic() - ts
    # print(f'gemm_py | dt: {dt}')

    ts = time.monotonic()
    C3 = gemm_pyx(A, B)
    dt = time.monotonic() - ts
    print(f'gemm_pyx | dt: {dt}')

    gemm_c = libgemm.gemm_c
    gemm_c.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t
    ]
    rows = A.shape[0]
    cols = B.shape[1]
    assert A.shape[1] == B.shape[0]
    inner = A.shape[1]
    C4 = np.zeros((rows, cols), dtype=np.double)
    ts = time.monotonic()
    gemm_c(A, B, C4, rows, cols, inner)
    dt = time.monotonic() - ts
    print(f'gemm_c | dt: {dt}')

    gemm_blas = libgemm.gemm_blas
    gemm_blas.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t
    ]
    rows = A.shape[0]
    cols = B.shape[1]
    assert A.shape[1] == B.shape[0]
    inner = A.shape[1]
    C5 = np.zeros((rows, cols), dtype=np.double)
    ts = time.monotonic()
    gemm_blas(A, B, C5, rows, cols, inner)
    dt = time.monotonic() - ts
    print(f'gemm_blas | dt: {dt}')

    ts = time.monotonic()
    C6 = gemm_cp(A, B)
    dt = time.monotonic() - ts
    print(f'gemm_cp | dt: {dt}')

    ts = time.monotonic()
    C7 = gemm_torch(A, B)
    dt = time.monotonic() - ts
    print(f'gemm_torch | dt: {dt}')

    ts = time.monotonic()
    C8 = gemm_da(A, B)
    dt = time.monotonic() - ts
    print(f'gemm_da | dt: {dt}')

    gemm_eigen = libgemm.gemm_eigen
    gemm_eigen.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.double, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t
    ]
    rows = A.shape[0]
    cols = B.shape[1]
    assert A.shape[1] == B.shape[0]
    inner = A.shape[1]
    C9 = np.zeros((rows, cols), dtype=np.double)
    ts = time.monotonic()
    gemm_eigen(A, B, C9, rows, cols, inner)
    dt = time.monotonic() - ts
    print(f'gemm_eigen | dt: {dt}')

if __name__ == '__main__':
    main()
