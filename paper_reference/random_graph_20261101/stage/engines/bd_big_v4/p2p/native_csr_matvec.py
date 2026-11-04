"""Parallelize independent CSR rows; preserve each row's float64 summation order."""
import numpy as np
from numba import njit,prange,set_num_threads

@njit(parallel=True,cache=True)
def csr_matvec(indptr,indices,data,vector):
    result=np.empty(len(indptr)-1,dtype=np.float64)
    for row in prange(len(indptr)-1):
        value=0.0
        for position in range(indptr[row],indptr[row+1]):
            value+=data[position]*vector[indices[position]]
        result[row]=value
    return result

def warmup(threads=3):
    set_num_threads(threads)
    for dtype in [np.int32,np.int64]:
        csr_matvec(np.array([0,1,2],dtype=dtype),np.array([0,1],dtype=dtype),
                   np.array([1.,2.]),np.array([.5,1.]))
