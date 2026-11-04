"""Full-linear BDSVM with lossless sparse storage and UNCHANGED statistic gossip.

Identity is a coordinate change from a full orthonormal linear pre-image basis.
Each message still logically contains the complete dense C_k and d_k, including
zeros, and is charged at the original dense float64 size. CSR is only storage
and local linear algebra; no factor, new reduction or hidden sender data is used.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator,cg
from src.unified_quality import LinearNoBiasBDSVM


class DenseWireMatrix:
    """A dense wire matrix whose zero entries need no resident RAM in simulator."""
    def __init__(self,matrix):
        self.csr=sp.csr_matrix(matrix,dtype=np.float64)
        self.csr.sum_duplicates();self.csr.eliminate_zeros();self.csr.sort_indices()
        self.shape=self.csr.shape
        self.nbytes=int(np.prod(self.shape))*8
        for buffer in [self.csr.data,self.csr.indices,self.csr.indptr]:buffer.flags.writeable=False
    @property
    def storage_nbytes(self):
        return self.csr.data.nbytes+self.csr.indices.nbytes+self.csr.indptr.nbytes


class IdentityBasis:
    def __init__(self,d):self.shape=(d,d)
    @property
    def T(self):return self
    def __matmul__(self,v):return np.asarray(v).copy()


class NativeSparseLinearBDSVM(LinearNoBiasBDSVM):
    def __init__(self,*args,cg_rtol=1e-8,cg_maxiter=5000,**kwargs):
        super().__init__(*args,**kwargs)
        self.cg_rtol=cg_rtol;self.cg_maxiter=cg_maxiter;self.cg_iterations_total=0
        self.last_linear_residual=None;self.max_local_statistic_storage=0
    def clone(self):
        return type(self)(gossip_k=self.gossip_k,P=self.P,C=self.C,rho=self.rho,
            gamma=self.gamma,gamma_mult=self.gamma_mult,preimage=self.preimage,
            arch_seed=self.arch_seed,ey_floor=self.ey_floor,kernel=self.kernel,
            cg_rtol=self.cg_rtol,cg_maxiter=self.cg_maxiter)
    def configure_network(self,node_id,n_global,n_workers,local_steps=None,step_scale=None):
        if self.kernel!='linear' or self.P!=self.d:
            raise ValueError('Lossless identity coordinates require full linear P=d')
        self.node_id=int(node_id);self.n_global=int(n_global);self.n_workers=int(n_workers)
        self.p=IdentityBasis(self.d);self.Km=self.X;self.Kpp=sp.eye(self.d,format='csr')
        self.kernel_ready=True;self._publish()
    def _publish(self):
        residual=1-self.y*(self.X@self.beta)
        a=np.where(residual<0.,0.,2*self.C/np.maximum(residual,self.ey_floor))
        statistic=DenseWireMatrix(self.X.T@self.X.multiply(a[:,None]))
        rhs=np.asarray(self.X.T@(a*self.y)).ravel()
        rhs.flags.writeable=False
        self.max_local_statistic_storage=max(self.max_local_statistic_storage,statistic.storage_nbytes)
        self.version+=1;self.contributions=dict(self.contributions)
        self.contributions[self.node_id]=(self.version,(statistic,rhs),float(self.n))
    def local_update(self):
        if len(self.contributions)<self.n_workers:return
        # Only received sufficient statistics are used here, never another
        # worker's training rows. Transport and version-merging stay inherited.
        A=self.Kpp.copy();b=np.zeros(self.P)
        for _,(matrix,rhs),_ in self.contributions.values():
            A=A+matrix.csr;b+=rhs
        diagonal=A.diagonal();assert np.all(diagonal>=1.)
        M=LinearOperator(A.shape,matvec=lambda v:v/diagonal,dtype=np.float64)
        def count(_):self.cg_iterations_total+=1
        options=dict(x0=self.beta,M=M,maxiter=self.cg_maxiter,callback=count)
        try:new,info=cg(A,b,rtol=self.cg_rtol,atol=0.,**options)
        except TypeError:new,info=cg(A,b,tol=self.cg_rtol,**options)
        residual=float(np.linalg.norm(A@new-b));tolerance=self.cg_rtol*float(np.linalg.norm(b))
        if info!=0 or residual>max(1e-10,2*tolerance):
            raise RuntimeError(f'Native sparse IRWLS numerical solve failed: info={info}, residual={residual}, tolerance={tolerance}')
        self.last_linear_residual=residual
        previous=self.beta;self.beta=self.rho*previous+(1-self.rho)*new
        self.rel_change=float(np.linalg.norm(self.beta-previous))/max(float(np.linalg.norm(previous)),1e-30)
        self._publish()
