"""Full-linear BDSVM with lossless storage and UNCHANGED statistic gossip.

Identity is a coordinate change from a full orthonormal linear pre-image basis.
Each message still logically contains the complete dense C_k and d_k, including
zeros, and is charged at the original dense float64 size. CSR/dense storage and
local solver choices do not send a factor or use hidden sender data/reductions.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator,cg
from src.unified_quality import LinearNoBiasBDSVM


class DenseWireMatrix:
    """Dense wire-size contract, independent of lossless local representation."""
    def __init__(self,matrix,storage='csr'):
        self.dense=None;self.csr=None
        if storage=='dense':
            self.dense=np.asarray(matrix.toarray() if sp.issparse(matrix) else matrix,dtype=np.float64)
            self.dense.flags.writeable=False;self.shape=self.dense.shape
        else:
            self.csr=sp.csr_matrix(matrix,dtype=np.float64)
            self.csr.sum_duplicates();self.csr.eliminate_zeros();self.csr.sort_indices()
            self.shape=self.csr.shape
            for buffer in [self.csr.data,self.csr.indices,self.csr.indptr]:buffer.flags.writeable=False
        self.nbytes=int(np.prod(self.shape))*8
    @property
    def storage_nbytes(self):
        return self.dense.nbytes if self.dense is not None else self.csr.data.nbytes+self.csr.indices.nbytes+self.csr.indptr.nbytes


class IdentityBasis:
    def __init__(self,d):self.shape=(d,d)
    @property
    def T(self):return self
    def __matmul__(self,v):return np.asarray(v).copy()


class NativeSparseLinearBDSVM(LinearNoBiasBDSVM):
    def __init__(self,*args,cg_rtol=1e-8,cg_maxiter=5000,
                 linear_solver='cg',preconditioner='jacobi',matrix_storage='csr',
                 matvec_backend='scipy',**kwargs):
        super().__init__(*args,**kwargs)
        if linear_solver not in ['cg','cholesky'] or preconditioner not in ['jacobi','none']:
            raise ValueError('Unsupported native linear-system solver configuration')
        self.linear_solver=linear_solver;self.preconditioner=preconditioner
        if matrix_storage not in ['csr','dense']:raise ValueError('Unsupported statistic storage')
        self.matrix_storage=matrix_storage
        if matvec_backend not in ['scipy','numba'] or (matvec_backend=='numba' and (matrix_storage!='csr' or linear_solver!='cg')):
            raise ValueError('Parallel CSR matvec requires CSR storage and CG')
        self.matvec_backend=matvec_backend
        self.cg_rtol=cg_rtol;self.cg_maxiter=cg_maxiter;self.cg_iterations_total=0
        self.linear_solves=0;self.statistic_formations=0
        self.last_linear_residual=None;self.max_local_statistic_storage=0
    def clone(self):
        return type(self)(gossip_k=self.gossip_k,P=self.P,C=self.C,rho=self.rho,
            gamma=self.gamma,gamma_mult=self.gamma_mult,preimage=self.preimage,
            arch_seed=self.arch_seed,ey_floor=self.ey_floor,kernel=self.kernel,
            cg_rtol=self.cg_rtol,cg_maxiter=self.cg_maxiter,
            linear_solver=self.linear_solver,preconditioner=self.preconditioner,matrix_storage=self.matrix_storage,
            matvec_backend=self.matvec_backend)
    def configure_network(self,node_id,n_global,n_workers,local_steps=None,step_scale=None):
        if self.kernel!='linear' or self.P!=self.d:
            raise ValueError('Lossless identity coordinates require full linear P=d')
        self.node_id=int(node_id);self.n_global=int(n_global);self.n_workers=int(n_workers)
        self.p=IdentityBasis(self.d);self.Km=self.X.toarray() if self.matrix_storage=='dense' else self.X
        self.Kpp=sp.eye(self.d,format='csr')
        self.kernel_ready=True;self._publish()
    def _publish(self):
        self.statistic_formations+=1
        residual=1-self.y*(self.X@self.beta)
        a=np.where(residual<0.,0.,2*self.C/np.maximum(residual,self.ey_floor))
        matrix=self.Km.T@(a[:,None]*self.Km) if self.matrix_storage=='dense' else self.X.T@self.X.multiply(a[:,None])
        statistic=DenseWireMatrix(matrix,storage=self.matrix_storage)
        rhs=np.asarray(self.X.T@(a*self.y)).ravel()
        rhs.flags.writeable=False
        self.max_local_statistic_storage=max(self.max_local_statistic_storage,statistic.storage_nbytes)
        self.version+=1;self.contributions=dict(self.contributions)
        self.contributions[self.node_id]=(self.version,(statistic,rhs),float(self.n))
    def local_update(self):
        if len(self.contributions)<self.n_workers:return
        # Only received sufficient statistics are used here, never another
        # worker's training rows. Transport and version-merging stay inherited.
        A=np.eye(self.P) if self.matrix_storage=='dense' else self.Kpp.copy();b=np.zeros(self.P)
        for _,(matrix,rhs),_ in self.contributions.values():
            if self.matrix_storage=='dense':A+=matrix.dense
            else:A=A+matrix.csr
            b+=rhs
        diagonal=A.diagonal();assert np.all(diagonal>=1.)
        self.linear_solves+=1
        if self.linear_solver=='cholesky':
            from scipy.linalg import cho_factor,cho_solve
            dense=np.array(A,order='F',copy=True) if self.matrix_storage=='dense' else A.toarray(order='F')
            factor=cho_factor(dense,lower=True,overwrite_a=True,check_finite=False)
            new=cho_solve(factor,b,check_finite=False);info=0
        else:
            M=LinearOperator(A.shape,matvec=lambda v:v/diagonal,dtype=np.float64) if self.preconditioner=='jacobi' else None
            operator=A
            if self.matvec_backend=='numba':
                from p2p.native_csr_matvec import csr_matvec
                operator=LinearOperator(A.shape,matvec=lambda v:csr_matvec(A.indptr,A.indices,A.data,np.asarray(v).reshape(-1)),dtype=np.float64)
            def count(_):self.cg_iterations_total+=1
            options=dict(x0=self.beta,M=M,maxiter=self.cg_maxiter,callback=count)
            try:new,info=cg(operator,b,rtol=self.cg_rtol,atol=0.,**options)
            except TypeError:new,info=cg(operator,b,tol=self.cg_rtol,**options)
        residual=float(np.linalg.norm(A@new-b));tolerance=self.cg_rtol*float(np.linalg.norm(b))
        if info!=0 or residual>max(1e-10,2*tolerance):
            raise RuntimeError(f'Native sparse IRWLS numerical solve failed: info={info}, residual={residual}, tolerance={tolerance}')
        self.last_linear_residual=residual
        previous=self.beta;self.beta=self.rho*previous+(1-self.rho)*new
        self.rel_change=float(np.linalg.norm(self.beta-previous))/max(float(np.linalg.norm(previous)),1e-30)
        self._publish()
