import unittest
import numpy as np
import scipy.sparse as sp
from src.peersim_python.core import Network,Control
from src.peersim_python.communication import numeric_payload_nbytes
from src.bdsvm_simulation import BDSVMSimulation
from src.config import CONFIG
from src.unified_quality import LinearNoBiasBDSVM
from p2p.native_sparse_bdsvm import DenseWireMatrix,NativeSparseLinearBDSVM

class NativeSparseTests(unittest.TestCase):
    def test_parallel_csr_matvec_preserves_empty_rows_and_row_summation(self):
        from p2p.native_csr_matvec import warmup,csr_matvec
        warmup(3)
        rng=np.random.RandomState(91)
        A=sp.random(150,170,density=.08,random_state=rng,format='csr',data_rvs=rng.standard_normal)
        A.data[:10]=0.;A[50:52]=0.;A.eliminate_zeros()
        for _ in range(10):
            vector=rng.standard_normal(A.shape[1])
            np.testing.assert_allclose(csr_matvec(A.indptr,A.indices,A.data,vector),A@vector,atol=1e-12,rtol=1e-12)

    def test_storage_is_sparse_but_dense_wire_zeros_are_charged(self):
        value=DenseWireMatrix(sp.eye(100,format='csr'))
        self.assertEqual(value.nbytes,80000)
        self.assertEqual(numeric_payload_nbytes({'stats':{0:(1,(value,np.ones(100)),10.)}}),80000+800+24)
        self.assertLess(value.storage_nbytes,value.nbytes/10)
        self.assertFalse(value.csr.data.flags.writeable)

    def test_native_gossip_trajectory_and_every_byte_equal_dense_full_basis(self):
        from p2p.native_csr_matvec import warmup
        warmup(3)
        rng=np.random.RandomState(24);X=sp.csr_matrix(rng.randn(100,4));y=np.where(rng.randn(100)>0,1.,-1.)
        shards=[dict(X_csr=X[ix],y=y[ix],X_test=X[:0],y_test=y[:0],n_local=len(ix)) for ix in np.array_split(np.arange(100),10)]
        c=dict(CONFIG);c.update(NUM_WORKERS=10,SEED=42,TOPOLOGY='ring',GOSSIP_K=2,
            WARM_START=False,LAMBDA=.1,STOP_ON_PLATEAU=False,STOP_ON_THRESHOLD=False)
        traces=[]
        variants=[(LinearNoBiasBDSVM,{}),(NativeSparseLinearBDSVM,{}),
                  (NativeSparseLinearBDSVM,{'preconditioner':'none'}),
                  (NativeSparseLinearBDSVM,{'linear_solver':'cholesky','preconditioner':'none'}),
                  (NativeSparseLinearBDSVM,{'linear_solver':'cholesky','preconditioner':'none','matrix_storage':'dense'}),
                  (NativeSparseLinearBDSVM,{'preconditioner':'none','matvec_backend':'numba'})]
        for protocol,solver_options in variants:
            rows=[]
            class Capture(Control):
                def execute(self):
                    ps=[Network.get(i).getProtocol(1) for i in range(10)]
                    rows.append((np.stack([p.p.T@p.beta for p in ps]),sum(p.comm_bytes for p in ps)))
                    return False
            params=dict(P=4,C=1/(2*.1*100),rho=.5,kernel='linear',preimage='unit',arch_seed=0,ey_floor=1e-4,gossip_k=2)
            params.pop('gossip_k')
            params.update(solver_options)
            assembly=BDSVMSimulation(c,shards,params=params,protocol_cls=protocol).build()
            assembly.sim.controls=[Capture()];assembly.sim.cycles=20;assembly.sim.nextExperiment();traces.append(rows)
        for alternative in traces[1:]:
            for (dense,dense_bytes),(sparse,sparse_bytes) in zip(traces[0],alternative):
                np.testing.assert_allclose(sparse,dense,atol=1e-8,rtol=1e-8)
                self.assertEqual(dense_bytes,sparse_bytes)

if __name__=='__main__':unittest.main()
