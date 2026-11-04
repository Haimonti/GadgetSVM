"""Optional compiled float64 local loops; no fastmath or internal RNG.

All row draws are supplied by the original caller. JIT warm-up is performed
before timing, uniformly across benchmark children.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def dual_steps(ptr, idx, val, y, alpha, w, draws, lam_n, sigma, step_scale):
    a, v = alpha.copy(), w.copy()
    for i in draws:
        norm, dot = 0., 0.
        for q in range(ptr[i], ptr[i+1]):
            norm += val[q]*val[q]; dot += v[idx[q]]*val[q]
        if norm == 0.:
            # This coordinate has no quadratic term, but its dual linear term
            # is still maximized at unsigned alpha=1. It changes no primal w.
            a[i] += step_scale * (y[i] - a[i])
            continue
        old = a[i]*y[i]
        new = min(1., max(0., old+(1-y[i]*dot)*lam_n/(sigma*norm)))
        delta = step_scale*(y[i]*new-a[i])
        a[i] += delta
        for q in range(ptr[i], ptr[i+1]):
            v[idx[q]] += sigma*delta/lam_n*val[q]
    return a, v


@njit(cache=True)
def pegasos_steps(ptr, idx, val, y, w, draws, lam, base):
    u=w.copy();scale=1.
    for j in range(len(draws)):
        i=draws[j]; eta=1./(lam*(base+j+1))
        margin=0.
        for q in range(ptr[i],ptr[i+1]):margin+=scale*u[idx[q]]*val[q]
        margin*=y[i]
        scale*=1.-eta*lam
        if abs(scale)<1e-12:
            u[:]=0.;scale=1.
        if margin<1.:
            for q in range(ptr[i],ptr[i+1]):u[idx[q]]+=(eta*y[i]/scale)*val[q]
    return scale*u


@njit(cache=True)
def cola_steps(ptr,idx,val,y,alpha,b,curvature,steps,state,d):
    a=alpha.copy();v=np.zeros(d);norms=np.zeros(len(y))
    for i in range(len(y)):
        for q in range(ptr[i],ptr[i+1]):
            v[idx[q]]+=a[i]*val[q];norms[i]+=val[q]*val[q]
    for _ in range(steps):
        state^=(state<<13)&0xffffffff;state^=state>>17
        state^=(state<<5)&0xffffffff;state&=0xffffffff
        i=(state%0x80000000)%len(y)
        if norms[i]==0:
            coefficient=y[i]*(y[i]+b[i])
            if coefficient>0:a[i]=y[i]
            elif coefficient<0:a[i]=0.
            continue
        dot=0.
        for q in range(ptr[i],ptr[i+1]):dot+=val[q]*v[idx[q]]
        change=(y[i]+b[i]-curvature*dot)/(curvature*norms[i])
        new=y[i]*min(1.,max(0.,y[i]*(a[i]+change)))
        for q in range(ptr[i],ptr[i+1]):v[idx[q]]+=(new-a[i])*val[q]
        a[i]=new
    return a


def warmup():
    ptr=np.array([0,2],dtype=np.int32);idx=np.array([0,1],dtype=np.int32)
    val=np.array([1.,2.]);y=np.array([1.]);a=np.array([0.]);w=np.zeros(2)
    draws=np.array([0,0],dtype=np.int64)
    dual_steps(ptr,idx,val,y,a,w,draws,1.,1.,1.)
    pegasos_steps(ptr,idx,val,y,w,draws,.1,0)
    cola_steps(ptr,idx,val,y,a,a,.1,2,42,2)
