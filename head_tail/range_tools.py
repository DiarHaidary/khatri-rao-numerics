"""Exact, memory-conscious operations for two-level-spectrum inputs."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]=os.environ.get('KR_BLAS_THREADS','4')
import numpy as np
from scipy.linalg import cholesky,solve_triangular,qr,svdvals
from scipy.linalg.lapack import dgeqrf,dorgqr

def thin_haar(matrix):
    """Positive-diagonal Gaussian QR, forming Q without allocating a full R."""
    a=np.asfortranarray(matrix)
    size=int(dgeqrf(a,lwork=-1,overwrite_a=True)[2][0])
    a,tau,_,info=dgeqrf(a,lwork=size,overwrite_a=True)
    if info: raise RuntimeError(('dgeqrf',info))
    signs=np.where(np.diag(a)>=0,1.,-1.)
    size=int(dorgqr(a,tau,lwork=-1,overwrite_a=True)[1][0])
    a,_,info=dorgqr(a,tau,lwork=size,overwrite_a=True)
    if info: raise RuntimeError(('dorgqr',info))
    a*=signs
    return a

def error_from_head_tail(head,tail,verify_qr=False):
    """Exact Woodbury identity; tail is rescaled in place by column norms.

    For Y=[H; T/sqrt(s)], let K=T^T T/s and C=I+H K^{-1} H^T.
    Squared optimal rank-r error is 1+(1-1/s) tr(C^{-1}).
    Normalizing complete columns changes no range and stabilizes K.
    """
    r,m=head.shape; s=tail.shape[0]
    if not (r<=m<=s): raise ValueError('This exact identity requires r <= m <= tail dimension.')
    scale=np.sqrt(np.sum(tail*tail,axis=0)/s)
    if not np.all(scale>0): raise ValueError('Zero tail column')
    h=np.asfortranarray(head/scale)
    tail/=scale
    k=np.empty((m,m),dtype=float,order='F')
    np.matmul(tail.T,tail,out=k); k/=s
    l=cholesky(k,lower=True,overwrite_a=True,check_finite=False)
    tail_margin=float(min(np.diag(l))/max(np.diag(l)))
    b=solve_triangular(l,h.T,lower=True,check_finite=False)
    c=np.empty((r,r),dtype=float,order='F')
    np.matmul(b.T,b,out=c); c.flat[::r+1]+=1
    lc=cholesky(c,lower=True,overwrite_a=True,check_finite=False)
    inverse=solve_triangular(lc,np.eye(r),lower=True,check_finite=False)
    deficit=float(np.sum(inverse*inverse))
    rank_error=1+(1-1/s)*deficit
    projector_error=rank_error-(m-r)/s
    info=dict(tail_cholesky_diagonal_ratio=tail_margin,
              head_cholesky_diagonal_ratio=float(min(np.diag(lc))/max(np.diag(lc))),
              qr_error_difference='',qr_rank_margin='')
    if verify_qr:
        y=np.empty((r+s,m),order='F')
        y[:r]=h; y[r:]=tail/np.sqrt(s)
        q,t=qr(y,mode='economic',overwrite_a=True,check_finite=False)
        comparison=1+(1-1/s)*(r-float(np.sum(q[:r]**2)))
        info['qr_error_difference']=abs(rank_error-comparison)
        diagonal=np.abs(np.diag(t))
        info['qr_rank_margin']=float(min(diagonal)/max(diagonal))
        assert info['qr_error_difference']<1e-7
        assert info['qr_rank_margin']>(r+s+m)*np.finfo(float).eps
    assert rank_error>=1 and projector_error>=-1e-10
    return dict(rankr_ratio=rank_error,projector_ratio=projector_error,qr_rank_margin=info['qr_rank_margin']),info

def validate():
    rng=np.random.default_rng(2026100906)
    r,m,P=16,49,16; s=2*m-r
    gaussian=rng.standard_normal((15*r,s))
    basis=thin_haar(gaussian.copy())
    standard,t=qr(gaussian,mode='economic')
    standard*=np.where(np.diag(t)>=0,1.,-1.)
    qr_difference=float(np.max(abs(basis-standard)))
    first=rng.standard_normal((m,r))
    factors=[rng.standard_normal((m,n)) for n in [2,2,4]]
    common=factors[0]
    for f in factors[1:]: common=(common[:,:,None]*f[:,None,:]).reshape(m,-1)
    actual=(first[:,:,None]*common[:,None,:]).reshape(m,-1)
    indices=np.setdiff1d(np.arange(r*P),np.arange(r)*P)
    direct_tail=basis.T@actual[:,indices].T
    blocked=np.zeros((s,m))
    for k in range(15): blocked+=(basis[k::15].T@first.T)*common[:,k+1]
    contraction_difference=float(np.max(abs(blocked-direct_tail)))
    head=actual[:,np.arange(r)*P].T
    result,diagnostic=error_from_head_tail(head,direct_tail.copy(),True)
    sigma=np.r_[np.ones(r),np.full(s,1/np.sqrt(s))]
    y=sigma[:,None]*np.vstack([head,direct_tail])
    q=qr(y,mode='economic')[0]
    singular=svdvals(q.T*sigma)
    direct=r+1-float(np.sum(singular[:r]**2))
    svd_difference=abs(result['rankr_ratio']-direct)
    assert qr_difference<1e-12 and contraction_difference<1e-11 and svd_difference<1e-10
    return dict(haar_vs_scipy_qr_error=qr_difference,blocked_contraction_error=contraction_difference,
                compressed_svd_error=svd_difference,**diagnostic)
