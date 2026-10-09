"""Exact-product head/tail sensitivity to common-mode dimensions.

Run: python mode_sensitivity.py --output rerun_sensitivity [--save-bases]
One fixed input per common-mode product. No independent-tail approximation.
Larger bases are regenerated from recorded seeds; --save-bases exports them.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse,csv,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import qr,svdvals

SEED=2026100905
RANK=32
WIDTH=3*RANK+1
SUPPORT=2*WIDTH
CASES=[(16,),(4,4),(16,16),(32,32),(2,2,4),(8,8,8)]

def product_rows(factors):
    result=factors[0]
    for f in factors[1:]:
        result=(result[:,:,None]*f[:,None,:]).reshape(len(result),-1)
    return result

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def tail_basis(product,basis_seeds):
    N=RANK*product; s=SUPPORT-RANK
    if product==16:
        original=Path(__file__).resolve().parents[1]/'head_tail/tail_basis_r32.npy'
        if original.exists():
            return np.load(original),dict(source='head_tail/tail_basis_r32.npy')
        seed=np.random.SeedSequence(2026100903).spawn(12)[4]
        info=dict(source='regenerated main-experiment basis',seed=2026100903,spawn_key=list(seed.spawn_key))
    else:
        seed=basis_seeds[product]
        info=dict(source='Gaussian QR',seed=SEED,spawn_key=list(seed.spawn_key))
    rng=np.random.default_rng(seed)
    q,t=qr(rng.standard_normal((N-RANK,s)),mode='economic',check_finite=False)
    q*=np.where(np.diag(t)>=0,1.,-1.)
    basis=np.zeros((N,s))
    indices=np.setdiff1d(np.arange(N),np.arange(RANK)*product)
    basis[indices]=q
    return basis,info

def measure(projected,verify=False):
    beta=1/np.sqrt(SUPPORT-RANK)
    sigma=np.r_[np.ones(RANK),np.full(SUPPORT-RANK,beta)]
    y=sigma[:,None]*projected
    q,t=qr(y,mode='economic',check_finite=False)
    diagonal=np.abs(np.diag(t))
    margin=float(min(diagonal)/max(diagonal))
    assert margin>(SUPPORT+WIDTH)*np.finfo(float).eps
    error=1+(1-beta**2)*(RANK-float(np.sum(q[:RANK]**2)))
    assert error>=1-1e-10
    diff=0.
    if verify:
        singular=svdvals(q.T*sigma,check_finite=False)
        direct=RANK+1-float(np.sum(singular[:RANK]**2))
        diff=abs(error-direct)
        assert diff<1e-10
    return float(error),margin,diff

def run(out,trials,save_bases):
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    child=np.random.SeedSequence(SEED).spawn(4)
    basis_seeds=dict(zip([256,512,1024],child[:3]))
    products=sorted({int(np.prod(c)) for c in CASES})
    bases={}; basis_info=[]
    for P in products:
        basis,info=tail_basis(P,basis_seeds)
        orth=float(np.linalg.norm(basis.T@basis-np.eye(SUPPORT-RANK),2))
        assert orth<1e-11
        assert np.max(np.abs(basis[np.arange(RANK)*P]))==0
        bases[P]=basis
        info.update(common_product=P,shape=list(basis.shape),orthogonality_error=orth,
                    sha256_float64_C=hashlib.sha256(basis.tobytes(order='C')).hexdigest())
        basis_info.append(info)
        if save_bases: np.save(out/f'tail_basis_product_{P}.npy',basis)
    # The same first coordinates are shared across mode-size cases, so the
    # raw head sketch is exactly the same for every size within a fixed order.
    rng=np.random.default_rng(child[3])
    rows=[]; worst_svd=0.; worst_tensor=0.; worst_scale_error=0.; worst_scaled_range=0.; worst_head=0.
    max_factors=[max(c[j] if len(c)>j else 1 for c in CASES) for j in range(3)]
    for trial in range(trials):
        first=rng.standard_normal((WIDTH,RANK))
        common=[rng.standard_normal((WIDTH,n)) for n in max_factors]
        heads={}
        for dims in CASES:
            P=int(np.prod(dims)); d=len(dims)+1; N=RANK*P
            factors=[first]+[common[j][:,:n] for j,n in enumerate(dims)]
            raw=product_rows(factors)
            projected=np.vstack([raw[:,np.arange(RANK)*P].T,bases[P].T@raw.T])/np.sqrt(WIDTH)
            head=projected[:RANK]
            if d in heads: worst_head=max(worst_head,float(np.max(np.abs(head-heads[d]))))
            else: heads[d]=head.copy()
            error,margin,check=measure(projected,trial==0)
            worst_svd=max(worst_svd,check)
            angular=np.ones(WIDTH)
            for f,n in zip(factors[1:],dims):
                angular*=n*f[:,0]**2/np.sum(f*f,axis=1)
            assert np.max(angular)<=P*(1+1e-12)
            if trial==0:
                explicit=first[0]
                for f in factors[1:]: explicit=np.kron(explicit,f[0])
                worst_tensor=max(worst_tensor,float(np.max(abs(explicit-raw[0]))))
                # Directly generate spherical factors and compare the actual
                # restricted sketches and approximation errors under the coupling.
                normalized=[]; scales=np.ones(WIDTH)
                for f in factors:
                    norm=np.linalg.norm(f,axis=1)
                    scales*=norm/np.sqrt(f.shape[1])
                    normalized.append(f*np.sqrt(f.shape[1])/norm[:,None])
                sph=product_rows(normalized)
                spherical=np.vstack([sph[:,np.arange(RANK)*P].T,bases[P].T@sph.T])/np.sqrt(WIDTH)
                worst_scale_error=max(worst_scale_error,float(np.max(abs(projected/scales[None,:]-spherical))))
                sph_error,_,_=measure(spherical)
                worst_scaled_range=max(worst_scaled_range,abs(error-sph_error))
            rows.append(dict(d=d,common_modes='x'.join(map(str,dims)),common_product=P,r=RANK,m=WIDTH,
                             ambient_dimension=N,support_dimension=SUPPORT,trial=trial,rankr_ratio=error,
                             qr_rank_margin=margin,max_angular_head_multiplier=float(np.max(angular))))
        if (trial+1)%32==0: print(f'Completed {trial+1}/{trials} paired trials',flush=True)
    assert worst_head<1e-12 and worst_scale_error<1e-10 and worst_scaled_range<1e-10
    save_csv(out/'sensitivity_trials.csv',rows)
    summary=[]
    for dims in CASES:
        key='x'.join(map(str,dims)); P=int(np.prod(dims))
        values=np.array([v['rankr_ratio'] for v in rows if v['common_modes']==key])
        summary.append(dict(d=len(dims)+1,common_modes=key,common_product=P,r=RANK,m=WIDTH,
                            ambient_dimension=RANK*P,support_dimension=SUPPORT,trials=trials,
                            error_q10=float(np.quantile(values,.1)),error_median=float(np.median(values)),
                            error_q90=float(np.quantile(values,.9)),error_mean=float(np.mean(values)),
                            angular_second_moment=float(np.prod([3*n/(n+2) for n in dims]))))
    save_csv(out/'sensitivity_summary.csv',summary)
    metadata=dict(seed=SEED,trials=trials,rank=RANK,sketch_size=WIDTH,support_dimension=SUPPORT,
                  common_mode_cases=[list(c) for c in CASES],python=platform.python_version(),
                  numpy=np.__version__,scipy=scipy.__version__,basis_info=basis_info,
                  explicit_tensor_error=worst_tensor,compressed_svd_error=worst_svd,
                  head_difference_within_order=worst_head,gaussian_spherical_coupling_error=worst_scale_error,
                  gaussian_spherical_approximation_difference=worst_scaled_range,
                  minimum_qr_rank_margin=min(v['qr_rank_margin'] for v in rows),seconds=time.perf_counter()-started,
                  coupling='A fixed tail basis per common-mode product. Nested common-factor Gaussian coordinates and the same first-mode draw are shared across cases within each trial. Cases have different ambient inputs; trials are independent.',
                  interpretation='Descriptive comparison on fixed inputs. Does not prove monotonicity in dimensions, a limiting error, or a universal head-to-tail ratio bound.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--trials',type=int,default=128)
    p.add_argument('--save-bases',action='store_true')
    a=p.parse_args(); run(a.output,a.trials,a.save_bases)
