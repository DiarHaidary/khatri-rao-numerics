"""Finite-sample regression inflation under Gaussian product weights.

Run: python regression_finite_size.py --output rerun_regression
For each design/weight draw, tr(C_m) integrates out the Gaussian residual
coordinates exactly. The Monte Carlo mean of tr(C_m) estimates expected
excess error with less noise than sampling residual coordinates too.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[key]='1'
import argparse,csv,json,platform,time
from pathlib import Path
import numpy as np

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    seed=2026100904; r=10; m=300; trials=4000
    rows=[]; summary=[]; start=time.perf_counter()
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    for d,rng in zip([1,2,3,4],streams):
        mu2=3**(d-1)
        values=[]
        for trial in range(trials):
            weights=np.prod(rng.standard_normal((d-1,m))**2,axis=0) if d>1 else np.ones(m)
            xi=rng.standard_normal((m,r))
            S=xi.T@(weights[:,None]*xi)
            B=np.linalg.solve(S,xi.T*weights)
            conditional=float(np.sum(B**2))
            normalized=m*conditional/(r*mu2)
            values.append(normalized)
            rows.append(dict(d=d,r=r,m=m,trial=trial,mu2=mu2,conditional_mean_excess=conditional,normalized_conditional_mean=normalized))
        se=float(np.std(values,ddof=1)/np.sqrt(trials))
        summary.append(dict(d=d,r=r,m=m,trials=trials,mu2=mu2,weight_fourth_moment=105**(d-1),
                            normalized_mean=float(np.mean(values)),monte_carlo_standard_error=se))
    dense_expected=m/(m-r-1)
    assert abs(summary[0]['normalized_mean']-dense_expected)<6*summary[0]['monte_carlo_standard_error']
    save_csv(out/'regression_trials.csv',rows); save_csv(out/'regression_summary.csv',summary)
    metadata=dict(seed=seed,rank=r,sketch_size=m,trials=trials,orders=[1,2,3,4],
                  python=platform.python_version(),numpy=np.__version__,seconds=time.perf_counter()-start,
                  estimator='Rao-Blackwellized: tr(C_m), exact conditional expected excess, averaged over independent weight/design draws.',
                  normalization='Divide estimated excess by r E[w^2]/m. This denominator is an asymptotic reference, not an exact mean formula.',
                  dense_exact_normalized_expectation=dense_expected,
                  uncertainty='Reported standard errors are Monte Carlo standard errors, not convergence-rate guarantees.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    run(p.parse_args().output)
