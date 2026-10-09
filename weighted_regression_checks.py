"""Check scale-mixture regression identities against explicit tensor sketches.

Run: python weighted_regression_checks.py
This is a numerical implementation check, not a proof or tail simulation.
"""
import json
import numpy as np

def main():
    rng=np.random.default_rng(2026100902)
    errors=[]
    covariance_errors=[]
    slack=[]
    for d in [2,3,4]:
        r,R,m=3,7,31
        dims=[R]+[2]*(d-1)
        N=int(np.prod(dims))
        support=np.zeros((N,R))
        for a in range(R): support[a*2**(d-1),a]=1
        q,_=np.linalg.qr(rng.standard_normal((R,r+1)))
        design=support@q[:,:r]@np.diag([1.,2.,.5])
        residual=1.7*support@q[:,r]
        truth=np.array([.4,-.8,1.2])
        response=design@truth+residual
        for trial in range(32):
            factors=[rng.standard_normal((n,m)) for n in dims]
            columns=[]
            for j in range(m):
                col=factors[0][:,j]
                for factor in factors[1:]: col=np.kron(col,factor[:,j])
                columns.append(col)
            omega=np.column_stack(columns)/np.sqrt(m)
            solution=np.linalg.lstsq(omega.T@design,omega.T@response,rcond=None)[0]
            actual=np.linalg.norm(design@solution-response)**2/np.linalg.norm(residual)**2-1
            weights=np.prod(np.array([f[0]**2 for f in factors[1:]]),axis=0)
            xi=(q[:,:r].T@factors[0]).T
            eta=q[:,r]@factors[0]
            S=xi.T@(weights[:,None]*xi)
            T=xi.T@((weights**2)[:,None]*xi)
            perturbation=np.linalg.solve(S,xi.T@(weights*eta))
            errors.append(abs(actual-np.dot(perturbation,perturbation)))
            B=np.linalg.solve(S,xi.T*weights)
            Sinv=np.linalg.inv(S)
            C=Sinv@T@Sinv
            covariance_errors.append(float(np.max(abs(C-B@B.T))))
            slack.append(float(np.linalg.eigvalsh(max(weights)*Sinv-C)[0]))
    result=dict(seed=2026100902,draws=len(errors),max_relative_error_identity_error=max(errors),
                max_conditional_covariance_identity_error=max(covariance_errors),
                min_weight_cap_covariance_slack=min(slack))
    assert max(errors)<1e-10 and max(covariance_errors)<1e-10 and min(slack)>-1e-10
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
