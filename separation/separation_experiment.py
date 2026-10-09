"""Reproducible common-factor spectrum / rank-r approximation experiment.

Only NumPy, SciPy, Matplotlib are required. No ambient tensor is materialized.
Run: python separation_experiment.py [--output DIRECTORY] [--plot-only]
"""
import os
# Avoid oversubscription; set these before importing numerical libraries.
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import argparse, csv, json, time, platform
from functools import reduce
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import eigvalsh, qr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 2026100703
RS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
SPECTRUM_COUNTS = {r: 96 if r<=512 else (64 if r==1024 else 32) for r in RS}
APPROX_COUNTS = {8:64,16:64,32:64,64:64,128:64,256:64,512:32}
COLORS = {0: '#29234F', 2: '#00B9D8', 3: '#254BE8', 4: '#EA5CB5'}
MARKERS = {0:'o', 2:'s', 3:'^', 4:'D'}
LABELS = {0: 'Dense Gaussian', 2: 'KR, d=2', 3: 'KR, d=3', 4: 'KR, d=4'}

def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def load_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]

def sample_scales(rng, m, d):
    return np.prod(rng.standard_normal((d-1, m)), axis=0)

def approximation(g, r, m):
    """A=diag(1_r, b I_(R-r)), b=(R-r)^(-1/2), tail squared norm=1.

    If Q spans A g, Q^T A^2 Q = b^2 I + (1-b^2) Q_head^T Q_head.
    Its largest r eigenvalues sum to r b^2+(1-b^2)||Q_head||_F^2.
    This computes rank-r error without subtracting two nearly equal SVDs.
    """
    R = g.shape[0]; b2 = 1/(R-r)
    y = g.copy(); y[r:] *= np.sqrt(b2)
    q, triangular = qr(y, mode='economic', check_finite=False)
    diagonal = np.abs(np.diag(triangular))
    # Inspect the actual unpivoted factor before using all m Q columns.
    # Column-normalized values are invariant to nonzero column rescalings;
    # the global ratio also reports the effect of highly uneven scales.
    min_relative_diagonal = float(np.min(diagonal/np.linalg.norm(y,axis=0)))
    global_diagonal_ratio = float(np.min(diagonal)/np.max(diagonal))
    rank_tolerance = float((R+m)*np.finfo(float).eps)
    assert min_relative_diagonal>rank_tolerance and global_diagonal_ratio>rank_tolerance
    capture = np.sum(q[:r]**2)
    rankr = 1+(1-b2)*(r-capture)
    projector = r+1 - (m*b2+(1-b2)*capture)
    return rankr, projector, dict(min_column_relative_R_diagonal=min_relative_diagonal,
        min_over_max_abs_R_diagonal=global_diagonal_ratio,rank_tolerance=rank_tolerance,
        numerical_full_column_rank=1)

def verify_small(rng):
    # Exact tensor contraction checked against explicit ambient probes.
    n, m = 3, 13
    tensor_error = 0.
    for d in (2,3,4):
        factors = [rng.standard_normal((n,m)) for _ in range(d)]
        explicit = np.column_stack([reduce(np.kron, [f[:,i] for f in factors]) for i in range(m)])
        support = np.column_stack([reduce(np.kron, [np.eye(n)[:,a]]+[np.eye(n)[:,0]]*(d-1)) for a in range(n)])
        restricted = factors[0]*np.prod([f[0] for f in factors[1:]], axis=0)
        tensor_error = max(tensor_error, float(np.max(np.abs(support.T@explicit-restricted))))
    r, m, R = 3, 10, 20
    g = rng.standard_normal((R,m)); b=1/np.sqrt(R-r)
    a = np.diag(np.r_[np.ones(r), np.full(R-r,b)])
    q = qr(a@g, mode='economic')[0]
    sv = np.linalg.svd(q.T@a, compute_uv=False)
    er = np.sum(a*a)-np.sum(sv[:r]**2)
    formula_error=abs(approximation(g,r,m)[0]-er)
    assert tensor_error < 1e-12 and formula_error < 1e-12
    return dict(explicit_tensor_max_error=tensor_error, rankr_formula_vs_svd_error=formula_error)

def simulate(out):
    start=time.perf_counter()
    # Separate reproducible streams prevent numerical implementation changes
    # in one experiment from changing the samples in another.
    rng_verify, rng_edges, rng_approx = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(3)]
    checks=verify_small(rng_verify)
    edges=[]
    for r in RS:
        m=3*r+1; trials=SPECTRUM_COUNTS[r]
        rank_start=time.perf_counter()
        for trial in range(trials):
            g=rng_edges.standard_normal((r,m))
            # d=2,3,4 use nested factors; all sketches share the same g.
            scalar_factors=rng_edges.standard_normal((3,m))
            for d in (0,2,3,4):
                scales=np.ones(m) if d==0 else np.prod(scalar_factors[:d-1],axis=0)
                w=scales**2; gw=g*scales
                gram=(gw@gw.T)/m
                # One reduction instead of two independent extremal calls.
                # No eigenvectors are requested.
                spectrum=eigvalsh(gram,check_finite=False,driver='evr')
                lo=float(spectrum[0]); hi=float(spectrum[-1])
                j=int(np.argmax(w)); heuristic=1+(r/m)*float(w[j])
                # Exact Rayleigh quotient in the heaviest column direction.
                x=g[:,j]/np.linalg.norm(g[:,j])
                rayleigh=float(np.dot(w,(g.T@x)**2)/m)
                edges.append(dict(r=r,m=m,d=d,trial=trial,trials=trials,lambda_min=lo,lambda_max=hi,wmax=float(w[j]),heavy_heuristic=heuristic,heavy_direction_rayleigh=rayleigh))
        save_csv(out/'spectrum_trials.csv',edges)
        print(f'spectrum r={r}, trials={trials}, seconds={time.perf_counter()-rank_start:.2f}',flush=True)
    save_csv(out/'spectrum_trials.csv',edges)
    summaries=[]
    for r in RS:
        for d in (0,2,3,4):
            subset=[row for row in edges if row['r']==r and row['d']==d]
            row=dict(r=r,m=3*r+1,d=d,trials=len(subset))
            for key in ('lambda_min','lambda_max','wmax','heavy_heuristic','heavy_direction_rayleigh'):
                for p in (10,50,90): row[f'{key}_q{p}']=float(np.percentile([x[key] for x in subset],p))
            row['heuristic_relative_error_median']=float(np.median([abs(x['heavy_heuristic']/x['lambda_max']-1) for x in subset]))
            summaries.append(row)
    save_csv(out/'spectrum_summary.csv',summaries)
    approximation_rows=[]; diagnostics=[]; maxdiff=0.
    counts=APPROX_COUNTS
    for r,trials in counts.items():
        m=3*r+1; R=2*m
        for trial in range(trials):
            g=rng_approx.standard_normal((R,m))
            rankr,projector,diag=approximation(g,r,m)
            diagnostics.append(dict(r=r,m=m,R=R,kind=0,trial=trial,**diag))
            for kind in (0,2,3,4):
                # Range equality is algebraic; do not duplicate the dense QR
                # for every coupled Gaussian sketch. Independently check QR
                # on one draw for every d and every r.
                diff=0.
                if kind and trial==0:
                    s=sample_scales(rng_approx,m,kind)
                    err,proj,diag=approximation(g*s,r,m)
                    diagnostics.append(dict(r=r,m=m,R=R,kind=kind,trial=trial,**diag))
                    diff=max(abs(err-rankr),abs(proj-projector)); maxdiff=max(maxdiff,diff)
                approximation_rows.append(dict(r=r,m=m,R=R,kind=kind,trial=trial,trials=trials,rankr_ratio=rankr,projector_ratio=projector,zero_ratio=r+1,bound=1.5,paired_check_absdiff=diff,paired_check_performed=int(kind!=0 and trial==0)))
            # Common-factor Rademacher other factors have scalar signs +-1.
            # Thus their range equals a dense iid sign range, all d.
            sign=rng_approx.choice(np.array([-1.,1.]),size=(R,m))
            er,ep,diag=approximation(sign,r,m)
            diagnostics.append(dict(r=r,m=m,R=R,kind=-1,trial=trial,**diag))
            approximation_rows.append(dict(r=r,m=m,R=R,kind=-1,trial=trial,trials=trials,rankr_ratio=er,projector_ratio=ep,zero_ratio=r+1,bound=1.5,paired_check_absdiff=0.,paired_check_performed=0))
        save_csv(out/'approximation_trials.csv',approximation_rows)
        save_csv(out/'approximation_QR_diagnostics.csv',diagnostics)
        print(f'approximation r={r}, trials={trials}',flush=True)
    save_csv(out/'approximation_trials.csv',approximation_rows)
    rows=[]
    for r,trials in counts.items():
        for kind in (0,2,3,4,-1):
            subset=[x for x in approximation_rows if x['r']==r and x['kind']==kind]
            row=dict(r=r,m=3*r+1,R=2*(3*r+1),kind=kind,trials=trials,zero_ratio=r+1,bound=1.5)
            for key in ('rankr_ratio','projector_ratio'):
                row[key+'_mean']=float(np.mean([x[key] for x in subset]))
                for p in (10,50,90): row[f'{key}_q{p}']=float(np.percentile([x[key] for x in subset],p))
            rows.append(row)
    save_csv(out/'approximation_summary.csv',rows)
    checks['max_coupled_gaussian_QR_error_difference']=maxdiff
    checks['approximation_QR_diagnostics_count']=len(diagnostics)
    checks['min_column_relative_R_diagonal']=min(row['min_column_relative_R_diagonal'] for row in diagnostics)
    checks['min_over_max_abs_R_diagonal']=min(row['min_over_max_abs_R_diagonal'] for row in diagnostics)
    checks['numerically_rank_deficient_QR_draws']=sum(1-row['numerical_full_column_rank'] for row in diagnostics)
    assert maxdiff<1e-8
    metadata=dict(seed=SEED,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,spectrum_r=RS,spectrum_trials=SPECTRUM_COUNTS,approximation_trials=counts,approximation_R='2m',tail_squared_Frobenius_norm=1,checks=checks,simulation_seconds=time.perf_counter()-start,quantile_method='NumPy linear',blas_threads=1)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')

def plot(out):
    s=load_csv(out/'spectrum_summary.csv'); a=load_csv(out/'approximation_summary.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10,'axes.titleweight':'bold','pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(9.2,6.1))
    for ax in axs.flat:
        ticks=RS[::2]
        ax.set_xscale('log',base=2); ax.set_xticks(ticks); ax.set_xticklabels(ticks)
        ax.grid(True,color='#DADDE0',alpha=.6,linewidth=.5); ax.set_axisbelow(True)
        ax.set_xlabel('Target rank r (m = 3r + 1)')
    def curve(ax,rows,key,color,label,marker='o'):
        x=[z['r'] for z in rows]; med=[z[key+'_q50'] for z in rows]
        ax.fill_between(x,[z[key+'_q10'] for z in rows],[z[key+'_q90'] for z in rows],color=color,alpha=.20,linewidth=0)
        ax.plot(x,med,color=color,marker=marker,markersize=3.5,linewidth=1.5,label=label)
    for d in (0,2,3,4):
        rows=[z for z in s if z['d']==d]
        if d:
            curve(axs[0,0],rows,'wmax',COLORS[d],LABELS[d],MARKERS[d])
        curve(axs[0,1],rows,'lambda_max',COLORS[d],LABELS[d],MARKERS[d])
        curve(axs[1,0],rows,'lambda_min',COLORS[d],LABELS[d],MARKERS[d])
        if d:
            axs[0,1].plot([z['r'] for z in rows],[z['heavy_heuristic_q50'] for z in rows],color=COLORS[d],linestyle=':',linewidth=1.3)
    axs[0,0].set(title='(a) Common-factor weights',ylabel=r'Maximum weight $\max_i w_i$',yscale='log')
    axs[0,1].set(title='(b) Upper spectral edge',ylabel=r'$\lambda_{\max}$',yscale='log')
    axs[1,0].set(title='(c) Lower spectral edge',ylabel=r'$\lambda_{\min}$',yscale='log')
    for ax,edge in ((axs[0,1],(1+np.sqrt(1/3))**2),(axs[1,0],(1-np.sqrt(1/3))**2)):
        ax.axhline(edge,color=COLORS[0],linestyle='--',linewidth=1)
    axs[0,1].text(.02,.96,'Dotted: one-heavy-column heuristic',transform=axs[0,1].transAxes,va='top',fontsize=8)
    axs[1,0].text(.97,.95,'Dashed: asymptotic dense-Gaussian edge',transform=axs[1,0].transAxes,va='top',ha='right',fontsize=7.5)
    dense=[z for z in a if z['kind']==0]
    sign=[z for z in a if z['kind']==-1]
    curve(axs[1,1],dense,'rankr_ratio',COLORS[0],'Gaussian / KR Gaussian','o')
    curve(axs[1,1],sign,'rankr_ratio','#FF925B','Rademacher common factor','s')
    axs[1,1].axhline(1.5,color='#7042DF',linestyle='--',linewidth=1.1)
    axs[1,1].text(.03,.995,'Gaussian expectation bound: 1.5',transform=axs[1,1].transAxes,va='top',fontsize=8,color='#7042DF')
    axs[1,1].text(.03,.07,r'$\tau_r(A)=1$; zero-output ratio $=r+1$',transform=axs[1,1].transAxes,fontsize=8)
    axs[1,1].set(title='(d) Rank-r approximation',ylabel=r'$\|A-\widehat A_r\|_F^2\,/\,\tau_r(A)$',ylim=(1.16,1.54),xlim=(7,570))
    axs[1,1].set_xticks(list(APPROX_COUNTS)); axs[1,1].set_xticklabels(list(APPROX_COUNTS))
    axs[1,1].legend(loc='center right',fontsize=7.8,frameon=True,facecolor='white',edgecolor='#DDDDDD')
    handles,labels=axs[0,1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.01),ncol=4,frameon=False)
    fig.text(.5,.005,'Lines: empirical medians; shading: trial 10%-90% quantiles (not confidence intervals).',ha='center',fontsize=8,color='#29234F')
    fig.tight_layout(rect=(0,.025,1,.965),h_pad=1.6,w_pad=3.2)
    fig.savefig(out/'kr_separation.pdf',bbox_inches='tight')
    fig.savefig(out/'kr_separation.png',dpi=200,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--plot-only',action='store_true')
    args=p.parse_args();args.output.mkdir(exist_ok=True,parents=True)
    if not args.plot_only: simulate(args.output)
    plot(args.output)
