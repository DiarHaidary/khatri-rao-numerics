"""Common-factor head with a fixed generic orthogonal tail.

Run: python head_tail_experiment.py --output rerun_head_tail
Uses NumPy, SciPy, and Matplotlib. Seed and input bases are fixed before
sketch draws. No exact output-law transfer is used.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import argparse, csv, json, platform, time, subprocess, sys
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import qr, svdvals, eigvalsh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=2026100903
RANKS=[8,16,32,64,128,256,512,1024]
COUNTS={8:128,16:128,32:128,64:128,128:96,256:64,512:32,1024:32}
FACTORS={2:[16],3:[4,4],4:[2,2,4]}
COLORS={0:'#29234F',2:'#00B9D8',3:'#254BE8',4:'#EA5CB5'}
MARKERS={0:'o',2:'s',3:'^',4:'D'}
LABELS={0:'Dense Gaussian',2:'KR, d=2',3:'KR, d=3',4:'KR, d=4'}

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def product_rows(fs):
    a=fs[0]
    for f in fs[1:]: a=(a[:,:,None]*f[:,None,:]).reshape(len(a),-1)
    return a

def measure(projected,r,m,verify):
    R=projected.shape[0]
    beta=1/np.sqrt(R-r)
    singular=np.r_[np.ones(r),np.full(R-r,beta)]
    y=singular[:,None]*projected
    q,t=qr(y,mode='economic',check_finite=False)
    diag=np.abs(np.diag(t))
    rank_margin=float(min(diag)/max(diag))
    assert rank_margin>(R+m)*np.finfo(float).eps
    head=float(np.sum(q[:r]**2))
    rank_error=1+(1-beta**2)*(r-head)
    projector_error=r+1-m*beta**2-(1-beta**2)*head
    check=0.
    if verify:
        sv=svdvals(q.T*singular,check_finite=False)
        check=abs(rank_error-(r+1-np.sum(sv[:r]**2)))
        assert check<1e-10
    eig=eigvalsh(projected[:r]@projected[:r].T,check_finite=False)
    assert rank_error>=1-1e-10 and projector_error>=-1e-10
    return dict(lambda_min=float(eig[0]),lambda_max=float(eig[-1]),
                rankr_ratio=float(rank_error),projector_ratio=float(projector_error),
                qr_rank_margin=rank_margin),check

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    seeds=np.random.SeedSequence(SEED).spawn(2*len(RANKS))
    rows=[]; orth_errors=[]; max_svd_check=0.; max_tensor_check=0.
    for k,r in enumerate([r for r in RANKS if r<=256]):
        m=3*r+1; N=16*r; R=2*m; s=R-r
        head_indices=np.arange(r)*16
        tail_indices=np.setdiff1d(np.arange(N),head_indices)
        basis_rng=np.random.default_rng(seeds[2*k])
        reduced,tri=qr(basis_rng.standard_normal((N-r,s)),mode='economic',check_finite=False)
        reduced*=np.where(np.diag(tri)>=0,1.,-1.)
        tail=np.zeros((N,s)); tail[tail_indices]=reduced
        orth_errors.append(float(np.linalg.norm(tail.T@tail-np.eye(s),2)))
        np.save(out/f'tail_basis_r{r}.npy',tail)
        streams=[np.random.default_rng(x) for x in seeds[2*k+1].spawn(5)]
        for trial in range(COUNTS[r]):
            first=streams[0].standard_normal((m,r))
            dense=np.vstack([first.T,streams[1].standard_normal((s,m))])/np.sqrt(m)
            observation,check=measure(dense,r,m,trial==0)
            max_svd_check=max(max_svd_check,check)
            rows.append(dict(r=r,m=m,ambient_dimension=N,support_dimension=R,d=0,trial=trial,**observation))
            for offset,d in enumerate([2,3,4]):
                common=[streams[offset+2].standard_normal((m,n)) for n in FACTORS[d]]
                probes=product_rows([first]+common)
                projected=np.vstack([probes[:,head_indices].T,tail.T@probes.T])/np.sqrt(m)
                if trial==0:
                    explicit=first[0]
                    for factor in common: explicit=np.kron(explicit,factor[0])
                    max_tensor_check=max(max_tensor_check,float(np.max(abs(explicit-probes[0]))))
                    assert np.allclose(projected[:r],first.T*np.prod([f[:,0] for f in common],axis=0)/np.sqrt(m),rtol=1e-12,atol=1e-12)
                observation,check=measure(projected,r,m,trial==0)
                max_svd_check=max(max_svd_check,check)
                rows.append(dict(r=r,m=m,ambient_dimension=N,support_dimension=R,d=d,trial=trial,**observation))
        print(f'Completed rank {r}: ambient {N}, {COUNTS[r]} trials per law',flush=True)
    save_csv(out/'head_tail_trials.csv',rows)
    summary=[]
    for d in [0,2,3,4]:
        for r in [r for r in RANKS if r<=256]:
            group=[v for v in rows if v['d']==d and v['r']==r]
            entry=dict(d=d,r=r,m=3*r+1,ambient_dimension=16*r,support_dimension=2*(3*r+1),trials=len(group))
            for key in ['lambda_min','lambda_max','rankr_ratio','projector_ratio']:
                values=np.array([v[key] for v in group])
                for name,p in [('q10',.1),('q50',.5),('q90',.9)]: entry[key+'_'+name]=float(np.quantile(values,p))
                entry[key+'_mean']=float(np.mean(values))
            entry['upper_above_4_fraction']=float(np.mean([v['lambda_max']>4 for v in group]))
            entry['relative_1p5_fraction']=float(np.mean([v['rankr_ratio']<=1.5 for v in group]))
            entry['joint_fraction']=float(np.mean([v['lambda_max']>4 and v['rankr_ratio']<=1.5 for v in group]))
            summary.append(entry)
    save_csv(out/'head_tail_summary.csv',summary)
    metadata=dict(seed=SEED,rank_values=[r for r in RANKS if r<=256],counts={r:COUNTS[r] for r in RANKS if r<=256},common_mode_dimensions=FACTORS,
                  ambient_multiplier=16,support_dimension='2m',sketch_size='3r+1',
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  matplotlib=matplotlib.__version__,basis_orthogonality_errors=orth_errors,
                  explicit_tensor_error=max_tensor_check,direct_compressed_svd_error=max_svd_check,
                  smallest_qr_rank_margin=min(v['qr_rank_margin'] for v in rows),seconds=time.perf_counter()-started,
                  coupling='One fixed tail basis per rank shared across orders. All laws share the first-mode Gaussian draw within each trial. Other factors and the dense tail are independent. Ranks use independent streams.',
                  interpretation='Finite-size observations, not a theorem probability. Quantile bands are trial distributions, not confidence intervals.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    subprocess.run([sys.executable,str(Path(__file__).with_name('extend_head_tail.py')),'--output',str(out),'--ranks','512','1024','--trials','32'],check=True)
    with (out/'head_tail_summary.csv').open(newline='',encoding='utf-8') as f:
        summary=[{k:(int(v) if k in ['d','r','m','ambient_dimension','support_dimension','trials'] else float(v)) for k,v in row.items()} for row in csv.DictReader(f)]
    metadata=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    plot(out,summary); inline(out,summary)
    print(json.dumps(metadata),flush=True)

def plot(out,summary):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(2,1,figsize=(10.5,8.2))
    for d in [0,2,3,4]:
        group=[v for v in summary if v['d']==d]
        for ax,key in zip(axes,['lambda_max','rankr_ratio']):
            xs=[v['r'] for v in group]
            ax.fill_between(xs,[v[key+'_q10'] for v in group],[v[key+'_q90'] for v in group],color=COLORS[d],alpha=.20,lw=0)
            ax.plot(xs,[v[key+'_q50'] for v in group],color=COLORS[d],marker=MARKERS[d],lw=1.8,label=LABELS[d])
            ax.set_xscale('log',base=2); ax.set_xticks(RANKS,labels=[str(r) for r in RANKS]); ax.grid(alpha=.2); ax.set_axisbelow(True)
            ax.set_xlabel('Target rank r (m = 3r + 1)')
    axes[0].set(title='(a) Common-factor head: upper edge',ylabel='Head Gram maximum eigenvalue',yscale='log')
    axes[0].axhline((1+1/np.sqrt(3))**2,color=COLORS[0],ls='--',lw=1)
    axes[1].set(title='(b) Same input: rank-r approximation',ylabel='Squared error / optimal tail')
    axes[1].axhline(1.5,color=COLORS[0],ls='--',lw=1)
    for ax in axes: ax.yaxis.labelpad=1
    axes[0].legend(frameon=False,fontsize=8)
    fig.tight_layout(); fig.savefig(out/'head_tail.pdf',bbox_inches='tight'); fig.savefig(out/'head_tail.png',dpi=190,bbox_inches='tight'); plt.close(fig)

def inline(out,summary):
    colors={0:'htInk',2:'htBlue',3:'htRose',4:'htViolet'}
    marks={0:'*',2:'square*',3:'triangle*',4:'diamond*'}
    lines=[r'\begin{figure}[!t]',r'\centering\begingroup',r'\definecolor{htInk}{HTML}{29234F}',r'\definecolor{htBlue}{HTML}{00B9D8}',r'\definecolor{htRose}{HTML}{254BE8}',r'\definecolor{htViolet}{HTML}{EA5CB5}',r'\begin{tikzpicture}',
           r'\begin{groupplot}[group style={group size=1 by 2,vertical sep=1.9cm},width=.86\linewidth,height=4.3cm,scale only axis,xmode=log,log basis x=2,xtick={8,16,32,64,128,256,512,1024},xticklabels={8,16,32,64,128,256,512,1024},xlabel={Target rank $r$ ($m=3r+1$)},grid=major,grid style={gray!18},tick label style={font=\scriptsize},label style={font=\small},ylabel style={xshift=6pt},title style={font=\small\bfseries},legend style={font=\scriptsize,draw=gray!25},legend columns=4]']
    for panel,key in enumerate(['lambda_max','rankr_ratio']):
        lines.append(r'\nextgroupplot[title={' + ('(a) Common-factor head' if panel==0 else '(b) Same input: approximation')+'},ylabel={'+(r'$\lambda_{\max}$ of head Gram' if panel==0 else r'$\|A-\widehat A_r\|_F^2/\tau_r(A)$')+'}'+(',ymode=log,legend to name=htLegend' if panel==0 else '')+']')
        for d in [0,2,3,4]:
            group=[v for v in summary if v['d']==d]
            def coords(q): return ' '.join(f"({v['r']},{v[key+'_'+q]:.10g})" for v in group)
            prefix=f'h{panel}{d}'
            for q in ['q10','q90']: lines.append(r'\addplot[draw=none,forget plot,name path='+prefix+q+'] coordinates {'+coords(q)+'};')
            lines.append(r'\addplot[draw=none,forget plot,fill='+colors[d]+r',fill opacity=.20] fill between[of='+prefix+'q10 and '+prefix+'q90];')
            lines.append(r'\addplot[color='+colors[d]+',thick,mark='+marks[d]+',mark size=1.7pt'+(',forget plot' if panel else '')+'] coordinates {'+coords('q50')+'};')
            if panel==0: lines.append(r'\addlegendentry{'+LABELS[d].replace('d=',r'$d=')+('$' if d else '')+'}')
        val=(1+1/np.sqrt(3))**2 if panel==0 else 1.5
        lines.append(r'\addplot[htInk,dashed,forget plot] coordinates {(8,'+str(val)+')(1024,'+str(val)+')};')
    lines.extend([r'\end{groupplot}',r'\node[anchor=south] at ([yshift=.8cm]group c1r1.north) {\pgfplotslegendfromname{htLegend}};',r'\end{tikzpicture}\endgroup',
        r'\caption{A common-factor head with a generic orthogonal tail. Here $N=16r$ (up to 16384), $R=2m$, and $m=3r+1$. The first $r$ right singular vectors form the Gaussian common-factor witness; the other $R-r$ vectors are a fixed Gaussian-QR basis in its orthogonal complement. Head singular values are one and tail singular values are $(R-r)^{-1/2}$, so $\tau_r(A)=1$ and zero output has ratio $r+1$. Mode dimensions are $(r,16)$, $(r,4,4)$, and $(r,2,2,4)$; the common-mode product is 16 throughout. One fixed input per rank is shared across all laws, with 128 trials through $r=64$, 96 at $r=128$, 64 at $r=256$, and 32 each at $r=512,1024$. Lines are medians and bands are empirical 10--90\% quantiles. The dashed line in (a) is the asymptotic dense spectral edge; in (b), it is the dense Gaussian expected-error bound $1.5$. These are not finite-draw bounds for the product sketches. Every sketch and its error are evaluated separately, without an exact output-law transfer.}',
        r'\label{fig:head-tail}',r'\end{figure}'])
    (out/'head_tail_inline.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    a=p.parse_args(); run(a.output)
