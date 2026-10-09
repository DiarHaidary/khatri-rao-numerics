"""Oversampling sweep on one fixed common-factor-head/generic-tail input.

Run: python oversampling_experiment.py --output rerun_oversampling
Uses actual product probes. All widths are nested prefixes on one input.
"""
import os
os.environ.setdefault('KR_BLAS_THREADS','1')
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]=os.environ['KR_BLAS_THREADS']
import argparse,csv,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np
import scipy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'head_tail'))
from range_tools import thin_haar,error_from_head_tail,validate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=2026100907
RANK=32
P=512
SUPPORT=1536
CONTROL_SUPPORT=480
RATIOS=[1.5,2,3,4,6,8,12]
WIDTHS=[int(x*RANK) for x in RATIOS]
FACTORS={2:[512],3:[16,32],4:[8,8,8]}
COLORS={0:'#29234F',2:'#00B9D8',3:'#254BE8',4:'#EA5CB5'}
MARKERS={0:'o',2:'s',3:'^',4:'D'}
LABELS={0:'Dense Gaussian',2:'KR, d=2',3:'KR, d=3',4:'KR, d=4'}

def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def product_rows(factors):
    result=factors[0]
    for f in factors[1:]: result=(result[:,:,None]*f[:,None,:]).reshape(len(result),-1)
    return result

def run(out,trials,support_rank=SUPPORT):
    out.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    validation=validate(); seeds=np.random.SeedSequence(SEED).spawn(6)
    basis_rng=np.random.default_rng(seeds[0])
    s=support_rank-RANK; N=RANK*P; maximum=max(WIDTHS)
    if s<maximum or support_rank>N: raise ValueError('Require max(m)+r <= R <= N for this exact solver.')
    basis=thin_haar(basis_rng.standard_normal((N-RANK,s)))
    orth=float(np.max(abs(basis.T@basis-np.eye(s))))
    digest=hashlib.sha256(basis.tobytes(order='C')).hexdigest()
    assert orth<1e-11
    streams=[np.random.default_rng(x) for x in seeds[1:]]
    rows=[]; diagnostics=[]; comparison=[]
    controls=[CONTROL_SUPPORT] if CONTROL_SUPPORT<support_rank else []
    for trial in range(trials):
        first=streams[0].standard_normal((maximum,RANK))
        dense=streams[1].standard_normal((s,maximum))
        for m in WIDTHS:
            error,info=error_from_head_tail(first[:m].T,dense[:,:m].copy(),trial==0)
            rows.append(dict(d=0,r=RANK,m=m,m_over_r=m/RANK,capture_fraction=m/support_rank,ambient_dimension=N,support_dimension=support_rank,trial=trial,**error))
            diagnostics.append(dict(d=0,m=m,trial=trial,**info))
            if m in [96,384]:
                comparison.append(dict(d=0,r=RANK,m=m,m_over_r=m/RANK,support_dimension=support_rank,capture_fraction=m/support_rank,trial=trial,rankr_ratio=error['rankr_ratio']))
                for control in controls:
                    small,_=error_from_head_tail(first[:m].T,dense[:control-RANK,:m].copy(),trial==0)
                    comparison.append(dict(d=0,r=RANK,m=m,m_over_r=m/RANK,support_dimension=control,capture_fraction=m/control,trial=trial,rankr_ratio=small['rankr_ratio']))
        common={d:product_rows([streams[j+2].standard_normal((maximum,n)) for n in FACTORS[d]]) for j,d in enumerate([2,3,4])}
        for d in [2,3,4]:
            head=first.T*common[d][:,0]
            actual_tail_coordinates=(first[:,:,None]*common[d][:,None,1:]).reshape(maximum,N-RANK)
            tail=basis.T@actual_tail_coordinates.T
            del actual_tail_coordinates
            for m in WIDTHS:
                error,info=error_from_head_tail(head[:,:m],tail[:,:m].copy(),trial==0 and d==4)
                rows.append(dict(d=d,r=RANK,m=m,m_over_r=m/RANK,capture_fraction=m/support_rank,ambient_dimension=N,support_dimension=support_rank,trial=trial,**error))
                diagnostics.append(dict(d=d,m=m,trial=trial,**info))
                if d==4 and m in [96,384]:
                    comparison.append(dict(d=d,r=RANK,m=m,m_over_r=m/RANK,support_dimension=support_rank,capture_fraction=m/support_rank,trial=trial,rankr_ratio=error['rankr_ratio']))
                    for control in controls:
                        small,_=error_from_head_tail(head[:,:m],tail[:control-RANK,:m].copy(),trial==0)
                        comparison.append(dict(d=d,r=RANK,m=m,m_over_r=m/RANK,support_dimension=control,capture_fraction=m/control,trial=trial,rankr_ratio=small['rankr_ratio']))
            del tail
        if (trial+1)%16==0: print(f'Oversampling: {trial+1}/{trials} paired trials; {time.perf_counter()-started:.1f}s',flush=True)
    save_csv(out/'oversampling_trials.csv',rows); save_csv(out/'solver_diagnostics.csv',diagnostics)
    save_csv(out/'rank_comparison_trials.csv',comparison)
    compared=[]
    for d in [0,4]:
        for m in [96,384]:
            for support in sorted(set([support_rank]+controls)):
                values=[v['rankr_ratio'] for v in comparison if v['d']==d and v['m']==m and v['support_dimension']==support]
                compared.append(dict(d=d,r=RANK,m=m,m_over_r=m/RANK,support_dimension=support,capture_fraction=m/support,
                                     trials=len(values),median=float(np.median(values)),q10=float(np.quantile(values,.1)),q90=float(np.quantile(values,.9))))
    save_csv(out/'rank_comparison_summary.csv',compared)
    largest_increase=max(float(np.max(np.diff([v['rankr_ratio'] for v in rows if v['d']==d and v['trial']==trial]))) for d in [0,2,3,4] for trial in range(trials))
    assert largest_increase<=1e-10
    summary=[]
    for d in [0,2,3,4]:
        for m in WIDTHS:
            group=[v for v in rows if v['d']==d and v['m']==m]
            values=np.array([v['rankr_ratio'] for v in group])
            item=dict(d=d,r=RANK,m=m,m_over_r=m/RANK,support_dimension=support_rank,capture_fraction=m/support_rank,trials=len(group),median=float(np.median(values)),
                      q10=float(np.quantile(values,.1)),q90=float(np.quantile(values,.9)),mean=float(np.mean(values)))
            for limit in [1.5,1.2,1.1]: item['fraction_below_'+str(limit).replace('.','p')]=float(np.mean(values<=limit))
            summary.append(item)
    save_csv(out/'oversampling_summary.csv',summary)
    metadata=dict(seed=SEED,rank=RANK,ambient_dimension=N,support_dimension=support_rank,common_product=P,
                  common_mode_dimensions=FACTORS,widths=WIDTHS,ratios=RATIOS,trials=trials,
                  basis_shape=[N-RANK,s],basis_seed_spawn_key=list(seeds[0].spawn_key),basis_sha256_compact_C=digest,
                  basis_gram_max_error=orth,validation=validation,
                  max_qr_error_difference=max(float(v['qr_error_difference']) for v in diagnostics if v['qr_error_difference']!=''),
                  minimum_tail_cholesky_ratio=min(v['tail_cholesky_diagonal_ratio'] for v in diagnostics),
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,
                  seconds=time.perf_counter()-started,blas_threads=int(os.environ['KR_BLAS_THREADS']),
                  maximum_error_increase_across_nested_widths=largest_increase,
                  minimum_capture_fraction=min(WIDTHS)/support_rank,maximum_capture_fraction=max(WIDTHS)/support_rank,
                  rank_controls=controls,rank_control_design='Smaller tail is the prefix of the larger fixed Haar tail. Same probes and head; each input has unit optimal squared tail.',
                  design=f'One fixed Haar tail and the same two-level spectrum for every width and law. Input rank is {support_rank}. Prefixes nested within each trial and law; first-mode Gaussian matrix shared across laws.',
                  interpretation='Empirical medians and trial quantiles on one fixed input. Threshold crossings do not establish theorem-level sample complexity.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    plot(out,summary); inline(out,summary,trials)
    print('Support-rank comparison:',json.dumps(compared,indent=2),flush=True)
    print('Sweep medians:',json.dumps([{k:v for k,v in row.items() if k in ['d','m_over_r','median','q90']} for row in summary]),flush=True)

def plot(out,summary):
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(figsize=(10,4.6))
    for d in [0,2,3,4]:
        group=[v for v in summary if v['d']==d]
        ax.fill_between(RATIOS,[v['q10'] for v in group],[v['q90'] for v in group],color=COLORS[d],alpha=.20,lw=0)
        ax.plot(RATIOS,[v['median'] for v in group],color=COLORS[d],marker=MARKERS[d],lw=1.8,label=LABELS[d])
    ax.axhline(1.5,color='#7042DF',ls='--',lw=1.2,label='Target relative error: 1.5')
    ax.set_xscale('log',base=2); ax.set_yscale('log')
    ax.set_xticks(RATIOS,labels=[str(v) for v in RATIOS])
    ax.set_yticks([1,1.5,2,3,5,10,15],labels=['1','1.5','2','3','5','10','15'])
    support=int(summary[0]['support_dimension'])
    ax.set(xlabel='Oversampling ratio m/r',ylabel='Squared rank-r error / optimal tail',title=f'Oversampling on one fixed input (r = 32; R = {support}; common-mode product 512)')
    ax.grid(alpha=.2); ax.set_axisbelow(True); ax.legend(fontsize=9,frameon=False)
    fig.tight_layout(); fig.savefig(out/'oversampling.pdf',bbox_inches='tight'); fig.savefig(out/'oversampling.png',dpi=190,bbox_inches='tight'); plt.close(fig)

def inline(out,summary,trials):
    support=int(summary[0]['support_dimension'])
    maximum_fraction=max(v['capture_fraction'] for v in summary)
    colors={0:'osInk',2:'osCyan',3:'osBlue',4:'osPink'}
    marks={0:'*',2:'square*',3:'triangle*',4:'diamond*'}
    parts=[r'\begin{figure}[H]',r'\centering\begingroup',r'\definecolor{osInk}{HTML}{29234F}',r'\definecolor{osCyan}{HTML}{00B9D8}',r'\definecolor{osBlue}{HTML}{254BE8}',r'\definecolor{osPink}{HTML}{EA5CB5}',r'\definecolor{osRef}{HTML}{7042DF}',r'\begin{tikzpicture}',
           r'\begin{axis}[width=.84\linewidth,height=5.3cm,scale only axis,xmode=log,log basis x=2,ymode=log,xtick={1.5,2,3,4,6,8,12},xticklabels={1.5,2,3,4,6,8,12},ytick={1,1.5,2,3,5,10,15},yticklabels={1,1.5,2,3,5,10,15},xlabel={Oversampling ratio $m/r$},ylabel={$\|A-\widehat A_r\|_F^2/\tau_r(A)$},ylabel style={xshift=6pt},grid=major,grid style={gray!18},tick label style={font=\scriptsize},label style={font=\small},legend style={font=\scriptsize,draw=none,at={(.97,.97)},anchor=north east}]']
    for d in [0,2,3,4]:
        group=[v for v in summary if v['d']==d]
        def coords(key): return ' '.join(f"({v['m_over_r']},{v[key]:.10g})" for v in group)
        for key in ['q10','q90']: parts.append(r'\addplot[draw=none,forget plot,name path=os'+str(d)+key+'] coordinates {'+coords(key)+'};')
        parts.append(r'\addplot[draw=none,forget plot,fill='+colors[d]+r',fill opacity=.20] fill between[of=os'+str(d)+'q10 and os'+str(d)+'q90];')
        parts.append(r'\addplot[color='+colors[d]+',thick,mark='+marks[d]+',mark size=1.8pt] coordinates {'+coords('median')+'};')
        parts.append(r'\addlegendentry{'+LABELS[d].replace('d=',r'$d=')+('$' if d else '')+'}')
    parts.extend([r'\addplot[osRef,dashed] coordinates {(1.5,1.5)(12,1.5)};',r'\addlegendentry{Target error $1.5$}',r'\end{axis}\end{tikzpicture}\endgroup',
                  r'\caption{Oversampling on one fixed input. Target rank is $r=32$, ambient dimension $N=16384$, and input rank $R='+str(support)+r'$ throughout the sweep. The head singular values are one and the tail singular values are $1/\sqrt{'+str(support-RANK)+r'}$, so the optimal squared tail is one. Common modes are $(512)$, $(16,32)$, and $(8,8,8)$ for orders $2,3,4$, all with product 512. Widths are nested prefixes of each trial and satisfy $m/R\le '+f'{maximum_fraction:g}'+r'$. Lines are medians and bands are empirical 10--90\% quantiles from '+str(trials)+r' independent trials per law. The dashed line marks relative error $1.5$; observed crossings are descriptive, not sample-complexity guarantees.}',
                  r'\label{fig:oversampling}',r'\end{figure}'])
    (out/'oversampling_inline.tex').write_text('\n'.join(parts)+'\n',encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent); p.add_argument('--trials',type=int,default=96)
    p.add_argument('--support-rank',type=int,default=SUPPORT)
    a=p.parse_args(); run(a.output,a.trials,a.support_rank)
