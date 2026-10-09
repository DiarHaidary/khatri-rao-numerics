"""Fixed generic-support approximation experiment (seed 2026100901).

Run: python generic_approximation.py --output DIRECTORY [--trials 128]
Requires NumPy and Matplotlib. All input bases are fixed before probe draws.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 2026100901
RANK, SUPPORT = 8, 48
WIDTHS = [12, 16, 24, 32, 40]
LAWS = ['Dense Gaussian', 'KR Gaussian', 'KR Rademacher']
COLORS = ['#29234F', '#254BE8', '#FF925B']
MARKERS = ['o', 's', '^']

def product_rows(factors):
    result = factors[0]
    for factor in factors[1:]:
        result = (result[:, :, None] * factor[:, None, :]).reshape(len(result), -1)
    return result

def errors(projected, verify=False):
    singular = np.r_[np.ones(RANK), np.full(SUPPORT-RANK, 1/np.sqrt(SUPPORT-RANK))]
    y = singular[:, None] * projected
    q, sv, _ = np.linalg.svd(y, full_matrices=False)
    cutoff = max(y.shape) * np.finfo(float).eps * sv[0]
    sample_rank = int(np.count_nonzero(sv > cutoff))
    q = q[:, :sample_rank]
    head_energy = np.sum(q[:RANK] ** 2)
    beta2 = 1/(SUPPORT-RANK)
    # Use the compressed SVD for every draw; do not assume full sample rank.
    compressed_sv = np.linalg.svd(q.T * singular, compute_uv=False)
    rankr = RANK + 1 - np.sum(compressed_sv[:RANK] ** 2)
    projector = RANK + 1 - np.sum(compressed_sv ** 2)
    check = 0.0
    if verify:
        b = q.T * singular
        ub, sb, vhb = np.linalg.svd(b, full_matrices=False)
        actual = q @ ((ub[:, :RANK] * sb[:RANK]) @ vhb[:RANK])
        direct = np.linalg.norm(np.diag(singular) - actual, 'fro') ** 2
        check = abs(rankr - direct)
        if sample_rank >= RANK:
            check = max(check, abs(rankr - (1+(1-beta2)*(RANK-head_energy))))
        assert check < 1e-10
    assert rankr >= 1-1e-10 and projector >= -1e-10 and projector <= rankr+1e-10
    return rankr, projector, sample_rank, check

def save_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def run(out, trials):
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    seeds = np.random.SeedSequence(SEED).spawn(5)
    rows, basis_errors = [], []
    max_check = max_tensor_check = 0.0
    for idx, (d, n) in enumerate([(2, 8), (3, 4)]):
        basis_rng = np.random.default_rng(seeds[idx])
        basis, triangular = np.linalg.qr(basis_rng.standard_normal((n**d, SUPPORT)))
        basis *= np.where(np.diag(triangular) >= 0, 1, -1)
        basis_errors.append(float(np.linalg.norm(basis.T@basis-np.eye(SUPPORT), 2)))
        np.save(out/f'fixed_basis_d{d}.npy', basis)
        streams = [np.random.default_rng(x) for x in seeds[idx+2].spawn(3)]
        for trial in range(trials):
            for law_idx, (law, rng) in enumerate(zip(LAWS, streams)):
                if law_idx == 0:
                    probes = rng.standard_normal((max(WIDTHS), n**d))
                else:
                    factors = [rng.standard_normal((max(WIDTHS), n)) if law_idx == 1
                               else rng.choice([-1., 1.], size=(max(WIDTHS), n)) for _ in range(d)]
                    probes = product_rows(factors)
                    if trial == 0:
                        explicit = factors[0][0]
                        for factor in factors[1:]:
                            explicit = np.kron(explicit, factor[0])
                        max_tensor_check = max(max_tensor_check, float(np.max(abs(probes[0]-explicit))))
                projected = basis.T @ probes.T
                for m in WIDTHS:
                    rankr, projector, sample_rank, check = errors(projected[:, :m]/np.sqrt(m), trial == 0)
                    max_check = max(max_check, check)
                    rows.append(dict(d=d, mode_dimension=n, r=RANK, support_dimension=SUPPORT,
                                     m=m, trial=trial, law=law, rankr_ratio=float(rankr),
                                     projector_ratio=float(projector), sample_rank=sample_rank))
    save_csv(out/'generic_trials.csv', rows)
    summaries = []
    for d in [2, 3]:
        for law in LAWS:
            for m in WIDTHS:
                group = [x for x in rows if x['d']==d and x['law']==law and x['m']==m]
                summary = dict(d=d, law=law, m=m, trials=len(group))
                for key in ['rankr_ratio', 'projector_ratio']:
                    values = [x[key] for x in group]
                    for label, quantile in [('q10', .1), ('q50', .5), ('q90', .9)]:
                        summary[f'{key}_{label}'] = float(np.quantile(values, quantile))
                    summary[f'{key}_mean'] = float(np.mean(values))
                summaries.append(summary)
    save_csv(out/'generic_summary.csv', summaries)
    metadata = dict(seed=SEED, python=platform.python_version(), numpy=np.__version__,
                    matplotlib=matplotlib.__version__, trials=trials, rank=RANK, support=SUPPORT,
                    widths=WIDTHS, ambient_dimension=64, orders=[2,3], mode_dimensions=[8,4],
                    basis_orthogonality_errors=basis_errors, explicit_tensor_error=max_tensor_check,
                    direct_svd_error=max_check, seconds=time.perf_counter()-start,
                    coupling='Nested widths within a trial and law; separate streams across laws and orders.',
                    interpretation='Empirical trial quantiles, not confidence intervals. No exact Gaussian range law is assumed.')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    plot(out, summaries)
    inline(out, summaries, trials)
    print(json.dumps(metadata))

def plot(out, summaries):
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype':42, 'ps.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(10,3.7),sharey=True)
    for ax, d in zip(axes, [2,3]):
        for law, color, marker in zip(LAWS, COLORS, MARKERS):
            group=[x for x in summaries if x['d']==d and x['law']==law]
            ax.fill_between(WIDTHS,[x['rankr_ratio_q10'] for x in group],[x['rankr_ratio_q90'] for x in group],color=color,alpha=.20,linewidth=0)
            ax.plot(WIDTHS,[x['rankr_ratio_q50'] for x in group],color=color,marker=marker,lw=1.8,label=law)
        ax.set(title=f'Generic fixed support, d = {d}',xlabel='Sketch size m',xticks=WIDTHS)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Squared rank-8 error / optimal tail')
    axes[1].legend(frameon=False,fontsize=9)
    fig.tight_layout()
    fig.savefig(out/'generic_approximation.pdf',bbox_inches='tight')
    fig.savefig(out/'generic_approximation.png',dpi=190,bbox_inches='tight')
    plt.close(fig)

def inline(out, summaries, trials):
    colors = ['genericInk','genericBlue','genericRose']
    marks = ['*','square*','triangle*']
    lines=[r'\begin{figure}[H]',r'\centering\begingroup',
           r'\definecolor{genericInk}{HTML}{29234F}',r'\definecolor{genericBlue}{HTML}{254BE8}',
           r'\definecolor{genericRose}{HTML}{FF925B}',r'\begin{tikzpicture}',
           r'\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.55cm},width=.39\linewidth,height=4.2cm,scale only axis,xlabel={Sketch size $m$},xtick={12,16,24,32,40},grid=major,grid style={gray!18},tick label style={font=\scriptsize},label style={font=\small},title style={font=\small\bfseries},legend style={font=\scriptsize,draw=none,at={(.97,.97)},anchor=north east}]']
    for d in [2,3]:
        extra=r',ylabel={$\|A-\widehat A_8\|_F^2/\tau_8(A)$}' if d==2 else ''
        lines.append(r'\nextgroupplot[title={Generic fixed support, $d='+str(d)+'$}'+extra+']')
        for li,(law,color,mark) in enumerate(zip(LAWS,colors,marks)):
            group=[x for x in summaries if x['d']==d and x['law']==law]
            def coords(k):
                return ' '.join(f"({x['m']},{x['rankr_ratio_'+k]:.10g})" for x in group)
            for q in ['q10','q90']:
                lines.append(r'\addplot[draw=none,forget plot,name path=g'+f'{d}{li}{q}'+'] coordinates {'+coords(q)+'};')
            lines.append(r'\addplot[draw=none,forget plot,fill='+color+r',fill opacity=.20] fill between[of=g'+f'{d}{li}q10 and g{d}{li}q90'+'];')
            lines.append(r'\addplot[color='+color+',thick,mark='+mark+',mark size=1.7pt] coordinates {'+coords('q50')+'};')
            if d==3:
                lines.append(r'\addlegendentry{'+law+'}')
    lines += [r'\end{groupplot}\end{tikzpicture}\endgroup',
              r'\caption{Approximation on two fixed generic supports, with no exact Gaussian range law assumed. Each input is $A=\Sigma V^\top$, where $V\in\mathbb R^{64\times48}$ is a fixed orthonormal Gaussian-QR basis drawn independently of the probes. The first eight singular values are one and the remaining forty are $1/\sqrt{40}$, so $\tau_8(A)=1$ and zero output has ratio nine. Mode dimensions are $8\times8$ and $4\times4\times4$. Lines are medians and bands are empirical 10--90\% quantiles from '+str(trials)+r' trials per law and order. Widths are nested within each trial; the three laws use independent streams.}',
              r'\label{fig:generic-approximation}',r'\end{figure}']
    (out/'generic_inline.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--trials',type=int,default=128)
    args=parser.parse_args()
    run(args.output,args.trials)
