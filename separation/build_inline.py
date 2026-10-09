"""Standard-library-only generator; all plot coordinates are embedded."""
import csv, hashlib
from pathlib import Path

root=Path(__file__).resolve().parent
def read(name):
    with (root/name).open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
s=read('spectrum_summary.csv'); a=read('approximation_summary.csv')
def coord(rows,key): return ' '.join(f"({z['r']},{z[key]})" for z in rows)
colors={0:'sepNavy',2:'sepCyan',3:'sepBlue',4:'sepPink',-1:'sepOrange'}
markers={0:'*',2:'square*',3:'triangle*',4:'diamond*',-1:'square*'}
names={0:'Dense Gaussian',2:'KR, $d=2$',3:'KR, $d=3$',4:'KR, $d=4$'}
parts=[r'''% Generated from summary CSVs by build_inline.py; no external data required.
% Preamble: \usepackage{tikz,pgfplots}; \usepgfplotslibrary{groupplots,fillbetween}
% \pgfplotsset{compat=1.18}
\begin{figure}[t]
\centering
\begingroup
\definecolor{sepNavy}{HTML}{29234F}
\definecolor{sepCyan}{HTML}{00B9D8}
\definecolor{sepBlue}{HTML}{254BE8}
\definecolor{sepPink}{HTML}{EA5CB5}
\definecolor{sepOrange}{HTML}{FF925B}
\definecolor{sepReference}{HTML}{7042DF}
\begin{tikzpicture}
\begin{groupplot}[
group style={group size=2 by 2,horizontal sep=2.10cm,vertical sep=2.05cm},
width=.370\linewidth,height=4.2cm,scale only axis,
xmode=log,log basis x=2,xmin=7,xmax=2280,
xtick={8,32,128,512,2048},xticklabels={8,32,128,512,2048},
xlabel={Target rank $r$ ($m=3r+1$)},
grid=major,grid style={gray!18},tick label style={font=\scriptsize},
label style={font=\small},title style={font=\small\bfseries},
legend style={font=\scriptsize,draw=gray!25,fill=white},
]
''']
def curve(rows,key,d,prefix,legend=None):
    color=colors[d]
    for p in (10,90):
        parts.append(f"\\addplot[draw=none,name path={prefix}{p},forget plot] coordinates {{{coord(rows,key+'_q'+str(p))}}};\n")
    parts.append(f"\\addplot[draw=none,fill={color},fill opacity=.20,forget plot] fill between[of={prefix}10 and {prefix}90];\n")
    parts.append(f"\\addplot[color={color},thick,mark={markers[d]},mark size=1.4pt] coordinates {{{coord(rows,key+'_q50')}}};\n")
    if legend: parts.append('\\addlegendentry{'+legend+'}\n')
    else: parts[-1]=parts[-1].replace('mark size=1.4pt]','mark size=1.4pt,forget plot]')
parts.append(r'\nextgroupplot[title={(a) Common-factor weights},ymode=log,ylabel={Maximum weight $\max_i w_i$}]'+'\n')
for d in (2,3,4): curve([z for z in s if int(z['d'])==d],'wmax',d,f'w{d}')
parts.append(r'\nextgroupplot[title={(b) Upper spectral edge},ymode=log,ylabel={$\lambda_{\max}$},legend to name=sepLegend,legend columns=4]'+'\n')
for d in (0,2,3,4):
    rows=[z for z in s if int(z['d'])==d]
    curve(rows,'lambda_max',d,f'u{d}',names[d])
    if d: parts.append(f"\\addplot[color={colors[d]},dotted,thick,forget plot] coordinates {{{coord(rows,'heavy_heuristic_q50')}}};\n")
parts.append(r'\addplot[sepNavy,dashed,forget plot] coordinates {(8,2.488033871712585)(2048,2.488033871712585)};'+'\n')
parts.append(r'\node[anchor=north west,font=\scriptsize] at (rel axis cs:0.02,.97) {Dotted: heavy-column heuristic};'+'\n')
parts.append(r'\nextgroupplot[title={(c) Lower spectral edge},ymode=log,ylabel={$\lambda_{\min}$}]'+'\n')
for d in (0,2,3,4): curve([z for z in s if int(z['d'])==d],'lambda_min',d,f'l{d}')
parts.append(r'\addplot[sepNavy,dashed,forget plot] coordinates {(8,.1786327949540818)(2048,.1786327949540818)};'+'\n')
parts.append(r'\node[anchor=north east,font=\scriptsize] at (rel axis cs:.98,.97) {Dashed: asymptotic dense edge};'+'\n')
parts.append(r'\nextgroupplot[title={(d) Rank-$r$ approximation},ylabel={$\|A-\widehat A_r\|_F^2/\tau_r(A)$},ymin=1.16,ymax=1.54,xmax=570,xtick={8,16,32,64,128,256,512},xticklabels={8,16,32,64,128,256,512},legend style={at={(.98,.5)},anchor=east}]'+'\n')
for d,name in ((0,'Gaussian / KR Gaussian'),(-1,'Rademacher common factor')):
    curve([z for z in a if int(z['kind'])==d],'rankr_ratio',d,f'a{abs(d)}',name)
parts.append(r'\addplot[sepReference,dashed,forget plot] coordinates {(8,1.5)(512,1.5)};'+'\n')
parts.append(r'\node[anchor=north west,font=\scriptsize,text=sepReference,inner sep=0pt] at (axis cs:8.5,1.528) {Gaussian expectation bound: $1.5$};'+'\n')
parts.append(r'\node[anchor=south west,font=\scriptsize] at (rel axis cs:.02,.02) {$\tau_r(A)=1$; zero-output ratio $=r+1$};'+'\n')
parts.append(r'''\end{groupplot}
\node[anchor=south] at ([yshift=1.05cm]group c1r1.north east) {\pgfplotslegendfromname{sepLegend}};
\end{tikzpicture}%
\endgroup
\caption{Embedding and approximation at $m=3r+1$ on common-factor supports.
Lines are medians; bands are empirical 10--90\% trial quantiles, not confidence
intervals. Panels (a)--(c) use 96 trials through $r=512$, 64 at $r=1024$,
and 32 at $r=2048$.
Weights $w_i$ are products of $d-1$ independent squared standard normals.
Dotted upper-edge curves show the heuristic $1+(r/m)\max_iw_i$;
the dashed spectral lines are asymptotic dense-Gaussian edges.
Panel (d) uses $R=2m$ singular values $1$ ($r$ repetitions) and
$(R-r)^{-1/2}$ ($R-r$ repetitions), so the tail mass is one and zero output
has ratio $r+1$. Coupled Gaussian outputs coincide by the exact range law;
the additional Rademacher output equals a dense sign rangefinder.
The $1.5$ line bounds Gaussian expected error, not every draw or sign-sketch
error. Approximation uses 64 trials through $r=256$ and 32 at $r=512$.}
\label{fig:kr-separation-numerics}
\end{figure}
''')
hashes=''.join(f'% SHA256 {name}: {hashlib.sha256((root/name).read_bytes()).hexdigest()}\n' for name in ('spectrum_summary.csv','approximation_summary.csv'))
(root/'separation_inline.tex').write_text(hashes+''.join(parts),encoding='utf-8')
print('Wrote separation_inline.tex')
