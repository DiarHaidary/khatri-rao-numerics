Expanded separation experiment
==============================
The separation/ subfolder contains the rank-varying experiment used in the
main manuscript: spectral ranks 8--2048, approximation ranks 8--512,
raw trials, QR diagnostics, reproducible Python, and embedded figure source.
See separation/README.txt. The files described below are the original
fixed-rank common-factor/Haar experiment, retained as a complementary experiment in this repository.

Reproducible Gaussian tensor-sketch experiment
=============================================

Files
-----
kr_embedding_experiment.py      Complete NumPy/Matplotlib simulation and plot source.
distortion_trials.csv           All 2,048 observations (2 orders x 2 geometries x 64 trials x 8 widths).
distortion_summary.csv          Empirical 10%, 50%, 90% quantiles and success fractions.
closed_certificates.csv         All certificate parameters and checked sufficient integer widths.
experiment_metadata.json       Seed, versions, settings, validation, and runtime.
haar_basis_d2.npy / d3.npy      The actual fixed orthonormal subspace bases.
kr_embedding_distortion.pdf    Vector figure, with embedded fonts.
kr_embedding_distortion.png    Raster preview of the same figure.
figure_caption.tex             Suggested manuscript figure and complete caption.
figure_inline.tex              Self-contained PGFPlots figure with every coordinate embedded.
build_inline_figure.py          Standard-library generator for the inline figure from the CSVs.

Run
---
Use Python 3.10+ with numpy and matplotlib installed:
    python kr_embedding_experiment.py
To preserve supplied outputs while rerunning:
    python kr_embedding_experiment.py --output rerun
To regenerate only the PDF/PNG from the existing data, without resampling:
    python kr_embedding_experiment.py --plot-only
The code requires no SciPy, external data, LaTeX installation, or GPU.
To regenerate the inline TeX fragment, run python build_inline_figure.py.
Its required preamble is graphicx, tikz, pgfplots, the groupplots and
fillbetween PGFPlots libraries, and compatibility level 1.18. The generated
fragment requires no external figures or data files at TeX compilation time.
The bundled Python used here was Python 3.12 with NumPy 2.5.3 and Matplotlib
3.11.2 from the pre-existing local scientific plotting environment. Actual versions
are recorded in experiment_metadata.json. Any compatible Python environment
with these two packages can run the commands above.

Design
------
Rank r=8; each mode dimension n=8; orders d=2,3; ambient N=n^d.
64 independent probe trials for each order; master seed 20261007.
Each order has a single fixed Haar subspace, sampled by a Gaussian QR with
positive diagonal convention. The basis is independent of the probe trials.
The other basis has columns e_a tensor e_1 tensor ... tensor e_1.
All coordinates of all factors are independent N(0,1). The common-factor
and Haar experiments share factor draws within each trial. Within each trial,
m=64,128,256,512,1024,2048,4096,8192 are nested prefixes. Thus the plotted
widths and geometries are coupled, while the 64 trials remain independent.
No random subspace is redrawn between trials; the experiment concerns
distortion on fixed subspaces, not an average over random subspace draws.

The measured quantity is ||U^T Omega Omega^T U-I_r||_2, the squared-norm
embedding distortion, where U is the fixed orthonormal subspace basis and
Omega contains the product probes scaled by m^{-1/2}. This equals
||Y^T Y/m-I_r||_2 for an internal array Y of projected probe rows; this
internal array does not use the manuscript's Y=A Omega notation.
Eigenvalues are recorded so either side of the deviation can
be inspected. Quantiles use NumPy's default linear interpolation. The shaded
bands describe the empirical trial distribution and are not confidence
intervals. Observed success fractions are descriptive; no finite-trial
probability guarantee is inferred from them.

Certificate
-----------
The computation uses the manuscript's explicit closed certificate:
K^2=8/3, theta=1-K^{-2}, v=[2 theta K^4]^d r,
c_star=sqrt(12 sqrt(3)/7), q=ceil(log(4r/delta)),
rho=c_star K^{2d}(2q)^{d-2}(r+2q)+1+4/(3v),
h_q=2(q-1)/sqrt(2q-3), eta=(r/delta)^{1/(2q)},
c_q=[(2q-1)!!]^{1/(2q)}, A=eta c_q sqrt(v), B=eta c_q h_q rho.
At epsilon=0.5 and delta=0.1, q=6. The integer width is
ceil(((A+sqrt(A^2+4 epsilon B))/(2 epsilon))^2).
This yields 33,968 for d=2 and 489,430 for d=3. The source checks that
the allowance is <=epsilon at this width and >epsilon at its predecessor.
The choice q=ceil(log(4r/delta)) is explicit and unoptimized. No constants
are fit to observations and no tighter certificate is implied.
Neither sufficient width was simulated. They appear in a separate log-width
strip, together with the actual sampled range, to avoid suggesting empirical
sharpness or extrapolated observations.

Selected observations
---------------------
At m=8192, common-factor/Haar median distortions are
0.100323/0.0699375 (d=2), and 0.174980/0.0793172 (d=3).
The first sampled widths with empirical success fraction >=0.90 at
epsilon=0.5 are 1024/256 for common-factor/Haar at d=2 and 2048/1024
at d=3. These are grid-dependent descriptive summaries of 64 trials;
they are not certified widths or estimates with a stated confidence level.
The common-factor geometry is more distorted than the chosen Haar geometry
in these finite-size data, especially at order three. Fixed rank cannot
demonstrate or distinguish r log^{d-1}(r) asymptotic rank dependence.

Verification and cost
---------------------
For d=2 and d=3, independently generated small instances (n=r=3, m=17)
compare vectorized product rows against explicit iterative numpy.kron rows,
the common-factor shortcut against the explicit ambient basis, and the
eigenvalue distortion against a direct spectral norm. Maximum errors are
0, 8.88e-16, and 2.22e-16, respectively. The saved Haar bases have checked
orthogonality error below 1e-12. In the local run, generation and numerical
analysis took about 1.6 seconds; plot export adds a few seconds. See metadata
for the measured time of the final run. The final vector PDF was rendered
with Poppler and visually inspected; all labels, bands, and markers are legible.
