Common-factor separation experiment (2026-10-07)
================================================

Purpose and files
-----------------
This reproducible experiment isolates the upper/lower spectrum and rank-r approximation
at linear sample count m=3r+1. The old fixed-r common-factor/Haar experiment
is complementary and has not been changed.

separation_experiment.py    Complete simulation, checks, and Matplotlib plotting.
build_inline.py            Standard-library generator for an embedded figure.
spectrum_trials.csv       Every individual spectral observation (3,072 rows).
spectrum_summary.csv      Empirical 10%, 50%, 90% quantiles and heuristic audit.
approximation_trials.csv  Every paired/output observation (2,080 rows).
approximation_summary.csv Empirical quantiles and means for both error objectives.
approximation_QR_diagnostics.csv Actual QR rank diagnostics (853 factorizations).
computed_summary.txt      Computed large-size results and checks.
report_results.py         Standard-library CSV verification and summary builder.
metadata.json             Seed, versions, trial counts, validation and runtime.
kr_separation.pdf          Matplotlib vector figure with embedded fonts.
kr_separation.png          Preview of the same figure.
separation_inline.tex      Fully embedded PGFPlots figure with caption and label.

Run with Python 3.10+ and NumPy, SciPy, Matplotlib:
    python separation_experiment.py
    python build_inline.py
    python report_results.py
To retain supplied results, use --output rerun; copy build_inline.py to that
directory if regenerating its embedded plot. --plot-only regenerates PDF/PNG
from existing summary data without drawing new samples.
The main manuscript can embed separation_inline.tex with no data files or PDF.
Its preamble needs graphicx, tikz, pgfplots, and PGFPlots libraries groupplots
and fillbetween, compat=1.18. 

Design and coupling
-------------------
Seed 2026100703; three NumPy SeedSequence child streams independently handle
validation, spectrum, and approximation. Trial index is paired only within
an experiment and target rank. Different ranks use fresh random draws;
this experiment does not use nested sketch prefixes.

Spectrum uses r=8,16,32,64,128,256,512,1024,2048 and m=3r+1. Each trial draws
G in R^{r x m} with independent standard normal coordinates. Dense Gaussian
uses G/sqrt(m). The d-factor product sketch on a common-factor coordinate
support has restricted matrix G diag(s)/sqrt(m), with
    s_i = product of d-1 independent N(0,1) scalars; w_i=s_i^2.
This is the actual projected product-probe law, not a fitted heavy-tailed law.
No full ambient tensor needs to be stored. Every sketch shares G within a
trial. KR d=2,3,4 additionally share nested scalar factors within that trial.
There are 96 independent trials per r through 512, 64 at 1024, and 32 at 2048.
Quantiles use NumPy linear interpolation. Bands describe trial distributions,
NOT confidence intervals or theorem-level probabilities. The plotted curves
are empirical medians, including in the approximation panel.

Each spectral row records lambda_min and lambda_max of GD^2G^T/m,
max w_i, the heuristic 1+(r/m)max w_i, and the exact Rayleigh quotient
in the heaviest-column direction. The Rayleigh quotient is an actual lower
bound for lambda_max. The heuristic is not a rigorous bound. Dense spectral
reference lines are the r/m -> 1/3 limits (1 +/- sqrt(1/3))^2;
they are neither finite-size deterministic bounds nor measured constants.

Approximation uses R=2m>m and a diagonal input with singular values
1 (r repetitions), b=(R-r)^(-1/2) (R-r repetitions). Its full right support
is the R-dimensional common-factor support. Thus rank(A)=R>m and
tau_r(A)=||A-A_r||_F^2=1; zero-output ratio is r+1 (9 through 513).
For Q spanning A G, define Ahat_r=Q (Q^T A)_r. The plotted objective is
||A-Ahat_r||_F^2 / tau_r(A). The projector objective is separately recorded;
it must not be confused with the rank-r output.

For numerical efficiency the rank-r error uses the exact identity
    ||A-Ahat_r||_F^2 = 1+(1-b^2)(r-||Q_head||_F^2).
This follows from Q^T A^2 Q=b^2 I+(1-b^2)Q_head^T Q_head: its top r
eigenvalues sum to r b^2+(1-b^2)||Q_head||_F^2. A direct small SVD checks
the identity. The projector error equals
    r+1 - [m b^2+(1-b^2)||Q_head||_F^2].

Gaussian KR rows couple to the same full-support G and report exactly the
same range-derived output, as guaranteed algebraically when every s_i!=0.
One independently recomputed weighted QR per d and r checks floating-point
agreement; the check flag distinguishes verified rows from copied algebraic
couplings. The maximum check difference is reported in metadata.json.
No implication of
identical floating-point conditioning for arbitrary ill-conditioned D is made.

The additional Rademacher common-factor experiment uses an independently
drawn dense R x m sign matrix. For this coordinate support, every other
factor projection is +/-1, so column scales are nonzero signs and the KR
range equals a dense sign range for all d. It has a different output law
from Gaussian; it is not another exact Gaussian-law demonstration.
The line1.5 is the Gaussian expected bound, not a proven sign-sketch bound
or a deterministic guarantee on each draw. This coordinate sign support
has constant weights and is not the manuscript's separate sign OSE witness.

Approximation uses 64 trials per r at 8,16,32,64,128,256, and 32 at 512.
These are descriptive trials, not estimates of a rare
failure probability. Gaussian output rows are algebraic coupled copies, not
four independent families of draws. The plotted Gaussian and sign outputs are similar
in this particular diagonal instance; this is an observation, not a law.

Validation
----------
Explicit ambient tensor contraction agrees with the common-factor shortcut
within8.88e-16 for d2,3,4. The rank-r error identity agrees with a direct SVD
within1.55e-15. Paired weighted QR differences and total simulation runtime
are recorded in metadata.json. The experiment uses single-threaded BLAS.
For every actual QR, approximation_QR_diagnostics.csv records
min_j |R_jj|/||Y_col_j||_2 and min_j |R_jj|/max_j |R_jj|.
Both must exceed (R+m)*machine_epsilon before all m Q columns are used.
This is a numerical rank diagnostic, not a condition-number theorem.
All 853 actual factorizations, including independently recomputed weighted
checks, must pass; report_results.py verifies this and the spectral Rayleigh
lower bound against every observation.
The final Matplotlib
PNG was visually inspected: labels, reference lines, medians, bands and
legend are clear and do not overlap. The vector PDF contains the same axes.
The inline PGFPlots figure is separately compiled/rendered for layout QA.

Submission/archive
------------------
Package code, CSVs, metadata and README with the manuscript or a public
archive chosen by the author. Exclude .python_packages, compiler logs and
preview intermediates. The public repository is linked in the top-level README.md.

Figure palette updated on 2026-10-09 to match the manuscript schematic.
Distinct markers accompany the brighter line colors. Raw data are unchanged.
