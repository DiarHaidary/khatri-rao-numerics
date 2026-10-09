Finite-sample check of the asymptotic regression inflation
=========================================================
Run: python regression_finite_size.py --output rerun_regression
Requires NumPy. Seed 2026100904; r=10; m=300; 4000 independent trials
for each d=1,2,3,4. The d=1 case is the unweighted Gaussian control.

For each draw of weights and design coordinates, the exact conditional
expected excess error is tr(C_m). Averaging this integrates out the
Gaussian residual coordinates and reduces Monte Carlo noise.
The reported normalized value divides this mean by r E[w^2]/m,
an asymptotic reference rather than an exact finite-sample formula.
The summary includes Monte Carlo standard errors. The unweighted
control is checked against its exact finite-sample normalized mean
m/(m-r-1). No convergence rate or finite-sample relative-error theorem
is inferred from these means.
