Generic-support approximation experiment
========================================
Run: python generic_approximation.py --output rerun
Requires Python 3.10+, NumPy, and Matplotlib.

Seed 2026100901. Target rank 8, full input support dimension 48,
ambient dimension 64. Orders 2 and 3 have mode dimensions 8 and 4.
Each input uses one independently drawn Gaussian-QR basis, saved as
fixed_basis_d2.npy / fixed_basis_d3.npy and held fixed across all trials.
Singular values: eight ones and forty values 1/sqrt(40).
Optimal squared rank-8 tail is 1; zero-output ratio is 9.

There are 128 trials for each order and law: dense Gaussian,
Gaussian product, and independent-coordinate Rademacher product.
Widths 12,16,24,32,40 use nested prefixes within a trial and law.
The different laws and orders use independent streams.
Every result is computed by an SVD of the actual sampled matrix and
a compressed SVD. Rank-deficient samples are handled explicitly.
No exact Gaussian range identity is used to produce any curve.

generic_trials.csv has all 3840 observations; generic_summary.csv has
the means and 10/50/90 percent trial quantiles. Shading represents the
empirical trial distribution, not a confidence interval.
metadata.json records versions, parameters, and numerical checks.
Explicit tensor products and direct SVD errors are checked at the
first trial of every law/order/width. These checks verify the
implementation, not asymptotic probability bounds.
generic_inline.tex contains all plot coordinates and the full caption.
