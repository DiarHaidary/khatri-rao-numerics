# Large-rank extension: same input family and exact error calculation

Figure 3 now contains ranks 8 through 1024 (ambient dimensions through
16384), in two vertically stacked wide panels. Original observations
through rank 256 are preserved byte for byte as a prefix of the raw CSV.
There are 32 new paired trials per law at each of ranks 512 and 1024.
The complete dataset has 2944 observations.

The input remains the same family: a common-factor head and a fixed
independent Haar tail, with N=16r, m=3r+1, R=2m, unit squared tail energy,
and zero-output ratio r+1. Common modes remain (16), (4,4), and (2,2,4).

## Exact operations

The compact tail basis has (15r) rows, omitting the zero head coordinates.
Its 15 blocks contract with the same first-mode Gaussian matrix, then
combine with the actual common-factor coordinates for each tensor order.
This is the exact product-probe contraction, not an independent-tail model.

For the two-level spectrum, write Y=[H; T/sqrt(s)] with s=R-r. Define

    K = T^T T / s,
    C = I_r + H K^(-1) H^T.

The Woodbury identity gives

    r - trace(P_Y[head,head]) = trace(C^(-1)).

Therefore the same optimal rank-r squared error is

    1 + (1 - 1/s) trace(C^(-1)).

All tested widths satisfy r <= m <= s and K is positive definite.
Complete columns are normalized by their tail norms before the calculation;
this preserves ranges and improves numerical conditioning. Cholesky solves
evaluate the formula without subtracting two large nearly equal traces.
Direct QR checks on dense and order-four trial zero at each large rank agree
within 8e-14. Independent small-instance checks also compare the blocked
contraction, Gaussian-QR basis, and compressed-SVD error.

## Files and reproduction

- head_tail_trials.csv and head_tail_summary.csv: full raw data and summaries.
- large_rank_512/1024_trials.csv: extension checkpoints, duplicated in the full CSV.
- large_rank_512/1024_diagnostics.csv: Cholesky diagnostics and actual QR checks.
- large_rank_512/1024.json: seeds, basis hashes, sampled-column orthogonality
  checks, completed trial counts, and random states for recovery.
- range_tools.py: exact error identity and independent checks.
- extend_head_tail.py: memory-conscious extension with checkpoints.

To reproduce the full experiment in a new output directory:

```sh
python head_tail/head_tail_experiment.py --output rerun_head_tail
```

The main script computes ranks through 256, then invokes the large-rank
extension. `KR_BLAS_THREADS` controls the large-rank BLAS thread count
(default 4). The six original bases through rank 256 are supplied. Larger
bases are regenerated from their recorded Gaussian-QR seeds and compact
matrix hashes, rather than stored as oversized binaries. Numerical hashes
can vary with cross-platform QR roundoff; the generation rule is fixed.

Blank qr_rank_margin entries mean no QR was performed for that draw.
Actual Cholesky diagnostics are in the corresponding diagnostics file.
These margins are implementation checks, not condition-number theorems.
All bands are empirical 10--90 percent trial quantiles, not confidence
intervals. The large ranks use only 32 trials per law.
