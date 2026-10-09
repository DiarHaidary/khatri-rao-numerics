# Oversampling sweep with explicit input-rank coverage

The current Figure 4 uses target rank r=32, ambient dimension N=16384,
and fixed input rank R=1536. Widths are 48,64,96,128,192,256,384, giving
m/r=1.5,2,3,4,6,8,12 and m/R from 1/32 to 1/4. The input and its
spectrum are fixed throughout this sweep. Head singular values are
one; all 1504 tail singular values are 1/sqrt(1504); the optimal squared
tail is one and zero-output ratio is 33.

Common modes are (512), (16,32), and (8,8,8), with product 512 for all
orders. Prefixes are nested within trials; first-mode draws are shared
across sketch laws. There are 96 independent trials per law and 2688
observations, with seed 2026100907. Every error decreases along its
nested prefixes. Quantile bands are trial spread, not confidence intervals.

| Sketch | First tested m/r with empirical 90th percentile <= 1.5 | Empirical success fraction |
| --- | ---: | ---: |
| Dense Gaussian | 3 | 0.979 |
| KR order 2 | 6 | 1.000 |
| KR order 3 | 6 | 0.958 |
| KR order 4 | 8 | 1.000 |

These are fixed-input finite-trial thresholds, not general guarantees.

## Paired input-rank comparison

Both inputs use the same head and probes. The R=480 tail is the first
448 columns of the fixed R=1536 Haar tail basis; each spectrum is
normalized to have optimal squared tail one. This controls changes in
tail orientation more closely than two unrelated input draws.

| m/r | Input rank R | m/R | Dense median | Order-4 median |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 480 | 0.2000 | 1.425 | 3.610 |
| 3 | 1536 | 0.0625 | 1.473 | 3.860 |
| 12 | 480 | 0.8000 | 1.019 | 1.039 |
| 12 | 1536 | 0.2500 | 1.070 | 1.137 |

There are 96 trials for every row. Raw paired observations are in
rank_comparison_trials.csv; summaries are in rank_comparison_summary.csv.
The old independent R=480 sweep is retained in legacy_R480/.

Figure 3 and the mode-size table use R=2m, so m/R=1/2. Their error
constants should not be directly compared with this sweep without
accounting for R. The table has m=97, while this sweep's m/r=3 point
has m=96. For the two-level dense Gaussian input, the paper gives the
finite-R upper bound

    1 + (1 - 1/(R-r)) ((R-m)/(R-r)) r/(m-r-1),

valid for r+1 < m <= R-r. With R=2m and m=3r+1, this bound tends to
1.3, helping explain the dense reference curve below the general 1.5 bound.

## Reproduce

```sh
python oversampling/oversampling_experiment.py --output rerun_oversampling --support-rank 1536
```

Requires NumPy, SciPy, and Matplotlib. The generic tail is generated
from its recorded Gaussian-QR seed; metadata includes its compact hash,
orthogonality checks, library versions, and numerical diagnostics.
Actual tensor probes are contracted directly. The exact two-level-spectrum
error identity is checked against QR on the first dense and order-four
draw at every width. No independent-tail surrogate is used.
