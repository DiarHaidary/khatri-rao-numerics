# Oversampling sweep on one fixed input

Target rank r=32, ambient dimension N=16384, input rank R=480. One fixed
Gaussian-QR tail is drawn independently of all probes and shared across
orders and widths. The head singular values are one and all 448 tail
singular values are 1/sqrt(448), giving unit optimal squared tail and
zero-output ratio 33. Common modes are (512), (16,32), and (8,8,8), with
product 512 for each order. This input is different from the R=194 input
used in the mode-sensitivity table.

The widths are 48,64,96,128,192,256,384, or m/r=1.5,2,3,4,6,8,12. Every
width uses a nested prefix of its trial and is below R, so increasing m
does not change the input or trivially recover its full range. There are
96 independent trials per law; seed 2026100907; 2688 observations total.

| Sketch | First tested m/r with empirical 90th percentile <= 1.5 | Empirical success fraction at that ratio |
| --- | ---: | ---: |
| Dense Gaussian | 3 | 1.000 |
| KR order 2 | 4 | 0.958 |
| KR order 3 | 6 | 1.000 |
| KR order 4 | 8 | 1.000 |

These are fixed-input finite-trial observations, not sample-complexity
guarantees or confidence intervals. No untested ratio is interpolated to
claim a threshold. Every raw error is monotone along its nested prefixes.

```sh
python oversampling/oversampling_experiment.py --output rerun_oversampling
```

Requires NumPy, SciPy, and Matplotlib. Exact product contractions and the
two-level-spectrum error identity are used; see head_tail/LARGE_RANK_README.md.
Direct QR checks cover the first dense and order-four draw at every width,
agreeing within 2.2e-14. Metadata includes the fixed-basis seed, compact hash,
orthogonality check, library versions, and numerical diagnostics. The input
basis is regenerated from its recorded seed rather than stored as a large
binary file. All data, summaries, and inline figure coordinates are supplied.
