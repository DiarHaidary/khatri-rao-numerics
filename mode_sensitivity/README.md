# Common-mode sensitivity of the head/tail experiment

This comparison uses the actual Gaussian product probes and one fixed
generic tail subspace for each common-mode product. It does not replace
the tail by an independent Gaussian surrogate.

All cases use target rank **32**, sketch size **97**, support dimension
**194**, unit optimal squared tail, and zero-output ratio **33**.
There are **128 independent trials per case**, with seed **2026100905**.
The different cases have different ambient inputs; the tail is fixed
before sampling and is never redrawn between trials.

| Order | Common modes | Common-mode product | Ambient dimension | Median relative error | Trial 10--90% quantiles |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 | 16 | 16 | 512 | 1.479 | 1.375--1.592 |
| 3 | 4 x 4 | 16 | 512 | 1.692 | 1.504--1.931 |
| 3 | 16 x 16 | 256 | 8192 | 1.975 | 1.718--2.303 |
| 3 | 32 x 32 | 1024 | 32768 | 2.028 | 1.747--2.388 |
| 4 | 2 x 2 x 4 | 16 | 512 | 1.843 | 1.577--2.229 |
| 4 | 8 x 8 x 8 | 512 | 16384 | 2.728 | 2.202--3.546 |

These are descriptive fixed-input results, not confidence intervals or a
theorem asserting that error must increase monotonically with mode size.
The larger tested modes give materially larger median errors. Thus the
original Figure 3 values with common-mode product 16 are not representative
of a worst case over unrestricted common-mode dimensions.

## What is bounded

Write each factor as its radius times a unit direction. Replacing Gaussian
factors by isotropic spherical factors multiplies every complete probe by
a nonzero scalar, so it leaves all sampled ranges and range-based outputs
unchanged, for any mode sizes and any fixed input.

For the common-factor head, after this normalization,

    w_ang = product_(j=2,...,d) n_j (u_1^(j))^2 <= P,
    P = product_(j=2,...,d) n_j,
    ||U_head^T z_spherical||^2 = r w_ang.

The angular multiplier has mean 1 and second moment

    E[w_ang^2] = product_(j=2,...,d) 3 n_j / (n_j + 2).

The cap is 16 in the original experiment. Larger finite mode sizes raise
this cap; they do not remove it at a fixed finite dimension. This is NOT
a uniform bound of 16 on the actual head-to-tail ratio. As the common
directions approach their first coordinate vectors, a probe approaches
the head subspace and its orthogonal-tail projection can approach zero.
Within-head/tail directions and their joint column geometry also matter;
one scalar ratio alone does not characterize the range output.

## Coupling and validation

Within each trial, cases use nested coordinates of the same common-mode
Gaussian factors and the same first-mode draw. Therefore the raw head
sketch is exactly identical across different mode sizes within a fixed
order. The measured maximum difference is zero. The tail always comes
from projecting the same actual product probe, preserving its dependence
on the head. The experiment confirms sensitivity but does not identify
correlation as the only mechanism or validate an independent-tail model.

Explicit tensor products, direct compressed SVDs, and an independently
formed Gaussian/spherical coupling check the implementation. Numerical
diagnostics are in metadata.json. The Gaussian/spherical approximation
differences are below 8e-15.

## Reproduce

```sh
python mode_sensitivity/mode_sensitivity.py --output rerun_sensitivity
```

Requires NumPy and SciPy. The product-16 input uses the same saved rank-32
tail basis as the main experiment. Larger bases are generated once from
recorded seeds. metadata.json records their dimensions, seed spawn keys,
library versions, and SHA-256 hashes of C-order float64 matrix bytes.
Use `--save-bases` to export those bases as .npy files. The repository keeps
the code, raw trials, summary, and generation records without duplicating
large regenerated basis files. Cross-platform QR roundoff may change basis
hashes slightly without materially changing the numerical comparison.

## Rank trend in the original experiment

With common-mode product fixed at 16, the original median approximation
errors appear to level off over the tested ranks. In particular, the
order-four median is 1.956630 at rank 128 and 1.941483 at rank 256. These
finite-trial observations do not establish a limiting error or a uniform
bound in rank. The original raw trials and curves are unchanged.
