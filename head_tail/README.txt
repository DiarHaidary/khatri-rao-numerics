Common-factor head with generic orthogonal tail
================================================
Run: python head_tail_experiment.py --output rerun_head_tail
Dependencies: NumPy, SciPy, Matplotlib. Seed: 2026100903.

Ranks: 8,16,32,64,128,256. Sketch size m=3r+1. Ambient dimension N=16r,
up to 4096. Input support dimension R=2m. The head basis consists of
e_a tensor e_1 tensor ... tensor e_1. A Gaussian-QR tail basis is drawn
once per rank in its orthogonal complement and is held fixed for all
trials and orders. The tail bases are included as .npy files.
Head singular values are one; the other R-r singular values equal
1/sqrt(R-r), so the best rank-r squared error is one and zero output
has ratio r+1. The input rank exceeds every sketch size.

Mode dimensions are (r,16), (r,4,4), and (r,2,2,4) for d=2,3,4.
This retains the common-factor head obstruction but gives a generic
tail support. No common-factor exact range law is used for the full input.
Each trial shares the first-mode Gaussian matrix across all sketch laws;
the other factors and the dense Gaussian tail use independent streams.
Different ranks use independent streams.

128 independent trials per law through r=64, 96 at r=128, 64 at r=256.
There are 2688 raw observations. Every actual sampled range is computed
with QR and checked for numerical rank. The two-level singular spectrum
gives an exact rank-r error identity, checked against a compressed SVD
on the first draw at every rank/order. Explicit tensor products check
the projected-row construction.

The plot reports medians and empirical 10--90 percent trial quantiles,
not confidence intervals. The horizontal approximation line 1.5 is a
dense Gaussian expectation guarantee, not a bound for each product-sketch
draw. The experiment intentionally uses a small practical sketch budget;
it does not numerically verify the conservative sufficient constants
of the fourth-moment theorem. Larger tensor orders can have median
errors above 1.5 at this budget.

Common-mode sizes matter: their product is 16 in every order here.
See ../mode_sensitivity/README.md for an exact-product comparison with
larger modes. The error medians appear to level off over the tested ranks;
no limiting error or uniform bound in rank is established.
