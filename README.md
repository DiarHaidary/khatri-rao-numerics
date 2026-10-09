# Khatri-Rao sketching numerics

Reproducible code and data accompanying Diar Heidary's manuscript,
*The Exact Logarithmic Cost of Khatri-Rao Subspace Embeddings*.

## Experiments

| Location | What it measures |
| --- | --- |
| Repository root | Fixed-rank common-factor versus generic Haar embedding distortion |
| `separation/` | Spectral extremes and rank-r approximation at m = 3r + 1 |
| `generic_support/` | Delocalized generic-support approximation control |
| `head_tail/` | Common-factor head with generic tail; ranks through 1024 and ambient dimension through 16384 |
| `regression_finite_size/` | Finite-sample check of asymptotic weighted-regression inflation |
| `mode_sensitivity/` | Fixed-input comparison across common-mode sizes, with paired identical head sketches |
| `oversampling/` | Fixed-input m/r sweep, with common-mode product 512 |
| `weighted_regression_checks.py` | Independent numerical checks of the weighted regression identities |

The folders include source, raw trials, summaries, fixed bases where applicable,
parameter records, and publication figures. No manuscript PDF is included.
The manuscript's plots embed their coordinates in LaTeX and require no external
figure files to compile. All computations use synthetic data.

## Reproduce

Use Python 3.10+ and install the dependencies:

```sh
python -m pip install -r requirements.txt
python kr_embedding_experiment.py --output rerun_embedding
python separation/separation_experiment.py --output rerun_separation
python generic_support/generic_approximation.py --output rerun_generic
python head_tail/head_tail_experiment.py --output rerun_head_tail
python regression_finite_size/regression_finite_size.py --output rerun_regression
python mode_sensitivity/mode_sensitivity.py --output rerun_sensitivity
python oversampling/oversampling_experiment.py --output rerun_oversampling
python weighted_regression_checks.py
```

The supplied results record their actual library versions in metadata JSON.
Run `python separation/separation_experiment.py --output separation --plot-only`
to redraw the separation plots without resampling. The inline figure generator
`python separation/build_inline.py` reads the supplied separation summaries.
The generic experiment writes its inline figure automatically.

See each README.txt for trial counts, coupling, objectives, validation, and
limitations. Quantile bands describe the empirical trial distribution; they
are not confidence intervals or proofs of asymptotic rates. The Gaussian
common-factor approximation curves in `separation/` coincide by an exact
range law. Every curve in `generic_support/` is computed separately.

## Manuscript revision

The 9 October 2026 revision uses vivid, consistent colors and distinct markers,
and adds the generic-support experiment. The original simulation data remain
unchanged. SHA256SUMS.txt records the files included in this snapshot.

Public repository: https://github.com/DiarHaidary/khatri-rao-numerics

## Second manuscript revision (9 October 2026)

The main numerical comparison beyond the exact law now uses `head_tail/`.
The earlier `generic_support/` experiment is retained as a delocalized control.
The common-factor experiment and the new head/tail experiment share the
reference palette in `figure_palette.json`: navy, cyan, blue, and pink.
Trial bands use opacity 0.20, with distinct markers on the median curves.
`regression_finite_size/` includes independently generated values supporting
the manuscript's finite-sample caveat.

## License

Code is released under the MIT License. Numerical data, metadata, figures,
LaTeX figure fragments, and explanatory documentation are released under
Creative Commons Attribution 4.0 International. See [license scope](LICENSING.md),
[MIT terms](LICENSE), and [CC BY 4.0 terms](LICENSE-DATA.txt).

## Extended numerical comparisons

Figure 3 now uses two wide stacked panels and includes r=512 and r=1024.
See `head_tail/LARGE_RANK_README.md` for the exact large-rank implementation
and validation. The common-mode comparison is a compact table in the paper,
and `oversampling/` supplies the added fixed-input oversampling panel.
Original rank-at-most-256 observations are retained unchanged.
