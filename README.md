# Khatri-Rao sketching numerics

Reproducible code and data accompanying Diar Heidary's manuscript,
*The Exact Logarithmic Cost of Khatri-Rao Subspace Embeddings*.

## Experiments

| Location | What it measures |
| --- | --- |
| Repository root | Fixed-rank common-factor versus generic Haar embedding distortion |
| `separation/` | Spectral extremes and rank-r approximation at m = 3r + 1 |
| `generic_support/` | Approximation on fixed generic supports beyond the exact common-factor law |
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
