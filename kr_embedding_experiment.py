"""Reproducible fixed-subspace Gaussian tensor-sketch experiment.

Run: python kr_embedding_experiment.py [--output DIRECTORY] [--trials 64]
Requires Python 3.10+, NumPy and Matplotlib. No SciPy is needed.
The rank is fixed: this experiment does not estimate an asymptotic rank rate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

# Avoid excessive BLAS threads for the many small matrices in this experiment.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter


def haar_basis(n_ambient: int, rank: int, rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((n_ambient, rank)), mode="reduced")
    signs = np.where(np.diag(r) >= 0, 1.0, -1.0)
    return q * signs


def product_rows(factors: list[np.ndarray]) -> np.ndarray:
    """Rows use lexicographic/C-order tensor coordinates."""
    rows = factors[0]
    for f in factors[1:]:
        rows = (rows[:, :, None] * f[:, None, :]).reshape(rows.shape[0], -1)
    return rows


def common_rows(factors: list[np.ndarray], rank: int) -> np.ndarray:
    weights = np.prod(np.stack([f[:, 0] for f in factors[1:]], axis=1), axis=1)
    return factors[0][:, :rank] * weights[:, None]


def verify_implementations(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    worst_projection_error = 0.0
    worst_common_error = 0.0
    worst_distortion_error = 0.0
    for d in (2, 3):
        n, rank, m = 3, 3, 17
        factors = [rng.standard_normal((m, n)) for _ in range(d)]
        fast = product_rows(factors)
        explicit = []
        for i in range(m):
            row = factors[0][i]
            for f in factors[1:]:
                row = np.kron(row, f[i])
            explicit.append(row)
        explicit = np.array(explicit)
        basis = haar_basis(n**d, rank, rng)
        actual = fast @ basis
        expected = explicit @ basis
        common_basis = np.zeros((n**d, rank))
        common_basis[np.arange(rank) * n**(d-1), np.arange(rank)] = 1
        common = common_rows(factors, rank)
        common_expected = explicit @ common_basis
        worst_projection_error = max(worst_projection_error, float(np.max(np.abs(actual-expected))))
        worst_common_error = max(worst_common_error, float(np.max(np.abs(common-common_expected))))
        eigen_dist = float(np.max(np.abs(np.linalg.eigvalsh(actual.T @ actual/m)-1)))
        direct_norm = float(np.linalg.norm(expected.T @ expected/m-np.eye(rank), 2))
        worst_distortion_error = max(worst_distortion_error, abs(eigen_dist-direct_norm))
        assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        assert np.allclose(common, common_expected, rtol=1e-12, atol=1e-12)
        assert math.isclose(eigen_dist, direct_norm, rel_tol=1e-12, abs_tol=1e-12)
    return {"projection_max_abs_error": worst_projection_error,
            "common_projection_max_abs_error": worst_common_error,
            "distortion_max_abs_error": worst_distortion_error,
            "verified_orders": [2, 3], "small_n": 3, "small_r": 3, "small_m": 17}


def closed_certificate(d, rank, epsilon, delta) -> dict:
    k2 = 8/3  # E exp(g^2/K^2)=2 for a standard normal g.
    theta = 1-1/k2
    q = math.ceil(math.log(4*rank/delta))
    v = (2*theta*k2*k2)**d * rank
    cstar = math.sqrt(12*math.sqrt(3)/7)
    raw_radius = cstar*k2**d*(2*q)**(d-2)*(rank+2*q)
    rho = raw_radius+1+4/(3*v)
    hq = 0 if q == 1 else 2*(q-1)/math.sqrt(2*q-3)
    eta = (rank/delta)**(1/(2*q))
    kappa = math.prod(range(1, 2*q, 2))
    cq = kappa**(1/(2*q))
    a = eta*cq*math.sqrt(v)
    b = eta*cq*hq*rho
    m = math.ceil(((a+math.sqrt(a*a+4*epsilon*b))/(2*epsilon))**2)
    allowance = lambda width: eta*cq*(math.sqrt(v/width)+hq*rho/width)
    assert allowance(m) <= epsilon
    assert allowance(m-1) > epsilon
    return {"d": d, "K_squared": k2, "theta": theta, "q": q,
            "v": v, "c_star": cstar, "raw_radius": raw_radius, "rho": rho,
            "h_q": hq, "eta_q": eta, "kappa_q": kappa, "c_q": cq,
            "A": a, "B": b, "m_cert": m, "allowance_m_cert": allowance(m),
            "allowance_previous_integer": allowance(m-1)}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path, trials=64, seed=20261007, rank=8, n=8):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    checks = verify_implementations(seed+991)
    mgrid = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    epsilon, delta = 0.5, 0.1
    certificates = [closed_certificate(d, rank, epsilon, delta) for d in (2, 3)]
    raw_records = []
    subspace_checks = {}
    for d in (2, 3):
        basis_rng = np.random.default_rng(np.random.SeedSequence([seed, d, 0]))
        basis = haar_basis(n**d, rank, basis_rng)
        np.save(output/f"haar_basis_d{d}.npy", basis)
        orth_error = float(np.linalg.norm(basis.T @ basis-np.eye(rank), 2))
        assert orth_error < 1e-12
        subspace_checks[str(d)] = {"ambient_dimension": n**d, "orthogonality_error": orth_error,
                                   "basis_seed_entropy": [seed, d, 0]}
        for trial in range(trials):
            rng = np.random.default_rng(np.random.SeedSequence([seed, d, trial+1]))
            factors = [rng.standard_normal((mgrid[-1], n)) for _ in range(d)]
            common = common_rows(factors, rank)
            random = np.empty((mgrid[-1], rank))
            for lo in range(0, mgrid[-1], 1024):
                hi = min(lo+1024, mgrid[-1])
                random[lo:hi] = product_rows([f[lo:hi] for f in factors]) @ basis
            grams = {"common_factor": np.zeros((rank, rank)), "haar": np.zeros((rank, rank))}
            lo = 0
            for m in mgrid:
                for geometry, rows in (("common_factor", common), ("haar", random)):
                    block = rows[lo:m]
                    grams[geometry] += block.T @ block
                    eig = np.linalg.eigvalsh(grams[geometry]/m)
                    distortion = float(np.max(np.abs(eig-1)))
                    raw_records.append({"d": d, "n": n, "N": n**d, "r": rank,
                                        "trial": trial, "geometry": geometry, "m": m,
                                        "distortion": distortion, "lambda_min": float(eig[0]),
                                        "lambda_max": float(eig[-1]),
                                        "success_at_epsilon": int(distortion <= epsilon)})
                lo = m
        print(f"Finished d={d}: {trials} trials, N={n**d}.", flush=True)
    summary = []
    for d in (2, 3):
        for geometry in ("common_factor", "haar"):
            for m in mgrid:
                records = [row for row in raw_records if row["d"] == d and row["geometry"] == geometry and row["m"] == m]
                values = np.array([row["distortion"] for row in records])
                q10, median, q90 = np.quantile(values, [0.1, 0.5, 0.9])
                summary.append({"d": d, "n": n, "N": n**d, "r": rank, "geometry": geometry,
                                "m": m, "trials": trials, "q10": float(q10),
                                "median": float(median), "q90": float(q90),
                                "success_fraction": float(np.mean(values <= epsilon))})
    write_csv(output/"distortion_trials.csv", raw_records)
    write_csv(output/"distortion_summary.csv", summary)
    write_csv(output/"closed_certificates.csv", certificates)
    metadata = {"seed": seed, "rank": rank, "mode_dimension": n, "orders": [2, 3],
                "trials_per_order": trials, "m_grid": mgrid, "epsilon": epsilon, "delta": delta,
                "subspace_rule": "One fixed independent Haar-distributed subspace for each order; common-factor span e_a tensor e_1 tensor ... tensor e_1.",
                "probe_rule": "Independent standard normal coordinates across rows and modes; common and Haar geometries share probe draws within each trial; widths are nested prefixes.",
                "distortion": "spectral norm of U^T Omega Omega^T U-I_r, where Omega contains product probes scaled by m^{-1/2} (squared-norm distortion)",
                "quantile_rule": "NumPy linear empirical quantiles at 0.10, 0.50, 0.90; these are trial spread, not confidence intervals.",
                "certificate_rule": "Manuscript explicit closed certificate, K^2=8/3, q=ceil(log(4r/delta)); no q optimization or fitted constants.",
                "numpy_version": np.__version__, "matplotlib_version": matplotlib.__version__,
                "verification": checks, "subspace_checks": subspace_checks,
                "simulation_seconds": time.perf_counter()-start}
    (output/"experiment_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    plot(summary, certificates, metadata, output)
    print(json.dumps({"simulation_seconds": metadata["simulation_seconds"], "certificates": certificates,
                      "verification": checks}, indent=2), flush=True)


def plot(summary, certificates, metadata, output):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9,
                         "axes.labelsize": 10, "axes.titlesize": 11,
                         "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.25, 4.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[4.1, 0.75], hspace=0.46, wspace=0.15)
    colors = {"common_factor": "#B65320", "haar": "#245EA6"}
    labels = {"common_factor": "Common factor", "haar": "Haar subspace"}
    main_axes = []
    for col, d in enumerate((2, 3)):
        ax = fig.add_subplot(gs[0, col])
        main_axes.append(ax)
        for geometry in ("common_factor", "haar"):
            rows = [row for row in summary if row["d"] == d and row["geometry"] == geometry]
            x = np.array([row["m"] for row in rows])
            y = np.array([row["median"] for row in rows])
            ax.fill_between(x, [row["q10"] for row in rows], [row["q90"] for row in rows],
                            color=colors[geometry], alpha=0.18, linewidth=0)
            ax.plot(x, y, "o-", color=colors[geometry], markersize=3.5, linewidth=1.5,
                    label=labels[geometry])
        ax.axhline(metadata["epsilon"], color="#565656", linestyle="--", linewidth=1,
                   label=r"$\varepsilon=0.5$")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlim(56, 9400)
        ax.set_xticks([64, 256, 1024, 4096, 8192], ["64", "256", "1024", "4096", "8192"])
        ax.tick_params(axis="x", labelsize=8)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="major", color="#D6D6D6", linewidth=0.5)
        ax.set_xlabel(r"Sketch width $m$")
        rank, n = metadata["rank"], metadata["mode_dimension"]
        ax.set_title(rf"$d={d}$: $r={rank}$, $n={n}$, $N={n**d}$")
        if col == 0:
            ax.set_ylabel(r"$\|U^\top\Omega\Omega^\top U-I_8\|_2$")
            ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D8D8D8")
        strip = fig.add_subplot(gs[1, col])
        mc = certificates[col]["m_cert"]
        strip.set_xscale("log")
        strip.set_xlim(50, 1e6)
        strip.set_ylim(0, 1)
        strip.plot([64, 8192], [0.45, 0.45], color="#858585", linewidth=6, solid_capstyle="butt")
        strip.axvline(mc, color="#202020", linestyle="--", linewidth=1)
        strip.text(math.sqrt(64*8192), 0.28, "Sampled widths", ha="center", va="top", fontsize=8)
        strip.text(mc, 0.70, rf"$m_{{\rm cert}}={mc:,}$", ha="right", va="bottom", fontsize=8)
        strip.set_yticks([])
        strip.set_xticks([1e2, 1e4, 1e6])
        strip.tick_params(axis="x", labelsize=8, length=3)
        strip.minorticks_off()
        for side in ("top", "right", "left"):
            strip.spines[side].set_visible(False)
    min_y = min(row["q10"] for row in summary)*0.8
    max_y = max(row["q90"] for row in summary)*1.25
    for ax in main_axes:
        ax.set_ylim(min_y, max_y)
    main_axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.91, bottom=0.105)
    fig.text(0.55, 0.025, f"Medians and 10%-90% trial bands; {metadata['trials_per_order']} independent trials. Certificate uses $\\delta=0.1$.",
             ha="center", fontsize=8)
    fig.savefig(output/"kr_embedding_distortion.pdf")
    fig.savefig(output/"kr_embedding_distortion.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--plot-only", action="store_true",
                        help="Regenerate figure from existing CSV/metadata without random sampling.")
    args = parser.parse_args()
    if args.plot_only:
        with (args.output/"distortion_summary.csv").open(newline="", encoding="utf-8") as f:
            summary = list(csv.DictReader(f))
        for row in summary:
            for key in ("d", "n", "N", "r", "m", "trials"):
                row[key] = int(row[key])
            for key in ("q10", "median", "q90", "success_fraction"):
                row[key] = float(row[key])
        with (args.output/"closed_certificates.csv").open(newline="", encoding="utf-8") as f:
            certificates = list(csv.DictReader(f))
        for row in certificates:
            row["d"], row["m_cert"] = int(row["d"]), int(row["m_cert"])
        metadata = json.loads((args.output/"experiment_metadata.json").read_text(encoding="utf-8"))
        plot(summary, certificates, metadata, args.output)
        print("Regenerated plot from saved CSV data; no random experiment run.")
    else:
        run(args.output, args.trials)
