"""Inspect complete CSVs and summarize computed claims, standard library only."""
import csv, json, math
from pathlib import Path
root=Path(__file__).resolve().parent
def read(name):
    with (root/name).open(newline='',encoding='utf-8') as f:
        return [{k:float(v) for k,v in z.items()} for z in csv.DictReader(f)]
s=read('spectrum_summary.csv'); a=read('approximation_summary.csv')
raw=read('spectrum_trials.csv'); q=read('approximation_QR_diagnostics.csv')
metadata=json.loads((root/'metadata.json').read_text())
assert all(math.isfinite(v) for row in raw+q for v in row.values())
assert all(row['lambda_min']>0 for row in raw)
assert all(row['heavy_direction_rayleigh']<=row['lambda_max']*(1+1e-10) for row in raw)
assert all(row['numerical_full_column_rank']==1 for row in q)
lines=['Computed results (all bands are empirical trial quantiles)',
       f"Simulation runtime: {metadata['simulation_seconds']:.3f} seconds",
       f'Spectrum observations: {len(raw)}',
       f'Approximation actual QR checks: {len(q)}',
       'Spectrum at largest rank, medians:']
largest=max(z['r'] for z in s)
for row in s:
    if row['r']==largest:
        lines.append(f"  r={int(largest)} d={int(row['d'])}: lower={row['lambda_min_q50']:.6g}, upper={row['lambda_max_q50']:.6g}, maxweight={row['wmax_q50']:.6g}, heuristic median relative error={100*row['heuristic_relative_error_median']:.4g}%")
lines.append('Approximation at largest rank:')
largest_a=max(z['r'] for z in a)
for row in a:
    if row['r']==largest_a and row['kind'] in (0,-1):
        lines.append(f"  r={int(largest_a)} kind={int(row['kind'])}: rank-r median={row['rankr_ratio_q50']:.8g}, mean={row['rankr_ratio_mean']:.8g}, projector median={row['projector_ratio_q50']:.8g}, zero={row['zero_ratio']:.0f}")
lines.append('All QR diagnostics:')
lines.append(f"  min column-relative |Rjj|={min(z['min_column_relative_R_diagonal'] for z in q):.8g}")
lines.append(f"  min global |Rjj|/max |Rjj|={min(z['min_over_max_abs_R_diagonal'] for z in q):.8g}")
lines.append('  numerical rank-deficient draws: 0')
lines.append(f"  max Gaussian paired QR output difference={metadata['checks']['max_coupled_gaussian_QR_error_difference']:.8g}")
lines.append('Median heuristic relative errors by rank (d3,d4):')
for rank in sorted(set(z['r'] for z in s)):
    vals=[next(z['heuristic_relative_error_median'] for z in s if z['r']==rank and z['d']==d) for d in (3,4)]
    lines.append(f"  r={int(rank)}: {100*vals[0]:.4g}%, {100*vals[1]:.4g}%")
text='\n'.join(lines)+'\n'
(root/'computed_summary.txt').write_text(text,encoding='utf-8')
print(text)
