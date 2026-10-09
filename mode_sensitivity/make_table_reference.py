"""Build the paper's mode-size table with its dense Gaussian reference.

The product summaries are unchanged. The dense row is taken from the
Figure 3 experiment at exactly r=32, m=97, R=194 (128 trials).
"""
from pathlib import Path
import csv

root=Path(__file__).resolve().parents[1]
def read(path):
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))
products=read(root/'mode_sensitivity/sensitivity_summary.csv')
dense=next(x for x in read(root/'head_tail/head_tail_summary.csv') if int(x['d'])==0 and int(x['r'])==32)
assert int(dense['m'])==97 and int(dense['support_dimension'])==194
rows=[dict(sketch='Dense Gaussian',d=0,common_modes='--',common_product='--',r=32,m=97,support_dimension=194,
           trials=dense['trials'],error_median=dense['rankr_ratio_q50'],error_q10=dense['rankr_ratio_q10'],
           error_q90=dense['rankr_ratio_q90'],source='head_tail/head_tail_summary.csv: d=0, r=32')]
for x in products:
    rows.append(dict(sketch='KR',d=x['d'],common_modes=x['common_modes'],common_product=x['common_product'],r=x['r'],m=x['m'],
                     support_dimension=x['support_dimension'],trials=x['trials'],error_median=x['error_median'],
                     error_q10=x['error_q10'],error_q90=x['error_q90'],source='mode_sensitivity/sensitivity_summary.csv'))
with (root/'mode_sensitivity/table_with_dense_reference.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print('Dense reference:',rows[0])
