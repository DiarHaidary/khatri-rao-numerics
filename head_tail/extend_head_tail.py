"""Extend the same fixed-input Gaussian head/tail experiment to large ranks.

Run: python extend_head_tail.py --output DIRECTORY --ranks 512 1024 --trials 32
Existing trials are preserved. Per-rank checkpoints permit resumption.
Large Haar bases are regenerated from seed, not uploaded as oversized files.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]=os.environ.get('KR_BLAS_THREADS','4')
import argparse,csv,gc,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import scipy
from scipy.linalg import eigvalsh
from range_tools import thin_haar,error_from_head_tail,validate

SEED=2026100903
ALL_RANKS=[8,16,32,64,128,256,512,1024]
FACTORS={2:[16],3:[4,4],4:[2,2,4]}

def save_csv(path,rows,fields=None):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0])); w.writeheader(); w.writerows(rows)

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))

def summarize(rows):
    result=[]
    for d in [0,2,3,4]:
        for r in sorted({int(v['r']) for v in rows}):
            group=[v for v in rows if int(v['d'])==d and int(v['r'])==r]
            item=dict(d=d,r=r,m=3*r+1,ambient_dimension=16*r,support_dimension=2*(3*r+1),trials=len(group))
            for key in ['lambda_min','lambda_max','rankr_ratio','projector_ratio']:
                values=np.array([float(v[key]) for v in group])
                for name,p in [('q10',.1),('q50',.5),('q90',.9)]: item[key+'_'+name]=float(np.quantile(values,p))
                item[key+'_mean']=float(np.mean(values))
            item['upper_above_4_fraction']=float(np.mean([float(v['lambda_max'])>4 for v in group]))
            item['relative_1p5_fraction']=float(np.mean([float(v['rankr_ratio'])<=1.5 for v in group]))
            item['joint_fraction']=float(np.mean([float(v['lambda_max'])>4 and float(v['rankr_ratio'])<=1.5 for v in group]))
            result.append(item)
    return result

def common_rows(fs):
    a=fs[0]
    for f in fs[1:]: a=(a[:,:,None]*f[:,None,:]).reshape(len(a),-1)
    return a

def run_rank(out,r,count):
    started=time.perf_counter()
    checkpoint=out/f'large_rank_{r}.json'
    csv_path=out/f'large_rank_{r}_trials.csv'
    diagnostic_path=out/f'large_rank_{r}_diagnostics.csv'
    if checkpoint.exists():
        previous=json.loads(checkpoint.read_text(encoding='utf-8'))
        assert previous['r']==r and previous['target_trials']==count
        if previous['completed_trials']==count:
            print(f'Rank {r}: complete checkpoint reused',flush=True)
            return read_csv(csv_path),previous
        rows=read_csv(csv_path); diagnostics=read_csv(diagnostic_path)
    else:
        previous=None; rows=[]; diagnostics=[]
    m=3*r+1; s=5*r+2; N=16*r
    seeds=np.random.SeedSequence(SEED).spawn(2*len(ALL_RANKS))
    index=ALL_RANKS.index(r)
    rng=np.random.default_rng(seeds[2*index])
    print(f'Rank {r}: constructing fixed Haar tail ({15*r} by {s})',flush=True)
    raw=rng.standard_normal((15*r,s))
    q=thin_haar(raw); del raw; gc.collect()
    digest=hashlib.sha256()
    for i in range(0,len(q),128): digest.update(q[i:i+128].tobytes(order='C'))
    chosen=np.unique(np.linspace(0,s-1,min(s,64),dtype=int))
    sample=q[:,chosen]
    orth=float(np.max(np.abs(sample.T@sample-np.eye(len(chosen)))))
    assert orth<1e-11
    blocks=[np.asfortranarray(q[k::15,:]) for k in range(15)]
    del q,sample; gc.collect()
    streams=[np.random.default_rng(x) for x in seeds[2*index+1].spawn(5)]
    begin=0
    if previous:
        assert previous['basis_sha256_compact_C']==digest.hexdigest()
        begin=previous['completed_trials']
        for rng,state in zip(streams,previous['rng_states']): rng.bit_generator.state=state
    elapsed_prior=previous.get('seconds',0) if previous else 0
    for trial in range(begin,count):
        first=streams[0].standard_normal((m,r))
        dense_tail=streams[1].standard_normal((s,m))
        dense_head=first.T
        obs,diag=error_from_head_tail(dense_head,dense_tail,trial==0)
        gram=first.T@first/m; ev=eigvalsh(gram,check_finite=False)
        rows.append(dict(r=r,m=m,ambient_dimension=N,support_dimension=2*m,d=0,trial=trial,
                         lambda_min=float(ev[0]),lambda_max=float(ev[-1]),**obs))
        diagnostics.append(dict(r=r,d=0,trial=trial,**diag))
        del dense_tail,gram,ev
        common={d:common_rows([streams[j+2].standard_normal((m,n)) for n in FACTORS[d]]) for j,d in enumerate([2,3,4])}
        tails={d:np.zeros((s,m)) for d in [2,3,4]}
        for k,block in enumerate(blocks):
            contraction=block.T@first.T
            for d in [2,3,4]: tails[d]+=contraction*common[d][:,k+1]
        del contraction
        for d in [2,3,4]:
            head=first.T*common[d][:,0]
            gram=head@head.T/m
            ev=eigvalsh(gram,check_finite=False)
            obs,diag=error_from_head_tail(head,tails[d],trial==0 and d==4)
            rows.append(dict(r=r,m=m,ambient_dimension=N,support_dimension=2*m,d=d,trial=trial,
                             lambda_min=float(ev[0]),lambda_max=float(ev[-1]),**obs))
            diagnostics.append(dict(r=r,d=d,trial=trial,**diag))
            del tails[d]
        del first,common,head,gram,ev; gc.collect()
        info=dict(r=r,target_trials=count,completed_trials=trial+1,seed=SEED,
                  basis_seed_spawn_key=list(seeds[2*index].spawn_key),basis_shape=[15*r,s],
                  basis_sha256_compact_C=digest.hexdigest(),sampled_64_column_orthogonality_error=orth,
                  basis_storage='Regenerated from seed; rows omit the head coordinates. Not stored as a large binary.',
                  rng_states=[x.bit_generator.state for x in streams],seconds=elapsed_prior+time.perf_counter()-started,
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                  blas_threads=int(os.environ.get('KR_BLAS_THREADS','4')),
                  error_solver='Exact Woodbury-Cholesky identity, with complete-column normalization. QR cross-check on dense and order-four trial zero.',
                  note='A blank qr_rank_margin means QR was not run for that draw; actual Cholesky diagnostics are in the accompanying diagnostics CSV.')
        save_csv(csv_path,rows); save_csv(diagnostic_path,diagnostics)
        checkpoint.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
        print(f'Rank {r}: {trial+1}/{count} paired trials; elapsed {info["seconds"]:.1f}s',flush=True)
    del blocks; gc.collect()
    return rows,info

def run(out,ranks,count):
    out.mkdir(parents=True,exist_ok=True)
    validation=validate(); print('Independent implementation checks:',json.dumps(validation),flush=True)
    existing=read_csv(out/'head_tail_trials.csv')
    fields=list(existing[0]); original_bytes=(out/'head_tail_trials.csv').read_bytes()
    extension_records=[]
    for r in ranks:
        if any(int(x['r'])==r for x in existing):
            checkpoint=out/f'large_rank_{r}.json'
            info=json.loads(checkpoint.read_text(encoding='utf-8'))
            assert info['completed_trials']==count and sum(int(x['r'])==r for x in existing)==4*count
            print(f'Rank {r}: existing completed observations preserved',flush=True)
            extension_records.append(info)
            continue
        rows,info=run_rank(out,r,count)
        with (out/'head_tail_trials.csv').open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writerows(rows)
        existing.extend(rows); extension_records.append(info)
        save_csv(out/'head_tail_summary.csv',summarize(existing))
    assert (out/'head_tail_trials.csv').read_bytes().startswith(original_bytes)
    metadata=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
    metadata['rank_values']=sorted({int(v['r']) for v in existing})
    for r in ranks: metadata['counts'][str(r)]=count
    metadata['large_rank_extension']=dict(validation=validation,ranks=ranks,
        checkpoint_files=[f'large_rank_{r}.json' for r in ranks],
        diagnostic_files=[f'large_rank_{r}_diagnostics.csv' for r in ranks],
        algorithm='Shared blocked contraction of actual product probes and an exact two-level-spectrum Woodbury error identity. No independent-tail surrogate.',
        old_observations='Original raw CSV preserved as a byte-for-byte prefix.')
    metadata['basis_orthogonality_errors_scope']='Full spectral norms through rank 256. Larger ranks have sampled-column checks recorded in per-rank checkpoint files.'
    metadata['smallest_qr_rank_margin_scope']='Only draws with an actual QR factorization; the large-rank Cholesky diagnostics are recorded separately.'
    metadata['large_rank_seconds']=sum(v['seconds'] for v in extension_records)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print('Large ranks merged successfully.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--ranks',type=int,nargs='+',default=[512,1024])
    parser.add_argument('--trials',type=int,default=32)
    args=parser.parse_args(); run(args.output,args.ranks,args.trials)
