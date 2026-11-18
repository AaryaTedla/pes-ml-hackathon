import sys, json, time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path('src').resolve()))
from pipeline import prepare_trip
from hmm_fixed import decode
from hmm import decode as decode_multihop
from evaluate import pointwise_accuracy, route_continuity, sequence_exact_rate
parser = argparse.ArgumentParser(description='Run from repository root; never overwrites output.')
parser.add_argument('--data-root', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
output = Path(args.output)
if output.exists():
    raise FileExistsError(output)
source = pd.read_csv('results/hmm/stand_in/hmm_eval_mock.csv', dtype={'trip_id':str})
refs = {part: pd.read_parquet(f'results/points_{part}.parquet') for part in ('train','student_test')}
cfg=json.loads(Path('artifacts/hmm_fixed_config.json').read_text())
multi_cfg=json.loads(Path('artifacts/hmm_config.json').read_text())
rows=[]
for row in source.itertuples():
    trip,f,_=prepare_trip(Path(args.data_root)/row.part/row.trip_id)
    s=(-f.dist_m/10+3*f.cos_head).to_numpy(float)
    key=(f.trip_id.astype(str)+'_'+f.point_idx.astype(str)).to_numpy()
    s=pd.Series(s); w=np.exp(s-s.groupby(key).transform('max'))
    f['prob']=(w/w.groupby(key).transform('sum')).to_numpy()
    pi,ai,p=f.point_idx.to_numpy(),f.arc_id.to_numpy(),f.prob.to_numpy()
    order=np.lexsort((-p,pi)); first=np.r_[True,pi[order][1:]!=pi[order][:-1]]
    clf=np.full(len(trip['track']),-1,dtype=np.int64); clf[pi[order][first]]=ai[order][first]
    pts=refs[row.part]; pts=pts[pts.trip_id.astype(str).str.zfill(8)==row.trip_id].sort_values('point_idx')
    true=pts.true_arc.to_numpy(); mask=pts.label_dist_m.to_numpy()<=60.
    baseline=pointwise_accuracy(clf,true,mask)
    if not np.isclose(baseline,row.acc_clf,rtol=0,atol=1e-12):
        raise ValueError(f'Baseline mismatch {row.trip_id}: {baseline} vs {row.acc_clf}')
    begin=time.perf_counter(); pred=decode(trip,f,cfg); elapsed=time.perf_counter()-begin
    begin=time.perf_counter(); multi=decode_multihop(trip,f,multi_cfg); multi_elapsed=time.perf_counter()-begin
    rows.append(dict(part=row.part,trip_id=row.trip_id,split=row.split,n_points=len(pred),
       acc_clf=baseline,acc_fixed=pointwise_accuracy(pred,true,mask),
       cont_clf=route_continuity(clf[clf>=0],trip['arcs']),cont_fixed=route_continuity(pred[pred>=0],trip['arcs']),
       fixed_runtime_s=elapsed,seq_clf=sequence_exact_rate(clf,true),seq_fixed=sequence_exact_rate(pred,true),
       acc_multihop=pointwise_accuracy(multi,true,mask),cont_multihop=route_continuity(multi[multi>=0],trip['arcs']),
       multihop_runtime_s=multi_elapsed,seq_multihop=sequence_exact_rate(multi,true)))
    if not np.isclose(rows[-1]['acc_multihop'],row.acc_hmm,rtol=0,atol=1e-12):
        raise ValueError(f'Multi-hop result mismatch {row.trip_id}')
if output.exists(): raise FileExistsError(output)
pd.DataFrame(rows).to_csv(output,index=False)
print('Baseline matched on all',len(rows),'trips')
print(pd.DataFrame(rows).groupby('split')[['acc_clf','acc_fixed','cont_clf','cont_fixed','fixed_runtime_s','seq_clf','seq_fixed']].mean().to_string())
print('Multi-hop:',source.groupby('split')[['acc_clf','acc_hmm','cont_clf','cont_hmm','hmm_runtime_s']].mean().to_string())
print(pd.DataFrame(rows).groupby('split')[['seq_multihop','multihop_runtime_s']].mean().to_string())
print('Split counts',source.groupby('split').size().to_dict())
