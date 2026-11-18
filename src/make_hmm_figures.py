"""Generate report-only HMM maps. Run from repo root; refuses existing outputs.

python src/make_hmm_figures.py --data-root <root> --tag mock_20261118_v1
Add --screenshots --browser <Chromium executable> to render PNGs with Playwright.
Only the two named training trips are read. References are loaded after decoding.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from pipeline import prepare_trip
from data_loader import load_trip
from hmm import decode as multihop
from hmm_fixed import decode as fixed
from evaluate import pointwise_accuracy, route_continuity
from viz import make_trip_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--screenshots', action='store_true')
    ap.add_argument('--browser')
    args = ap.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', args.tag):
        ap.error('tag must contain letters, digits, underscores or hyphens')
    root = Path(__file__).resolve().parents[1]
    out = root/'results/maps'
    if list(out.glob(f'hmm_{args.tag}_*')):
        raise FileExistsError('Choose a fresh tag: existing figures will not be overwritten')
    out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(root/'results/points_train.parquet')
    configs = {name: json.loads((root/f'artifacts/{filename}').read_text()) for name, filename in
               [('fixed','hmm_fixed_config.json'), ('multihop','hmm_config.json')]}
    manifest = dict(probabilities='mock softmax(-dist_m/10 + 3*cos_head)',
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root,text=True).strip(),
                    configs=configs, cases=[], note='Report-only reference layers; no evaluator data. Accuracy mask: label_dist_m <= 60. Continuity: changed arcs, shared evaluate.py. Both cases are multi-hop TUNE trips; illustrations, not held-out evidence.')
    pages = []
    for tid in ['00000010', '00000078']:
        trip_dir = Path(args.data_root)/'train'/tid
        trip, feats, _ = prepare_trip(trip_dir)
        logits = (-feats.dist_m/10 + 3*feats.cos_head).to_numpy(float)
        groups = feats.point_idx.to_numpy()
        weights = np.exp(logits - pd.Series(logits).groupby(groups).transform('max').to_numpy())
        feats['prob'] = weights/pd.Series(weights).groupby(groups).transform('sum').to_numpy()
        pi, ai, prob = [feats[c].to_numpy() for c in ['point_idx','arc_id','prob']]
        order=np.lexsort((-prob,pi)); first=np.r_[True,pi[order][1:]!=pi[order][:-1]]
        baseline=np.full(len(trip['track']),-1,dtype=int)
        baseline[pi[order][first]]=ai[order][first]
        preds={'classifier':baseline,'fixed':fixed(trip,feats,configs['fixed']),
               'multihop':multihop(trip,feats,configs['multihop'])}
        # Decoding is complete before any reference file is opened.
        ref=load_trip(trip_dir,with_route=True)
        truth=labels[labels.trip_id.astype(str).str.zfill(8)==tid].sort_values('point_idx')
        if len(truth)!=len(baseline): raise ValueError('Reference alignment failure')
        true=truth.true_arc.to_numpy(); covered=truth.label_dist_m.to_numpy()<=60
        corrected=(baseline!=true)&(preds['fixed']==true)&covered
        regressed=(baseline==true)&(preds['fixed']!=true)&covered
        target=corrected if tid=='00000010' else regressed
        # Choose the 41-observation window containing the most target changes.
        center=int(np.argmax(np.convolve(target.astype(int),np.ones(41,dtype=int),mode='same')))
        lo,hi=max(0,center-20),min(len(baseline),center+21)
        # Include the actual arcs so a displaced GPS point cannot hide the road choice.
        ids=np.unique(np.r_[true[lo:hi],*[p[lo:hi] for p in preds.values()]])
        ids=ids[ids>=0]
        coords=np.vstack([trip['track'][lo:hi,:2],trip['nodes'][np.unique(trip['arcs'][ids])]])
        pad=.0003
        bounds=[[float(coords[:,1].min()-pad),float(coords[:,0].min()-pad)],
                [float(coords[:,1].max()+pad),float(coords[:,0].max()+pad)]]
        case=dict(trip_id=tid,detail_point_range=[lo,hi-1],detail_bounds=bounds,
                  corrected_points=int(corrected.sum()),regressed_points=int(regressed.sum()),metrics={},files=[])
        children=[]
        for key,pred in preds.items():
            accuracy=pointwise_accuracy(pred,true,covered)
            continuity=route_continuity(pred[pred>=0],trip['arcs'])
            case['metrics'][key]=dict(accuracy=accuracy,continuity=continuity,missing=int((pred<0).sum()))
            filename=f'hmm_{args.tag}_{tid}_{key}.html'
            path=out/filename
            label={'classifier':'Classifier only','fixed':'Fixed HMM','multihop':'Multi-hop HMM (API)'}[key]
            title=f'{label} | accuracy {accuracy:.2%}'
            make_trip_map(ref,str(path),pred_arcs=pred[pred>=0],title=title,offline=True,
                          point_errors=(pred!=true)&covered)
            text=path.read_text(encoding='utf-8')
            mapname=re.search(r'var (map_\w+) = L.map',text).group(1)
            detail=f"\nif(new URLSearchParams(location.search).get('view')==='detail') {{ {mapname}.fitBounds({json.dumps(bounds)}, {{padding:[24,24],animate:false}}); }}\n"
            a,b=text.rsplit('</script>',1)
            path.write_text(a+detail+'</script>'+b,encoding='utf-8')
            children.append((filename,label,accuracy,continuity))
            case['files'].append(filename)
        filename=f'hmm_{args.tag}_{tid}_comparison.html'
        title='HMM improvement example' if tid=='00000010' else 'Fixed-transition failure example'
        cards=''.join(f'<section><h2>{label}</h2><p>Accuracy {acc:.2%} &nbsp; Direct continuity {cont:.2%}</p><iframe title="{label}" src="{file}"></iframe></section>' for file,label,acc,cont in children)
        document='<!doctype html><html><head><meta charset="utf-8"><title>'+title+'</title><style>body{margin:0;font:15px Arial;background:#f5f7fa;color:#17212b}header{padding:18px 24px 12px}h1{margin:0 0 7px;font-size:25px}header p{margin:4px 0}main{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;padding:0 12px}section{background:white;border:1px solid #cbd5df;border-radius:6px;overflow:hidden}h2{font-size:18px;margin:12px 14px 6px}section p{margin:0 14px 10px;font-size:14px}iframe{width:100%;height:calc(100vh - 230px);min-height:480px;border:0}a{color:#124ca0}</style></head><body><header><h1>'+title+' | train '+tid+'</h1><p>Mock probabilities; derived reference labels. Both examples are multi-hop TUNE trips.</p><p>Blue dots: GPS. Grey: nearby roads. Green: reference. Red: predicted arcs. Toggle layers to inspect overlap.</p><p><a href="?view=overview">Full trip</a> &nbsp;|&nbsp; <a href="?view=detail">Detail: points '+str(lo)+'-'+str(hi-1)+'</a></p></header><main>'+cards+'</main><script>if(new URLSearchParams(location.search).get("view")==="detail"){document.querySelectorAll("iframe").forEach(f=>f.src=f.getAttribute("src")+"?view=detail");}</script></body></html>'
        document=document.replace('Toggle layers to inspect overlap.',
            'Orange rings: wrong arc vs derived reference (including reverse-direction errors). Offline: no external basemap.')
        (out/filename).write_text(document,encoding='utf-8')
        pages.append(out/filename)
        case['files'].append(filename)
        pd.DataFrame(dict(point_idx=np.arange(len(baseline)),**preds)).to_csv(out/f'hmm_{args.tag}_{tid}_predictions.csv',index=False)
        manifest['cases'].append(case)
    if args.screenshots:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser=p.chromium.launch(executable_path=args.browser,headless=True)
            context=browser.new_context(viewport=dict(width=2100,height=1100),device_scale_factor=1,offline=True)
            context.route('http://**/*', lambda route: route.abort())
            context.route('https://**/*', lambda route: route.abort())
            page=context.new_page()
            for file in pages:
                for view in ['overview','detail']:
                    errors=[]
                    page.on('pageerror',lambda e: errors.append(str(e)))
                    page.goto(file.as_uri()+'?view='+view,wait_until='networkidle',timeout=60000)
                    for frame in page.frames[1:]:
                        frame.wait_for_selector('.leaflet-control-layers',timeout=30000)
                        frame.wait_for_function('!document.querySelector(".leaflet-zoom-anim")',timeout=10000)
                        frame.wait_for_function('Array.from(document.querySelectorAll(".leaflet-tile")).every(i=>i.complete)',timeout=20000)
                    if errors: raise RuntimeError(errors)
                    page.screenshot(path=str(file.with_name(file.stem+'_'+view+'.png')),full_page=True)
            browser.close()
    manifest['sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.glob(f'hmm_{args.tag}_*') if f.is_file()}
    (out/f'hmm_{args.tag}_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest['cases'],indent=2))


if __name__=='__main__':
    main()
