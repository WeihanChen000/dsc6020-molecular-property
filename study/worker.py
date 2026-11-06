"""Run one isolated, predetermined job; preserve every validation candidate."""
import argparse,json,time
import numpy as np,pandas as pd
from core import ROOT,PROTOCOL,NEURAL,RIDGES,features,train_neural,train_tabular,restore,score,dump,signature

def run(job):
    sig=signature();cache={};frame=pd.read_csv(ROOT/job['partition_file'])
    sample=pd.read_csv(ROOT/'inputs'/job['dataset']/'samples.csv');assert list(sample.sample_id)==list(frame.sample_id)
    y=frame.y_true.to_numpy();train=np.asarray(job['train_indices']);valid=np.flatnonzero(frame.partition.eq('valid'))
    assert set(train)<=set(np.flatnonzero(frame.partition.eq('train')))
    for route in job['routes']:
        dest=ROOT/'results'/job['id']/route;dest.mkdir(parents=True,exist_ok=True)
        sf=dest/'signature.txt'
        if sf.exists():assert sf.read_text()==sig,'Source/protocol changed; refuse resume'
        sf.write_text(sig)
        if (dest/'complete.json').exists():continue
        X=features(job['dataset'],route,cache)
        candidates=[{'lr':.0003}] if route=='dmpnn_budget' else list(PROTOCOL['rf']['candidates']) if route in ['rf','ecfp_rf'] else [{'alpha':a} for a in PROTOCOL['ridge']['alphas']] if route in RIDGES else [{'lr':a} for a in PROTOCOL['dmpnn' if route=='dmpnn' else 'mlp']['learning_rates']]
        selection_seed=job['train_seeds'][0];records=[];started=time.perf_counter()
        print(job['id'],route,'train n=',len(train),flush=True)
        def fit(i,param,seed):
            folder=dest/f'trial{i}_seed{seed}';folder.mkdir(exist_ok=True)
            if (folder/'fit.json').exists():return json.loads((folder/'fit.json').read_text())
            return (train_neural if route in NEURAL else train_tabular)(route,X,y,train,valid,param,seed,folder)
        for i,param in enumerate(candidates):
            meta=fit(i,param,selection_seed);records.append({'candidate':i,'params':param,'valid_rmse':meta['valid_rmse']})
        if route in RIDGES and np.argmin([r['valid_rmse'] for r in records])==len(candidates)-1:
            for a in PROTOCOL['ridge']['boundary_extension']:
                i=len(candidates);param={'alpha':a};candidates.append(param);meta=fit(i,param,selection_seed)
                records.append({'candidate':i,'params':param,'valid_rmse':meta['valid_rmse']})
        chosen=int(np.argmin([r['valid_rmse'] for r in records]))
        dump(dest/'selection.json',{'chosen':chosen,'candidate':candidates[chosen],'trials':records,'selection_seed':selection_seed,
                                   'boundary':bool(route in RIDGES and chosen in [0,len(candidates)-1])})
        rows=[]
        for seed in job['train_seeds']:
            meta=fit(chosen,candidates[chosen],seed);folder=dest/f'trial{chosen}_seed{seed}'
            for part in ['valid','test']:
                idx=np.flatnonzero(frame.partition.eq(part));pred=restore(route,folder,X,idx)
                out=frame.iloc[idx][['sample_id','y_true']].copy();out['y_pred']=pred;out.to_csv(dest/f'seed{seed}_{part}_predictions.csv',index=False)
                rows.append({k:v for k,v in job.items() if k not in ['train_indices','routes','train_seeds']}|{'model':route,'train_seed':seed,'partition':part,
                    'train_n':len(train),'evaluation_n':len(idx),**score(y[idx],pred),'fit_seconds':meta['fit_seconds'],
                    'best_epoch':meta.get('best_epoch'),'epochs_run':meta.get('epochs_run'),'near_epoch_limit':meta.get('near_epoch_limit',False)})
                if route=='dmpnn_budget':
                    prior=restore(route,folder,X,idx,'budget300.pt');old=out.copy();old['y_pred']=prior;old.to_csv(dest/f'seed{seed}_{part}_budget300_predictions.csv',index=False)
        pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
        dump(dest/'complete.json',{'signature':sig,'wall_seconds':time.perf_counter()-started,'selection':'validation_only'})
        print('Completed',job['id'],route,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args()
    run(next(j for j in json.loads((ROOT/'jobs.json').read_text()) if j['id']==args.job))
