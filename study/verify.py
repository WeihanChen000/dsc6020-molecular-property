"""Reload selected models and independently audit membership, preprocessing and metrics."""
import argparse,json
import joblib,numpy as np,pandas as pd,torch
from core import ROOT,NEURAL,GRAPH,features,restore,score,sha,signature,dump

def main(stage):
    manifest=json.loads((ROOT/'input_manifest.json').read_text())
    assert manifest['protocol_sha256']==sha(ROOT/'protocol.json')
    assert manifest['jobs_sha256']==sha(ROOT/'jobs.json')
    for item in manifest['files']:assert sha(ROOT/item['path'])==item['sha256'],item['path']
    jobs=json.loads((ROOT/'jobs.json').read_text()); cache={};checked=0;rows=0;fits=0;caps=[];boundaries=[]
    decision=ROOT/'optional_stage_decision.json'
    skipped=['E'] if decision.exists() and not json.loads(decision.read_text())['execute'] else []
    for dataset in ['ESOL','Lipophilicity']:
        for kind in ['random_group','random_scaffold']:
            sets=[]
            for job in [j for j in jobs if j['stage']=='B' and j['dataset']==dataset and j['split_type']==kind]:
                f=pd.read_csv(ROOT/job['partition_file']);sets.append(frozenset(f.loc[f.partition.eq('test'),'sample_id']))
            assert len(set(sets))==3,'Split seeds did not change held-out molecules'
    for job in [j for j in jobs if (stage=='all' and j['stage'] not in skipped) or j['stage']==stage]:
        frame=pd.read_csv(ROOT/job['partition_file']);train=np.array(job['train_indices']);y=frame.y_true.to_numpy()
        assert len(train)==len(set(train)) and set(train)<=set(np.flatnonzero(frame.partition.eq('train')))
        assert frame.groupby('connectivity').partition.nunique().max()==1
        if 'scaffold' in job['split_type']:assert frame.groupby('scaffold').partition.nunique().max()==1
        groups=set(frame.connectivity.iloc[train]);assert set(np.flatnonzero(frame.connectivity.isin(groups)))==set(train)
        if job['fraction']==.25:
            other=next(j for j in jobs if j['stage']==job['stage'] and j['dataset']==job['dataset'] and j['split_type']==job['split_type'] and j['fraction']==.5 and j['subset_seed']==job['subset_seed'])
            assert set(train)<=set(other['train_indices'])
        for route in job['routes']:
            dest=ROOT/'results'/job['id']/route
            assert json.loads((dest/'complete.json').read_text())['signature']==signature()
            sel=json.loads((dest/'selection.json').read_text());assert sel['chosen']==int(np.argmin([r['valid_rmse'] for r in sel['trials']]))
            if sel['boundary']:boundaries.append(f"{job['id']}/{route}")
            X=features(job['dataset'],route,cache);table=pd.read_csv(dest/'metrics.csv')
            assert sorted(table.train_seed.unique())==job['train_seeds']
            for fitpath in dest.glob('trial*/fit.json'):
                fits+=1;fit=json.loads(fitpath.read_text())
                if route in NEURAL:
                    h=pd.read_csv(fitpath.parent/'history.csv');assert int(h.loc[h.valid_rmse.idxmin(),'epoch'])==fit['best_epoch']
                    np.testing.assert_allclose(h.valid_rmse.min(),fit['valid_rmse'],atol=1e-7)
            for seed in job['train_seeds']:
                folder=dest/f"trial{sel['chosen']}_seed{seed}";fit=json.loads((folder/'fit.json').read_text())
                if fit.get('near_epoch_limit'):caps.append(f"{job['id']}/{route}/seed{seed}")
                if route in NEURAL:
                    np.testing.assert_allclose([fit['y_mean'],fit['y_scale']],[y[train].mean(),max(y[train].std(),1e-8)])
                    prep=None if route in GRAPH else joblib.load(folder/'preprocessor.joblib')
                else:prep=joblib.load(folder/'model.joblib')
                if prep is not None:
                    expected=np.array([np.nanmedian(c) if np.isfinite(c).any() else 0 for c in X[train].T])
                    np.testing.assert_allclose(prep[0].statistics_,expected)
                    if route not in ['rf','ecfp_rf']:
                        np.testing.assert_allclose(prep[1].mean_,prep[0].transform(X[train]).mean(axis=0,dtype=np.float64),atol=1e-8)
                checked+=1
                for part in ['valid','test']:
                    idx=np.flatnonzero(frame.partition.eq(part));p=pd.read_csv(dest/f'seed{seed}_{part}_predictions.csv')
                    assert list(p.sample_id)==list(frame.sample_id.iloc[idx]) and p.sample_id.is_unique
                    np.testing.assert_allclose(p.y_true,y[idx]);fresh=restore(route,folder,X,idx)
                    np.testing.assert_allclose(fresh,p.y_pred,rtol=1e-6,atol=1e-6)
                    s=score(y[idx],p.y_pred.to_numpy());r=table[(table.partition==part)&(table.train_seed==seed)].iloc[0]
                    np.testing.assert_allclose([s['rmse'],s['mae']],[r.rmse,r.mae],atol=1e-8)
                    if part=='valid' and seed==sel['selection_seed']:np.testing.assert_allclose(s['rmse'],sel['trials'][sel['chosen']]['valid_rmse'],atol=1e-6)
                    if route=='dmpnn_budget':
                        prior=pd.read_csv(dest/f'seed{seed}_{part}_budget300_predictions.csv')
                        np.testing.assert_allclose(restore(route,folder,X,idx,'budget300.pt'),prior.y_pred,atol=1e-6,rtol=1e-6)
                    rows+=len(p)
    report={'status':'passed','stage':stage,'skipped_optional_stages':skipped,'selected_models_reloaded':checked,'fits_audited':fits,'prediction_rows_checked':rows,'selected_near_epoch_limit':caps,'ridge_boundary_optima':boundaries,'source_signature':signature(),'scope':'Internal implementation consistency; not evidence of external generalization or teacher approval.'}
    dump(ROOT/f'verification_{stage}.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=list('ADBCE')+['all'],required=True);main(p.parse_args().stage)
