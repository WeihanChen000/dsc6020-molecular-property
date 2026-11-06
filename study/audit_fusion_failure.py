"""Diagnose the retained extreme prediction without refitting, removing data or tuning."""
import json,joblib,subprocess,sys
import numpy as np,pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from core import ROOT,features,dump

def main():
    context='Lipophilicity_random_group_split3103';dataset='Lipophilicity';job=next(j for j in json.loads((ROOT/'jobs.json').read_text()) if j['id']=='E_'+context)
    frame=pd.read_csv(ROOT/job['partition_file']);tr=np.array(job['train_indices']);te=np.flatnonzero(frame.partition.eq('test'))
    names=json.loads((ROOT/'inputs'/dataset/'descriptor_names.json').read_text())+[f'LM_{i}' for i in range(768)]
    raw=pd.read_csv(ROOT/'provenance/raw_data/Lipophilicity.csv');rows=[];routes=[];cache={};raw_check=None
    for route in ['descriptor_ridge','fusion_ridge']:
        base=ROOT/'results'/job['id']/route;sel=json.loads((base/'selection.json').read_text());model=joblib.load(base/f'trial{sel["chosen"]}_seed0'/'model.joblib')
        X=features(dataset,route,cache);pred=model.predict(X[te]);e=(pred-frame.y_true.iloc[te].to_numpy())**2;k=int(np.argmax(e));i=te[k];sid=frame.sample_id.iloc[i]
        z=model[:-1].transform(X[[i]])[0];contrib=z*model[-1].coef_;np.testing.assert_allclose(contrib.sum()+model[-1].intercept_,pred[k],rtol=1e-8)
        saved=pd.read_csv(base/'seed0_test_predictions.csv').set_index('sample_id');np.testing.assert_allclose(saved.loc[sid,'y_pred'],pred[k])
        top=np.argsort(abs(contrib))[::-1];assert names[top[0]]=='Ipc'
        if raw_check is None:
            # Upstream feature extraction used the default numeric runtime, while core
            # caps compute threads at four. Isolate extraction, record both configurations.
            code="""import sys,json
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from threadpoolctl import threadpool_limits,threadpool_info
s=sys.stdin.read();get=lambda:Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s))['Ipc']
r={'default':get(),'default_blas_threads':[x['num_threads'] for x in threadpool_info() if x['user_api']=='blas'],'by_thread_limit':{}}
for n in [1,2,4,8]:
 with threadpool_limits(n):r['by_thread_limit'][str(n)]=get()
print(json.dumps(r))"""
            raw_check=json.loads(subprocess.check_output([sys.executable,'-c',code],input=raw.smiles.iloc[int(frame.source_row.iloc[i])],text=True))
        raw_ipc=raw_check['default'];np.testing.assert_allclose(raw_ipc,X[i,top[0]],rtol=1e-8)
        for col in top[:10]:rows.append({'route':route,'sample_id':sid,'feature':names[col],'raw':X[i,col],'train_min':np.nanmin(X[tr,col]),'train_median':np.nanmedian(X[tr,col]),'train_max':np.nanmax(X[tr,col]),'standardized':z[col],'coefficient':model[-1].coef_[col],'prediction_contribution':contrib[col]})
        routes.append({'route':route,'sample_id':sid,'target':float(frame.y_true.iloc[i]),'prediction':float(pred[k]),'selected_alpha':sel['candidate']['alpha'],'valid_rmse':sel['trials'][sel['chosen']]['valid_rmse'],'test_rmse':float(np.sqrt(e.mean())),'largest_error_fraction_of_test_SSE':float(e[k]/e.sum()),'raw_ipc_recomputed':float(raw_ipc)})
    pd.DataFrame(rows).to_csv(ROOT/'analysis/fusion_failure_contributions.csv',index=False)
    dump(ROOT/'analysis/fusion_failure_audit.json',{'status':'passed','context':context,'test_n':len(te),'routes':routes,'raw_ipc_runtime_sensitivity':raw_check,'scope':'Post-hoc arithmetic decomposition of saved predictions. Default upstream runtime reproduces cache; descriptor magnitude varies with thread limit. No refitting, clipping, exclusion or replacement of reported test metrics.'})
    print(json.dumps(routes,indent=2))
if __name__=='__main__':main()
