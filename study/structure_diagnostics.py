"""Post-hoc descriptive diagnostic only; not used in training or selecting configurations."""
import json
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
from core import ROOT,score

def main():
    jobs=json.loads((ROOT/'jobs.json').read_text());rows=[];groups=[];overlap=[];ranges=[]
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    for dataset in ['ESOL','Lipophilicity']:
        samples=pd.read_csv(ROOT/'inputs'/dataset/'samples.csv')
        gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
        fps=[gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in samples.connectivity]
        contexts=[j for j in jobs if j['dataset']==dataset and j['stage']=='B']+[j for j in jobs if j['dataset']==dataset and j['stage']=='A' and j['fraction']==1]
        sets={}
        for j in contexts:
            f=pd.read_csv(ROOT/j['partition_file']);tr=np.array(j['train_indices']);te=np.flatnonzero(f.partition.eq('test'));trainfps=[fps[k] for k in tr];sims=[]
            sets[j['context']]=set(f.sample_id.iloc[te])
            for i in te:
                sim=DataStructs.BulkTanimotoSimilarity(fps[i],trainfps);nearest=int(np.argmax(sim));sims.append(sim[nearest])
                rows.append({'dataset':dataset,'context':j['context'],'split_type':j['split_type'],'sample_id':f.sample_id.iloc[i],'nearest_training_id':f.sample_id.iloc[tr[nearest]],'similarity':sim[nearest],'y_true':f.y_true.iloc[i],'nearest_training_y':f.y_true.iloc[tr[nearest]],'train_target_min':f.y_true.iloc[tr].min(),'train_target_max':f.y_true.iloc[tr].max()})
            bins=np.digitize(sims,[.3,.6],right=False)
            for m in ['rf','ecfp_rf','dmpnn','ridge','mlp']:
                base=ROOT/'results'/j['id']/m if j['stage']=='B' or m=='ecfp_rf' else ROOT/'inherited_v2'/m/j['context']
                p=pd.read_csv(base/'seed0_test_predictions.csv');assert list(p.sample_id)==list(f.sample_id.iloc[te])
                outside=(p.y_true<f.y_true.iloc[tr].min())|(p.y_true>f.y_true.iloc[tr].max())
                for label,mask in [('inside_training_target_range',~outside),('outside_training_target_range',outside)]:
                    if mask.any():ranges.append({'dataset':dataset,'context':j['context'],'model':m,'group':label,'n':int(mask.sum()),**score(p.loc[mask,'y_true'].to_numpy(),p.loc[mask,'y_pred'].to_numpy())})
                for b,label in enumerate(['[0,0.3)','[0.3,0.6)','[0.6,1]']):
                    idx=np.flatnonzero(bins==b)
                    if len(idx):groups.append({'dataset':dataset,'context':j['context'],'split_type':j['split_type'],'model':m,'similarity_bin':label,'n':len(idx),'target_mean':p.y_true.iloc[idx].mean(),**score(p.y_true.iloc[idx].to_numpy(),p.y_pred.iloc[idx].to_numpy())})
        for a in sets:
            for b in sets:
                if a<b:overlap.append({'dataset':dataset,'context_a':a,'context_b':b,'shared_test_n':len(sets[a]&sets[b]),'test_jaccard':len(sets[a]&sets[b])/len(sets[a]|sets[b])})
    detail=pd.DataFrame(rows);detail.to_csv(out/'nearest_training_similarity.csv',index=False)
    detail.groupby(['dataset','split_type','context']).similarity.agg(['mean','median','min','max']).reset_index().to_csv(out/'similarity_summary.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'similarity_error_groups.csv',index=False);pd.DataFrame(overlap).to_csv(out/'test_set_overlap.csv',index=False)
    pd.DataFrame(ranges).to_csv(out/'target_range_error_groups.csv',index=False)
    print('Descriptive structural diagnostics:',len(rows),'test/context observations')
if __name__=='__main__':main()
