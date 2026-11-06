"""Check the numerical claims curated for this fixed study, independently of prose rendering."""
import json
import numpy as np,pandas as pd
from core import ROOT,dump
def main():
    a=ROOT/'analysis';cross=pd.read_csv(a/'cross_split_individual.csv')
    w=cross.pivot(index=['dataset','split_type','split_seed'],columns='model',values='rmse')
    assert len(w)==12 and (w.mlp<w.ridge).all() and (w.ecfp_rf>w.rf).all()
    core=['rf','ecfp_rf','dmpnn','ridge','mlp'];winners=w[core].idxmin(axis=1)
    assert list(winners.loc[('ESOL','random_group')])==['dmpnn']*3
    assert list(winners.loc[('ESOL','random_scaffold')])==['mlp','dmpnn','rf']
    anchors=pd.read_csv(a/'anchor_main_results.csv').pivot(index=['dataset','split_type'],columns='model',values='rmse_mean')
    assert (anchors.descriptor_mlp<anchors.mlp).all() and (anchors.descriptor_mlp<anchors.descriptor_ridge).all()
    assert (anchors.ecfp_rf>anchors.rf).all()
    learning=pd.read_csv(a/'learning_curve_individual.csv');low=learning[learning.fraction.lt(1)]
    low_wide=low.pivot(index=['dataset','split_type','fraction','subset_seed'],columns='model',values='rmse')
    assert (low_wide.loc['ESOL',core].idxmin(axis=1)=='rf').all()
    assert (low_wide.loc['Lipophilicity',core].idxmin(axis=1)=='dmpnn').all()
    b=pd.read_csv(a/'budget_sensitivity.csv');t=b[b.partition.eq('test')].set_index('train_seed')
    assert t.loc[0,'delta_rmse']<0
    np.testing.assert_allclose(t.loc[[1,2],'delta_rmse'],0,atol=1e-12)
    cases=[('ESOL','scaffold','ESOL_00718',[-7.9620,-6.0090,-10.1520,-5.9611,-4.9292,-10.2064]),('Lipophilicity','random','Lipophilicity_03148',[1.8676,2.3945,2.1437,1.9566,3.2671,1.8554])]
    for dataset,kind,sid,expected in cases:
        predictions=[]
        for route in core+['descriptor_mlp']:
            base=ROOT/'results'/f'A_{dataset}_{kind}_full'/route if route in ['ecfp_rf','descriptor_mlp'] else ROOT/'inherited_v2'/route/f'{dataset}_{kind}_full'
            predictions.append(pd.read_csv(base/'seed0_test_predictions.csv').set_index('sample_id').loc[sid,'y_pred'])
        np.testing.assert_allclose(predictions,expected,atol=5.1e-5,rtol=0)
    report={'status':'passed','claims':['12 new LM-MLP RMSE values below LM-Ridge','ECFP-RF above RDKit-RF on all original mean/new split settings','Original RDKit-MLP below RDKit-Ridge and LM-MLP','Specified ESOL winners verified among five core routes','ESOL small-data RF and Lipophilicity small-data D-MPNN winners verified for all twelve subsets each','Within-run budget effect for all three seeds verified','Two case tables matched to saved predictions'],'scope':'Curated claims for this fixed result set; a new training run requires a new interpretation review.'}
    dump(ROOT/'report_claim_verification.json',report);print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
