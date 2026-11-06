"""Produce transparent descriptive comparisons; no tests used to choose models."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import ROOT,score,dump

LABEL={'rf':'RDKit + RF','ecfp_rf':'ECFP count + RF','dmpnn':'D-MPNN','ridge':'MoLFormer + Ridge','mlp':'MoLFormer + MLP','descriptor_ridge':'RDKit + Ridge','descriptor_mlp':'RDKit + MLP','fusion_ridge':'RDKit + MoLFormer + Ridge'}
CORE=['rf','ecfp_rf','dmpnn','ridge','mlp']
OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True);FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})

def save(df,name):df.to_csv(OUT/(name+'.csv'),index=False);return df
def summary(df,keys):
    return df.groupby(keys,dropna=False).agg(rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),repeats=('rmse','size')).reset_index()
def paired(df,index,reference,other):
    wide=df.pivot(index=index,columns='model',values='rmse').reset_index()
    wide['comparison']=other+' minus '+reference;wide['delta_rmse']=wide[other]-wide[reference]
    return wide[index+['comparison','delta_rmse']]

def main():
    frames=[]
    for p in (ROOT/'inherited_v2').glob('*/*/metrics.csv'):
        t=pd.read_csv(p);t['stage']='v2';t['origin']=str(p.relative_to(ROOT));frames.append(t)
    for p in (ROOT/'results').glob('*/*/metrics.csv'):
        t=pd.read_csv(p);t['origin']=str(p.relative_to(ROOT));frames.append(t)
    allrows=save(pd.concat(frames,ignore_index=True),'all_metrics');test=allrows[allrows.partition.eq('test')]
    anchor=test[test.fraction.eq(1)&test.split_type.isin(['random','scaffold'])&~test.stage.eq('D')]
    main=save(summary(anchor,['dataset','split_type','model']),'anchor_main_results')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,((dataset,kind),g) in zip(axes.flat,main.groupby(['dataset','split_type'])):
        g=g.set_index('model').reindex(CORE).reset_index();ax.barh([LABEL[x] for x in g.model],g.rmse_mean,xerr=g.rmse_sd.fillna(0),capsize=3)
        ax.set(title=f'{dataset} / original {kind}',xlabel='Test RMSE (lower is better)');ax.invert_yaxis()
    fig.suptitle('Original partitions: training-seed variation; Ridge has one deterministic fit');fig.tight_layout();fig.savefig(FIG/'01_anchor_results.png');plt.close(fig)
    heads=main[main.model.isin(['descriptor_ridge','descriptor_mlp','ridge','mlp'])]
    save(heads,'head_representation_comparison')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(dataset,g) in zip(axes,heads.groupby('dataset')):
        for m in ['descriptor_ridge','descriptor_mlp','ridge','mlp']:
            t=g[g.model.eq(m)].set_index('split_type').reindex(['random','scaffold']);ax.plot([0,1],t.rmse_mean,'o-',label=LABEL[m])
        ax.set(title=dataset,xticks=[0,1],xticklabels=['Original random','Original scaffold'],ylabel='Test RMSE')
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(FIG/'02_heads.png');plt.close(fig)
    cross=test[test.stage.eq('B')| (test.stage.eq('E')&test.split_type.isin(['random_group','random_scaffold']))]
    save(cross,'cross_split_individual');cross_summary=save(summary(cross,['dataset','split_type','model']),'cross_split_summary')
    comparisons=[paired(cross,['dataset','split_type','split_seed'],'rf',m) for m in ['ecfp_rf','dmpnn','ridge','mlp']]
    save(pd.concat(comparisons),'cross_split_paired_differences')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,((dataset,kind),g) in zip(axes.flat,cross.groupby(['dataset','split_type'])):
        for i,m in enumerate(CORE):
            values=g[g.model.eq(m)].sort_values('split_seed').rmse.to_numpy();ax.scatter([i]*len(values),values);ax.plot([i-.2,i+.2],[values.mean()]*2,color='black')
        ax.set(title=f'{dataset} / {kind}',ylabel='Test RMSE',xticks=range(5),xticklabels=[LABEL[m] for m in CORE]);ax.tick_params(axis='x',rotation=35)
    fig.suptitle('Three shared partitions; dots = split seeds, training seed fixed at 0');fig.tight_layout();fig.savefig(FIG/'03_split_variation.png');plt.close(fig)
    small=test[(test.fraction.lt(1)|((test.fraction.eq(1))&test.train_seed.eq(0)))&test.split_type.isin(['random','scaffold'])&test.model.isin(CORE)&~test.stage.eq('D')]
    save(small,'learning_curve_individual');curve=save(summary(small,['dataset','split_type','fraction','model']),'learning_curve_summary')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,((dataset,kind),g) in zip(axes.flat,curve.groupby(['dataset','split_type'])):
        for m in CORE:
            t=g[g.model.eq(m)].sort_values('fraction');ax.errorbar(t.fraction*100,t.rmse_mean,yerr=t.rmse_sd.fillna(0),marker='o',capsize=3,label=LABEL[m])
        ax.set(title=f'{dataset} / original {kind}',xlabel='Fraction of original training partition (%)',ylabel='Test RMSE',xticks=[25,50,100])
    axes[0,1].legend(fontsize=7);fig.suptitle('Nested training subsets; fixed validation/test; 100% endpoint shown once');fig.tight_layout();fig.savefig(FIG/'04_small_data.png');plt.close(fig)
    budget=[]
    for p in (ROOT/'results').glob('D_*/dmpnn_budget/metrics.csv'):
        t=pd.read_csv(p)
        for _,r in t.iterrows():
            seed=int(r.train_seed);prior=pd.read_csv(p.parent/f'seed{seed}_{r.partition}_budget300_predictions.csv');s=score(prior.y_true.to_numpy(),prior.y_pred.to_numpy())
            old=pd.read_csv(ROOT/'inherited_v2/dmpnn/Lipophilicity_random_full'/f'seed{seed}_{r.partition}_predictions.csv')
            end=pd.read_csv(p.parent/f'seed{seed}_{r.partition}_predictions.csv');final_score=score(end.y_true.to_numpy(),end.y_pred.to_numpy())['rmse']
            budget.append({'train_seed':seed,'partition':r.partition,'rmse_budget300':s['rmse'],'rmse_final':final_score,'delta_rmse':final_score-s['rmse'],'v2_prediction_max_abs_difference':float(np.max(np.abs(prior.y_pred-old.y_pred))),'best_epoch_final':r.best_epoch,'epochs_run':r.epochs_run,'near_epoch_limit':r.near_epoch_limit})
    save(pd.DataFrame(budget),'budget_sensitivity')
    trajectories=[]
    oldbase=ROOT/'inherited_v2/dmpnn/Lipophilicity_random_full';oldchoice=json.loads((oldbase/'selection.json').read_text())['chosen']
    for seed in [0,1,2]:
        oldh=pd.read_csv(oldbase/f'trial{oldchoice}_seed{seed}'/'history.csv')
        newh=pd.read_csv(ROOT/'results'/f'D_Lipophilicity_random_budget_seed{seed}'/'dmpnn_budget'/f'trial0_seed{seed}'/'history.csv')
        joined=oldh.merge(newh,on='epoch',suffixes=('_v2','_v3'));delta=(joined.valid_rmse_v3-joined.valid_rmse_v2).abs();different=joined.loc[delta>1e-6,'epoch']
        trajectories.append({'seed':seed,'shared_epochs':len(joined),'first_epoch_absolute_difference_over_1e_6':int(different.iloc[0]) if len(different) else None,'first_epoch_validation_difference':float(delta.iloc[0]),'maximum_validation_difference':float(delta.max())})
    save(pd.DataFrame(trajectories),'budget_trajectory_comparison')
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),sharey=True)
    for seed,ax in enumerate(axes):
        folder=ROOT/'results'/f'D_Lipophilicity_random_budget_seed{seed}'/'dmpnn_budget'/f'trial0_seed{seed}'
        h=pd.read_csv(folder/'history.csv');ax.plot(h.epoch,h.valid_rmse,lw=.7,alpha=.5,label='Validation RMSE');ax.plot(h.epoch,h.valid_rmse.cummin(),lw=1.4,label='Best so far');ax.axvline(300,ls='--',color='grey');ax.set(title=f'Training seed {seed}',xlabel='Epoch',ylabel='Validation RMSE')
    axes[-1].legend(fontsize=7);fig.suptitle('Finite GNN budget check: validation only, same training trajectory');fig.tight_layout();fig.savefig(FIG/'06_budget_curves.png');plt.close(fig)
    fusion=test[test.model.isin(['descriptor_ridge','ridge','fusion_ridge'])&test.fraction.eq(1)]
    if 'fusion_ridge' in set(fusion.model):
        f=fusion.pivot(index=['dataset','split_type','context'],columns='model',values='rmse').reset_index();f['fusion_minus_rdkit']=f.fusion_ridge-f.descriptor_ridge;f['fusion_minus_lm']=f.fusion_ridge-f.ridge;save(f,'fusion_comparison')
        fig,axes=plt.subplots(1,2,figsize=(12,5));zoom=f[['fusion_minus_rdkit','fusion_minus_lm']].abs().max(axis=1)<=1
        for ax,df,title in [(axes[0],f,'All settings (symmetric log axes)'),(axes[1],f[zoom],f'Linear detail: both absolute changes <= 1 ({zoom.sum()}/{len(f)})')]:
            for dataset,color in [('ESOL','tab:blue'),('Lipophilicity','tab:orange')]:
                for old,marker in [(True,'s'),(False,'o')]:
                    g=df[(df.dataset==dataset)&(df.split_type.isin(['random','scaffold'])==old)]
                    ax.scatter(g.fusion_minus_rdkit,g.fusion_minus_lm,color=color,marker=marker,label=dataset+(' original' if old else ' new splits'))
            ax.axhline(0,color='grey',lw=1);ax.axvline(0,color='grey',lw=1);ax.set(xlabel='Fusion RMSE minus RDKit RMSE',ylabel='Fusion RMSE minus MoLFormer RMSE',title=title)
        axes[0].set_xscale('symlog',linthresh=.1);axes[0].set_yscale('symlog',linthresh=.1);axes[0].relim();axes[0].update_datalim(f[['fusion_minus_rdkit','fusion_minus_lm']].to_numpy());axes[0].autoscale_view();axes[0].margins(.08)
        extreme=f.loc[f.fusion_minus_lm.idxmax()];axes[0].annotate(extreme.context.replace('Lipophilicity_random_group_split','Lipo split '),(extreme.fusion_minus_rdkit,extreme.fusion_minus_lm),xytext=(8,-8),textcoords='offset points',fontsize=8)
        axes[1].legend(fontsize=8);fig.suptitle('Feature fusion: lower-left means improvement over both; extreme failure retained');fig.tight_layout();fig.savefig(FIG/'05_fusion.png');plt.close(fig)
    fits=[]
    for p in (ROOT/'results').glob('*/*/trial*/fit.json'):
        r=json.loads(p.read_text());r['path']=str(p.relative_to(ROOT));r['stage']=p.parts[-4].split('_')[0];fits.append(r)
    fitdf=save(pd.DataFrame(fits),'fit_inventory');save(fitdf.groupby(['stage','route']).agg(fits=('fit_seconds','size'),fit_seconds_sum=('fit_seconds','sum')).reset_index(),'compute_inventory')
    executions=[]
    for p in ROOT.glob('stage_*_execution.json'):
        r=json.loads(p.read_text());executions.append({'stage':r['stage'],'jobs':r['jobs'],'failed_jobs':len(r['failures']),'wall_minutes':r['wall_seconds']/60,'new_fits':int(fitdf.stage.eq(r['stage']).sum())})
    save(pd.DataFrame(executions).sort_values('stage'),'stage_execution_summary')
    jobs=json.loads((ROOT/'jobs.json').read_text());label_budget=[]
    for j in jobs:
        if j['stage'] not in ['A','C'] or j['fraction']==1:continue
        f=pd.read_csv(ROOT/j['partition_file']);nvalid=int(f.partition.eq('valid').sum())
        label_budget.append({'dataset':j['dataset'],'split_type':j['split_type'],'fraction':j['fraction'],'subset_seed':j['subset_seed'],'training_labels':len(j['train_indices']),'validation_labels':nvalid,'training_plus_validation_labels':len(j['train_indices'])+nvalid})
    save(pd.DataFrame(label_budget),'small_data_label_budget')
    transitions=[]
    for (dataset,kind,m),g in small.groupby(['dataset','split_type','model']):
        full=g[g.fraction.eq(1)].iloc[0]
        for seed in [100,101,102]:
            low=g[(g.fraction.lt(1))&(g.subset_seed.eq(seed))].set_index('fraction')
            transitions.append({'dataset':dataset,'split_type':kind,'model':m,'subset_seed':seed,'rmse25':low.loc[.25,'rmse'],'rmse50':low.loc[.5,'rmse'],'rmse100_shared_endpoint':full.rmse,'delta50_minus25':low.loc[.5,'rmse']-low.loc[.25,'rmse'],'delta100_minus50':full.rmse-low.loc[.5,'rmse']})
    save(pd.DataFrame(transitions),'learning_curve_paired_changes')
    # Examples are selected explicitly by absolute error, never called activity cliffs.
    examples=[]
    for dataset in ['ESOL','Lipophilicity']:
        for kind in ['random','scaffold']:
            ctx=f'{dataset}_{kind}_full';frame=pd.read_csv(ROOT/'inputs'/dataset/f'{kind}_split.csv').set_index('sample_id')
            preds={}
            for m in CORE:
                p=ROOT/'results'/f'A_{ctx}'/m/'seed0_test_predictions.csv' if m=='ecfp_rf' else ROOT/'inherited_v2'/m/ctx/'seed0_test_predictions.csv'
                preds[m]=pd.read_csv(p).set_index('sample_id')
            ids=set()
            for m,p in preds.items():ids.update((p.y_pred-p.y_true).abs().nlargest(3).index)
            for sid in sorted(ids):
                r={'dataset':dataset,'split_type':kind,'sample_id':sid,'smiles':frame.loc[sid,'smiles'],'y_true':frame.loc[sid,'y_true'],'selection_rule':'union of each core model top-3 absolute errors, seed0'}
                for m,p in preds.items():r[m+'_prediction']=p.loc[sid,'y_pred'];r[m+'_absolute_error']=abs(p.loc[sid,'y_pred']-r['y_true'])
                examples.append(r)
    save(pd.DataFrame(examples),'error_examples')
    print('Analysis saved:',OUT,'new fits',len(fits),'metric rows',len(allrows))

if __name__=='__main__':main()
