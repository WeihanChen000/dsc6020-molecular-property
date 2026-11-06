"""Rebuild scores, paired counts and the report figure from public predictions."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS=["dmpnn","molformer_mlp","train_mean","rdkit_mlp","without_ipc"]
SPECIAL="Lipophilicity_03592"


def scores(part,model):
    err=(part[model]-part.y_true).abs()
    return {"rmse":np.sqrt((err**2).mean()),"mae":err.mean(),"median_ae":err.median(),
            "max_ae":err.max(),"max_error_id":part.loc[err.idxmax(),"sample_id"]}


def summarize(folder):
    data=pd.read_csv(folder/"predictions.csv")
    table, paired, coverage=[],[],[]
    assert not data.duplicated(["context","sample_id"]).any()
    for ctx,g in data.groupby("context"):
        key={"dataset":g.dataset.iloc[0],"context":ctx,"split_type":g.split_type.iloc[0],"seed":int(g.seed.iloc[0])}
        for model in MODELS:
            table.append(key|{"model":model,"n":len(g)}|scores(g,model))
        comparison=[("all",g)]
        if SPECIAL in set(g.sample_id):
            comparison.append(("other_419",g[g.sample_id.ne(SPECIAL)]))
        for label,part in comparison:
            a=scores(part,"rdkit_mlp")
            b=scores(part,"without_ipc")
            ea=(part.rdkit_mlp-part.y_true).abs()
            eb=(part.without_ipc-part.y_true).abs()
            paired.append(key|{"subset":label,"n":len(part),"closer_without_ipc":(eb<ea).sum(),
                        "ties":(eb==ea).sum()}|{"old_"+k:v for k,v in a.items()}|{"new_"+k:v for k,v in b.items()})
    results=pd.DataFrame(table)
    results.to_csv(folder/"scores.csv",index=False)
    pd.DataFrame(paired).to_csv(folder/"ablation.csv",index=False)
    scaffold=data[data.split_type.eq("random_scaffold")]
    for ds,g in scaffold.groupby("dataset"):
        for name,part in list(g.groupby("seed"))+[("union",g)]:
            diff=(part.dmpnn-part.y_true).abs()-(part.molformer_mlp-part.y_true).abs()
            coverage.append(dict(dataset=ds,seed=name,test_occurrences=len(part),distinct_records=part.sample_id.nunique(),
             distinct_scaffolds=part.scaffold.nunique(),gnn_closer=int((diff<0).sum()),ties=int((diff==0).sum())))
    pd.DataFrame(coverage).to_csv(folder/"coverage.csv",index=False)
    fig,axes=plt.subplots(2,2,figsize=(9.2,5.5),layout="constrained",width_ratios=[1.25,1])
    for (ds,g),(ax,bx) in zip(scaffold.groupby("dataset"),axes):
        agg=results[(results.dataset==ds)&(results.split_type=="random_scaffold")]
        for offset,model,color,label in [(-.08,"dmpnn","#166355","D-MPNN"),(.08,"molformer_mlp","#a45224","Frozen MoLFormer + MLP")]:
            vals=agg[agg.model==model].sort_values("seed")
            ax.scatter(np.arange(3)+offset,vals.rmse,s=42,color=color,label=label)
        ax.set(xticks=range(3),xticklabels=["3101","3102","3103"],title=ds+"  /  test RMSE",ylabel="RMSE")
        ax.set_xlim(-.4,2.4)
        ax.grid(axis="y",alpha=.2)
        counts=[v for v in coverage if v["dataset"]==ds and v["seed"]!="union"]
        for i,c in enumerate(counts):
            rate=c["gnn_closer"]/c["test_occurrences"]
            bx.barh(i,rate,color="#166355",height=.46)
            bx.text(rate+.015,i,f'{c["gnn_closer"]}/{c["test_occurrences"]}',va="center",fontsize=9)
        bx.axvline(.5,ls=":",lw=1,color="black")
        bx.set(yticks=range(3),yticklabels=["3101","3102","3103"],xlim=(0,1),
               xticks=[0,.5,1],xticklabels=["0%","50%","100%"],title="Smaller absolute error with D-MPNN")
        bx.invert_yaxis()
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=2,frameon=False)
    fig.savefig(folder/"gnn_molformer.png",dpi=175)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    summarize(p.parse_args().folder)
