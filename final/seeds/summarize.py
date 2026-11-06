"""Score matched predictions without pooling different training runs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(predictions, out):
    data=pd.read_csv(predictions)
    out.mkdir(parents=True, exist_ok=True)
    assert not data.duplicated(["context","train_seed","sample_id"]).any()
    results=[]
    for (dataset,split,ctx),group in data.groupby(["dataset","split_seed","context"]):
        assert set(group.train_seed)=={0,1,2}
        reference=group[group.train_seed.eq(0)].set_index("sample_id").sort_index()
        for seed,part in group.groupby("train_seed"):
            part=part.set_index("sample_id").sort_index()
            pd.testing.assert_frame_equal(reference[["scaffold","y_true"]],part[["scaffold","y_true"]])
            err=part[["dmpnn","molformer_mlp"]].sub(part.y_true,axis="index")
            assert np.isfinite(err).all().all()
            mse_by_scaffold=err.pow(2).groupby(part.scaffold).mean()
            metrics={"rmse":np.sqrt(err.pow(2).mean()),"mae":err.abs().mean(),
                     "scaffold_rmse":np.sqrt(mse_by_scaffold.mean())}
            row={"dataset":dataset,"split_seed":split,"context":ctx,"train_seed":seed,
                 "n":len(part),"scaffolds":len(mse_by_scaffold),
                 "gnn_closer":int((err.dmpnn.abs()<err.molformer_mlp.abs()).sum()),
                 "record_ties":int((err.dmpnn.abs()==err.molformer_mlp.abs()).sum()),
                 "gnn_better_scaffolds":int((mse_by_scaffold.dmpnn<mse_by_scaffold.molformer_mlp).sum())}
            for name,values in metrics.items():
                row["gnn_"+name]=values.dmpnn
                row["lm_"+name]=values.molformer_mlp
                row["gnn_advantage_"+name]=values.molformer_mlp-values.dmpnn
            results.append(row)
    scores=pd.DataFrame(results)
    assert len(scores)==18 and len(data)==4788
    scores.to_csv(out/"scores.csv",index=False)
    subset=["gnn_rmse","lm_rmse","gnn_scaffold_rmse","lm_scaffold_rmse",
            "gnn_advantage_rmse","gnn_advantage_scaffold_rmse"]
    grouped=scores.groupby(["dataset","split_seed"])[subset].agg(["mean","std","min","max"])
    grouped.columns=["_".join(col) for col in grouped.columns]
    grouped.reset_index().to_csv(out/"split_summary.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(10,4.1),sharex=True,sharey=True,layout="constrained")
    labels=[f"{ds}  {s}" for ds,s in scores[["dataset","split_seed"]].drop_duplicates().itertuples(index=False,name=None)]
    for ax,metric,title in zip(axes,["rmse","scaffold_rmse"],["Equal weight per record","Equal weight per scaffold"]):
        ax.axvline(0,color="#555555",lw=1)
        ax.axhline(2.5,color="#dddddd",lw=1)
        for seed,marker,color in [(0,"o","#336b77"),(1,"s","#c16d37"),(2,"^","#665183")]:
            points=scores[scores.train_seed.eq(seed)]
            ax.scatter(points["gnn_advantage_"+metric],np.arange(6)+(seed-1)*.16,
                       marker=marker,s=38,color=color,label=f"Training seed {seed}")
        ax.set_title(title)
        ax.set_xlabel("RMSE difference (MoLFormer minus D-MPNN)")
        ax.grid(axis="x",alpha=.18)
    axes[0].set_yticks(np.arange(6),labels)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=3,frameon=False)
    fig.savefig(out/"seed_comparison.png",dpi=170)
    plt.close(fig)
    return scores


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("predictions",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    summarize(args.predictions,args.out)
