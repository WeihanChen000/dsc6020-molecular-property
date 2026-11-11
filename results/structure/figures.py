"""Draw both figures from the published summary tables."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent
DATASETS=["ESOL","Lipophilicity","FreeSolv"]
UNITS=["logS","logD","kcal/mol"]
MODELS=["RF","D-MPNN","MoLFormer","Qwen"]
SEEDS=[3101,3102,3103]
PALETTE=["#0C6D80","#AF5309","#634D93"]
MARKERS=["o","s","^"]


def split_figure():
    gap=pd.read_csv(OUT/"split_changes.csv")
    fig,axes=plt.subplots(3,4,figsize=(10,7.3),sharey="row")
    fig.subplots_adjust(left=.11,right=.98,top=.88,bottom=.115,wspace=.15,hspace=.36)
    for row,(ds,unit) in enumerate(zip(DATASETS,UNITS)):
        for col,model in enumerate(MODELS):
            ax=axes[row,col]
            g=gap[(gap.dataset==ds)&(gap.model==model)].set_index("seed")
            for i,seed in enumerate(SEEDS):
                vals=g.loc[seed,["rmse_random","rmse_scaffold"]]
                ax.plot(np.array([0,1])+(i-1)*.032,vals,color=PALETTE[i],marker=MARKERS[i],
                        lw=1.6,ms=5.7,zorder=3,linestyle=["-","--","-."][i])
            ax.set(xlim=(-.18,1.18),xticks=[0,1],xticklabels=["Random","Scaffold"])
            ax.grid(axis="y",alpha=.2)
            ax.spines[["top","right"]].set_visible(False)
            if not col:
                ax.set_ylabel(f"{ds}\nRMSE ({unit})",fontweight="semibold",labelpad=8)
            if not row:
                ax.set_title(model,fontweight="semibold")
    handles=[Line2D([],[],color=c,marker=m,linestyle=l,label=str(s))
             for c,m,l,s in zip(PALETTE,MARKERS,["-","--","-."],SEEDS)]
    fig.legend(handles=handles,loc="lower center",ncol=3,title="Split seed",frameon=False)
    fig.suptitle("Test RMSE on group-random and scaffold splits",fontsize=16,fontweight="semibold",y=.985)
    fig.savefig(OUT/"generalization.png",dpi=180,facecolor="white")
    plt.close(fig)


def similarity_figure():
    data=pd.read_csv(OUT/"bins_pooled.csv")
    fig,axes=plt.subplots(3,2,figsize=(9.8,9.2),layout="constrained")
    for r,(ds,unit) in enumerate(zip(DATASETS,UNITS)):
        task=data[data.dataset==ds]
        vmax=task[MODELS].max().max()
        for c,(split,title) in enumerate([("random_group","Group Random"),("random_scaffold","Scaffold")]):
            ax=axes[r,c]
            g=task[task.split==split].set_index("similarity_bin").loc[["[0,0.3)","[0.3,0.6)","[0.6,1]"]]
            vals=g[MODELS].to_numpy().T
            cmap=plt.get_cmap("Blues")
            ax.imshow(vals,aspect="auto",vmin=0,vmax=vmax,cmap=cmap)
            for (i,j),v in np.ndenumerate(vals):
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",
                        color="white" if v/vmax>.63 else "#10283b",fontsize=11)
            ax.set_yticks(range(4),MODELS)
            ax.set_xticks(range(3),[f"{k}\n{n} / {s}" for k,n,s in zip(g.index,g.n_records,g.n_scaffolds)])
            ax.tick_params(length=0)
            ax.set_title(f"{ds} ({unit})  ·  {title}",fontsize=11,fontweight="semibold")
    fig.suptitle("MAE by nearest-training Tanimoto similarity\n"
                 "Columns show similarity band, then test observations / distinct scaffolds",
                 fontsize=13)
    fig.savefig(OUT/"similarity_mae.png",dpi=160,facecolor="white")
    plt.close(fig)

if __name__=="__main__":
    split_figure()
    similarity_figure()
