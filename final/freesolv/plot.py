"""Plot paired scaffold-holdout RMSE differences for the three physical properties."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
old = pd.read_csv(HERE.parent/"seeds/results/scores.csv")
new = pd.read_csv(HERE/"results/scores.csv").query("split == 'random_scaffold'")
wide = new.pivot(index=["split_seed","seed"],columns="model",values=["rmse","scaffold_rmse"])
extra = pd.DataFrame({
    "gnn_advantage_rmse":wide["rmse"]["mlp"]-wide["rmse"]["dmpnn"],
    "gnn_advantage_scaffold_rmse":wide["scaffold_rmse"]["mlp"]-wide["scaffold_rmse"]["dmpnn"],
}).reset_index().rename(columns={"seed":"train_seed"}).assign(dataset="FreeSolv")
table = pd.concat([old,extra])
fig, axes = plt.subplots(1,3,figsize=(10,3.4),sharey=True)
for ax,(task,unit) in zip(axes,[("ESOL","log mol/L"),("Lipophilicity","logD"),("FreeSolv","kcal/mol")]):
    g=table[table.dataset==task]
    ax.axvline(0,color="0.5",lw=0.9)
    for col,offset,color,marker,label in [
        ("gnn_advantage_rmse",0.11,"#2162a0","o","Per molecule"),
        ("gnn_advantage_scaffold_rmse",-0.11,"#a4511c","s","Equal scaffold weight")]:
        ypos=3103-g.split_seed+offset+0.045*(g.train_seed-1)
        ax.scatter(g[col],ypos,c=color,s=27,marker=marker,label=label,alpha=0.85)
    ax.set_title(task)
    ax.set_xlabel(f"RMSE difference ({unit})")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_yticks([0,1,2],["3103","3102","3101"])
axes[0].set_ylabel("Holdout split")
fig.legend(*axes[0].get_legend_handles_labels(),loc="lower center",ncol=2,frameon=False)
fig.subplots_adjust(left=.085,right=.985,bottom=.32,top=.86,wspace=.29)
fig.savefig(HERE/"results/physical_tasks.png",dpi=165,facecolor="white")
