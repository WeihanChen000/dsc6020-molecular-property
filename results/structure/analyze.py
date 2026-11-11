"""Rebuild train-only Morgan nearest neighbours, conditional MAE, and paired split scores."""
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parent
SEEDS = (3101, 3102, 3103)
SPLITS = ("random_group", "random_scaffold")
MODELS = ["RF", "D-MPNN", "MoLFormer", "Qwen"]
BINS = ["[0,0.3)", "[0.3,0.6)", "[0.6,1]"]
FOLD = ["dataset", "split", "seed"]
KEY = FOLD + ["sample_id"]


def nearest_table():
    inputs = pd.read_csv(OUT/"folds.csv.gz")
    old = pd.read_csv(RESULTS.parent/"analysis/nearest_training_similarity.csv")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    records, audits = [], []
    for ds, frame in inputs.groupby("dataset", sort=False):
        frame = frame.set_index("sample_id")
        assert frame.index.is_unique and frame[["connectivity","scaffold","y_true"]].notna().all().all()
        fp = {i:gen.GetFingerprint(Chem.MolFromSmiles(s)) for i,s in frame.connectivity.items()}
        for split in SPLITS:
            for seed in SEEDS:
                col = f"{split}_{seed}"
                assert set(frame[col]) == {"train","valid","test"}
                assert frame.groupby("connectivity")[col].nunique().max() == 1
                if split == "random_scaffold":
                    assert frame.groupby("scaffold")[col].nunique().max() == 1
                tr,te = (frame[frame[col]==p] for p in ["train","test"])
                # Keep archived source order to reproduce np.argmax's first-maximum rule.
                train_fp = [fp[i] for i in tr.index]
                previous = old[old.context==f"{ds}_{split}_split{seed}"].set_index("sample_id")
                if ds != "FreeSolv":
                    assert previous.index.is_unique and set(previous.index)==set(te.index)
                differences, id_matches = [], 0
                for i,row in te.iterrows():
                    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp[i],train_fp))
                    k = int(sims.argmax())
                    best, train_id = sims[k],tr.index[k]
                    if ds != "FreeSolv":
                        saved = previous.loc[i]
                        assert saved.nearest_training_id in tr.index
                        differences.append(abs(best-saved.similarity))
                        assert np.isclose(tr.loc[saved.nearest_training_id,"y_true"],saved.nearest_training_y)
                        assert np.isclose(saved.y_true,row.y_true) and differences[-1]<1e-12
                        assert train_id==saved.nearest_training_id
                        id_matches += 1
                    records.append(dict(dataset=ds,split=split,seed=seed,sample_id=i,scaffold=row.scaffold,
                                        y_true=row.y_true,nearest_train_id=train_id,nearest_train_y=tr.y_true.iloc[k],
                                        similarity=best,ties_at_max=int((sims==best).sum()),
                                        similarity_bin=BINS[np.digitize(best,[0.3,0.6])]))
                audits.append(dict(dataset=ds,split=split,seed=seed,n_train=len(tr),n_test=len(te),
                                   old_ids_matched=id_matches,max_similarity_diff=max(differences,default=np.nan)))
    return pd.DataFrame(records), pd.DataFrame(audits)


def analyze():
    nn,audit = nearest_table()
    pred = pd.read_csv(RESULTS/"test_predictions.csv")
    assert not pred.duplicated(KEY+["model"]).any() and set(pred.model)==set(MODELS)
    wide = pred.pivot(index=KEY,columns="model",values="prediction").reset_index()
    assert pred.groupby(KEY).y_true.nunique().eq(1).all()
    wide = wide.merge(pred[KEY+["y_true"]].drop_duplicates(),on=KEY,validate="one_to_one")
    data = nn.merge(wide,on=KEY,how="outer",validate="one_to_one",indicator=True)
    assert len(data)==3576 and data._merge.eq("both").all() and data[MODELS].notna().all().all()
    np.testing.assert_allclose(data.y_true_x,data.y_true_y,atol=1e-12,rtol=0)
    score_rows=[]
    for key,g in data.groupby(FOLD):
        for model in MODELS:
            err=g[model]-g.y_true_x
            score_rows.append(dict(zip(FOLD,key),model=model,rmse=np.sqrt((err**2).mean()),mae=err.abs().mean()))
    scores = pd.DataFrame(score_rows)
    check=scores.merge(pd.read_csv(RESULTS/"all_scores.csv"),on=FOLD+["model"],validate="one_to_one")
    assert len(check)==72
    np.testing.assert_allclose(check[["rmse","mae"]],check[["test_rmse","test_mae"]],rtol=0,atol=1e-10)
    # Counts use all observations in the bin once, irrespective of model.
    for model in MODELS:
        data[model]=(data[model]-data.y_true_x).abs()
    for by,filename in [(["dataset","split"],"bins_pooled.csv"),(FOLD,"bins_by_seed.csv")]:
        rows=[]
        for key,g in data.groupby(by):
            for label in BINS:
                sub=g[g.similarity_bin==label]
                rows.append(dict(zip(by,key),similarity_bin=label,n_records=len(sub),n_ids=sub.sample_id.nunique(),
                                 n_scaffolds=sub.scaffold.nunique(),n_seeds=sub.seed.nunique(),
                                 **sub[MODELS].mean().to_dict()))
        summary=pd.DataFrame(rows)
        assert summary.n_records.sum()==len(data)
        summary.to_csv(OUT/filename,index=False)
    a=scores.query("split=='random_group'").drop(columns="split")
    b=scores.query("split=='random_scaffold'").drop(columns="split")
    gap=a.merge(b,on=["dataset","model","seed"],suffixes=("_random","_scaffold"),validate="one_to_one")
    for metric in ["rmse","mae"]:
        gap[metric+"_change_pct"]=100*(gap[metric+"_scaffold"]/gap[metric+"_random"]-1)
    gap.to_csv(OUT/"split_changes.csv",index=False)
    nn.to_csv(OUT/"nearest_test.csv",index=False)
    audit.to_csv(OUT/"audit.csv",index=False)

if __name__=="__main__":
    analyze()
