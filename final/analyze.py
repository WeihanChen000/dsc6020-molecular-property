"""Collect verified predictions from v3, v4 and the Ipc ablation."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from summarize import summarize

HERE = Path(__file__).resolve().parent


def collect(root, runs, full, out):
    sys.path.insert(0, str(root))
    import core
    out.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in json.loads((root/"jobs.json").read_text()) if j["stage"] == "B"]
    rows, trials, inventory = [], [], []
    for j in jobs:
        ds, ctx = j["dataset"], j["context"]
        split = pd.read_csv(root/j["partition_file"])
        test = split[split.partition.eq("test")].copy()
        train = split[split.partition.eq("train")]
        assert split.sample_id.is_unique and split.groupby("connectivity").partition.nunique().max()==1
        if j["split_type"]=="random_scaffold":
            assert split.groupby("scaffold").partition.nunique().max()==1
        test["train_mean"] = train.y_true.mean()
        inputs = root/"inputs"/ds
        feat = core.features(ds, "descriptor_mlp", {})
        names = json.loads((inputs/"descriptor_names.json").read_text())
        x = np.delete(feat,names.index("Ipc"),axis=1)
        dest = runs/"no_ipc"/ctx
        selection = json.loads((dest/"complete.json").read_text())
        best = selection["chosen"]
        assert np.argmin([m["valid_rmse"] for m in selection["candidates"]])==best
        for i,r in enumerate(selection["candidates"]):
            history=pd.read_csv(dest/f"trial{i}/history.csv")
            assert np.isclose(history.valid_rmse.min(),r["valid_rmse"]) and r["input_dim"]==209
            trials.append(dict(context=ctx,selected=i==best,**r))
        for part in ["valid","test"]:
            idx=np.flatnonzero(split.partition.eq(part))
            saved=pd.read_csv(dest/f"{part}.csv")
            assert list(saved.sample_id)==list(split.sample_id.iloc[idx])
            np.testing.assert_array_equal(saved.y_true,split.y_true.iloc[idx])
            np.testing.assert_allclose(core.restore("descriptor_mlp",dest/f"trial{best}",x,idx),
                                       saved.y_pred,rtol=1e-6,atol=1e-6)
        old = full/ctx/"test_predictions.csv"
        if not old.exists():
            old = full/ctx/"test.csv"
        for name,path in {
            "dmpnn":root/"results"/j["id"]/"dmpnn/seed0_test_predictions.csv",
            "molformer_mlp":root/"results"/j["id"]/"mlp/seed0_test_predictions.csv",
            "rdkit_mlp":old,"without_ipc":dest/"test.csv"}.items():
            pred=pd.read_csv(path).set_index("sample_id",verify_integrity=True)
            assert set(pred.index)==set(test.sample_id)
            pred=pred.loc[test.sample_id]
            np.testing.assert_array_equal(pred.y_true,test.y_true)
            assert np.isfinite(pred.y_pred).all()
            test[name]=pred.y_pred.to_numpy()
        rows.append(test.assign(context=ctx,dataset=ds,split_type=j["split_type"],seed=j["split_seed"]))
        groups=split.scaffold.value_counts()
        fixed=groups[groups>int(.1*len(split))]
        inventory.append(dict(dataset=ds,context=ctx,dataset_n=len(split),dataset_scaffolds=len(groups),
             training_n=len(train),test_n=len(test),test_scaffolds=test.scaffold.nunique(),
             forced_training_scaffolds=len(fixed),forced_training_rows=int(fixed.sum())))
    data=pd.concat(rows,ignore_index=True)
    assert len(data)==3192 and len(trials)==24 and len(inventory)==12
    cols=["dataset","context","split_type","seed","sample_id","scaffold","y_true","train_mean",
          "dmpnn","molformer_mlp","rdkit_mlp","without_ipc"]
    data[cols].to_csv(out/"predictions.csv",index=False)
    pd.DataFrame(inventory).to_csv(out/"partitions.csv",index=False)
    pd.DataFrame(trials).to_csv(out/"training.csv",index=False)
    summarize(out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ["study","runs","full","out"]:
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    collect(a.study.resolve(),a.runs,a.full,a.out)
