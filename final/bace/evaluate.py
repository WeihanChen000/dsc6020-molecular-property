"""Open the BACE test labels after verifying both validation selections."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from train import HERE,dataset,digest


def score(frame,pred):
    err=frame[pred]-frame.y_true
    return {"rmse":float(np.sqrt(err.pow(2).mean())),
            "scaffold_rmse":float(np.sqrt(err.pow(2).groupby(frame.scaffold_id).mean().mean())),
            "mae":float(err.abs().mean())}


def evaluate(a):
    sys.path.insert(0,str(a.study.resolve()))
    import core
    data=dataset(a.raw,a.study)
    locked={}
    fits=[]
    for route in ["dmpnn","mlp"]:
        base=a.runs/route
        record=json.loads((base/"selection.json").read_text())
        assert record["route"]==route
        for source in [a.raw,HERE/"PROTOCOL.md",HERE/"partitions.csv",HERE/"partition.py",HERE/"train.py",a.study/"core.py",a.study/"protocol.json"]:
            assert record["hashes"][source.name]==digest(source)
        trials=record["seed0_candidates"]
        winner=min(trials,key=lambda t:t["valid_rmse"])
        assert [t["lr"] for t in trials]==[.001,.0003] and all(t["seed"]==0 for t in trials)
        assert [t["seed"] for t in record["selected"]]==[0,1,2]
        for t in trials+record["selected"][1:]:
            folder=base/f"seed{t['seed']}_lr{t['lr']}"
            v=pd.read_csv(folder/"validation.csv")
            np.testing.assert_array_equal(v.sample_id,data.sample_id[data.partition.eq("valid")])
            assert np.isclose(np.sqrt(((v.y_pred-v.y_true)**2).mean()),t["valid_rmse"])
            assert np.isclose(pd.read_csv(folder/"history.csv").valid_rmse.min(),t["valid_rmse"])
            fits.append({"route":route,**t,"selected":t["lr"]==winner["lr"]})
        for t in record["selected"]:
            assert t["lr"]==winner["lr"]
            folder=base/f"seed{t['seed']}_lr{t['lr']}"
            locked[f"{route}/{folder.name}"]={f.name:digest(f) for f in folder.iterdir()
                                              if f.name in {"model.pt","preprocessor.joblib"}}
    a.out.mkdir(parents=True,exist_ok=True)
    assert len(locked)==6 and len(fits)==8
    core.dump(a.out/"checkpoint_lock.json",locked)
    labels=pd.read_csv(a.raw,usecols=["pIC50"]).pIC50.loc[data.index].to_numpy()
    assert np.isfinite(labels).all()
    for t in fits:
        folder=a.runs/t["route"]/f"seed{t['seed']}_lr{t['lr']}"
        v=pd.read_csv(folder/"validation.csv")
        np.testing.assert_array_equal(v.y_true,labels[data.partition.eq("valid")])
        assert np.isclose(t["y_mean"],labels[data.partition.eq("train")].mean())
    test=np.flatnonzero(data.partition.eq("test"))
    ref=data.iloc[test]
    base=pd.DataFrame({"sample_id":ref.sample_id.to_numpy(),
        "scaffold_id":[hashlib.sha256(s.encode()).hexdigest()[:18] for s in ref.scaffold],
        "y_true":labels[test],"train_mean":labels[data.partition.eq("train")].mean()})
    assert len(base)==300 and base.scaffold_id.nunique()==ref.scaffold.nunique()==124
    with np.load(a.runs/"mlp/embeddings.npz") as cache:
        np.testing.assert_array_equal(cache["sample_id"],data.sample_id)
        inputs={"mlp":cache["X"],"dmpnn":core.make_graphs(data.connectivity)}
    parts,rows=[],[]
    for seed in [0,1,2]:
        frame=base.copy().assign(seed=seed)
        for route,name in [("dmpnn","dmpnn"),("mlp","molformer_mlp")]:
            t=next(t for t in fits if t["route"]==route and t["seed"]==seed and t["selected"])
            folder=a.runs/route/f"seed{seed}_lr{t['lr']}"
            # Export float32 predictions with enough decimal places for a CSV-only rescore.
            frame[name]=np.asarray(core.restore(route,folder,inputs[route],test),dtype=np.float64)
        assert np.isfinite(frame[["dmpnn","molformer_mlp"]]).all().all()
        row={"seed":seed,"records":len(frame),"scaffolds":frame.scaffold_id.nunique()}
        for model in ["dmpnn","molformer_mlp","train_mean"]:
            row.update({model+"_"+key:value for key,value in score(frame,model).items()})
        diff=(frame.dmpnn-frame.y_true).abs()-(frame.molformer_mlp-frame.y_true).abs()
        row.update(gnn_closer=int((diff<0).sum()),ties=int((diff==0).sum()))
        rows.append(row)
        parts.append(frame)
    scores=pd.DataFrame(rows)
    def wins(model,other):
        return all((scores[model+"_"+m]<scores[k+"_"+m]).all()
                   for m in ["rmse","scaffold_rmse"] for k in [other,"train_mean"])
    result="supported" if wins("dmpnn","molformer_mlp") else "reversed" if wins("molformer_mlp","dmpnn") else "mixed"
    scores.to_csv(a.out/"scores.csv",index=False)
    pd.concat(parts,ignore_index=True).to_csv(a.out/"predictions.csv",index=False)
    pd.DataFrame(fits).to_csv(a.out/"training.csv",index=False)
    core.dump(a.out/"verification.json",{"verdict":result,"fits":len(fits),"selected_models":len(locked),
        "split_sha256":digest(HERE/"partitions.csv"),"train_n":1053,"valid_n":150,"test_n":300,
        "test_scaffolds":124,"token_audit":json.loads((a.runs/"mlp/token_audit.json").read_text())})
    print(scores.round(4).to_string(index=False),result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ["raw","study","runs","out"]:p.add_argument("--"+arg,type=Path,required=True)
    evaluate(p.parse_args())
