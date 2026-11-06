"""Train a preregistered BACE pipeline. Test labels are skipped at CSV read time."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from partition import build

HERE=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dataset(raw,study):
    data=build(raw,study)
    split=pd.read_csv(HERE/"partitions.csv")
    pd.testing.assert_frame_equal(data[["sample_id","partition"]].reset_index(drop=True),split)
    return data


def embeddings(core, data, modeldir, out, manifest):
    os.environ["HF_MODULES_CACHE"]=str((out/"module_cache").resolve())
    from transformers import AutoModel,AutoTokenizer
    info=json.loads(manifest.read_text())
    assert info["revision"]=="7b12d946c181a37f6012b9dc3b002275de070314"
    for f in info["files"]:
        assert digest(modeldir/f["filename"])==f["sha256"]
    torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(modeldir,trust_remote_code=True,local_files_only=True)
    model,loading=AutoModel.from_pretrained(modeldir,trust_remote_code=True,local_files_only=True,
                  use_safetensors=True,deterministic_eval=True,output_loading_info=True)
    assert not loading["missing_keys"] and not loading.get("mismatched_keys")
    model.eval().requires_grad_(False)
    texts=data.connectivity.tolist()
    tokens=tokenizer(texts,truncation=False,padding=False)["input_ids"]
    lengths=np.array([len(t) for t in tokens])
    unknown=[t.count(tokenizer.unk_token_id) for t in tokens]
    order=np.argsort(lengths,kind="stable")
    x=np.empty((len(texts),model.config.hidden_size),dtype=np.float32)
    with torch.inference_mode():
        for first in range(0,len(texts),16):
            ids=order[first:first+16]
            batch=tokenizer([texts[i] for i in ids],padding=True,truncation=False,return_tensors="pt")
            x[ids]=model(**batch).pooler_output.numpy()
    assert np.isfinite(x).all()
    np.savez_compressed(out/"embeddings.npz",X=x,sample_id=data.sample_id.to_numpy(dtype=str))
    core.dump(out/"token_audit.json",{"revision":info["revision"],"n":len(x),"dimension":x.shape[1],
            "max_length":int(lengths.max()),"over_202":int((lengths>202).sum()),
            "unknown_tokens":int(sum(unknown)),"rows_with_unknown":int(np.count_nonzero(unknown))})
    return x


def run(a):
    sys.path.insert(0,str(a.study.resolve()))
    import core
    data=dataset(a.raw,a.study)
    dev=data[data.partition.ne("test")]
    allowed=set(dev.index)
    labels=pd.read_csv(a.raw,usecols=["pIC50"],skiprows=lambda i:i>0 and i-1 not in allowed)
    assert len(labels)==len(dev) and np.isfinite(labels.pIC50).all()
    y=np.full(len(data),np.nan)
    y[data.partition.ne("test")]=labels.pIC50
    train=np.flatnonzero(data.partition.eq("train"))
    valid=np.flatnonzero(data.partition.eq("valid"))
    out=a.out/a.route
    out.mkdir(parents=True,exist_ok=True)
    hashes={p.name:digest(p) for p in [a.raw,HERE/"PROTOCOL.md",HERE/"partitions.csv",HERE/"partition.py",
                                       Path(__file__),a.study/"core.py",a.study/"protocol.json"]}
    marker=out/"selection.json"
    if marker.exists():
        assert json.loads(marker.read_text())["hashes"]==hashes
        return
    x=core.make_graphs(data.connectivity) if a.route=="dmpnn" else embeddings(core,data,a.model,out,a.manifest)
    def fit(seed,lr):
        folder=out/f"seed{seed}_lr{lr}"
        folder.mkdir(exist_ok=True)
        meta=core.train_neural(a.route,x,y,train,valid,{"lr":lr},seed,folder)
        history=pd.read_csv(folder/"history.csv")
        pred=core.restore(a.route,folder,x,valid)
        assert np.isclose(history.valid_rmse.min(),meta["valid_rmse"]) and np.isclose(core.score(y[valid],pred)["rmse"],meta["valid_rmse"])
        pd.DataFrame({"sample_id":data.sample_id.iloc[valid],"y_true":y[valid],"y_pred":pred}).to_csv(folder/"validation.csv",index=False)
        print(a.route,seed,lr,meta["best_epoch"],meta["valid_rmse"],flush=True)
        return meta
    rates=core.PROTOCOL[a.route]["learning_rates"]
    assert rates==[.001,.0003]
    candidates=[fit(0,lr) for lr in rates]
    best=candidates[int(np.argmin([m["valid_rmse"] for m in candidates]))]
    chosen=[best]+[fit(s,best["lr"]) for s in [1,2]]
    core.dump(marker,{"route":a.route,"hashes":hashes,"seed0_candidates":candidates,"selected":chosen})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("route",choices=["mlp","dmpnn"])
    for arg in ["raw","study","out"]:
        p.add_argument("--"+arg,type=Path,required=True)
    for arg in ["model","manifest"]:p.add_argument("--"+arg,type=Path)
    run(p.parse_args())
