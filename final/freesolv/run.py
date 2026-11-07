"""Freeze memberships, fit on development data, then score the saved FreeSolv models."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = Path(__file__).resolve().parent
RAW_SHA = "ab5895d914ee87cb563bd7b9611e869527bba45bec6b014d34dc495a0f9dcb72"
ROUTES = ("dmpnn", "mlp", "rf")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv(table, path):
    table.to_csv(path, index=False, lineterminator="\n")


def layout(args):
    assert sha(args.raw) == RAW_SHA
    raw = pd.read_csv(args.raw, usecols=["smiles"])
    mols = [Chem.MolFromSmiles(s) for s in raw.smiles]
    data = pd.DataFrame({
        "sample_id": [f"FS_{i:04}" for i in raw.index],
        "connectivity": [Chem.MolToSmiles(m, isomericSmiles=False) for m in mols],
        "scaffold": [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
                     or "__ACYCLIC__" for m in mols],
    })
    assert len(data) == 642 and data.scaffold.nunique() == 63
    frames = []
    for kind, key in (("random_group", "connectivity"), ("random_scaffold", "scaffold")):
        groups = list(data.groupby(key, sort=True).indices.values())
        capacity = len(data)//10
        eligible = [g for g in groups if len(g) <= capacity]
        for seed in (3101, 3102, 3103):
            split = np.full(len(data), "train", dtype=object)
            spare = dict(test=capacity, valid=capacity)
            for j in np.random.default_rng(seed).permutation(len(eligible)):
                g = eligible[j]
                for part in spare:
                    if len(g) <= spare[part]:
                        split[g] = part
                        spare[part] -= len(g)
                        break
            frame = data.assign(split=kind, split_seed=seed, partition=split)
            for group in ("connectivity", key):
                assert frame.groupby(group).partition.nunique().max() == 1
            frames.append(frame)
    return data, pd.concat(frames, ignore_index=True)


def features(args, core, data):
    started = time.perf_counter()
    if args.route == "dmpnn":
        x = core.make_graphs(data.connectivity)
    elif args.route == "mlp":
        sys.path.insert(0, str(HERE.parent/"bace"))
        from train import embeddings
        x = embeddings(core, data, args.model, args.work/args.route, args.manifest)
    else:
        desc = [Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s), missingVal=np.nan)
                for s in data.connectivity]
        x = pd.DataFrame(desc).to_numpy(dtype=float)
        x[~np.isfinite(x) | (np.abs(x)>np.finfo(np.float32).max)] = np.nan
        np.save(args.work/"rf/features.npy", x)
    core.dump(args.work/args.route/"preparation.json",
              {"seconds": time.perf_counter()-started, "records": len(x)})
    return x


def fit(args, core, data, partitions):
    route = args.route
    root = args.work/route
    root.mkdir(parents=True, exist_ok=True)
    x = features(args, core, data)
    candidates = (core.PROTOCOL["rf"]["candidates"] if route == "rf" else
                  [{"lr": lr} for lr in core.PROTOCOL[route]["learning_rates"]])
    trainer = core.train_tabular if route == "rf" else core.train_neural
    rows = []
    for (kind, split_seed), group in partitions.groupby(["split","split_seed"]):
        train = np.flatnonzero(group.partition.eq("train"))
        valid = np.flatnonzero(group.partition.eq("valid"))
        dev = np.sort(np.r_[train, valid])
        allowed = set(dev)
        y = np.full(len(data), np.nan)
        y[dev] = pd.read_csv(args.raw,usecols=["expt"],skiprows=lambda i:i>0 and i-1 not in allowed).expt

        def trial(seed, index):
            folder = root/f"{kind}_{split_seed}"/f"seed{seed}_candidate{index}"
            folder.mkdir(parents=True, exist_ok=True)
            meta = trainer(route,x,y,train,valid,candidates[index],seed,folder)
            pred = core.restore(route,folder,x,valid)
            np.testing.assert_allclose(core.score(y[valid],pred)["rmse"],meta["valid_rmse"],atol=1e-5)
            row = dict(model=route,split=kind,split_seed=split_seed,seed=seed,candidate=index,
                       params=json.dumps(candidates[index]),valid_rmse=meta["valid_rmse"],
                       fit_seconds=meta["fit_seconds"],best_epoch=meta.get("best_epoch"),
                       epochs_run=meta.get("epochs_run"),selected=False,
                       path=folder.relative_to(args.work).as_posix())
            rows.append(row)
            return row

        fits = [trial(0,i) for i in range(len(candidates))]
        winner = min(fits, key=lambda r:r["valid_rmse"])
        winner["selected"] = True
        if kind == "random_scaffold":
            for seed in (1,2):
                trial(seed,winner["candidate"])["selected"] = True
        print(route,kind,split_seed,"selected",winner["candidate"],flush=True)
    csv(pd.DataFrame(rows), root/"training.csv")


def evaluate(args,core,data,partitions):
    y = pd.read_csv(args.raw,usecols=["expt"]).expt.to_numpy()
    training = pd.concat([pd.read_csv(args.work/m/"training.csv") for m in ROUTES])
    selected = training[training.selected]
    assert len(training)==60 and len(selected)==36
    z = np.load(args.work/"mlp/embeddings.npz")
    np.testing.assert_array_equal(z["sample_id"],data.sample_id)
    features_by_model = {"mlp":z["X"],"rf":np.load(args.work/"rf/features.npy"),
                         "dmpnn":core.make_graphs(data.connectivity)}
    scores, predictions = [], []
    for (kind,split_seed),g in partitions.groupby(["split","split_seed"]):
        idx = np.flatnonzero(g.partition.eq("test"))
        dev = np.flatnonzero(g.partition.eq("train"))
        s = data.iloc[idx].scaffold.to_numpy()
        for (model,seed), run in selected[(selected.split==kind)&(selected.split_seed==split_seed)].groupby(["model","seed"]):
            row, = list(run.itertuples())
            p = core.restore(model,args.work/row.path,features_by_model[model],idx)
            predictions.append(pd.DataFrame(dict(split=kind,split_seed=split_seed,model=model,seed=seed,
                                      sample_id=data.sample_id.iloc[idx],scaffold=s,expt=y[idx],prediction=p)))
        predictions.append(pd.DataFrame(dict(split=kind,split_seed=split_seed,model="train_mean",seed=0,
                                  sample_id=data.sample_id.iloc[idx],scaffold=s,expt=y[idx],prediction=y[dev].mean())))
    predictions = pd.concat(predictions,ignore_index=True)
    for keys,g in predictions.groupby(["split","split_seed","model","seed"]):
        errors = g.prediction-g.expt
        scores.append(dict(zip(["split","split_seed","model","seed"],keys),n=len(g),
                           scaffolds=g.scaffold.nunique(),**core.score(g.expt,g.prediction),
                           scaffold_rmse=np.sqrt(errors.pow(2).groupby(g.scaffold).mean().mean())))
    for name,table in (("training",training),("scores",pd.DataFrame(scores)),("predictions",predictions)):
        csv(table,args.results/(name+".csv"))
    prep = {route:json.loads((args.work/route/"preparation.json").read_text()) for route in ROUTES}
    prep["tokens"] = json.loads((args.work/"mlp/token_audit.json").read_text())
    core.dump(args.results/"features.json",prep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=["prepare","fit","evaluate"])
    for key in ["raw","study","work","results","model","manifest"]:
        parser.add_argument("--"+key,type=Path,required=key not in ("model","manifest"))
    parser.add_argument("--route",choices=ROUTES)
    args=parser.parse_args()
    sys.path.insert(0,str(args.study.resolve()))
    import core
    data,partitions=layout(args)
    args.results.mkdir(parents=True,exist_ok=True)
    manifest=args.results/"partitions.csv"
    if args.phase=="prepare":
        csv(partitions,manifest)
    else:
        pd.testing.assert_frame_equal(partitions,pd.read_csv(manifest,keep_default_na=False))
        if args.phase=="fit":
            fit(args,core,data,partitions)
        else:
            evaluate(args,core,data,partitions)
