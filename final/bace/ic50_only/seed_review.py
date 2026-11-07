"""Audit stored validation checkpoints and decompose the change in paired test errors."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS = ["qwen_mlp", "molformer_mlp"]


def checkpoints(a, labels):
    rows = []
    selected_validation = []
    valid = labels.query("partition == 'valid'").set_index("sample_id")
    for scope, model, root in [
        ("mixed", MODELS[0], a.qwen), ("mixed", MODELS[1], a.molformer),
        ("exact", MODELS[0], a.exact/MODELS[0]), ("exact", MODELS[1], a.exact/MODELS[1])]:
        lock = json.loads((root/"selection.json").read_text())
        selected = {(r["seed"], r["lr"]) for r in lock["selected"]}
        assert len(selected) == 3 and len({lr for _, lr in selected}) == 1
        ids = valid.index if scope == "mixed" else valid.index[valid.exact_ic50]
        for c in lock["candidates"] + lock["selected"][1:]:
            seed, lr = c["seed"], c["lr"]
            folder = root/f"seed{seed}_lr{lr}"
            h = pd.read_csv(folder/"history.csv").set_index("epoch")
            best, stop = int(h.valid_rmse.idxmin()), int(h.index[-1])
            assert (best, stop) == (c["best_epoch"], c["epochs_run"]) and stop-best == 40
            v = pd.read_csv(folder/"validation.csv").set_index("sample_id").loc[:, ["y_true", "prediction" if scope=="exact" else "y_pred"]]
            assert v.index.is_unique and set(v.index) == set(ids)
            np.testing.assert_allclose(v.y_true, valid.loc[v.index].pIC50, atol=1e-12, rtol=0)
            sq = (v.iloc[:,1]-v.y_true)**2
            np.testing.assert_allclose(np.sqrt(sq.mean()), h.loc[best,"valid_rmse"], atol=1e-6, rtol=0)
            if scope == "exact" and (seed, lr) in selected:
                selected_validation.append(v.rename(columns={"prediction":model})[[model]].assign(seed=seed).reset_index())
            rows.append({"scope":scope,"model":model,"seed":seed,"lr":lr,"selected":(seed,lr) in selected,
                         "best_epoch":best,"stop_epoch":stop,"valid_best":h.loc[best,"valid_rmse"],
                         "valid_shared147":np.sqrt(sq.loc[valid.index[valid.exact_ic50]].mean()),
                         "valid_first":h.iloc[0].valid_rmse,"valid_final":h.iloc[-1].valid_rmse,
                         "train_first":h.iloc[0].train_mse_standardized,"train_final":h.iloc[-1].train_mse_standardized})
    q, m = [pd.concat([f for f in selected_validation if model in f]) for model in MODELS]
    paired = q.merge(m, on=["seed","sample_id"],validate="one_to_one").join(valid.pIC50,on="sample_id")
    paired["gap"] = (paired.qwen_mlp-paired.pIC50)**2 - (paired.molformer_mlp-paired.pIC50)**2
    return pd.DataFrame(rows), paired


def contributions(labels):
    raw = pd.read_csv(HERE/"results/predictions.csv")
    ref = pd.read_csv(HERE/"results/scores.csv").set_index(["seed","model"])
    key = ["seed","sample_id","scaffold","group","y_true"]
    p = raw[raw.model.isin(MODELS)].pivot(index=key, columns="model", values=["mixed","exact"])
    p.columns = ["_".join(parts) for parts in p.columns]
    p = p.reset_index()
    test = labels.query("partition == 'test' and exact_ic50").set_index("sample_id")
    for seed, g in p.groupby("seed"):
        assert len(g) == len(test) == 268 and set(g.sample_id) == set(test.index) and g.scaffold.nunique() == 163
        np.testing.assert_allclose(g.y_true, test.loc[g.sample_id].pIC50, rtol=0, atol=1e-12)
        for scope in ["mixed","exact"]:
            for model in MODELS:
                mse = ((g[scope+"_"+model]-g.y_true)**2).mean()
                np.testing.assert_allclose(mse, ref.at[(seed,model),scope+"_rmse"]**2, atol=1e-12)
    for scope in ["mixed", "exact"]:
        sq_q = (p[scope+"_qwen_mlp"]-p.y_true)**2
        sq_m = (p[scope+"_molformer_mlp"]-p.y_true)**2
        p[scope+"_gap"] = sq_q-sq_m
    p["swing"] = p.exact_gap-p.mixed_gap
    skeleton = p.groupby(["seed","scaffold"],as_index=False).agg(
        first_id=("sample_id","min"), n=("sample_id","size"), mixed_gap=("mixed_gap","sum"),
        exact_gap=("exact_gap","sum"), swing=("swing","sum"))
    return p, skeleton


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["exact","qwen","molformer","out"]:
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    labels = pd.read_csv(HERE.parent/"qwen/assay_audit/labels.csv")
    fits, validation = checkpoints(args, labels)
    molecules, skeletons = contributions(labels)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("checkpoints",fits),("validation",validation),("molecules",molecules),("scaffolds",skeletons)]:
        frame.to_csv(args.out/f"{name}.csv", index=False, lineterminator="\n")
