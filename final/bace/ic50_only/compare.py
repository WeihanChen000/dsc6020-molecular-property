"""Replay all selected exact-IC50 fits and pair them with the archived mixed-label fits."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fit_exact import HERE, SIM, MODELS, arguments, digest, features, members


def evaluate(a):
    core, data = members(a)
    selected = {}
    trials = []
    for model in MODELS:
        root = a.runs/model
        lock = json.loads((root/"selection.json").read_text())
        assert lock["model"] == model and lock["protocol"] == digest(HERE/"PROTOCOL.md")
        for name, checksum in lock["files"].items():
            assert digest(root/name) == checksum
        selected[model] = lock["selected"]
        assert [r["seed"] for r in selected[model]] == [0,1,2]
        best_lr = min(lock["candidates"], key=lambda r:r["valid_rmse"])["lr"]
        assert all(r["lr"] == best_lr for r in selected[model])
        for row in lock["candidates"] + selected[model][1:]:
            trials.append({"model":model, "selected":row["lr"] == best_lr, **row})

    test = np.flatnonzero(data.keep & data.partition.eq("test"))
    ids = data.iloc[test].sample_id.tolist()
    y = pd.read_csv(a.raw, usecols=["pIC50"]).pIC50.iloc[data.index[test]].to_numpy()
    qwen = pd.read_csv(HERE.parent/"qwen/results/predictions.csv")
    mixed = pd.read_csv(SIM/"results/grouped_predictions.csv").merge(
        qwen, on=["seed","sample_id","scaffold","group","y_true","molformer_mlp"], validate="one_to_one")
    mixed = mixed[mixed.sample_id.isin(ids)].set_index(["seed","sample_id"]).sort_index()
    assert len(mixed) == 804
    old_scores = pd.read_csv(HERE.parent/"qwen/assay_audit/scores.csv").query(
        "scope == 'exact_ic50'").set_index(["seed","model"])
    records, scores = [], []
    for model in MODELS:
        x = features(core, data, a, model)
        for row in selected[model]:
            seed = row["seed"]
            old = mixed.loc[seed].loc[ids]
            np.testing.assert_array_equal(old.scaffold, data.iloc[test].scaffold)
            np.testing.assert_allclose(old.y_true, y, rtol=0, atol=1e-12)
            path = a.runs/model/f"seed{seed}_lr{row['lr']}"
            pred = core.restore(row["route"], path, x, test).astype(float)
            frame = old[["scaffold","group","y_true",model]].rename(columns={model:"mixed"}).reset_index()
            frame["exact"] = pred
            frame["seed"] = seed
            frame["model"] = model
            records.append(frame)
            result = {"seed":seed, "model":model, "n":len(old), "scaffolds":old.scaffold.nunique()}
            for scope in ["mixed","exact"]:
                metrics = core.score(y,frame[scope])
                metrics["scaffold_rmse"] = np.sqrt((frame[scope]-y).pow(2).groupby(frame.scaffold).mean().mean())
                for key,value in metrics.items():
                    result[f"{scope}_{key}"] = value
                    if scope == "mixed":
                        np.testing.assert_allclose(value, old_scores.at[(seed,model),key], atol=1e-12, rtol=0)
                    else:
                        result["change_"+key] = value-result["mixed_"+key]
            scores.append(result)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("scores",pd.DataFrame(scores)), ("predictions",pd.concat(records)),
                        ("training",pd.DataFrame(trials))]:
        frame.to_csv(a.out/f"{name}.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    parser = arguments()
    parser.add_argument("--out", type=Path, required=True)
    evaluate(parser.parse_args())
