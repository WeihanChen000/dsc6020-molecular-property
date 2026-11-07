"""Fit three exact-IC50 pipelines using the published similarity split and CPU trainer."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / "similarity"
sys.path.insert(0, str(SIM))
from fit import digest, inputs, setup

MODELS = ("dmpnn", "molformer_mlp", "qwen_mlp")


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["study", "raw", "qwen", "molformer", "runs"]:
        p.add_argument("--"+key, type=Path, required=True)
    return p


def members(a):
    core, data = setup(a)
    labels = HERE.parent / "qwen/assay_audit/labels.csv"
    assert digest(labels) == "1f5ecf0220c88b276d949b52ed616ffac68d6e5e0a6f4beabe2aedaebd824330"
    assert digest(a.study/"core.py") == "8048aea406dd5cbc8b255e08b055ccf6eec0e6eadc2c415155c7dce00fca6194"
    assert digest(a.study/"protocol.json") == "d7710207d0492e3ee735fbb863f3b9dc4a6fad530a375d109f7d95ab5d0a0b22"
    audited = pd.read_csv(labels, usecols=["sample_id", "group", "partition", "exact_ic50"])
    for key in ["sample_id","partition","group"]:
        np.testing.assert_array_equal(audited[key], data[key])
    data["keep"] = audited.exact_ic50.to_numpy()
    assert data[data.keep].partition.value_counts().to_dict() == {"train":992, "valid":147, "test":268}
    return core, data


def features(core, data, a, model):
    if model != "qwen_mlp":
        return inputs(core, "dmpnn" if model == "dmpnn" else "mlp", data, a.molformer)
    assert digest(a.qwen) == "7dcaa4cc8095d98b46176a4471cd13bca776100068fce1f9343e4923c874d6b8"
    with np.load(a.qwen) as z:
        np.testing.assert_array_equal(z["sample_id"], data.sample_id)
        assert z["X"].shape == (len(data), 2048)
        return z["X"]


def train(a):
    core, data = members(a)
    tr = np.flatnonzero(data.keep & data.partition.eq("train"))
    va = np.flatnonzero(data.keep & data.partition.eq("valid"))
    visible = data.keep & data.partition.ne("test")
    rows = set(data.index[visible])
    y = np.full(len(data), np.nan)
    y[visible] = pd.read_csv(a.raw, usecols=["pIC50"], skiprows=lambda i: i>0 and i-1 not in rows).pIC50
    assert np.isfinite(y).sum() == 1139
    for model in MODELS:
        root = a.runs/model
        marker = root/"selection.json"
        if marker.exists():
            saved = json.loads(marker.read_text())
            assert saved["protocol"] == digest(HERE/"PROTOCOL.md")
            for name, checksum in saved["files"].items():
                assert digest(root/name) == checksum
            continue
        x = features(core, data, a, model)
        route = "dmpnn" if model == "dmpnn" else "mlp"

        def candidate(seed, lr):
            path = root/f"seed{seed}_lr{lr}"
            path.mkdir(parents=True, exist_ok=True)
            result = core.train_neural(route, x, y, tr, va, {"lr":lr}, seed, path)
            pred = core.restore(route, path, x, va)
            np.testing.assert_allclose(core.score(y[va], pred)["rmse"], result["valid_rmse"], atol=1e-6)
            pd.DataFrame({"sample_id":data.iloc[va].sample_id, "y_true":y[va], "prediction":pred}).to_csv(
                path/"validation.csv", index=False, lineterminator="\n")
            print(model, seed, lr, result["best_epoch"], result["valid_rmse"], flush=True)
            return result

        grid = [candidate(0, lr) for lr in (0.001, 0.0003)]
        best = min(grid, key=lambda r:r["valid_rmse"])
        chosen = [best]+[candidate(seed, best["lr"]) for seed in (1,2)]
        files = {p.relative_to(root).as_posix():digest(p) for p in sorted(root.glob("seed*/*"))}
        core.dump(marker, {"model":model, "protocol":digest(HERE/"PROTOCOL.md"), "files":files,
                           "candidates":grid, "selected":chosen})


if __name__ == "__main__":
    train(arguments().parse_args())
