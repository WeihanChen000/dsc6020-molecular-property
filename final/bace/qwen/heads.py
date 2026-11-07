"""Fit CPU heads and replay their predictions against the archived comparison."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / "similarity"
sys.path.insert(0, str(SIM))
from fit import digest, setup


def fit_heads(a, core, data, x, sources):
    start = time.perf_counter()
    train = np.flatnonzero(data.partition == "train")
    valid = np.flatnonzero(data.partition == "valid")
    visible = data.partition.ne("test")
    source_rows = set(data.index[visible])
    labels = pd.read_csv(a.raw, usecols=["pIC50"], skiprows=lambda i: i > 0 and i-1 not in source_rows)
    y = np.full(len(data), np.nan)
    y[visible] = labels.pIC50.values
    assert np.isfinite(y[visible]).all()

    def train_one(seed, lr):
        folder = a.runs / f"seed{seed}_lr{lr}"
        folder.mkdir(parents=True, exist_ok=True)
        meta = core.train_neural("mlp", x, y, train, valid, {"lr": lr}, seed, folder)
        pred = core.restore("mlp", folder, x, valid)
        np.testing.assert_allclose(core.score(y[valid], pred)["rmse"], meta["valid_rmse"], atol=1e-6)
        pd.DataFrame({"sample_id": data.iloc[valid].sample_id, "y_true": y[valid],
                      "y_pred": pred.astype(float)}).to_csv(folder / "validation.csv", index=False)
        print(seed, lr, meta["best_epoch"], meta["valid_rmse"], flush=True)
        return meta

    candidates = [train_one(0, lr) for lr in (0.001, 0.0003)]
    winner = min(candidates, key=lambda r: r["valid_rmse"])
    selected = [winner] + [train_one(seed, winner["lr"]) for seed in (1, 2)]
    files = {p.relative_to(a.runs).as_posix(): digest(p) for p in sorted(a.runs.glob("seed*/*"))}
    core.dump(a.runs / "selection.json", {"files": files, "sources": sources, "selected": selected,
              "candidates": candidates, "head_wall_seconds": time.perf_counter()-start,
              "python": platform.python_version(), "torch": str(torch.__version__), "threads": torch.get_num_threads()})


def evaluate(a, core, data, x, locked):
    for name, checksum in locked["files"].items():
        assert digest(a.runs / name) == checksum
    ids = np.flatnonzero(data.partition == "test")
    labels = pd.read_csv(a.raw, usecols=["pIC50"]).iloc[data.index[ids]].pIC50.values
    prior = pd.read_csv(SIM / "results/grouped_predictions.csv")
    previous_scores = pd.read_csv(SIM / "results/scores.csv").query("split == 'grouped' and model == 'molformer_mlp'").set_index("seed")
    scores, records = [], []
    for row in locked["selected"]:
        seed, lr = row["seed"], row["lr"]
        table = prior.loc[prior.seed == seed, ["sample_id", "scaffold", "group", "seed", "y_true", "molformer_mlp"]].copy()
        np.testing.assert_array_equal(table.sample_id, data.iloc[ids].sample_id)
        np.testing.assert_array_equal(table.scaffold, data.iloc[ids].scaffold)
        np.testing.assert_allclose(table.y_true, labels, atol=1e-12, rtol=0)
        table["qwen_mlp"] = core.restore("mlp", a.runs / f"seed{seed}_lr{lr}", x, ids).astype(float)
        records.append(table)
        for model in ["qwen_mlp", "molformer_mlp"]:
            residual = table[model] - table.y_true
            metrics = core.score(table.y_true, table[model])
            metrics["scaffold_rmse"] = float(np.sqrt(residual.pow(2).groupby(table.scaffold).mean().mean()))
            if model == "molformer_mlp":
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, previous_scores.loc[seed, key], atol=1e-10, rtol=0)
            scores.append({"seed": seed, "model": model, **metrics})
    a.out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("predictions", pd.concat(records)), ("scores", pd.DataFrame(scores)),
                        ("training", pd.DataFrame(locked["candidates"] + locked["selected"][1:]))]:
        frame.to_csv(a.out / f"{name}.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fit", "evaluate"])
    for arg in ["raw", "study", "features", "runs", "out"]:
        parser.add_argument("--"+arg, type=Path, required=True)
    args = parser.parse_args()
    core, data = setup(args)
    cache = args.features / "vectors.npz"
    cache_hash = digest(cache)
    assert cache_hash == json.loads((args.features / "gpu.json").read_text())["cache_sha256"]
    with np.load(cache, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["sample_id"], data.sample_id)
        x = archive["X"]
    assert x.shape == (1503, 2048) and np.isfinite(x).all()
    sources = {f.name: digest(f) for f in [HERE / "PROTOCOL.md", args.study / "core.py", args.study / "protocol.json", SIM / "partitions.csv"]}
    sources["vectors.npz"] = cache_hash
    marker = args.runs / "selection.json"
    if args.action == "fit" and not marker.exists():
        fit_heads(args, core, data, x, sources)
    locked = json.loads(marker.read_text())
    assert locked["sources"] == sources
    if args.action == "evaluate":
        evaluate(args, core, data, x, locked)
