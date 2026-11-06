"""Fit a new predictor using only the grouped training and validation labels."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from partition_groups import HERE, manifest, partition

EMBEDDINGS_SHA = "f67d87948ce5b243e9ec44ea2e9f5bf7619ff7e40e2767035b4cc7746a0f0fbb"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(args):
    sys.path.insert(0, str(args.study.resolve()))
    import core
    data = partition(args.raw, args.study)
    stored = pd.read_csv(HERE / "partitions.csv", dtype={"group": data.group.dtype})
    pd.testing.assert_frame_equal(manifest(data), stored)
    original = json.loads((HERE.parent / "results/verification.json").read_text())
    assert digest(HERE.parent / "partitions.csv") == original["split_sha256"]
    np.testing.assert_array_equal(data.original_partition, pd.read_csv(HERE.parent / "partitions.csv").partition)
    return core, data


def inputs(core, route, data, cache):
    if route == "dmpnn":
        return core.make_graphs(data.connectivity)
    assert digest(cache) == EMBEDDINGS_SHA
    with np.load(cache) as vectors:
        np.testing.assert_array_equal(vectors["sample_id"], data.sample_id)
        return vectors["X"]


def lock(study):
    files = [HERE / n for n in ["fit.py", "partition_groups.py", "partitions.csv", "PROTOCOL.md"]]
    files.extend([study / "core.py", study / "protocol.json", HERE.parent / "partition.py"])
    return {f.name: digest(f) for f in files}


def run(a):
    core, data = setup(a)
    source_hashes = lock(a.study)
    root = a.runs / a.route
    marker = root / "selection.json"
    if marker.exists():
        assert json.loads(marker.read_text())["sources"] == source_hashes
        return
    visible = data.partition.ne("test")
    source_rows = set(data.index[visible])
    labels = pd.read_csv(a.raw, usecols=["pIC50"], skiprows=lambda i: i > 0 and i-1 not in source_rows)
    y = np.full(len(data), np.nan)
    y[visible] = labels.pIC50.to_numpy()
    assert np.isfinite(y[visible]).all()
    train = np.flatnonzero(data.partition == "train")
    valid = np.flatnonzero(data.partition == "valid")
    x = inputs(core, a.route, data, a.cache)

    def fit(seed, lr):
        folder = root / f"seed{seed}_lr{lr}"
        folder.mkdir(parents=True, exist_ok=True)
        meta = core.train_neural(a.route, x, y, train, valid, {"lr": lr}, seed, folder)
        predictions = core.restore(a.route, folder, x, valid)
        np.testing.assert_allclose(core.score(y[valid], predictions)["rmse"], meta["valid_rmse"], atol=1e-6)
        pd.DataFrame({"sample_id": data.iloc[valid].sample_id, "y_true": y[valid],
                      "y_pred": predictions.astype(float)}).to_csv(folder / "validation.csv", index=False)
        print(a.route, seed, lr, meta["best_epoch"], meta["valid_rmse"], flush=True)
        return meta

    candidates = [fit(0, lr) for lr in (0.001, 0.0003)]
    winner = min(candidates, key=lambda row: row["valid_rmse"])
    selected = [winner] + [fit(seed, winner["lr"]) for seed in (1, 2)]
    files = {str(f.relative_to(root).as_posix()): digest(f) for f in sorted(root.glob("seed*/*")) if f.is_file()}
    core.dump(marker, {"sources": source_hashes, "candidates": candidates, "selected": selected, "files": files})


def paths(parser):
    for key in ["raw", "study", "runs", "cache"]:
        parser.add_argument("--" + key, type=Path, required=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", choices=["dmpnn", "mlp"])
    paths(parser)
    run(parser.parse_args())
