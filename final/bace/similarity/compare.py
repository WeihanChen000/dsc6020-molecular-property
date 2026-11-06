"""Verify completed validation selections, then compare the two BACE holdouts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fit import HERE, digest, inputs, lock, paths, setup


def selections(a, data):
    records = {}
    for route in ("dmpnn", "mlp"):
        root = a.runs / route
        selection = json.loads((root / "selection.json").read_text())
        assert selection["sources"] == lock(a.study)
        for name, expected in selection["files"].items():
            assert digest(root / name) == expected
        candidates, selected = selection["candidates"], selection["selected"]
        assert [(m["seed"], m["lr"]) for m in candidates] == [(0, .001), (0, .0003)]
        best = min(candidates, key=lambda m: m["valid_rmse"])
        assert selected[0] == best and [m["seed"] for m in selected] == [0, 1, 2]
        assert all(m["lr"] == best["lr"] for m in selected)
        trials = candidates + selected[1:]
        for m in trials:
            folder = root / f"seed{m['seed']}_lr{m['lr']}"
            val = pd.read_csv(folder / "validation.csv")
            np.testing.assert_array_equal(val.sample_id, data.sample_id[data.partition == "valid"])
            np.testing.assert_allclose(np.sqrt(((val.y_pred-val.y_true)**2).mean()), m["valid_rmse"])
            history = pd.read_csv(folder / "history.csv")
            assert np.isclose(history.valid_rmse.min(), m["valid_rmse"])
            assert len(history) == m["epochs_run"]
        records[route] = trials
    return records


def compare(a):
    core, data = setup(a)
    trials = selections(a, data)
    labels = pd.read_csv(a.raw, usecols=["pIC50"]).pIC50.loc[data.index].to_numpy()
    assert np.isfinite(labels).all()
    for route, runs in trials.items():
        for m in runs:
            folder = a.runs / route / f"seed{m['seed']}_lr{m['lr']}"
            np.testing.assert_allclose(pd.read_csv(folder / "validation.csv").y_true, labels[data.partition == "valid"])
            assert np.isclose(m["y_mean"], labels[data.partition == "train"].mean())

    old = pd.read_csv(HERE.parent / "results/predictions.csv")
    old_neighbors = pd.read_csv(HERE.parent / "review/neighbors.csv").set_index("sample_id")
    test_idx = np.flatnonzero(data.partition == "test")
    test = data.iloc[test_idx]
    base = test[["sample_id", "scaffold", "group"]].reset_index(drop=True).copy()
    base["y_true"] = labels[test_idx]
    base["train_mean"] = labels[data.partition == "train"].mean()
    frames = [base.assign(seed=s) for s in range(3)]
    training = []
    for route, fits in trials.items():
        x = inputs(core, route, data, a.cache)
        best_lr = min(fits[:2], key=lambda m: m["valid_rmse"])["lr"]
        for m in fits:
            selected = m["lr"] == best_lr
            training.append({**m, "selected": selected})
            if selected:
                folder = a.runs / route / f"seed{m['seed']}_lr{m['lr']}"
                name = "dmpnn" if route == "dmpnn" else "molformer_mlp"
                frames[m["seed"]][name] = core.restore(route, folder, x, test_idx).astype(float)
    new = pd.concat(frames, ignore_index=True)
    assert np.isfinite(new[["dmpnn", "molformer_mlp"]]).all().all()
    scores, structures, context, bands = [], [], [], []
    for split, predictions, scaffold in [("original", old, "scaffold_id"), ("grouped", new, "scaffold")]:
        parts = data[split + "_partition"]
        members = data.loc[parts == "test"]
        sim = members[split + "_nearest_train"]
        if split == "original":
            np.testing.assert_allclose(sim, old_neighbors.loc[members.sample_id].max_similarity)
        for _, row in members.iterrows():
            structures.append({"split": split, "sample_id": row.sample_id,
                               "max_training_similarity": row[split + "_nearest_train"]})
        for part in ("train", "valid", "test"):
            mask = parts == part
            y = labels[mask]
            context.append({"split": split, "partition": part, "n": len(y), "scaffolds": data.scaffold[mask].nunique(),
                            "target_mean": y.mean(), "target_sd": y.std(ddof=1), "target_min": y.min(), "target_max": y.max()})
        for seed, frame in predictions.groupby("seed"):
            np.testing.assert_array_equal(frame.sample_id, members.sample_id)
            np.testing.assert_allclose(frame.y_true, labels[parts == "test"])
            for model in ("dmpnn", "molformer_mlp", "train_mean"):
                err = frame[model] - frame.y_true
                scores.append({"split": split, "seed": seed, "model": model, "rmse": np.sqrt(err.pow(2).mean()),
                               "scaffold_rmse": np.sqrt(err.pow(2).groupby(frame[scaffold]).mean().mean()),
                               "mae": err.abs().mean()})
                for band, mask in [("<0.3", sim < .3), ("[0.3,0.6)", (sim >= .3) & (sim < .6)),
                                   (">=0.6", sim >= .6)]:
                    subset = err.iloc[np.flatnonzero(mask)]
                    bands.append({"split": split, "seed": seed, "model": model, "band": band, "n": len(subset),
                                  "rmse": np.sqrt(subset.pow(2).mean()), "mae": subset.abs().mean()})
    scores = pd.DataFrame(scores)
    reference = pd.read_csv(HERE.parent / "results/scores.csv").set_index("seed")
    for r in scores[scores.split == "original"].itertuples():
        for k in ["rmse", "scaffold_rmse", "mae"]:
            assert np.isclose(getattr(r, k), reference.loc[r.seed, r.model + "_" + k])
    metrics = scores[scores.split == "grouped"].set_index(["model", "seed"])[["rmse", "scaffold_rmse"]]
    def wins(model, rival):
        return (metrics.loc[model] < metrics.loc[rival]).all().all() and (metrics.loc[model] < metrics.loc["train_mean"]).all().all()
    verdict = "retained" if wins("molformer_mlp", "dmpnn") else "reversed" if wins("dmpnn", "molformer_mlp") else "mixed"
    structural = pd.DataFrame(structures)
    shared_ids = set(old.sample_id) & set(new.sample_id)
    shared = []
    for split, frame in [("original", old), ("grouped", new)]:
        for seed, block in frame[frame.sample_id.isin(shared_ids)].groupby("seed"):
            assert len(block) == len(shared_ids)
            for model in ("dmpnn", "molformer_mlp"):
                err = block[model] - block.y_true
                shared.append({"split": split, "seed": seed, "model": model, "n": len(block),
                               "rmse": np.sqrt(err.pow(2).mean()), "mae": err.abs().mean()})
    summary = []
    for name, table in structural.groupby("split", sort=False):
        s = table.max_training_similarity
        summary.append({"split": name, "mean": s.mean(), "median": s.median(), "max": s.max(),
                        "below_0.3": (s < .3).sum(), "0.3_to_below_0.6": ((s >= .3) & (s < .6)).sum(),
                        "at_least_0.6": (s >= .6).sum()})
    a.out.mkdir(parents=True, exist_ok=True)
    for name, table in [("scores", scores), ("grouped_predictions", new), ("neighbors", structural),
                        ("similarity_summary", pd.DataFrame(summary)), ("training", pd.DataFrame(training)),
                        ("partitions_summary", pd.DataFrame(context)), ("shared_test", pd.DataFrame(shared)),
                        ("similarity_band_scores", pd.DataFrame(bands))]:
        table.to_csv(a.out / (name + ".csv"), index=False, lineterminator="\n")
    core.dump(a.out / "verification.json", {"verdict": verdict, "fits": len(training), "selected": 6,
              "shared_test_records": int((data.partition.eq("test") & data.original_partition.eq("test")).sum()),
              "grouped_test_components": int(test.group.nunique()), "largest_test_component": int(test.group.value_counts().max()),
              "selection_sha256": {r: digest(a.runs / r / "selection.json") for r in trials}})
    print(scores.round(4).to_string(index=False), verdict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    paths(parser)
    parser.add_argument("--out", type=Path, required=True)
    compare(parser.parse_args())
