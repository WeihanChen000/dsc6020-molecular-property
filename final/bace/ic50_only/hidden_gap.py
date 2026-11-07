"""Decompose the mean of five individual neighbor predictions minus a target prediction."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch

from equal_activity_pair import CACHES, HERE, PAIR, digest


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5)


def decompose(net, scaled, meta):
    with torch.inference_mode():
        inputs = torch.tensor(scaled, dtype=torch.float32)
        hidden = net.net[:-1](inputs).numpy().astype(float)
        predictions = (net(inputs) * meta["y_scale"] + meta["y_mean"]).numpy().astype(float)
        readout = net.net[-1].weight[0].numpy().astype(float) * meta["y_scale"]

    # Independent float64 forward pass, starting at the saved first-layer weights.
    state = net.state_dict()
    z = inputs.numpy().astype(float) @ state["net.0.weight"].numpy().T + state["net.0.bias"].numpy()
    manual_h = np.maximum(0, z)
    intercept = state["net.3.bias"].item() * meta["y_scale"] + meta["y_mean"]
    check(manual_h @ readout + intercept, predictions)
    contributions = (hidden[1:] - hidden[0]) * readout
    check((manual_h[1:] - manual_h[0]) * readout, contributions)
    check(contributions.sum(axis=1), predictions[1:] - predictions[0])
    return predictions, hidden, readout, contributions


def run(args):
    sys.path.insert(0, str(args.study.resolve()))
    from core import MLP

    source = HERE / "equal_activity_pair"
    neighbors = pd.read_csv(source / "neighbors.csv")
    previous = pd.read_csv(source / "neighborhoods.csv").set_index(["metric", "case", "k"])
    targets = pd.read_csv(source / "seed_predictions.csv").set_index(["model", "seed"])
    units, cases, pairs = [], [], []
    for model, checksum in CACHES.items():
        cache = getattr(args, model)
        assert digest(cache) == checksum
        with np.load(cache) as data:
            ids = pd.Index(data["sample_id"])
            x = data["X"]
        assert ids.is_unique
        root = args.heads / (model + "_mlp")
        lock = json.loads((root / "selection.json").read_text())
        assert sorted(item["seed"] for item in lock["selected"]) == [0, 1, 2]
        for selected in lock["selected"]:
            seed = selected["seed"]
            folder = root / f"seed{seed}_lr{selected['lr']}"
            for name in ("model.pt", "preprocessor.joblib"):
                assert digest(folder / name) == lock["files"][f"{folder.name}/{name}"]
            scaler = joblib.load(folder / "preprocessor.joblib")
            checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
            net = MLP(x.shape[1]).eval()
            net.load_state_dict(checkpoint["state_dict"])
            metric = model + "_scaled"
            for case in PAIR:
                members = neighbors.loc[(neighbors.metric == metric) & (neighbors.case == case)
                                        & (neighbors["rank"] <= 5)].sort_values("rank")
                reference = previous.loc[(metric, case, 5)]
                assert members.train_id.nunique() == len(members) == reference["n"] == 5
                pos = ids.get_indexer([case, *members.train_id])
                assert pos.min() >= 0
                pred, h, w, individual = decompose(net, scaler.transform(x[pos]), checkpoint["metadata"])
                c = individual.mean(axis=0)
                gap = pred[1:].mean() - pred[0]
                check(pred[0], targets.loc[(model, seed), "pred_" + case.split("_")[1]])
                check(pred[1:].mean(), reference[f"mean_fit_seed{seed}"])
                check(c.sum(), gap)
                key = dict(model=model, seed=seed, case=case)
                for member, yhat, row in zip(members.train_id, pred[1:], individual, strict=True):
                    pairs.append(dict(**key, train_id=member, neighbor_pred=yhat,
                                      neighbor_minus_target=yhat-pred[0], unit_sum=row.sum()))
                for j in range(len(w)):
                    units.append(dict(**key, unit=j+1, effective_weight=w[j], target_h=h[0,j],
                                      mean_neighbor_h=h[1:,j].mean(), active_neighbors=(h[1:,j]>0).sum(),
                                      neighbors_with_positive_contribution=(individual[:,j]>0).sum(),
                                      contribution=c[j]))
                order = np.argsort(-c, kind="stable")
                cases.append(dict(**key, target_pred=pred[0], mean_neighbor_pred=pred[1:].mean(),
                                  gap=gap, positive_sum=c[c>0].sum(), negative_sum=c[c<0].sum(),
                                  top_unit=order[0]+1, top_contribution=c[order[0]],
                                  top3_sum=c[order[:3]].sum(), bottom_unit=order[-1]+1,
                                  bottom_contribution=c[order[-1]], reconstructed=c.sum()))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("units", units), ("summary", cases), ("neighbors", pairs)):
        pd.DataFrame(rows).to_csv(args.out / (name + ".csv"), index=False, lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("qwen", "molformer", "heads", "study", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    run(parser.parse_args())
