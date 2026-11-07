"""Replay first-layer inputs for the two 44 nM cases, Qwen seed 0 only."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from hidden_gap import CACHES, HERE, PAIR, check, digest


UNITS = np.array([56, 11, 64])  # Published one-based IDs, in output-contribution order.


def threshold_path(x, z, unit, weight, bias, delta, reference_dims):
    """Find the first prefix that activates a unit or enters its neighbors' interval."""
    low, high = z[1:].min(), z[1:].max()
    activation_goal = unit in (56, 11)
    direction = 1 if activation_goal or z[0] < low else -1
    order = np.argsort(-direction * delta, kind="stable")
    path = z[0] + np.r_[0., np.cumsum(delta[order])]
    reached = path > 0 if activation_goal else (path >= low) & (path <= high)
    crossings = np.flatnonzero(reached)
    if not len(crossings):
        raise ValueError(f"No prefix reached the requested threshold for unit {unit}")
    k = int(crossings[0])
    chosen = order[:k]
    mean_x = x[1:].mean(axis=0)
    # Replay every published prefix with an actual modified vector in float32.
    batch = np.tile(x[0], (k+1, 1))
    for step, coordinate in enumerate(chosen, start=1):
        batch[step:, coordinate] = mean_x[coordinate]

    def replay(values):
        return torch.nn.functional.linear(torch.as_tensor(values, dtype=torch.float32),
                                          weight[None, :], bias.reshape(1)).flatten().numpy()

    direct = replay(batch)
    check(direct, path[:k+1])
    direct_hit = direct > 0 if activation_goal else (direct >= low) & (direct <= high)
    np.testing.assert_array_equal(np.flatnonzero(direct_hit), [k])
    check(path[-1], z[1:].mean())
    transferred = x[0].copy()
    if reference_dims is None:
        reference_dims = chosen
    transferred[reference_dims] = mean_x[reference_dims]
    ref_z = replay(transferred)[0]
    stats = dict(goal="activate" if activation_goal else "neighbor_range", min_k=k,
                 start_z=z[0], neighbor_min_z=low, neighbor_max_z=high,
                 previous_z=path[k-1] if k else None, first_hit_z=path[k],
                 crossing_dimension=int(chosen[-1])+1 if k else None, all_replaced_z=path[-1],
                 z_with_0638_dims=ref_z)
    steps = pd.DataFrame(dict(k=np.arange(k+1), dimension=pd.array([None, *(chosen+1)], dtype="Int64"),
                              shift=np.r_[0.,delta[chosen]], preactivation=path[:k+1],
                              direct_preactivation=direct, reached=direct_hit))
    return stats, steps, chosen


def run(args):
    root = args.heads / "qwen_mlp"
    lock = json.loads((root / "selection.json").read_text())
    selected, = [row for row in lock["selected"] if row["seed"] == 0]
    folder = root / f"seed0_lr{selected['lr']}"
    for name in ("preprocessor.joblib", "model.pt"):
        assert digest(folder / name) == lock["files"][f"{folder.name}/{name}"]
    assert digest(args.qwen) == CACHES["qwen"]
    scaler = joblib.load(folder / "preprocessor.joblib")
    checkpoint = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    weights = state["net.0.weight"][UNITS - 1].double().numpy()
    biases = state["net.0.bias"][UNITS - 1].double().numpy()
    output_weights = state["net.3.weight"][0, UNITS - 1].double().numpy() * checkpoint["metadata"]["y_scale"]
    with np.load(args.qwen) as cache:
        ids = pd.Index(cache["sample_id"])
        vectors = cache["X"]
    assert ids.is_unique and vectors.shape[1] == weights.shape[1] == 2048
    members = pd.read_csv(HERE / "equal_activity_pair/neighbors.csv")
    members = members[(members.metric == "qwen_scaled") & (members["rank"] <= 5)]
    archived = pd.read_csv(HERE / "hidden_gap/units.csv").query("model == 'qwen' and seed == 0")
    archived = archived.set_index(["case", "unit"])
    summaries, samples, dimensions = [], [], []
    thresholds, prefixes = [], []
    dims_0638 = {}
    for case in PAIR:
        near = members[members.case == case].sort_values("rank")
        assert near.train_id.nunique() == len(near) == 5 and case not in set(near.train_id)
        names = [case, *near.train_id]
        positions = ids.get_indexer(names)
        assert positions.min() >= 0
        inputs = torch.tensor(scaler.transform(vectors[positions]), dtype=torch.float32)
        x = inputs.double().numpy()
        mean_x = x[1:].mean(axis=0)
        actual = torch.nn.functional.linear(inputs, state["net.0.weight"], state["net.0.bias"])
        for i, unit in enumerate(UNITS):
            weight = weights[i]
            weighted = x * weight
            z = weighted.sum(axis=1) + biases[i]
            check(z, actual[:, unit-1].numpy())
            h = np.maximum(z, 0)
            ref = archived.loc[(case, unit)]
            check([h[0], h[1:].mean()], [ref.target_h, ref.mean_neighbor_h])
            check((h[1:].mean() - h[0]) * output_weights[i], ref.contribution)
            gaps = weighted[1:] - weighted[0]
            check(gaps.sum(axis=1), z[1:] - z[0])
            delta = gaps.mean(axis=0)
            check(delta.sum(), z[1:].mean() - z[0])
            key = dict(case=case, unit=unit)
            stats, steps, chosen = threshold_path(x, z, unit, state["net.0.weight"][unit-1],
                                                  state["net.0.bias"][unit-1], delta,
                                                  dims_0638.get(unit))
            if case == PAIR[0]:
                assert case == "BACE_0638"
                dims_0638[unit] = chosen
            thresholds.append(dict(**key, **stats))
            prefixes.append(steps.assign(**key))
            # Order by support for the observed direction. Retain opposing dimensions as well.
            direction = np.sign(delta.sum())
            order = np.argsort(-direction * delta, kind="stable")
            top = order[:3]
            replaced = x[0].copy()
            replaced[top] = mean_x[top]
            replaced_z = replaced @ weight + biases[i]
            check(replaced_z, z[0] + delta[top].sum())
            summaries.append(dict(**key, bias=biases[i], target_z=z[0], mean_neighbor_z=z[1:].mean(),
                                  min_neighbor_z=z[1:].min(), max_neighbor_z=z[1:].max(),
                                  neighbor_minus_target_z=delta.sum(), top3_shift=delta[top].sum(),
                                  target_z_with_top3_replaced=replaced_z))
            for sample_id, preactivation, activation in zip(names, z, h, strict=True):
                samples.append(dict(**key, sample_id=sample_id, preactivation=preactivation,
                                    activation=activation))
            dimensions.append(pd.DataFrame(dict(**key, dimension=order+1, weight=weight[order],
                                           target_input=x[0,order], mean_neighbor_input=mean_x[order],
                                           target_term=weighted[0,order], mean_neighbor_minus_target_term=delta[order],
                                           neighbors_with_same_gap_sign=(np.sign(gaps[:,order]) ==
                                                                         np.sign(delta[order])).sum(axis=0))))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in (("summary", pd.DataFrame(summaries)), ("samples", pd.DataFrame(samples)),
                        ("dimensions", pd.concat(dimensions, ignore_index=True)),
                        ("thresholds", pd.DataFrame(thresholds)),
                        ("prefixes", pd.concat(prefixes, ignore_index=True))):
        table.to_csv(args.out / (name + ".csv"), index=False, float_format="%.12g", lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("qwen", "heads", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    run(parser.parse_args())
