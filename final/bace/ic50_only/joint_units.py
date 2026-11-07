"""Shared coordinate substitutions in the frozen Qwen seed-0 regression head."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from input_gap import CACHES, HERE, PAIR, UNITS, check, digest


def shortest_mask(gains, required):
    # Each row of gains @ mask >= required represents one preactivation boundary.
    n = gains.shape[1]
    relaxation = linprog(np.ones(n), A_ub=-gains, b_ub=-required, bounds=(0, 1), method="highs")
    assert relaxation.success
    multipliers = np.maximum(-relaxation.ineqlin.marginals, 0)
    lower = multipliers @ required - np.maximum(multipliers @ gains - 1, 0).sum()
    log = dict(fractional_lower_bound=lower, multipliers=multipliers.tolist(), fixed_sizes=[])
    # At fixed k, maximize the smallest signed distance to the four boundaries.
    matrix = np.block([[gains, -np.ones((4, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    bounds = Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf])
    for k in range(int(np.ceil(lower-1e-8)), n+1):
        cons = LinearConstraint(matrix, np.r_[required, k], np.r_[np.full(4, np.inf), k])
        result = milp(np.r_[np.zeros(n), -1], integrality=np.r_[np.ones(n), 0], bounds=bounds,
                      constraints=cons, options={"mip_rel_gap": 0.0})
        if not result.success:
            raise RuntimeError(result.message)
        mask = result.x[:-1] > 0.5
        margins = gains @ mask - required
        assert mask.sum() == k
        check(margins.min(), -result.fun)
        upper = -float(result.mip_dual_bound)
        log["fixed_sizes"].append(dict(k=k, achieved=float(margins.min()), upper_bound=upper,
                                       nodes=int(result.mip_node_count)))
        if (margins[:2] > 1e-6).all() and (margins[2:] >= 0).all():
            # A separate feasibility solve relaxes every boundary by 1e-5 and
            # explicitly tests all smaller cardinalities, including zero.
            smaller = LinearConstraint(np.vstack((gains, np.ones(n))),
                                       np.r_[required-1e-5, -np.inf],
                                       np.r_[np.full(4,np.inf), k-1])
            audit = milp(np.zeros(n), integrality=np.ones(n), bounds=Bounds(0,1),
                         constraints=smaller)
            if audit.status != 2:
                raise RuntimeError(f"Smaller-set infeasibility unconfirmed: {audit.message}")
            log["smaller_check"] = dict(at_most=k-1, relaxed_by=1e-5, status="infeasible")
            return mask, log
        if upper >= -1e-6:
            raise RuntimeError("Borderline margin: cannot exclude this size at the solver tolerance.")
    raise RuntimeError("No feasible mask")


def run(args):
    sys.path.insert(0, str(args.study.resolve()))
    from core import MLP
    folder = args.heads / "qwen_mlp"
    lock = json.loads((folder / "selection.json").read_text())
    selected, = [row for row in lock["selected"] if row["seed"] == 0]
    folder = folder / f"seed0_lr{selected['lr']}"
    for name in ("model.pt", "preprocessor.joblib"):
        assert digest(folder/name) == lock["files"][f"{folder.name}/{name}"]
    assert digest(args.qwen) == CACHES["qwen"]
    with np.load(args.qwen) as cache:
        ids, vectors = pd.Index(cache["sample_id"]), cache["X"]
    assert ids.is_unique
    scaler = joblib.load(folder/"preprocessor.joblib")
    ckpt = torch.load(folder/"model.pt", map_location="cpu", weights_only=True)
    net = MLP(vectors.shape[1]).eval()
    net.load_state_dict(ckpt["state_dict"])
    meta = ckpt["metadata"]
    indices = UNITS - 1
    state = net.state_dict()
    weights = state["net.0.weight"].double().numpy()
    bias = state["net.0.bias"].double().numpy()
    readout = state["net.3.weight"][0].double().numpy() * meta["y_scale"]
    members = pd.read_csv(HERE/"equal_activity_pair/neighbors.csv")
    ref = pd.read_csv(HERE/"equal_activity_pair/seed_predictions.csv").query("model=='qwen' and seed==0").iloc[0]
    steps = pd.read_csv(HERE/"input_gap/prefixes.csv").query("case=='BACE_0638' and k>0")
    singles = {int(j): sorted(g.dimension.astype(int)) for j, g in steps.groupby("unit")}
    lists = {j: set(d) for j, d in singles.items()}
    union = set.union(*lists.values())
    proof = dict(scipy=scipy.__version__, individual_dimensions=singles, union_size=len(union),
                 intersections={f"{a}_{b}": sorted(lists[a] & lists[b]) for a,b in ((56,11),(56,64),(11,64))},
                 three_way=sorted(set.intersection(*lists.values())), cases={})
    rows, masks = [], {}
    for case in PAIR:
        neighbors = members[(members.case==case) & (members.metric=="qwen_scaled") & (members["rank"]<=5)]
        assert neighbors.train_id.nunique() == len(neighbors) == 5
        positions = ids.get_indexer([case, *neighbors.train_id])
        assert positions.min() >= 0
        x = scaler.transform(vectors[positions]).astype(np.float32).astype(float)
        avg = x[1:].mean(axis=0).astype(np.float32).astype(float)
        pre = x @ weights.T + bias
        start = pre[0,indices]
        low, high = pre[1:,indices[2]].min(), pre[1:,indices[2]].max()
        d = weights[indices] * (avg-x[0])
        gains = np.vstack((d[0], d[1], -d[2], d[2]))
        required = np.array([-start[0], -start[1], start[2]-high, low-start[2]])
        masks[case], record = shortest_mask(gains, required)
        record["dimensions"] = (np.flatnonzero(masks[case])+1).tolist()
        proof["cases"][case] = record
        scenarios = {"original": np.zeros(len(avg), dtype=bool), "own_shortest": masks[case]}
        if case=="BACE_0638":
            scenarios["individual_union"] = np.isin(np.arange(1,len(avg)+1), sorted(union))
        else:
            scenarios["use_0638_mask"] = masks["BACE_0638"]
        for mode, mask in scenarios.items():
            changed = np.where(mask, avg, x[0])
            z = weights @ changed + bias
            h = np.maximum(0,z)
            with torch.inference_mode():
                tensor = torch.tensor(changed[None,:],dtype=torch.float32)
                direct_z = net.net[0](tensor)[0].numpy()
                pred = (net(tensor)*meta["y_scale"] + meta["y_mean"]).item()
            check(z, direct_z)
            check(pred, h@readout + state["net.3.bias"].item()*meta["y_scale"]+meta["y_mean"])
            contributions = (h-np.maximum(pre[0],0))*readout
            baseline = ref["pred_"+case.split("_")[1]]
            check(contributions.sum(), pred-baseline)
            a,b,c = direct_z[indices]
            hit = bool(a>0 and b>0 and low<=c<=high)
            if mode=="own_shortest":
                assert hit
            rows.append(dict(case=case, mode=mode, k=int(mask.sum()), z56=a,z11=b,z64=c,
                             lower64=low,upper64=high,all_met=hit,prediction=pred,change=pred-baseline,
                             change_three=contributions[indices].sum(),
                             change_other=contributions.sum()-contributions[indices].sum()))
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"solutions.json").write_text(json.dumps(proof,indent=2)+"\n",encoding="utf-8",newline="\n")
    pd.DataFrame(rows).to_csv(args.out/"replay.csv",index=False,lineterminator="\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("qwen","heads","study","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())
