"""Audit high-activity underprediction with fixed, activity-matched BACE controls."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
MODELS = ["qwen_mlp", "molformer_mlp"]
FOUR = ["BACE_0638", "BACE_0679", "BACE_0741", "BACE_0795"]


def match(samples):
    cases = samples.loc[FOUR]
    candidates = samples.loc[~samples.group.isin(cases.group)].sort_index()
    assert not set(candidates.scaffold) & set(cases.scaffold)
    # One representative per scaffold and target. Match three distinct scaffolds
    # to each target while minimizing the sum of absolute activity differences.
    best = []
    for case, row in cases.iterrows():
        ranked = candidates.assign(distance=(candidates.y_true-row.y_true).abs())
        ranked = ranked.sort_values("distance", kind="stable").drop_duplicates("scaffold")
        best.append(ranked.reset_index().set_index("scaffold").sort_index())
    costs = np.stack([r.distance.to_numpy() for r in best])
    slots, columns = linear_sum_assignment(np.repeat(costs, 3, axis=0))
    pairs = []
    for slot, col in zip(slots, columns):
        i = slot//3
        control = best[i].iloc[col]
        pairs.append({"target":FOUR[i], "target_y":cases.iloc[i].y_true, "control":control.sample_id,
                      "control_y":control.y_true, "control_scaffold":control.name,
                      "control_group":control.group, "abs_difference":control.distance})
    return pd.DataFrame(pairs).sort_values(["target","abs_difference","control"])


def run(out):
    labels = pd.read_csv(HERE.parent/"qwen/assay_audit/labels.csv").query("exact_ic50")
    tr = labels.query("partition == 'train'").pIC50
    test = labels.query("partition == 'test'").set_index("sample_id")
    assert len(tr) == 992 and len(test) == 268
    threshold, upper = tr.mean(), tr.quantile(.75)
    d = pd.read_csv(HERE/"results/predictions.csv").query("model in @MODELS").copy()
    samples = d.query("seed == 0 and model == 'qwen_mlp'").set_index("sample_id")[["group","scaffold","y_true"]].sort_index()
    samples.index.name = "sample_id"
    assert samples.index.is_unique and set(samples.index) == set(test.index)
    np.testing.assert_allclose(samples.y_true, test.loc[samples.index].pIC50, rtol=0, atol=1e-12)
    assert samples.loc[FOUR,"scaffold"].nunique() == 1 and samples.loc[FOUR,"group"].nunique() == 1
    pairs = match(samples)
    assert len(pairs) == pairs.control_scaffold.nunique() == pairs.control.nunique() == 12
    ref_scaffold = samples.loc[FOUR[0],"scaffold"]
    ref_group = samples.loc[FOUR[0],"group"]
    high = samples.y_true > threshold
    outside = samples.scaffold != ref_scaffold
    scopes = {"all":samples.index, "low":samples.index[~high], "high":samples.index[high],
              "high_other_scaffolds":samples.index[high & outside],
              "upper_other_scaffolds":samples.index[(samples.y_true>=upper) & outside],
              "all_other_scaffolds":samples.index[outside],
              "other_than_four":samples.index.difference(FOUR), "four":FOUR, "controls":pairs.control}
    rows = []
    for (model,seed), block in d.groupby(["model","seed"]):
        b = block.set_index("sample_id").sort_index()
        pd.testing.assert_frame_equal(b[samples.columns], samples)
        for scope, ids in scopes.items():
            errors = (b.exact-b.y_true).loc[ids]
            group_bias = errors.groupby(b.scaffold).mean()
            rows.append({"scope":scope,"model":model,"seed":seed,"n":len(errors),"scaffolds":len(group_bias),
                         "under":int((errors<0).sum()),"bias":errors.mean(),"mae":errors.abs().mean(),
                         "rmse":np.sqrt(errors.pow(2).mean()),"sse":errors.pow(2).sum(),
                         "scaffolds_under_mean":int((group_bias<0).sum()),"equal_scaffold_bias":group_bias.mean(),
                         "high_cut":threshold,"upper_cut":upper})
    summary = pd.DataFrame(rows)
    lookup = d[["sample_id","seed","model","exact"]]
    matched_predictions = pairs.merge(lookup.rename(columns={"sample_id":"control","exact":"control_pred"}), on="control")
    matched_predictions = matched_predictions.merge(lookup.rename(columns={"sample_id":"target","exact":"target_pred"}),
                                                    on=["target","model","seed"], validate="many_to_one")
    assert len(matched_predictions) == 72
    # As a check on the choice of three controls, also use every eligible molecule
    # within 0.10 pIC50 of each target. These windows may overlap.
    windows = []
    for case in FOUR:
        y = samples.at[case,"y_true"]
        mask = (samples.group != ref_group) & ((samples.y_true-y).abs()<=.10)
        for (model,seed), block in d.groupby(["model","seed"]):
            b = block.set_index("sample_id")
            for design, ids in [("matched",pairs.loc[pairs.target==case,"control"]),
                                ("window_0.10",samples.index[mask])]:
                comp = b.loc[ids]
                windows.append({"target":case,"model":model,"seed":seed,"design":design,"controls_n":len(comp),
                                "target_y":y,"target_pred":b.at[case,"exact"],"target_bias":b.at[case,"exact"]-y,
                                "control_bias":(comp.exact-comp.y_true).mean()})
    out.mkdir(parents=True,exist_ok=True)
    for name, table in [("summary",summary),("matches",pairs),("matched_predictions",matched_predictions),
                        ("paired_bias",pd.DataFrame(windows))]:
        table.to_csv(out/f"{name}.csv",index=False,lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
