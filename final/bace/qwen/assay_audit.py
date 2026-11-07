"""Audit source measurements and rescore saved predictions on exact IC50 records."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from training_neighbors import HERE, SOURCE_SHA, source_records

SIM = HERE.parent / "similarity"
MODELS = ["qwen_mlp", "molformer_mlp", "dmpnn"]


def audit_labels(raw_path, supplement):
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == SOURCE_SHA
    raw = pd.read_csv(raw_path, usecols=["CID", "pIC50"])
    raw.index = [f"BACE_{i:04d}" for i in raw.index]
    assert raw.CID.is_unique
    source = source_records(supplement, raw)
    split = pd.read_csv(SIM / "partitions.csv").set_index("sample_id")
    assert split.index.is_unique and split.partition.value_counts().to_dict() == {"train":1053,"valid":150,"test":300}
    labels = split.join(raw.CID).join(source.drop(columns=["PUBMED_ID", "COMPOUND_KEY"]))
    labels["pIC50"] = pd.to_numeric(labels.pIC50)
    value = pd.to_numeric(labels.STANDARD_VALUE)
    unit = labels.STANDARD_UNITS.map({"nM": 1e-9, "uM": 1e-6})
    assert value.dropna().gt(0).all() and unit.isna().equals(labels.STANDARD_UNITS.isna())
    assert labels.CID.notna().all() and labels.pIC50.notna().all() and set(labels.RELATION) == {"=", ">"}
    labels["log_from_value"] = -np.log10(value * unit)
    delta = (labels.log_from_value - labels.pIC50).abs()
    assert delta.max() < 0.0001
    equal_ic50 = labels.STANDARD_TYPE.eq("IC50") & labels.RELATION.eq("=")
    labels["exact_ic50"] = equal_ic50 & labels.log_from_value.notna()
    labels["category"] = labels.STANDARD_TYPE + " " + labels.RELATION
    labels.loc[equal_ic50 & ~labels.exact_ic50, "category"] = "IC50 = incomplete source"
    return labels


def matched_predictions(labels):
    keys = ["seed", "sample_id"]
    q = pd.read_csv(HERE / "results/predictions.csv").set_index(keys).sort_index()
    old = pd.read_csv(SIM / "results/grouped_predictions.csv").set_index(keys).sort_index()
    assert q.index.is_unique and old.index.is_unique
    pd.testing.assert_frame_equal(q[["y_true","group","scaffold","molformer_mlp"]],
                                  old[["y_true","group","scaffold","molformer_mlp"]])
    test = labels.query("partition == 'test'").sort_index()
    assert set(q.index.get_level_values("seed")) == {0,1,2}
    for _, frame in q.groupby(level="seed"):
        assert frame.droplevel("seed").index.equals(test.index)
        np.testing.assert_allclose(frame.y_true, test.pIC50, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(frame.group, test.group)
    return q.join(old.dmpnn).join(test[["category","exact_ic50"]], on="sample_id")


def score(pred):
    rows = []
    scopes = {"all": pred, "exact_ic50": pred[pred.exact_ic50],
              "ic50_equal_including_incomplete": pred[pred.category.str.startswith("IC50 =")]}
    for scope, sub in scopes.items():
        for seed, frame in sub.groupby(level="seed"):
            for model in MODELS:
                err = frame[model] - frame.y_true
                assert np.isfinite(err).all()
                sq = err.pow(2)
                rows.append({"scope":scope, "seed":seed, "model":model, "n":len(frame),
                             "scaffolds":frame.scaffold.nunique(), "rmse":np.sqrt(sq.mean()),
                             "mae":err.abs().mean(), "scaffold_rmse":np.sqrt(sq.groupby(frame.scaffold).mean().mean())})
    scores = pd.DataFrame(rows)
    metrics = ["rmse","mae","scaffold_rmse"]
    original = pd.concat([pd.read_csv(HERE / "results/scores.csv"),
        pd.read_csv(SIM / "results/scores.csv").query("split == 'grouped' and model == 'dmpnn'")])
    keys = ["seed","model"]
    np.testing.assert_allclose(scores.query("scope == 'all'").set_index(keys).sort_index()[metrics],
                               original.set_index(keys).sort_index()[metrics], atol=1e-12, rtol=0)
    return scores


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["raw", "supplement", "out"]:
        p.add_argument("--"+key, type=Path, required=True)
    args = p.parse_args()
    labels = audit_labels(args.raw, args.supplement)
    counts = pd.crosstab(labels.category, labels.partition).reindex(columns=["train","valid","test"])
    pred = matched_predictions(labels)
    scores = score(pred)
    squared = pred[MODELS].sub(pred.y_true, axis=0).pow(2)
    squared["category"] = pred.category
    by_type = squared.groupby(["seed","category"]).sum()
    by_type["n"] = squared.groupby(["seed","category"]).size()
    by_type["qwen_minus_molformer_sse"] = by_type.qwen_mlp - by_type.molformer_mlp
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in [("labels", labels.reset_index()), ("counts", counts.reset_index()), ("scores", scores),
                        ("squared_errors_by_type", by_type.reset_index())]:
        table.to_csv(args.out / (name+".csv"), index=False, lineterminator="\n")
