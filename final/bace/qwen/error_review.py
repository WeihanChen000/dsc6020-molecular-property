"""Trace the paired squared-error difference on the published BACE holdout."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def review(out):
    rows = pd.read_csv(HERE / "results/predictions.csv")
    reference = rows.query("seed == 0").set_index("sample_id").sort_index()
    scores = pd.read_csv(HERE / "results/scores.csv").set_index(["seed", "model"])
    means = pd.read_csv(HERE / "results/training.csv").y_mean.unique()
    assert len(means) == 1 and not (rows.y_true == means[0]).any()
    assert set(rows.seed) == {0, 1, 2} and len(rows) == 900
    assert len(reference) == 300 and reference.index.is_unique and reference.scaffold.nunique() == 183
    rows["activity"] = np.where(rows.y_true < means[0], "low", "high")
    rows["q_error"] = rows.qwen_mlp - rows.y_true
    rows["m_error"] = rows.molformer_mlp - rows.y_true
    rows["delta_se"] = rows.q_error**2 - rows.m_error**2
    summaries = []
    for seed, frame in rows.groupby("seed"):
        fixed = ["y_true", "scaffold", "group"]
        pd.testing.assert_frame_equal(frame.set_index("sample_id").sort_index()[fixed], reference[fixed])
        for model, error in [("qwen_mlp", frame.q_error), ("molformer_mlp", frame.m_error)]:
            published = scores.loc[(seed, model)]
            np.testing.assert_allclose([np.sqrt((error**2).mean()), error.abs().mean()],
                                       [published.rmse, published.mae], atol=1e-12, rtol=0)
        for side, part in frame.groupby("activity"):
            summaries.append({"seed": seed, "activity": side, "n": len(part),
                              "q_bias": part.q_error.mean(), "m_bias": part.m_error.mean(),
                              "q_over": (part.q_error > 0).sum(), "m_over": (part.m_error > 0).sum(),
                              "delta_sse": part.delta_se.sum(), "train_mean": means[0]})
    rows = rows.sort_values(["seed", "delta_se", "sample_id"], ascending=[True, False, True])
    rows["rank"] = rows.groupby("seed").cumcount() + 1
    top = rows[rows["rank"] <= 10].groupby("seed").sample_id.agg(set)
    recurring = top[1] & top[2]
    rows["shared_top10_seeds12"] = rows.sample_id.isin(recurring)
    cohorts = []
    for (seed, shared), part in rows.groupby(["seed", "shared_top10_seeds12"]):
        cohorts.append({"seed": seed, "shared_top10_seeds12": shared, "n": len(part),
                        "scaffolds": part.scaffold.nunique(), "delta_sse": part.delta_se.sum(),
                        "q_rmse": np.sqrt((part.q_error**2).mean()),
                        "m_rmse": np.sqrt((part.m_error**2).mean())})
    skeletons = rows.groupby(["seed", "scaffold"], as_index=False).agg(
        n=("sample_id", "size"), first_id=("sample_id", "min"), delta_sse=("delta_se", "sum"))
    skeletons = skeletons.sort_values(["seed", "delta_sse", "first_id"], ascending=[True, False, True])
    skeletons["rank"] = skeletons.groupby("seed").cumcount() + 1
    np.testing.assert_allclose(skeletons.groupby("seed").delta_sse.sum(), rows.groupby("seed").delta_se.sum())
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [("molecules", rows), ("scaffolds", skeletons), ("activity", pd.DataFrame(summaries)),
                        ("shared_cases", pd.DataFrame(cohorts))]:
        frame.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    review(parser.parse_args().out)
