"""Describe seed-zero rank flips and score the same scaffolds at all three seeds."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from scaffold_weights import HERE, MODELS


def profile(source, out):
    predictions = pd.read_csv(source / "grouped_predictions.csv")
    neighbors = pd.read_csv(source / "neighbors.csv").query("split == 'grouped'")
    assert len(neighbors) == neighbors.sample_id.nunique() == 300
    scores = pd.read_csv(source / "scores.csv").query("split == 'grouped'").set_index(["seed", "model"])
    detail, reference = [], None
    assert predictions.train_mean.nunique() == 1
    train_mean = predictions.train_mean.iloc[0]
    for seed, frame in predictions.groupby("seed", sort=True):
        frame = frame.merge(neighbors[["sample_id", "max_training_similarity"]], on="sample_id",
                            validate="one_to_one").sort_values("sample_id")
        members = frame[["sample_id", "scaffold", "group", "y_true"]].reset_index(drop=True)
        if reference is None:
            reference = members
        else:
            pd.testing.assert_frame_equal(reference, members)
        assert len(frame) == 300 and frame.scaffold.nunique() == 183
        residual = frame[list(MODELS)].sub(frame.y_true, axis=0)
        mse = residual.pow(2).groupby(frame.scaffold).mean()
        full = np.sqrt(mse.mean())
        for model in MODELS:
            np.testing.assert_allclose(full[model], scores.loc[(seed, model), "scaffold_rmse"], atol=1e-12, rtol=0)
        full_gap = full.dmpnn - full.molformer_mlp
        assert full_gap > 0
        omitted = np.sqrt((mse.sum() - mse) / (len(mse) - 1))
        gap = omitted.dmpnn - omitted.molformer_mlp
        result = pd.DataFrame({"seed": seed, "dmpnn_mse": mse.dmpnn, "lm_mse": mse.molformer_mlp,
                               "dmpnn_bias": residual.dmpnn.groupby(frame.scaffold).mean(),
                               "lm_bias": residual.molformer_mlp.groupby(frame.scaffold).mean(),
                               "full_gap": full_gap, "omitted_gap": gap, "influence": full_gap - gap})
        result["rank_for_lm"] = result.influence.rank(ascending=False, method="min").astype(int)
        result["rank_absolute"] = result.influence.abs().rank(ascending=False, method="min").astype(int)
        detail.append(result.reset_index())
    assert [int(d.seed.iloc[0]) for d in detail] == [0, 1, 2]
    effects = pd.concat(detail)
    selected = set(detail[0].loc[detail[0].omitted_gap < 0, "scaffold"])
    groups = frame.groupby("scaffold")
    features = groups.agg(example_id=("sample_id", "min"), n=("sample_id", "size"), component=("group", "first"),
                          mean_pIC50=("y_true", "mean"), mean_similarity=("max_training_similarity", "mean"))
    assert groups.group.nunique().max() == 1
    mols = [Chem.MolFromSmiles("" if s == "__ACYCLIC__" else s) for s in features.index]
    features["heavy_atoms"] = [m.GetNumHeavyAtoms() for m in mols]
    features["rings"] = [m.GetRingInfo().NumRings() for m in mols]
    features["distance_to_train_mean"] = (features.mean_pIC50 - train_mean).abs()
    features["flip_at_seed0"] = features.index.isin(selected)
    effects = effects.merge(features.reset_index(), on="scaffold", validate="many_to_one")
    out.mkdir(parents=True, exist_ok=True)
    effects.sort_values(["seed", "rank_for_lm"]).to_csv(out / "scaffold_influence.csv", index=False, lineterminator="\n")
    summaries = []
    for flag, cohort in effects[effects.seed == 0].groupby("flip_at_seed0"):
        towards = (train_mean - cohort.mean_pIC50) * cohort.dmpnn_bias > 0
        summaries.append({"flip_at_seed0": flag, "scaffolds": len(cohort), "records": cohort.n.sum(),
                          "singletons": (cohort.n == 1).sum(), "components": cohort.component.nunique(),
                          "bias_toward_training_mean": towards.sum(), "mean_similarity_below_0.3": (cohort.mean_similarity < .3).sum(),
                          **{"median_" + key: cohort[key].median() for key in
                             ["mean_pIC50", "distance_to_train_mean", "mean_similarity", "heavy_atoms", "rings"]}})
    pd.DataFrame(summaries).to_csv(out / "flip_cohorts.csv", index=False, lineterminator="\n")
    wide = effects[effects.flip_at_seed0].pivot(index="example_id", columns="seed", values="influence")
    wide.sort_values(0, ascending=False).add_prefix("influence_seed").to_csv(out / "flip13.csv", lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    profile(args.source, args.out)
