"""Post-hoc BACE checks. Keep the published split and all neural predictions fixed."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from partition import SOURCE_SHA

HERE = Path(__file__).resolve().parent
MODELS = ["dmpnn", "molformer_mlp", "nearest_train"]


def find_neighbors(raw, membership):
    if hashlib.sha256(raw.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("Unexpected BACE source file")
    source = pd.read_csv(raw, usecols=["mol", "pIC50"])
    source.index = [f"BACE_{i:04d}" for i in source.index]
    assert membership.index.is_unique
    assert membership.partition.value_counts().to_dict() == {"train": 1053, "valid": 150, "test": 300}
    train = source.loc[membership.index[membership.partition == "train"]]
    test = source.loc[membership.index[membership.partition == "test"]]
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)

    def fingerprint(smiles):
        connectivity = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
        return generator.GetFingerprint(Chem.MolFromSmiles(connectivity))

    fingerprints = [fingerprint(smiles) for smiles in train.mol]
    rows = []
    for sample_id, smiles in test.mol.items():
        similarities = np.array(DataStructs.BulkTanimotoSimilarity(fingerprint(smiles), fingerprints))
        highest = similarities.max()
        matched = train.iloc[np.flatnonzero(similarities == highest)]
        rows.append({"sample_id": sample_id, "max_similarity": highest,
                     "training_neighbors": "|".join(matched.index), "ties_at_max": len(matched),
                     "nearest_train": matched.pIC50.mean()})
    return pd.DataFrame(rows), test.pIC50


def analyze(raw, output):
    membership = pd.read_csv(HERE / "partitions.csv", index_col="sample_id")
    neighbors, labels = find_neighbors(raw, membership)
    published = pd.read_csv(HERE / "results/predictions.csv")
    original = pd.read_csv(HERE / "results/scores.csv").set_index("seed")
    assert set(published.seed) == {0, 1, 2}
    strata = ["all", "[0,0.3)", "[0.3,0.6)", "[0.6,1]"]
    neighbors["similarity_band"] = np.select(
        [neighbors.max_similarity < 0.3, neighbors.max_similarity < 0.6], strata[1:3], default=strata[3])
    scores, omissions = [], []
    for seed, predictions in published.groupby("seed"):
        frame = predictions.merge(neighbors, on="sample_id", validate="one_to_one")
        assert len(frame) == len(predictions) == len(labels) == 300
        np.testing.assert_allclose(frame.y_true, labels.loc[frame.sample_id], rtol=0, atol=1e-12)
        squared = frame[MODELS].sub(frame.y_true, axis=0).pow(2)
        for band in strata:
            selected = frame.index if band == "all" else frame.index[frame.similarity_band == band]
            errors = squared.loc[selected]
            for model in MODELS:
                rmse = np.sqrt(errors[model].mean())
                macro = np.sqrt(errors[model].groupby(frame.scaffold_id).mean().mean())
                if band == "all" and model != "nearest_train":
                    np.testing.assert_allclose(
                        [rmse, macro], original.loc[seed, [model + "_rmse", model + "_scaffold_rmse"]],
                        rtol=0, atol=1e-12)
                scores.append({"seed": seed, "band": band, "records": len(errors), "model": model,
                               "rmse": rmse, "scaffold_rmse": macro})
        groups = squared.groupby(frame.scaffold_id)
        totals, sizes, means = groups.sum(), groups.size(), groups.mean()
        assert len(sizes) == 124
        for scaffold, size in sizes.items():
            record_rmse = np.sqrt((totals.sum() - totals.loc[scaffold]) / (len(frame) - size))
            macro_rmse = np.sqrt((means.sum() - means.loc[scaffold]) / (len(sizes) - 1))
            omissions.append({"seed": seed, "omitted_scaffold": scaffold, "omitted_records": size,
                              "record_gap_gnn_minus_lm": record_rmse.dmpnn - record_rmse.molformer_mlp,
                              "scaffold_gap_gnn_minus_lm": macro_rmse.dmpnn - macro_rmse.molformer_mlp})
    output.mkdir(exist_ok=True, parents=True)
    for table, name in [(neighbors, "neighbors"), (pd.DataFrame(scores), "scores"),
                        (pd.DataFrame(omissions), "omit_one_scaffold")]:
        table.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.raw, args.out)
