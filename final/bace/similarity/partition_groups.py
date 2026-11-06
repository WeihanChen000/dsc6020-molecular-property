"""Build a label-blind BACE split with a strict cross-partition similarity cap."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from partition import build


def partition(raw, study):
    data = build(raw, study)
    original = data.partition.to_numpy(copy=True)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in data.connectivity]
    similarity = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
    scaffold = data.scaffold.to_numpy()
    links = (similarity >= 0.6) | (scaffold[:, None] == scaffold)
    _, group = connected_components(links, directed=False)
    data["group"] = group
    data["partition"] = "train"
    space = {"test": 300, "valid": 150}
    for key in np.random.default_rng(8101).permutation(np.unique(group)):
        members = data.index[group == key]
        for part in space:
            if len(members) <= space[part]:
                data.loc[members, "partition"] = part
                space[part] -= len(members)
                break
    assert not any(space.values())
    new = data.partition.to_numpy()
    cross = new[:, None] != new
    assert not links[cross].any()
    for name, parts in [("original", original), ("grouped", new)]:
        data[name + "_partition"] = parts
        data[name + "_nearest_train"] = similarity[:, parts == "train"].max(axis=1)
    return data


def manifest(data):
    return data[["sample_id", "group", "partition"]].reset_index(drop=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["raw", "study", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    data = partition(a.raw, a.study)
    manifest(data).to_csv(a.out, index=False, lineterminator="\n")
    print(data.groupby("partition").agg(records=("group", "size"), groups=("group", "nunique"),
                                         scaffolds=("scaffold", "nunique")))
    for name in ["original", "grouped"]:
        test = data[data[name + "_partition"] == "test"]
        print(name, test[name + "_nearest_train"].describe())
