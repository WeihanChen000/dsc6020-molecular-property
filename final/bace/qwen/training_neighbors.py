"""Look up training analogues for the eight previously identified BACE cases."""
import argparse
import hashlib
from pathlib import Path
import re
import sys
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from partition import SOURCE_SHA


def source_records(supplement, raw):
    assert hashlib.md5(supplement.read_bytes()).hexdigest() == "aa483ebebcb0b18fa5d639700db32f60"
    with ZipFile(supplement) as z:
        sdf = z.read("bace1547.sdf").decode().replace("\r", "")
    records = []
    for block in sdf.split("$$$$"):
        fields = dict(re.findall(r"^>\s+<([^>]+)>\n([^\n]+)", block, re.M))
        if "CID" in fields:
            records.append({k: fields.get(k) for k in ["CID", "PUBMED_ID", "COMPOUND_KEY", "RELATION",
                                                      "STANDARD_VALUE", "STANDARD_UNITS", "STANDARD_TYPE", "pIC50"]})
    source = pd.DataFrame(records).set_index("CID").loc[raw.CID].copy()
    source.index = raw.index
    np.testing.assert_allclose(source.pIC50.astype(float), raw.pIC50, rtol=0, atol=1e-6)
    return source


def cosine_matrix(path, checksum, train_ids, test_ids):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
    with np.load(path) as cache:
        keys = pd.Index(cache["sample_id"])
        assert keys.is_unique
        x = cache["X"].astype(float)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    tr, te = keys.get_indexer(train_ids), keys.get_indexer(test_ids)
    assert min(tr.min(), te.min()) >= 0
    return x[te] @ x[tr].T


def run(a):
    assert hashlib.sha256(a.raw.read_bytes()).hexdigest() == SOURCE_SHA
    raw = pd.read_csv(a.raw, usecols=["CID", "mol", "pIC50"])
    raw.index = [f"BACE_{i:04d}" for i in raw.index]
    split = pd.read_csv(HERE.parent / "similarity/partitions.csv").set_index("sample_id")
    train = raw.loc[split.index[split.partition == "train"]]
    reviewed = pd.read_csv(HERE / "error_review/molecules.csv")
    dev_mean = train.pIC50.mean()
    pred = reviewed.groupby("sample_id").agg(q_pred=("qwen_mlp", "mean"), m_pred=("molformer_mlp", "mean"),
                 y=("y_true", "first"), case=("shared_top10_seeds12", "first"))
    assert set(pred.index) == set(split.index[split.partition == "test"])
    np.testing.assert_allclose(raw.loc[pred.index, "pIC50"], pred.y, rtol=0, atol=1e-12)
    low = pred[pred.y < dev_mean].copy()
    cases = low[low.case]
    other = low[~low.case]
    assert (len(train), len(cases), len(other)) == (1053, 8, 195)
    source = source_records(a.supplement, raw)
    low["source_relation"] = source.RELATION
    low["source_type"] = source.STANDARD_TYPE
    train_papers = source.loc[train.index].PUBMED_ID.value_counts()
    pmid = source.loc[low.index].PUBMED_ID
    low["train_same_pmid"] = pmid.map(train_papers).fillna(0).where(pmid.notna()).astype("Int64")
    rows, cols = linear_sum_assignment(abs(cases.y.to_numpy()[:, None] - other.y.to_numpy()))
    matched = pd.DataFrame({"case_id": cases.index[rows], "control_id": other.index[cols]})
    low["matched_control"] = low.index.isin(matched.control_id)
    low["q_bias"] = low.q_pred - low.y
    low["m_bias"] = low.m_pred - low.y
    raw["input_smiles"] = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False) for s in raw.mol]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fps = {k: gen.GetFingerprint(Chem.MolFromSmiles(raw.at[k, "input_smiles"])) for k in split.index}
    train_fps = [fps[k] for k in train.index]
    morg = np.array([DataStructs.BulkTanimotoSimilarity(fps[k], train_fps) for k in low.index])
    published = pd.read_csv(HERE.parent / "similarity/results/neighbors.csv").query("split == 'grouped'").set_index("sample_id")
    np.testing.assert_allclose(morg.max(axis=1), published.loc[low.index].max_training_similarity)
    matrices = {"morgan": morg,
        "qwen": cosine_matrix(a.qwen, "7dcaa4cc8095d98b46176a4471cd13bca776100068fce1f9343e4923c874d6b8", train.index, low.index),
        "molformer": cosine_matrix(a.molformer, "f67d87948ce5b243e9ec44ea2e9f5bf7619ff7e40e2767035b4cc7746a0f0fbb", train.index, low.index)}
    neighbors = []
    for method, matrix in matrices.items():
        for (sample, case), sims in zip(low.case.items(), matrix, strict=True):
            order = np.argsort(-sims, kind="stable")
            nearest = order[sims[order] == sims[order[0]]]
            top = order[sims[order] >= sims[order[4]]]
            low.loc[sample, method+"_best"] = sims[order[0]]
            low.loc[sample, method+"_nn_y"] = train.iloc[nearest].pIC50.mean()
            low.loc[sample, method+"_top5_y"] = train.iloc[top].pIC50.mean()
            if case:
                for j in top:
                    key = train.index[j]
                    neighbors.append({"case_id": sample, "case_cid": raw.at[sample,"CID"], "case_y": low.at[sample,"y"],
                        "method": method, "rank": 1+int((sims > sims[j]).sum()), "similarity": sims[j],
                        "train_id": key, "train_cid": raw.at[key,"CID"], "train_y": train.at[key,"pIC50"],
                        "train_type": source.at[key,"STANDARD_TYPE"], "train_relation": source.at[key,"RELATION"],
                        "case_smiles": raw.at[sample,"input_smiles"], "train_smiles": raw.at[key,"input_smiles"]})
    a.out.mkdir(exist_ok=True, parents=True)
    nn = pd.DataFrame(neighbors)
    focus = source.loc[cases.index].copy()
    focus["train_same_pmid"] = low.loc[cases.index, "train_same_pmid"]
    scale = focus.STANDARD_UNITS.map({"nM": 1e-9, "uM": 1e-6})
    assert set(focus.STANDARD_TYPE) == {"Ki", "IC50"}
    np.testing.assert_allclose(-np.log10(focus.STANDARD_VALUE.astype(float)*scale), cases.y, atol=1e-6, rtol=0)
    for name, frame in [("low_activity", low.rename_axis("sample_id").reset_index()), ("neighbors", nn),
                        ("source", focus.rename_axis("sample_id").reset_index()), ("matched", matched)]:
        frame.to_csv(a.out / (name+".csv"), index=False, lineterminator="\n")
    pair = ["BACE_0242", "BACE_0243"]
    near_pair = nn[nn.case_id.isin(pair) & (nn.method == "morgan") & (nn["rank"] == 1)]
    ids = pair + near_pair.train_id.drop_duplicates().tolist()
    mols = [Chem.MolFromSmiles(raw.at[k, "input_smiles"]) for k in ids]
    legends = [f"{k}  ({split.at[k,'partition']})  p{source.at[k,'STANDARD_TYPE']} {raw.at[k,'pIC50']:.2f}" for k in ids]
    Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(460,280), legends=legends).save(a.out/"pair.png")
    print("0242/0243 Tanimoto:", DataStructs.TanimotoSimilarity(fps[pair[0]], fps[pair[1]]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["raw", "supplement", "qwen", "molformer", "out"]:
        p.add_argument("--"+key, type=Path, required=True)
    run(p.parse_args())
