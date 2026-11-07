"""Compare training neighbors and replay frozen heads for two measured 44 nM cases."""
import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator
from sklearn.metrics.pairwise import cosine_similarity

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"qwen"))
from training_neighbors import source_records
from fit_exact import digest

PAIR = ["BACE_0638","BACE_0070"]
CACHES = {"qwen":"7dcaa4cc8095d98b46176a4471cd13bca776100068fce1f9343e4923c874d6b8",
          "molformer":"f67d87948ce5b243e9ec44ea2e9f5bf7619ff7e40e2767035b4cc7746a0f0fbb"}


def run(a):
    sys.path.insert(0,str(a.study.resolve()))
    import core
    from partition import SOURCE_SHA
    assert digest(a.raw) == SOURCE_SHA
    raw = pd.read_csv(a.raw,usecols=["mol","CID","pIC50"])
    raw.index = [f"BACE_{i:04d}" for i in raw.index]
    assays = source_records(a.supplement,raw)
    labels = pd.read_csv(HERE.parent/"qwen/assay_audit/labels.csv").set_index("sample_id")
    train = labels.query("partition == 'train' and exact_ic50").sort_index()
    targets = labels.loc[PAIR]
    assert len(train)==992 and train.index.is_unique and targets.exact_ic50.all() and targets.partition.eq("test").all()
    assert targets.group.nunique()==2 and not train.group.isin(targets.group).any()
    source_pair = assays.loc[PAIR]
    assert source_pair.STANDARD_TYPE.eq("IC50").all() and source_pair.RELATION.eq("=").all()
    assert source_pair.STANDARD_UNITS.eq("uM").all() and source_pair.STANDARD_VALUE.astype(float).eq(.044).all()
    np.testing.assert_allclose(-np.log10(.044e-6),targets.pIC50,rtol=0,atol=1e-6)
    np.testing.assert_array_equal(labels.CID,raw.loc[labels.index,"CID"])
    np.testing.assert_allclose(train.pIC50,raw.loc[train.index,"pIC50"],atol=1e-12,rtol=0)
    ids = pd.Index(PAIR + train.index.tolist())
    smiles = raw.loc[ids,"mol"].map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=False))
    mols = smiles.map(Chem.MolFromSmiles)
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    fingerprints = list(map(fp.GetFingerprint,mols))
    sims = {"Morgan":np.array([DataStructs.BulkTanimotoSimilarity(f,fingerprints[2:]) for f in fingerprints[:2]])}
    prior = pd.read_csv(HERE/"results/predictions.csv").set_index(["model","seed","sample_id"]).sort_index()
    comparisons, fitted = [], {}
    for model, checksum in CACHES.items():
        cache = getattr(a,model)
        assert digest(cache)==checksum
        with np.load(cache) as z:
            index = pd.Index(z["sample_id"])
            positions = index.get_indexer(ids)
            assert index.is_unique and positions.min()>=0
            x = z["X"][positions]
        sims[model] = cosine_similarity(x[:2].astype(float),x[2:].astype(float))
        root = a.heads/(model+"_mlp")
        lock = json.loads((root/"selection.json").read_text())
        for r in lock["selected"]:
            seed = r["seed"]
            folder = root/f"seed{seed}_lr{r['lr']}"
            for filename in ["model.pt","preprocessor.joblib"]:
                assert digest(folder/filename)==lock["files"][(folder.name+"/"+filename)]
            pred = core.restore("mlp",folder,x,np.arange(len(ids))).astype(float)
            archived = prior.loc[(model+"_mlp",seed)].loc[PAIR,"exact"].values
            np.testing.assert_allclose(pred[:2],archived,atol=1e-5,rtol=0)
            fitted[(model,seed)] = pred[2:]
            comparisons.append({"model":model,"seed":seed,"best_epoch":r["best_epoch"],
                "y":targets.pIC50.iloc[0],"pred_0638":archived[0],"pred_0070":archived[1],
                "gap_0070_minus_0638":archived[1]-archived[0]})
            processed = joblib.load(folder/"preprocessor.joblib").transform(x).astype(float)
            post = cosine_similarity(processed[:2],processed[2:])
            if seed == 0:
                sims[model+"_scaled"] = post
            else:
                np.testing.assert_allclose(post,sims[model+"_scaled"],atol=1e-12)

    assert targets.pIC50.nunique()==1
    rows, stats = [], []
    for metric, matrix in sims.items():
        for case, scores in zip(PAIR,matrix,strict=True):
            order = np.argsort(-scores,kind="stable")
            for k in (1,5,10):
                top = order[scores[order]>=scores[order[k-1]]]
                y = train.iloc[top].pIC50
                summary = {"case":case,"metric":metric,"k":k,"n":len(y),"mean_y":y.mean(),"min_y":y.min(),
                           "max_y":y.max(),"at_least_target":(y>=targets.at[case,"pIC50"]).sum()}
                model = metric.split("_")[0]
                if model in CACHES:
                    for seed in (0,1,2):
                        summary[f"mean_fit_seed{seed}"] = fitted[(model,seed)][top].mean()
                stats.append(summary)
            for i in order[scores[order]>=scores[order[9]]]:
                key = train.index[i]
                rows.append({"case":case,"metric":metric,"rank":1+int((scores>scores[i]).sum()),"train_id":key,
                             "similarity":scores[i],"train_y":train.at[key,"pIC50"],"input_smiles":smiles[key]})
    n = pd.DataFrame(rows)
    a.out.mkdir(parents=True,exist_ok=True)
    meta = assays.loc[PAIR].assign(CID=raw.loc[PAIR,"CID"],input_smiles=smiles.loc[PAIR]).rename_axis("sample_id").reset_index()
    for name, table in [("measurements",meta),("neighbors",n),("neighborhoods",pd.DataFrame(stats)),
                        ("seed_predictions",pd.DataFrame(comparisons))]:
        table.to_csv(a.out/(name+".csv"),index=False,lineterminator="\n")
    # Each column is one test case followed by its nearest neighbors.
    keys, legends = list(PAIR), [f"{k} | test | {targets.at[k,'pIC50']:.2f}" for k in PAIR]
    for method in ("Morgan","qwen","molformer"):
        for case in PAIR:
            first = n[(n.case==case)&(n.metric==method)&(n["rank"]==1)].iloc[0]
            keys.append(first.train_id)
            legends.append(f"{first.train_id} | {method} | {first.train_y:.2f}")
    Draw.MolsToGridImage(list(mols.loc[keys]),molsPerRow=2,subImgSize=(440,210),legends=legends).save(a.out/"structures.png")
    print("Pair Morgan Tanimoto:", DataStructs.TanimotoSimilarity(*fingerprints[:2]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["raw","supplement","qwen","molformer","study","heads","out"]:
        p.add_argument("--"+key,type=Path,required=True)
    run(p.parse_args())
