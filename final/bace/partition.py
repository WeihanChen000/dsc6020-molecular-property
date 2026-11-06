"""Audit input connectivity and construct a single label-blind scaffold holdout."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

SOURCE_SHA = "f3fb9ce90bada3e2bd6148b0df13f8f8145a357bf87df0dd5b391ede974fc737"


def build(raw,study):
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==SOURCE_SHA
    frame=pd.read_csv(raw,usecols=["mol"])
    mols=[Chem.MolFromSmiles(s) for s in frame.mol]
    assert all(m is not None for m in mols)
    frame["connectivity"]=[Chem.MolToSmiles(m,isomericSmiles=False) for m in mols]
    frame["scaffold"]=[MurckoScaffold.MurckoScaffoldSmiles(mol=m,includeChirality=False) or "__ACYCLIC__" for m in mols]
    frame["sample_id"]=[f"BACE_{i:04d}" for i in frame.index]
    for ds in ["ESOL","Lipophilicity"]:
        previous=pd.read_csv(study/"inputs"/ds/"samples.csv",usecols=["connectivity"])
        frame["overlap_"+ds]=frame.connectivity.isin(previous.connectivity)
    kept=frame[~(frame.overlap_ESOL|frame.overlap_Lipophilicity)].copy()
    groups=list(kept.groupby("scaffold",sort=True).groups.values())
    remaining={"test":len(kept)//5,"valid":len(kept)//10}
    kept["partition"]="train"
    for i in np.random.default_rng(7101).permutation(len(groups)):
        indices=groups[i]
        for part in ["test","valid"]:
            if len(indices)<=remaining[part]:
                kept.loc[indices,"partition"]=part
                remaining[part]-=len(indices)
                break
    assert kept.groupby("scaffold").partition.nunique().max()==1
    assert kept.groupby("connectivity").partition.nunique().max()==1
    assert len(kept)==1503 and kept.scaffold.nunique()==667
    return kept


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ["raw","study","out"]: p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args()
    data=build(a.raw,a.study)
    data[["sample_id","partition"]].to_csv(a.out,index=False,lineterminator="\n")
    print(data.groupby("partition").agg(rows=("sample_id","size"),scaffolds=("scaffold","nunique")))
