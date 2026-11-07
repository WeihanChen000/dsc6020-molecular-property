"""Trace the ten members of the BACE_0638 scaffold using saved predictions."""
import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"qwen"))
from training_neighbors import SOURCE_SHA, source_records


def review(a):
    assert hashlib.sha256(a.raw.read_bytes()).hexdigest() == SOURCE_SHA
    raw = pd.read_csv(a.raw, usecols=["CID", "mol", "pIC50"])
    raw.index = [f"BACE_{i:04d}" for i in raw.index]
    assays = source_records(a.supplement, raw)
    archive = pd.read_csv(HERE/"seed_review/molecules.csv")
    scaffold, = archive.loc[archive.sample_id == "BACE_0638", "scaffold"].unique()
    rows = archive[archive.scaffold == scaffold].copy()
    ids = sorted(rows.sample_id.unique())
    assert len(ids) == 10 and len(rows) == 30 and not rows.duplicated(["seed","sample_id"]).any()
    assert set(rows.seed) == {0,1,2}
    records = assays.loc[ids]
    assert records.STANDARD_TYPE.eq("IC50").all() and records.RELATION.eq("=").all() and records.STANDARD_UNITS.eq("uM").all()
    np.testing.assert_allclose(-np.log10(records.STANDARD_VALUE.astype(float)*1e-6), raw.loc[ids,"pIC50"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(rows.y_true, raw.loc[rows.sample_id,"pIC50"], rtol=0, atol=1e-12)
    for scope in ("mixed","exact"):
        q = rows[f"{scope}_qwen_mlp"] - rows.y_true
        m = rows[f"{scope}_molformer_mlp"] - rows.y_true
        np.testing.assert_allclose(q*q-m*m, rows[f"{scope}_gap"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(rows.exact_gap-rows.mixed_gap, rows.swing, atol=1e-12, rtol=0)
    totals = rows.groupby("seed")[["mixed_gap","exact_gap","swing"]].sum()
    previous = pd.read_csv(HERE/"seed_review/scaffolds.csv").query("first_id == 'BACE_0638'").set_index("seed")
    np.testing.assert_allclose(totals, previous.loc[totals.index, totals.columns])

    smiles = {k: Chem.MolToSmiles(Chem.MolFromSmiles(raw.at[k,"mol"]), isomericSmiles=False) for k in ids}
    template = Chem.MolFromSmiles(smiles["BACE_1310"])
    rdDepictor.Compute2DCoords(template)
    mols, notes, highlights, legends = [], [], [], []
    for k in ids:
        mol = Chem.MolFromSmiles(smiles[k])
        assert MurckoScaffold.MurckoScaffoldSmiles(mol=mol) == scaffold
        match = mol.GetSubstructMatch(template)
        assert len(match) == template.GetNumAtoms()
        extra = sorted(set(range(mol.GetNumAtoms())) - set(match))
        additions = Chem.MolFragmentToSmiles(mol, atomsToUse=extra) if extra else ""
        attachments = [i for i, atom in enumerate(match)
                       if any(n.GetIdx() in extra for n in mol.GetAtomWithIdx(atom).GetNeighbors())]
        rdDepictor.GenerateDepictionMatching2DStructure(mol, template, list(enumerate(match)))
        mols.append(mol)
        highlights.append(extra)
        legends.append(f"{k}   pIC50 {raw.at[k,'pIC50']:.2f}")
        notes.append({"sample_id":k, "CID":raw.at[k,"CID"], "input_smiles":smiles[k],
                      "added_fragments":additions, "template_atom_indices":",".join(map(str,attachments)),
                      "ic50_uM":records.at[k,"STANDARD_VALUE"], "pIC50":raw.at[k,"pIC50"],
                      "source_compound":records.at[k,"COMPOUND_KEY"], "pmid":records.at[k,"PUBMED_ID"]})
    a.out.mkdir(parents=True,exist_ok=True)
    Draw.MolsToGridImage(mols, molsPerRow=2, subImgSize=(440,215), legends=legends,
                         highlightAtomLists=highlights).save(a.out/"structures.png")
    pd.DataFrame(notes).to_csv(a.out/"structures.csv",index=False,lineterminator="\n")
    rows.drop(columns=["scaffold","group"]).sort_values(["sample_id","seed"]).to_csv(
        a.out/"predictions.csv",index=False,lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["raw","supplement","out"]:
        parser.add_argument("--"+arg,type=Path,required=True)
    review(parser.parse_args())
