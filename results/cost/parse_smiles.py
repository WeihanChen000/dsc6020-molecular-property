"""Parse with the training RDKit version when its optional descriptor DLL is unavailable."""
import json
import sys

from rdkit import Chem, rdBase
from rdkit.Chem.SpacialScore import SPS

print(rdBase.rdkitVersion, flush=True)
for line in sys.stdin:
    request = json.loads(line)
    mols = [Chem.MolFromSmiles(smiles) for smiles in request["smiles"]]
    response = {"mols": [mol.ToBinary().hex() for mol in mols]}
    if request["include_sps"]:
        response["sps"] = [SPS(mol) for mol in mols]
    print(json.dumps(response), flush=True)
