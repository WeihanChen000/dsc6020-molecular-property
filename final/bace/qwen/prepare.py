"""Export only the structures and IDs used by the existing BACE comparison."""
import argparse
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from partition import build


def prepare(raw, study, out):
    data = build(raw, study)
    members = pd.read_csv(HERE.parent / "similarity/partitions.csv")
    assert data.sample_id.tolist() == members.sample_id.tolist()
    out.parent.mkdir(parents=True, exist_ok=True)
    data[["sample_id", "connectivity"]].to_csv(out, index=False, lineterminator="\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["raw", "study", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    prepare(a.raw, a.study, a.out)
