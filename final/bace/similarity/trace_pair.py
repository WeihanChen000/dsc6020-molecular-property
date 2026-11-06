"""Trace two project row IDs through DeepChem to the authors' SDF by CID."""
import argparse
import csv
import hashlib
import math
import re
from pathlib import Path
from zipfile import ZipFile


def trace(raw, supplement, out):
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == "f3fb9ce90bada3e2bd6148b0df13f8f8145a357bf87df0dd5b391ede974fc737"
    assert hashlib.md5(supplement.read_bytes()).hexdigest() == "aa483ebebcb0b18fa5d639700db32f60"
    with raw.open(encoding="utf-8", newline="") as f:
        table = list(csv.DictReader(f))
    with ZipFile(supplement) as z:
        sdf = z.read("bace1547.sdf").decode().replace("\r", "")
    targets = {table[i]["CID"]: i for i in (202, 203)}
    rows = []
    for record in sdf.split("$$$$"):
        fields = dict(re.findall(r"^>\s+<([^>]+)>\n([^\n]+)", record, re.M))
        if fields.get("CID") not in targets:
            continue
        idx = targets[fields["CID"]]
        assert fields["STANDARD_UNITS"] == "uM" and fields["STANDARD_TYPE"] == "IC50"
        log_activity = -math.log10(float(fields["STANDARD_VALUE"]) * 1e-6)
        assert math.isclose(log_activity, float(table[idx]["pIC50"]), abs_tol=2e-7)
        assert fields["pIC50"] == table[idx]["pIC50"]
        rows.append({"sample_id": f"BACE_{idx:04d}", **{k: fields[k] for k in [
            "CID", "PARENT_CMPD_ID", "COMPOUND_KEY", "PUBMED_ID", "RELATION", "STANDARD_TYPE",
            "STANDARD_VALUE", "STANDARD_UNITS", "pIC50"]}})
    assert len(rows) == len({r["CID"] for r in rows}) == 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "supplement", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    trace(args.raw, args.supplement, args.out)
