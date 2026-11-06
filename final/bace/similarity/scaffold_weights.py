"""Explain the two RMSE rankings from the published paired predictions alone."""
import argparse
import csv
from collections import defaultdict
from math import fsum, isclose, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("dmpnn", "molformer_mlp")


def read(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def explain(records, seed, scores):
    by_scaffold = defaultdict(list)
    for row in records:
        by_scaffold[row["scaffold"]].append(row)
    n, k = len(records), len(by_scaffold)
    assert len({r["sample_id"] for r in records}) == n == 300 and k == 183
    detail = []
    for scaffold, rows in sorted(by_scaffold.items()):
        mse = {model: fsum((float(r[model]) - float(r["y_true"]))**2 for r in rows) / len(rows)
               for model in MODELS}
        gap = mse["dmpnn"] - mse["molformer_mlp"]
        size = len(rows)
        detail.append({"scaffold": scaffold, "example_id": min(r["sample_id"] for r in rows),
                       "records": size, **mse, "group_mse_gap": gap,
                       "record_contribution": size / n * gap, "scaffold_contribution": gap / k,
                       "weight_shift": (1 / k - size / n) * gap})
    summary = {"seed": seed}
    for model in MODELS:
        official = scores[seed, model]
        for metric, result in [("rmse", sqrt(fsum(r["records"] * r[model] for r in detail) / n)),
                               ("scaffold_rmse", sqrt(fsum(r[model] for r in detail) / k))]:
            assert isclose(result, float(official[metric]), rel_tol=0, abs_tol=1e-12)
    record_total = fsum(r["record_contribution"] for r in detail)
    scaffold_total = fsum(r["scaffold_contribution"] for r in detail)
    for row in detail:
        row["omit_record_mse_gap"] = (record_total - row["record_contribution"]) * n / (n - row["records"])
        row["omit_scaffold_mse_gap"] = (scaffold_total - row["scaffold_contribution"]) * k / (k - 1)
    summary.update(record_mse_gap=record_total, scaffold_mse_gap=scaffold_total,
                   reweighting_change=scaffold_total-record_total)
    for unit in ["record", "scaffold"]:
        gaps = [r["omit_" + unit + "_mse_gap"] for r in detail]
        summary[unit + "_omissions_gnn_wins"] = sum(g < 0 for g in gaps)
        summary[unit + "_omissions_lm_wins"] = sum(g > 0 for g in gaps)
        summary[unit + "_omissions_ties"] = sum(g == 0 for g in gaps)
    return detail, summary


def run(source, out):
    records = read(source / "grouped_predictions.csv")
    official = {(int(r["seed"]), r["model"]): r for r in read(source / "scores.csv") if r["split"] == "grouped"}
    summaries = []
    out.mkdir(parents=True, exist_ok=True)
    for seed in [0, 1, 2]:
        detail, summary = explain([r for r in records if int(r["seed"]) == seed], seed, official)
        summaries.append(summary)
        if seed == 0:
            write(out / "seed0_scaffolds.csv", sorted(detail, key=lambda r: -abs(r["weight_shift"])))
    write(out / "seed_sensitivity.csv", summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.out)
