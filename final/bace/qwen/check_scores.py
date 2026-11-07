"""Recompute each published score from matched predictions, using only Python."""
import csv
from collections import defaultdict
from math import isclose, sqrt
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).resolve().parent / "results"


def read(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


if __name__ == "__main__":
    records = read("predictions.csv")
    scores = read("scores.csv")
    assert len(records) == 900 and len(scores) == 6
    assert {(r["model"], r["seed"]) for r in scores} == {
        (model, seed) for model in ["qwen_mlp", "molformer_mlp"] for seed in ["0", "1", "2"]}
    expected_ids = {r["sample_id"] for r in records}
    for row in scores:
        subset = [r for r in records if r["seed"] == row["seed"]]
        assert len(subset) == len(expected_ids) == 300
        assert {r["sample_id"] for r in subset} == expected_ids
        residuals = []
        scaffold_errors = defaultdict(list)
        for r in subset:
            residual = float(r[row["model"]]) - float(r["y_true"])
            residuals.append(residual)
            scaffold_errors[r["scaffold"]].append(residual**2)
        assert len(scaffold_errors) == 183
        checked = {"rmse": sqrt(mean(e**2 for e in residuals)), "mae": mean(map(abs, residuals)),
                   "scaffold_rmse": sqrt(mean(map(mean, scaffold_errors.values())))}
        for key, value in checked.items():
            assert isclose(value, float(row[key]), abs_tol=1e-12, rel_tol=0)
    print("Verified 18 metrics across three matched sets of 300 predictions.")
