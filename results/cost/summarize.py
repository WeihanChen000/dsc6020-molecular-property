"""Join historical candidate logs to the 72 latency measurements and official test scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
KEY = ["dataset", "split", "seed", "model"]
ORDER = ["RF", "D-MPNN", "MoLFormer", "Qwen"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_ledger(project):
    records = []
    baseline = pd.read_csv(project / "revised_gpu_pilot/audit/runs.csv")
    for row in baseline.itertuples(index=False):
        weight = project / row.checkpoint_rel
        model = {"rf":"RF", "dmpnn":"D-MPNN"}[row.model]
        trials = list(weight.parent.parent.glob("seed0_candidate*/fit.json" if row.dataset == "FreeSolv"
                                                 else "trial*_seed0/fit.json"))
        assert len(trials) == row.candidate_count
        elapsed = 0.
        for trial in sorted(trials):
            meta = read(trial)
            assert meta["route"] == row.model and meta["seed"] == 0
            selected = trial.parent == weight.parent
            elapsed += meta["fit_seconds"]
            if selected:
                np.testing.assert_allclose(meta["fit_seconds"], row.fit_seconds)
            records.append(dict(dataset=row.dataset, split=row.split, seed=row.split_seed, model=model,
                                candidate=trial.parent.name, attempt=1, selected=selected, outcome="ok",
                                fit_s=meta["fit_seconds"], fit_lower_bound=False, search_elapsed_s=meta["fit_seconds"],
                                elapsed_scope="fit_log", epochs=meta.get("epochs_run"), best_epoch=meta.get("best_epoch")))
        np.testing.assert_allclose(elapsed, row.candidate_fit_seconds_sum)
    locks = pd.read_csv(RESULTS / "validation/selected_checkpoints.csv")
    selected_paths = set(locks.checkpoint)
    for seed in [3101, 3102, 3103]:
        runs = project / f"formal_{seed}" / "runs"
        for attempt in sorted(runs.glob("*/*/*/attempt_*")):
            ds, split, candidate, attempt_id = attempt.relative_to(runs).parts
            selected = (attempt / "best.pt").relative_to(project).as_posix() in selected_paths
            model = "MoLFormer" if candidate.startswith("M") else "Qwen"
            history = read(attempt / "history.json")
            fit_s = sum(h["train_s"] for h in history)
            partial = (attempt / "zero_gradient.json").exists()
            if (attempt / "result.json").is_file():
                meta = read(attempt / "result.json")
                np.testing.assert_allclose(fit_s, meta["train_s"])
                elapsed = meta["wall_s"]
                scope = "attempt_wall"
                status = meta.get("status", "ok")
            else:
                assert partial and not selected
                elapsed = sum(h["train_s"] + h["valid_s"] + h["save_s"] for h in history)
                scope, status = "completed_epoch_lower_bound", "failed_zero_gradient"
            records.append(dict(dataset=ds, split=split, seed=seed, model=model, candidate=candidate,
                                attempt=attempt_id, selected=selected, outcome=status, fit_s=fit_s,
                                fit_lower_bound=partial, search_elapsed_s=elapsed, elapsed_scope=scope,
                                epochs=len(history), best_epoch=min(history, key=lambda h:h["valid_rmse"])["epoch"],
                                training_peak_alloc_MiB=max(h["gpu_allocated_MiB"] for h in history)))
    ledger = pd.DataFrame(records)
    assert ledger.groupby(KEY).selected.sum().eq(1).all()
    assert len(ledger.groupby(KEY)) == 72
    assert ledger.groupby("model").candidate.count().to_dict() == {"D-MPNN":36, "MoLFormer":37, "Qwen":108, "RF":54}
    return ledger


def collect(project, work):
    ledger = candidate_ledger(project)
    bench = pd.DataFrame([read(p) for p in sorted(work.glob("*.json"))])
    bench.loc[bench.model.isin(["MoLFormer","Qwen"]), "inference_features"] = "fresh"
    assert len(bench) == 72 and not bench.duplicated(KEY).any()
    scores = pd.read_csv(RESULTS / "all_scores.csv")
    table = scores.merge(bench, on=KEY, validate="one_to_one")
    table["device"] = np.where(table.model.isin(["RF","D-MPNN"]), "Intel Core Ultra 9 275HX (4 threads)",
                               "NVIDIA GeForce RTX 5060 Laptop GPU (8 GB)")
    assert len(table) == 72 and table.n.eq(table.n_test).all() and table.max_prediction_delta.max() < 1e-5
    sums = ledger.groupby(KEY).agg(candidate_count=("candidate", "nunique"), attempts=("attempt", "count"),
                                   search_elapsed_s=("search_elapsed_s","sum"), search_fit_s=("fit_s","sum"),
                                   elapsed_lower_bound=("elapsed_scope",lambda v:"completed_epoch_lower_bound" in set(v)))
    selected = ledger[ledger.selected].set_index(KEY)[["fit_s","search_elapsed_s","training_peak_alloc_MiB"]]
    table = table.join(sums, on=KEY).join(selected.rename(columns={"fit_s":"selected_fit_s",
                                 "search_elapsed_s":"selected_recorded_elapsed_s"}), on=KEY)
    ledger.to_csv(HERE / "candidate_costs.csv", index=False)
    table.to_csv(HERE / "fold_costs.csv", index=False)

    def aggregate(group):
        n = group.n.sum()
        return pd.Series(dict(device=group.device.iloc[0], batches=" / ".join(map(str,sorted(group.batch.unique()))),
                         folds=len(group), test_appearances=n, candidates=group.candidate_count.sum(),
                         attempts=group.attempts.sum(), search_recorded_h=group.search_elapsed_s.sum()/3600,
                         selected_fit_h=group.selected_fit_s.sum()/3600, load_median_s=group.load_s.median(),
                         preprocess_ms_per_mol=1000*group.preprocess_s.sum()/n,
                         inference_ms_per_mol=1000*group.infer_s.sum()/n,
                         warm_total_ms_per_mol=1000*(group.preprocess_s.sum()+group.infer_s.sum())/n,
                         molecules_per_s=n/group.infer_s.sum(),
                         peak_alloc_MiB=group.infer_peak_alloc_MiB.max(),
                         peak_reserved_MiB=group.infer_peak_reserved_MiB.max(),
                         selected_train_peak_MiB=group.training_peak_alloc_MiB.max()))

    overview = pd.DataFrame([dict(model=m, **aggregate(table[table.model==m])) for m in ORDER])
    overview.to_csv(HERE / "model_costs.csv",index=False)
    groups = []
    for key, group in table.groupby(["dataset","split","model"],sort=False):
        assert len(group) == 3
        groups.append(dict(zip(["dataset","split","model"],key), **aggregate(group),
                           rmse=group.test_rmse.mean(), mae=group.test_mae.mean()))
    detail = pd.DataFrame(groups)
    official = detail.merge(pd.read_csv(RESULTS / "summary.csv"), on=["dataset","split","model"],validate="one_to_one")
    np.testing.assert_allclose(official[["rmse","mae"]], official[["test_rmse_mean","test_mae_mean"]])
    detail.to_csv(HERE / "cost_and_error.csv", index=False)
    return overview, detail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    parser.add_argument("work",type=Path)
    args = parser.parse_args()
    collect(args.project, args.work)
