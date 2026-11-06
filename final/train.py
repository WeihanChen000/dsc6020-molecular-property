"""Train both descriptor variants with the original v3 partitions and trainer."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def run(args):
    root = args.study.resolve()
    sys.path.insert(0, str(root))
    import core

    jobs = [j for j in json.loads((root / "jobs.json").read_text()) if
            j["stage"] == "B" and j["dataset"] == args.dataset]
    assert len(jobs) == 6
    inputs = root / "inputs" / args.dataset
    names = json.loads((inputs / "descriptor_names.json").read_text())
    x = core.features(args.dataset, "descriptor_mlp", {})
    assert names.count("Ipc") == 1 and x.shape[1] == len(names) == 210
    if args.variant == "no_ipc":
        x = np.delete(x, names.index("Ipc"), axis=1)
    samples = pd.read_csv(inputs / "samples.csv")
    for j in jobs:
        out = args.output / args.variant / j["context"]
        out.mkdir(parents=True, exist_ok=True)
        split = pd.read_csv(root / j["partition_file"])
        assert split.sample_id.is_unique and split.sample_id.equals(samples.sample_id)
        np.testing.assert_array_equal(split.y_true, samples.y_true)
        train = np.array(j["train_indices"])
        assert np.array_equal(np.sort(train), np.flatnonzero(split.partition.eq("train")))
        assert j["train_seeds"] == [0]
        valid = np.flatnonzero(split.partition.eq("valid"))
        y = split.y_true.to_numpy()
        provenance = {p.name: core.sha(p) for p in [inputs/"features.npz", inputs/"descriptor_names.json",
                      root/j["partition_file"], root/"jobs.json", root/"core.py", root/"protocol.json",
                      HERE/"PROTOCOL.md", Path(__file__)]}
        marker = out / "complete.json"
        if marker.exists():
            assert json.loads(marker.read_text())["hashes"] == provenance
            continue
        trials = []
        for i, lr in enumerate(core.PROTOCOL["mlp"]["learning_rates"]):
            dest = out / f"trial{i}"
            dest.mkdir(exist_ok=True)
            trials.append(core.train_neural("descriptor_mlp", x, y, train, valid, {"lr":lr}, 0, dest))
        best = min(range(len(trials)), key=lambda i: trials[i]["valid_rmse"])
        for part in ["valid", "test"]:
            ids = np.flatnonzero(split.partition.eq(part))
            pred = core.restore("descriptor_mlp", out/f"trial{best}", x, ids)
            assert np.isfinite(pred).all()
            if part == "valid":
                assert np.isclose(core.score(y[ids], pred)["rmse"], trials[best]["valid_rmse"])
            split.iloc[ids][["sample_id","y_true"]].assign(y_pred=pred).to_csv(out/f"{part}.csv", index=False)
        core.dump(marker, {"hashes":provenance,"chosen":best,"candidates":trials,"dimension":x.shape[1]})
        print("Completed",j["context"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["ESOL","Lipophilicity"])
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--variant", choices=["no_ipc","all"], default="no_ipc")
    parser.add_argument("--output", type=Path, default=HERE/"runs")
    run(parser.parse_args())
