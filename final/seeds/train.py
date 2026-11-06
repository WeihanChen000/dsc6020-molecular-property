"""Repeat the two seed-0-selected pipelines on fixed scaffold partitions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def run(root, output, split_seed):
    sys.path.insert(0, str(root.resolve()))
    import core

    jobs = [j for j in json.loads((root/"jobs.json").read_text())
            if j["stage"] == "B" and j["split_type"] == "random_scaffold" and j["split_seed"] == split_seed]
    assert len(jobs) == 2
    for j in jobs:
        frame = pd.read_csv(root/j["partition_file"])
        src = root/"inputs"/j["dataset"]
        samples = pd.read_csv(src/"samples.csv")
        assert frame.sample_id.is_unique and frame.sample_id.equals(samples.sample_id)
        assert frame.y_true.equals(samples.y_true) and frame.groupby("scaffold").partition.nunique().max()==1
        train = np.array(j["train_indices"])
        valid = np.flatnonzero(frame.partition.eq("valid"))
        assert np.array_equal(np.sort(train), np.flatnonzero(frame.partition.eq("train")))
        for route in ["mlp","dmpnn"]:
            base=root/"results"/j["id"]/route
            selection=json.loads((base/"selection.json").read_text())
            assert selection["selection_seed"]==0
            assert selection["chosen"]==np.argmin([r["valid_rmse"] for r in selection["trials"]])
            param=selection["candidate"]
            assert param==selection["trials"][selection["chosen"]]["params"]
            x=core.features(j["dataset"],route,{})
            files=[root/"core.py",root/"protocol.json",root/"jobs.json",root/j["partition_file"],
                   src/"samples.csv",base/"selection.json",Path(__file__),Path(__file__).with_name("PROTOCOL.md")]
            if route=="mlp":
                files.append(src/"embeddings.npz")
            hashes={p.name:core.sha(p) for p in files}
            for seed in [1,2]:
                dest=output/j["context"]/route/f"seed{seed}"
                dest.mkdir(parents=True,exist_ok=True)
                marker=dest/"complete.json"
                if marker.exists():
                    assert hashes==json.loads(marker.read_text())["hashes"]
                    continue
                print("Starting",j["context"],route,seed,param,flush=True)
                meta=core.train_neural(route,x,frame.y_true.to_numpy(),train,valid,param,seed,dest)
                for part in ["valid","test"]:
                    ids=np.flatnonzero(frame.partition.eq(part))
                    pred=core.restore(route,dest,x,ids)
                    assert np.isfinite(pred).all()
                    if part=="valid":
                        assert np.isclose(core.score(frame.y_true.iloc[ids],pred)["rmse"],meta["valid_rmse"])
                    frame.iloc[ids][["sample_id","y_true"]].assign(y_pred=pred).to_csv(dest/f"{part}.csv",index=False)
                core.dump(marker,{"hashes":hashes,"fit":meta,"chosen_by_seed":0})
                print("Completed",j["context"],route,seed,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--split",type=int,choices=[3101,3102,3103],required=True)
    a=p.parse_args()
    run(a.study,a.out,a.split)
