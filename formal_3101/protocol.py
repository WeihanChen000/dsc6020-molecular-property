"""Prepare a locked, seed-specific queue using only training and validation rows."""
from pathlib import Path
import hashlib
import json
import os
import pandas as pd

SOURCE = Path(__file__).resolve().parent
PROJECT = SOURCE.parent
SEED = int(os.environ.get("DSC6020_SPLIT_SEED", "3101"))
if SEED not in (3101, 3102, 3103):
    raise ValueError(f"Unsupported split seed: {SEED}")
ROOT = PROJECT / f"formal_{SEED}"
SETS = {"FreeSolv": (514, 64), "ESOL": (904, 112), "Lipophilicity": (3360, 420)}
SPLITS = ("random_group", "random_scaffold")
CANDIDATES = {
    "M1": ("native", 3e-5, 0, 0), "M2": ("native", 1e-4, 0, 0),
    "Q1": ("masked_mean", 1e-4, 8, 16), "Q2": ("last", 1e-4, 8, 16),
    "Q3": ("masked_mean", 1e-4, 16, 16), "Q4": ("masked_mean", 1e-4, 8, 32),
    "Q5": ("masked_mean", 1e-4, 16, 32), "Q6": ("masked_mean", 3e-5, 8, 16)
}


def save_json(path, obj):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def pick(path, indexes, columns):
    allowed = set(indexes)
    return pd.read_csv(path, usecols=columns, skiprows=lambda n: n > 0 and n-1 not in allowed)


def prepare(check_only=False):
    if not check_only:
        (ROOT/"inputs").mkdir(parents=True,exist_ok=True)
    manifests = {}
    for ds, counts in SETS.items():
        for split in SPLITS:
            if ds == "FreeSolv":
                path = PROJECT.parent/"dsc6020-molecular-property/final/freesolv/results/partitions.csv"
                flags = pd.read_csv(path, usecols=["partition","split","split_seed"])
                chosen = flags.partition.isin(["train","valid"]) & flags.split.eq(split) & flags.split_seed.eq(SEED)
                data = pick(path, flags.index[chosen], ["sample_id","connectivity","scaffold","partition"])
                raw_ids = data.sample_id.str.removeprefix("FS_").astype(int)
                raw = pick(PROJECT/"study_v7/SAMPL.csv",raw_ids,["expt"])
                raw.index = sorted(raw_ids)
                data["y_true"] = raw.expt.loc[raw_ids].to_numpy()
            else:
                path = PROJECT/f"study_v3/inputs/{ds}/{split}_{SEED}_split.csv"
                flags = pd.read_csv(path, usecols=["partition"])
                data = pick(path, flags.index[flags.partition.isin(["train","valid"])],
                            ["sample_id","connectivity","scaffold","y_true","partition"])
            assert tuple(data.partition.value_counts()[["train","valid"]]) == counts
            assert set(data.partition)=={"train","valid"} and data.sample_id.is_unique
            assert data.y_true.notna().all() and data.connectivity.notna().all()
            key = "scaffold" if split=="random_scaffold" else "connectivity"
            assert data.groupby(key,dropna=False).partition.nunique().max()==1
            name = ds+"_"+split
            content = data.to_csv(index=False, lineterminator="\n").encode("utf-8")
            dest = ROOT/"inputs"/(name+".csv")
            if dest.exists() or check_only:
                if dest.read_bytes()!=content:
                    raise ValueError(f"Input changed: {name}")
            else:
                dest.write_bytes(content)
            manifests[name] = dict(train=counts[0], valid=counts[1], sha256=hashlib.sha256(content).hexdigest())
    queue = [(ds,split,c) for ds in SETS for split in SPLITS for c in CANDIDATES]
    protocol = dict(split_seed=SEED,train_seed=0,epochs=20,patience=10,
                    criteria="unrounded validation RMSE in original units; earliest epoch; first listed candidate",
                    candidates=CANDIDATES, queue=queue,inputs=manifests, test_access=False)
    path = ROOT/"protocol.json"
    if path.exists() or check_only:
        assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(protocol))
    else:
        save_json(path,protocol)
    return queue


if __name__=="__main__":
    print(f"{len(prepare())} train/valid fits")
