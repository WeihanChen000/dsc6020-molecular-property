"""Check selections and reload all new checkpoints before scoring them."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from summarize import summarize


def collect(root,runs,reference,out):
    sys.path.insert(0,str(root.resolve()))
    import core
    out.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(reference)
    jobs=[j for j in json.loads((root/"jobs.json").read_text()) if j["stage"]=="B" and j["split_type"]=="random_scaffold"]
    inputs,records,inventory={},[],[]
    for j in jobs:
        ctx=j["context"]
        frame=pd.read_csv(root/j["partition_file"])
        test=frame[frame.partition.eq("test")].set_index("sample_id")
        assert test.index.is_unique and frame.groupby("scaffold").partition.nunique().max()==1
        baseline=old[old.context.eq(ctx)].set_index("sample_id").loc[test.index]
        np.testing.assert_array_equal(test.y_true,baseline.y_true)
        for seed in [0,1,2]:
            table=test[["scaffold","y_true"]].copy()
            for route,col in [("dmpnn","dmpnn"),("mlp","molformer_mlp")]:
                base=root/"results"/j["id"]/route
                choice=json.loads((base/"selection.json").read_text())
                if seed==0:
                    saved=pd.read_csv(base/"seed0_test_predictions.csv")
                    np.testing.assert_allclose(saved.set_index("sample_id").loc[test.index,"y_pred"],baseline[col],rtol=1e-7)
                else:
                    folder=runs/ctx/route/f"seed{seed}"
                    record=json.loads((folder/"complete.json").read_text())["fit"]
                    assert record["seed"]==seed and record["lr"]==choice["candidate"]["lr"]
                    curve=pd.read_csv(folder/"history.csv")
                    assert len(curve)==record["epochs_run"] and int(curve.loc[curve.valid_rmse.idxmin(),"epoch"])==record["best_epoch"]
                    x=core.features(j["dataset"],route,inputs)
                    for part in ["valid","test"]:
                        ix=np.flatnonzero(frame.partition.eq(part))
                        saved=pd.read_csv(folder/f"{part}.csv")
                        assert list(saved.sample_id)==list(frame.sample_id.iloc[ix])
                        np.testing.assert_array_equal(saved.y_true,frame.y_true.iloc[ix])
                        np.testing.assert_allclose(core.restore(route,folder,x,ix),saved.y_pred,rtol=1e-6,atol=1e-6)
                        if part=="valid":
                            assert np.isclose(core.score(saved.y_true,saved.y_pred)["rmse"],record["valid_rmse"])
                    inventory.append({"context":ctx,"dataset":j["dataset"],"split_seed":j["split_seed"],**record})
                assert saved.sample_id.is_unique and set(saved.sample_id)==set(test.index)
                pred=saved.set_index("sample_id").loc[test.index]
                np.testing.assert_array_equal(test.y_true,pred.y_true)
                table[col]=pred.y_pred
            records.append(table.reset_index().assign(dataset=j["dataset"],context=ctx,split_seed=j["split_seed"],train_seed=seed))
    assert len(inventory)==24 and len(records)==18
    pd.concat(records,ignore_index=True).to_csv(out/"predictions.csv",index=False)
    pd.DataFrame(inventory).to_csv(out/"training.csv",index=False)
    summarize(out/"predictions.csv",out)
    core.dump(out/"verification.json",{"new_checkpoints":24,"new_validation_files_rescored":24,
              "new_test_files_rescored":24,"seed0_prediction_files_checked_against_previous_report":12,
              "paired_split_and_seed_conditions":18,"paired_test_record_appearances":4788})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for arg in ["study","runs","reference","out"]:
        p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args()
    collect(a.study,a.runs,a.reference,a.out)
