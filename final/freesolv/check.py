"""Recompute coverage and test scores from the published CSVs, without model weights."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def verify(folder):
    parts = pd.read_csv(folder/"partitions.csv")
    preds = pd.read_csv(folder/"predictions.csv")
    scores = pd.read_csv(folder/"scores.csv")
    fits = pd.read_csv(folder/"training.csv")
    keys = ["split","split_seed","model","seed"]
    assert not preds.duplicated(keys+["sample_id"]).any()
    assert len(scores)==42 and not scores.duplicated(keys).any()
    assert len(fits)==60 and fits.selected.sum()==36
    coverage = []
    for (kind,seed), frame in parts.groupby(keys[:2]):
        assert len(frame)==frame.sample_id.nunique()==642
        assert frame.groupby("connectivity").partition.nunique().max()==1
        train = frame[frame.partition=="train"]
        test = frame[frame.partition=="test"]
        coverage.append(dict(split=kind,split_seed=seed,train=len(train),
                             valid=(frame.partition=="valid").sum(),test=len(test),
                             test_scaffolds=test.scaffold.nunique(),
                             shared_with_train=len(set(test.scaffold)&set(train.scaffold)),
                             largest_test_group=test.scaffold.value_counts().max()))
        if kind=="random_scaffold":
            assert frame.groupby("scaffold").partition.nunique().max()==1
            assert set(frame[frame.scaffold.isin(["__ACYCLIC__","c1ccccc1"])].partition)=={"train"}
        subset = preds[(preds.split==kind)&(preds.split_seed==seed)]
        assert subset.groupby("sample_id").expt.nunique().max()==1
        for model in ["rf","mlp","dmpnn","train_mean"]:
            rows = subset[subset.model==model]
            expected = {0,1,2} if kind=="random_scaffold" and model!="train_mean" else {0}
            assert set(rows.seed)==expected
            for s,g in rows.groupby("seed"):
                assert set(g.sample_id)==set(test.sample_id)
                delta = g.prediction-g.expt
                actual = [np.sqrt(np.mean(delta**2)),np.mean(abs(delta)),
                          np.sqrt((delta**2).groupby(g.scaffold).mean().mean())]
                sel = scores[(scores.split==kind)&(scores.split_seed==seed)&(scores.model==model)&(scores.seed==s)]
                np.testing.assert_allclose(actual, sel[["rmse","mae","scaffold_rmse"]].iloc[0],rtol=1e-8)
        for model,g in fits[(fits.split==kind)&(fits.split_seed==seed)].groupby("model"):
            options = g[g.seed==0].sort_values(["valid_rmse","candidate"])
            assert len(options)==(3 if model=="rf" else 2)
            assert set(g.loc[g.selected,"candidate"])=={options.iloc[0].candidate}
    print(pd.DataFrame(coverage).to_string(index=False))
    print(f"Verified {len(scores)} score rows, {len(preds)} predictions and {len(fits)} fits.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    verify(parser.parse_args().results)
