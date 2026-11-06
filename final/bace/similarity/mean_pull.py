"""Post-hoc record-level diagnostic on the full grouped BACE holdout."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAIR = {"BACE_0202", "BACE_0203"}


def summarize(source, out):
    predictions = pd.read_csv(source)
    reference = predictions[predictions.seed == 0].set_index("sample_id").sort_index()
    assert len(reference) == 300 and reference.index.is_unique and PAIR <= set(reference.index)
    assert reference.train_mean.nunique() == 1 and set(predictions.seed) == {0, 1, 2}
    center = reference.train_mean.iloc[0]
    squared_error = reference[["dmpnn", "molformer_mlp"]].sub(reference.y_true, axis=0).pow(2)
    mse = squared_error.groupby(reference.scaffold).mean()
    gaps = mse.dmpnn - mse.molformer_mlp
    flip_scaffolds = gaps.index[gaps > gaps.sum()]
    assert len(flip_scaffolds) == 13
    summaries = []
    for seed, block in predictions.groupby("seed"):
        frame = block.set_index("sample_id").sort_index()
        pd.testing.assert_frame_equal(frame[["y_true", "train_mean", "scaffold"]], reference[["y_true", "train_mean", "scaffold"]])
        for model in ("dmpnn", "molformer_mlp"):
            y, pred = frame.y_true, frame[model]
            error = pred - y
            assert np.isfinite(pred).all() and (y != center).all()
            direction = error * (center - y) > 0
            closer = (pred - center).abs() < (y - center).abs()
            between = direction & ((pred - center) * (y - center) >= 0)
            for scope, included in [("all", y.notna()), ("without_pair", ~frame.index.isin(PAIR)),
                                    ("outside_flip13", ~frame.scaffold.isin(flip_scaffolds))]:
                for side, mask in [("all", included), ("low", included & (y < center)), ("high", included & (y > center))]:
                    summaries.append({"scope": scope, "seed": seed, "model": model, "side": side,
                                      "n": int(mask.sum()), "toward": int(direction[mask].sum()),
                                      "between": int(between[mask].sum()), "closer": int(closer[mask].sum()),
                                      "mean_error": error[mask].mean(), "mae": error[mask].abs().mean()})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out, index=False, lineterminator="\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=HERE / "results/grouped_predictions.csv")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    summarize(args.source, args.out)
