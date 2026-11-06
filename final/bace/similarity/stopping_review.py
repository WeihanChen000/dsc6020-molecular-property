"""Reconstruct stopping decisions and the saved predictors' spread without retraining."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARCHIVE_SHA = "f9056fb1bf0a05174b73e5c6cc29f7ea93a7565eef936f29ee804895ec54ec2a"


def distribution(y, pred, center):
    mean = pred.mean()
    return {"n": len(pred), "mean": mean, "sd": pred.std(ddof=1),
            "minimum": pred.min(), "maximum": pred.max(),
            "within_half_unit": int(((pred-center).abs() <= .5).sum()),
            "target_mean": y.mean(), "target_sd": y.std(ddof=1),
            "mean_pred_low": pred[y < center].mean(), "mean_pred_high": pred[y > center].mean(),
            "rmse": np.sqrt(((pred-y)**2).mean()),
            "constant_train_rmse": np.sqrt(((center-y)**2).mean()),
            "constant_pred_mean_rmse": np.sqrt(((mean-y)**2).mean())}


def analyze(archive, out):
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ARCHIVE_SHA
    training = pd.read_csv(HERE / "results/training.csv").query("selected")
    tests = pd.read_csv(HERE / "results/grouped_predictions.csv")
    split = pd.read_csv(HERE / "partitions.csv")
    scores = pd.read_csv(HERE / "results/scores.csv").query("split == 'grouped'").set_index(["model", "seed"])
    config = json.loads((HERE.parents[2] / "study/protocol.json").read_text())
    assert len(training) == 6
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4), sharex="col", sharey="col", layout="constrained")
    colors = ["#276b82", "#97733d", "#af4161"]
    stops, spreads = [], []
    with ZipFile(archive) as z:
        for t in training.itertuples():
            model = "dmpnn" if t.route == "dmpnn" else "molformer_mlp"
            folder = f"{t.route}/seed{t.seed}_lr{t.lr}"
            with z.open(folder + "/history.csv") as f:
                history = pd.read_csv(f).set_index("epoch")
            np.testing.assert_array_equal(history.index, np.arange(1, t.epochs_run+1))
            assert history.valid_rmse.idxmin() == t.best_epoch
            patience = config[t.route]["patience"]
            assert t.epochs_run - t.best_epoch == patience
            best = history.loc[t.best_epoch]
            last = history.iloc[-1]
            stops.append({"model": model, "seed": t.seed, "lr": t.lr, "best_epoch": t.best_epoch,
                          "stop_epoch": t.epochs_run, "patience": patience, "valid_rmse_best": best.valid_rmse,
                          "valid_rmse_last": last.valid_rmse, "valid_rmse_epoch3": history.loc[3, "valid_rmse"],
                          "valid_rmse_best_after_selected": history.loc[t.best_epoch+1:, "valid_rmse"].min(),
                          "train_std_mse_best_epoch": best.train_mse_standardized,
                          "train_std_mse_last_epoch": last.train_mse_standardized})
            with z.open(folder + "/validation.csv") as f:
                valid = pd.read_csv(f)
            test = tests[tests.seed == t.seed]
            for part, frame, pred in [("valid", valid, valid.y_pred), ("test", test, test[model])]:
                np.testing.assert_array_equal(frame.sample_id, split.sample_id[split.partition == part])
                stats = distribution(frame.y_true, pred, t.y_mean)
                expected = t.valid_rmse if part == "valid" else scores.loc[(model, t.seed), "rmse"]
                np.testing.assert_allclose(stats["rmse"], expected, atol=1e-12, rtol=0)
                spreads.append({"model": model, "seed": t.seed, "partition": part, "train_mean": t.y_mean, **stats})
            ax, ecdf = axes[int(t.route == "mlp")]
            color = colors[t.seed]
            ax.plot(history.index, history.valid_rmse, color=color, lw=1.4, label=f"Seed {t.seed}")
            ax.scatter(t.best_epoch, best.valid_rmse, color=color, s=28, zorder=3)
            quantile = np.arange(1, len(test)+1) / len(test)
            ecdf.step(np.sort(test[model]), quantile, where="post", color=color, lw=1.5)
            if t.seed == 0:
                ecdf.step(np.sort(test.y_true), quantile, where="post", color="#555555", linestyle="--", label="Measured")
                ecdf.axvline(t.y_mean, color="#777777", linestyle=":", label="Training mean")
    for i, route in enumerate(["D-MPNN", "Frozen MoLFormer + MLP"]):
        axes[i, 0].set(title=route, ylabel="Validation RMSE")
        axes[i, 1].set(title=route, ylabel="Cumulative fraction")
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[0, 1].legend(loc="upper left", fontsize=8)
    axes[1, 0].set_xlabel("Epoch (dot = saved checkpoint)")
    axes[1, 1].set_xlabel("Test prediction or measured pIC50")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stops).to_csv(out / "early_stops.csv", index=False, lineterminator="\n")
    pd.DataFrame(spreads).to_csv(out / "prediction_spread.csv", index=False, lineterminator="\n")
    fig.savefig(out / "stopping_and_spread.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.archive, args.out)
