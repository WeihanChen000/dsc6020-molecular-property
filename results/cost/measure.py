"""Replay locked checkpoints. Run the CPU and GPU passes separately, on an idle machine."""
import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RESULTS = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def timed(function, cuda=False):
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    if cuda:
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def gpu_route(path, model, texts):
    import gpu

    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    is_qwen = model == "Qwen"
    tokenizer, body, head, _ = gpu.load_model(cfg["route"], rank=cfg["rank"], alpha=cfg["alpha"])
    if is_qwen:
        params = {k: p for k, p in body.named_parameters() if p.requires_grad}
        assert params.keys() == ck["body"].keys()
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(ck["body"][name])
    else:
        body.load_state_dict(ck["body"], strict=True)
    head.load_state_dict(ck["head"], strict=True)
    body.eval()
    head.eval()
    batch = cfg["valid_batch"]

    def prepare():
        return [tokenizer(texts[i:i+batch], padding=True, add_special_tokens=not is_qwen,
                          truncation=False, return_tensors="pt") for i in range(0, len(texts), batch)]

    @torch.inference_mode()
    def predict(batches):
        chunks = []
        for tokens in batches:
            device_tokens = {k: v.to("cuda") for k, v in tokens.items()}
            pred = gpu.predict(body, head, device_tokens, cfg["pooling"])[0]
            chunks.append(pred.cpu().numpy())
        return np.concatenate(chunks).astype(np.float64) * cfg["scale"] + cfg["mean"]

    return batch, prepare, predict


def cpu_route(path, model, texts, names, fragment_file, parser):
    import core
    import joblib
    from rdkit import Chem

    def molecules():
        if parser is None:
            return [Chem.MolFromSmiles(s) for s in texts]
        print(json.dumps({"smiles":texts, "include_sps":model == "RF"}), file=parser.stdin, flush=True)
        response = json.loads(parser.stdout.readline())
        mols = [Chem.Mol(bytes.fromhex(m)) for m in response["mols"]]
        if model == "RF":
            for mol, sps in zip(mols, response["sps"], strict=True):
                mol.SetDoubleProp("_training_SPS", sps)
        return mols

    if model == "D-MPNN":
        ck = torch.load(path, map_location="cpu", weights_only=True)
        net = core.DMPNN()
        net.load_state_dict(ck["state_dict"])
        net.eval()
        meta = ck["metadata"]

        def prepare():
            from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
            featurizer = SimpleMoleculeMolGraphFeaturizer()
            graphs = [featurizer(mol) for mol in molecules()]
            return [core.batch_graph(graphs, range(i, min(i + 128, len(graphs))))
                    for i in range(0, len(graphs), 128)]

        @torch.inference_mode()
        def predict(batches):
            return np.concatenate([net(b).numpy() for b in batches]) * meta["y_scale"] + meta["y_mean"]

        return 128, prepare, predict

    from rdkit.Chem import Descriptors, Fragments
    pipeline = joblib.load(path)

    descriptor_map = dict(Descriptors.descList)
    # Restore the SMARTS definitions used to produce the training matrix.
    def patterns(file):
        result = {}
        for line in Path(file).read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= 3 and not line.startswith("#"):
                result[fields[0].replace("=", "_").replace("-", "_")] = fields[2].strip()
        return result

    installed = patterns(Fragments.defaultPatternFileName)
    for name, smarts in patterns(fragment_file).items():
        if name in names and smarts != installed.get(name):
            pattern = Chem.MolFromSmarts(smarts)
            assert pattern is not None
            descriptor_map[name] = lambda mol, p=pattern: len(mol.GetSubstructMatches(p, uniquify=True))
    if parser is not None:
        descriptor_map["SPS"] = lambda mol: mol.GetDoubleProp("_training_SPS")
    funcs = [descriptor_map[name] for name in names]
    assert len(funcs) == pipeline.n_features_in_

    def describe(mol):
        values = []
        for func in funcs:
            try:
                values.append(func(mol))
            except Exception:
                values.append(np.nan)  # Same missing-value rule as CalcMolDescriptors.
        return values

    def prepare():
        x = np.asarray([describe(mol) for mol in molecules()], dtype=float)
        x[~np.isfinite(x) | (np.abs(x) > np.finfo(np.float32).max)] = np.nan
        return pipeline[0].transform(x)

    return len(texts), prepare, pipeline[-1].predict


def measure(project, case, folds, ref, is_gpu, parser=None):
    model, seed, ds, split = case.model, case.seed, case.dataset, case.split
    selection = folds[(folds.dataset == ds) & (folds[f"{split}_{seed}"] == "test")]
    refs = ref[(ref.dataset == ds) & (ref.seed == seed) & (ref.split == split) & (ref.model == model)]
    assert selection.sample_id.is_unique and refs.sample_id.is_unique and set(refs.sample_id) == set(selection.sample_id)
    refs = refs.set_index("sample_id").loc[selection.sample_id]
    np.testing.assert_allclose(refs.y_true, selection.y_true, atol=1e-10, rtol=0)
    texts = selection.connectivity.tolist()
    names = None
    if model == "RF":
        names = json.loads((project / "study_v3/inputs/ESOL/descriptor_names.json").read_text())
    if model == "RF" and ds != "FreeSolv":
        raw = pd.read_csv(project / "data" / ("delaney-processed.csv" if ds == "ESOL" else "Lipophilicity.csv"))
        rows = selection.sample_id.str.split("_").str[-1].astype(int)
        texts = raw.iloc[rows].smiles.tolist()
    path = project / case.checkpoint
    weight_hash = digest(path)
    if is_gpu:
        assert weight_hash == case.sha256
        torch.manual_seed(0)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        factory = lambda: gpu_route(path, model, texts)
    else:
        fragments = project / "study_v2/.venv/Lib/site-packages/rdkit/Data/FragmentDescriptors.csv"
        factory = lambda: cpu_route(path, model, texts, names, fragments, parser)
    (batch, prepare, predict), load_s = timed(factory, is_gpu)
    data, prep_s = timed(prepare, is_gpu)
    first_predictions = predict(data)
    raw_delta = float(np.max(abs(first_predictions - refs.prediction.to_numpy())))
    feature_source = "fresh"
    assert raw_delta < 1e-5, (ds, split, seed, model, raw_delta)
    if is_gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    diff = 0.
    for _ in range(3):
        predictions, seconds = timed(lambda: predict(data), is_gpu)
        times.append(seconds)
        diff = max(diff, float(np.max(abs(predictions - refs.prediction.to_numpy()))))
        np.testing.assert_allclose(predictions, refs.prediction, atol=1e-5, rtol=0)
    return dict(dataset=ds, split=split, seed=int(seed), model=model, n=len(texts),
                batch=batch, weight_sha256=weight_hash, load_s=load_s, preprocess_s=prep_s,
                infer_s=statistics.median(times), **{f"infer_rep{i+1}_s": s for i,s in enumerate(times)},
                infer_peak_alloc_MiB=torch.cuda.max_memory_allocated()/2**20 if is_gpu else None,
                infer_peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20 if is_gpu else None,
                max_prediction_delta=diff, torch_version=torch.__version__,
                fresh_input_max_prediction_delta=raw_delta, inference_features=feature_source,
                rdkit_version=None if is_gpu else sys.modules["rdkit"].__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Original local experiment archive")
    parser.add_argument("route", choices=["gpu", "cpu"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--rdkit-package", type=Path, help="Optional existing RDKit distribution for CPU")
    parser.add_argument("--parser-python", type=Path, help="Interpreter with the training RDKit SMILES parser")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    is_gpu = args.route == "gpu"
    sys.path.insert(0, str(args.project / ("revised_gpu_pilot" if is_gpu else "study_v3")))
    if is_gpu:
        import gpu
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device required")
        torch.cuda.init()
        import bitsandbytes
        cases = pd.read_csv(RESULTS / "validation/selected_checkpoints.csv")
    else:
        if args.rdkit_package:
            sys.path.insert(0, str(args.rdkit_package.resolve()))
        import core
        import chemprop
        from rdkit.Chem import Descriptors
        cases = pd.read_csv(args.project / "revised_gpu_pilot/audit/runs.csv")
        assert set(cases.status) == {"reusable"}
        cases = cases.rename(columns={"split_seed":"seed", "checkpoint_rel":"checkpoint"})
        cases["model"] = cases.model.map({"rf":"RF", "dmpnn":"D-MPNN"})
    folds = pd.read_csv(RESULTS / "structure/folds.csv.gz")
    ref = pd.read_csv(RESULTS / "test_predictions.csv")
    worker = None
    parser_version = None
    try:
        if args.parser_python:
            worker = subprocess.Popen([str(args.parser_python.resolve()), str(Path(__file__).with_name("parse_smiles.py"))],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
            parser_version = worker.stdout.readline().strip()
            assert parser_version == "2023.09.6"
        for case in cases.itertuples(index=False):
            dest = args.out / f"{case.dataset}_{case.split}_{case.seed}_{case.model}.json"
            if dest.exists():
                continue
            row = measure(args.project, case, folds, ref, is_gpu, worker)
            row["parser_version"] = parser_version
            dest.write_text(json.dumps(row, indent=2, allow_nan=False))
            print(dest.stem, round(row["infer_s"], 4), flush=True)
            gc.collect()
    finally:
        if worker:
            worker.stdin.close()
            worker.wait()


if __name__ == "__main__":
    main()
