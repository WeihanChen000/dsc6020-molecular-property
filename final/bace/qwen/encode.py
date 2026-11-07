"""Extract last-layer mean-pooled Qwen features on one GPU and measure resource use."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import threading
import time

import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, Qwen3Model

HERE = Path(__file__).resolve().parent


def gpu_reading():
    fields = "memory.used,memory.total,driver_version"
    return subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=" + fields,
                                    "--format=csv,noheader,nounits"], text=True).strip().split(", ")


def encode(a):
    pinned = json.loads((HERE / "model_manifest.json").read_text())
    for item in pinned["files"]:
        path = a.model / item["name"]
        assert path.stat().st_size == item["bytes"]
        with path.open("rb") as stream:
            if "sha256" in item:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"]
            else:
                raw = stream.read()
                assert hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest() == item["git_blob"]
    with a.structures.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["sample_id"] for r in rows]
    assert len(set(ids)) == len(ids) == 1503
    before, total, driver = gpu_reading()
    board_samples = [int(before)]
    done = threading.Event()

    def monitor():
        while not done.wait(.3):
            board_samples.append(int(gpu_reading()[0]))

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    start = time.perf_counter()
    try:
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, padding_side="right")
        model, info = Qwen3Model.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": "cuda:0"},
                                                attn_implementation="sdpa", local_files_only=True,
                                                output_loading_info=True)
        assert not info["missing_keys"] and not info["mismatched_keys"]
        model.eval().requires_grad_(False)
        torch.cuda.synchronize()
        loaded = time.perf_counter()
        tokens = tokenizer([r["connectivity"] for r in rows], add_special_tokens=False, truncation=False)["input_ids"]
        lengths = np.array(list(map(len, tokens)))
        assert lengths.min() > 0 and lengths.max() < model.config.max_position_embeddings
        order = np.argsort(lengths, kind="stable")
        x = np.empty((len(ids), model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for i in range(0, len(ids), 8):
                indices = order[i:i+8]
                batch = tokenizer.pad({"input_ids": [tokens[j] for j in indices]}, padding=True,
                                      return_attention_mask=True, return_tensors="pt").to("cuda")
                hidden = model(**batch, use_cache=False).last_hidden_state.float()
                mask = batch.attention_mask.unsqueeze(-1)
                x[indices] = ((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy()
        torch.cuda.synchronize()
        end = time.perf_counter()
        board_samples.append(int(gpu_reading()[0]))
    finally:
        done.set()
        watcher.join()
    assert np.isfinite(x).all()
    a.out.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(a.out / "vectors.npz", sample_id=np.asarray(ids), X=x)
    result = {"model_revision": pinned["revision"], "gpu": torch.cuda.get_device_name(), "driver": driver,
              "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "transformers": transformers.__version__, "dtype": "bfloat16", "batch": 8,
              "n": len(ids), "dim": x.shape[1], "min_tokens": int(lengths.min()), "max_tokens": int(lengths.max()),
              "load_seconds": loaded-start, "encode_seconds": end-loaded,
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20, "board_total_mib": int(total),
              "board_before_mib": int(before), "board_sampled_peak_mib": max(board_samples),
              "board_sample_count": len(board_samples), "cache_sha256": hashlib.sha256((a.out/"vectors.npz").read_bytes()).hexdigest()}
    (a.out / "gpu.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["model", "structures", "out"]:
        p.add_argument("--" + key, type=Path, required=True)
    encode(p.parse_args())
