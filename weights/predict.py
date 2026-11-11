"""Load one released DSC6020 checkpoint and predict from connectivity SMILES."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def read_checkpoint(package):
    with ZipFile(package) as z:
        meta = json.loads(z.read("checkpoint.json"))
        payload = z.read("best.pt")
    keys = ("model", "dataset", "split", "seed")
    with (REPO / "results/validation/selected_checkpoints.csv").open(newline="") as f:
        matches = [row for row in csv.DictReader(f) if all(row[k] == str(meta[k]) for k in keys)]
    if len(matches) != 1 or hashlib.sha256(payload).hexdigest() != matches[0]["sha256"]:
        raise ValueError("Checkpoint differs from the published selection.")
    ck = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    cfg = ck["config"]
    row = matches[0]
    if (cfg["candidate"], ck["epoch"], cfg["split_seed"], cfg["dataset"], cfg["split"], cfg["route"]) != (
            row["candidate"], int(row["epoch"]), int(row["seed"]), row["dataset"], row["split"], row["model"].lower()):
        raise ValueError("Checkpoint metadata does not match the selected run.")
    return ck


def load_checkpoint(package, device="auto", base_dir=None):
    ck = read_checkpoint(package)
    cfg = ck["config"]
    qwen = cfg["route"] == "qwen"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if qwen and (device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise ValueError("Qwen replay requires a BF16-capable CUDA GPU. MoLFormer also supports CPU.")
    if qwen:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig, Qwen3Model
        from huggingface_hub import snapshot_download

        source = json.loads((REPO / "revised_gpu_pilot/qwen/source.json").read_text())
        assert cfg["source"] == source["revision"]
        base = base_dir or snapshot_download(source["repository"], revision=source["revision"],
                                             allow_patterns=[f["name"] for f in source["files"]])
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        body = Qwen3Model.from_pretrained(base, quantization_config=quant, device_map={"":"cuda:0"},
                                          dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True)
        body.config.use_cache = False
        body = prepare_model_for_kbit_training(body, use_gradient_checkpointing=False)
        body = get_peft_model(body, LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=0.,
                                              target_modules=["q_proj","v_proj"], bias="none"))
        params = {n:p for n,p in body.named_parameters() if p.requires_grad}
        if params.keys() != ck["body"].keys():
            raise ValueError("LoRA parameter names do not match the saved adapter.")
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(ck["body"][name])
    else:
        base = base_dir or HERE / "molformer-runtime"
        config = AutoConfig.from_pretrained(base, trust_remote_code=True, local_files_only=True)
        config.deterministic_eval = True
        body = AutoModel.from_config(config, trust_remote_code=True)
        body.load_state_dict(ck["body"], strict=True)
        body = body.to(device)
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=not qwen, local_files_only=True, padding_side="right")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    head = nn.Sequential(nn.Linear(body.config.hidden_size,128), nn.ReLU(), nn.Linear(128,1)).to(device)
    head.load_state_dict(ck["head"], strict=True)
    body.eval()
    head.eval()

    @torch.inference_mode()
    def predict(smiles):
        result = []
        batch = cfg["valid_batch"]
        for begin in range(0,len(smiles),batch):
            tokens = tok(smiles[begin:begin+batch], padding=True, truncation=False,
                         add_special_tokens=not qwen, return_tensors="pt").to(device)
            output = body(**tokens)
            if cfg["pooling"] == "native":
                vector = output.pooler_output
            elif cfg["pooling"] == "last":
                vector = output.last_hidden_state[torch.arange(len(tokens.input_ids),device=device),
                                                  tokens.attention_mask.sum(1)-1]
            elif cfg["pooling"] == "masked_mean":
                mask = tokens.attention_mask.unsqueeze(-1)
                vector = (output.last_hidden_state.float()*mask).sum(1)/mask.sum(1)
            else:
                raise ValueError(f"Unknown pooling: {cfg['pooling']}")
            scaled = head(vector.float()).flatten().cpu().double()
            result.extend((scaled*cfg["scale"]+cfg["mean"]).tolist())
        return result
    return predict, cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package",type=Path,help="A downloaded checkpoint ZIP")
    parser.add_argument("--smiles",nargs="+",required=True,help="Canonical non-isomeric connectivity SMILES")
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--base-dir",type=Path,help="Optional local snapshot of the exact upstream revision")
    args = parser.parse_args()
    predict, cfg = load_checkpoint(args.package, args.device, args.base_dir)
    for smiles, value in zip(args.smiles, predict(args.smiles)):
        print(f"{cfg['dataset']}\t{smiles}\t{value:.6f}")
