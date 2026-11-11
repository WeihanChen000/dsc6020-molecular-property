"""One locked candidate on one prefiltered train/validation split. No test entry point."""
import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time

from protocol import ROOT, PROJECT, SEED, SETS, SPLITS, CANDIDATES, save_json

sys.path.insert(0, str(PROJECT/"revised_gpu_pilot"))
import gpu
from gpu import np, pd, torch


def fit(ds, split, candidate, micro, dest):
    start = time.perf_counter()
    dest.mkdir(parents=True)
    inp = ROOT/"inputs"/f"{ds}_{split}.csv"
    proto=json.loads((ROOT/"protocol.json").read_text(encoding="utf-8"))
    digest=hashlib.sha256(inp.read_bytes()).hexdigest()
    assert proto["split_seed"]==SEED and digest==proto["inputs"][f"{ds}_{split}"]["sha256"]
    data = pd.read_csv(inp)
    assert set(data.partition)=={"train","valid"}
    train = data[data.partition=="train"].reset_index(drop=True)
    valid = data[data.partition=="valid"].reset_index(drop=True)
    assert (len(train),len(valid)) == SETS[ds]
    isq = candidate.startswith("Q")
    route = "qwen" if isq else "molformer"
    effective = 4 if isq else 8
    val_batch = 4 if isq else 8
    pool,lr,rank,alpha = CANDIDATES[candidate]
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tok,body,head,_ = gpu.load_model(route,rank=rank,alpha=alpha)
    params = [p for p in body.parameters() if p.requires_grad]
    assert all(p.device.type=="cuda" for p in list(body.parameters())+list(head.parameters()))
    if isq:
        assert all(("lora_" in n)==p.requires_grad for n,p in body.named_parameters())
    optimizer = torch.optim.AdamW(params+list(head.parameters()),lr=lr,weight_decay=0.01)
    mean,scale=float(train.y_true.mean()),float(train.y_true.std(ddof=0))
    targets = ((train.y_true.to_numpy()-mean)/scale).astype(np.float32)
    train_text=train.connectivity.to_numpy()
    valid_text=valid.connectivity.to_numpy()
    valid_y=valid.y_true.to_numpy(dtype=np.float64)
    def tokens(texts):
        return tok(list(texts),padding=True,add_special_tokens=not isq,truncation=False,return_tensors="pt").to("cuda")
    @torch.inference_mode()
    def evaluate():
        body.eval()
        head.eval()
        parts = [gpu.predict(body,head,tokens(valid_text[i:i+val_batch]),pool)[0].cpu().numpy()
                 for i in range(0,len(valid),val_batch)]
        predictions = np.concatenate(parts).astype(np.float64)*scale+mean
        assert np.isfinite(predictions).all()
        return predictions
    config=dict(dataset=ds,split=split,split_seed=SEED,candidate=candidate,route=route,
                lr=lr,pooling=pool,rank=rank,alpha=alpha,train_seed=0,micro=micro,
                effective_batch=effective,valid_batch=val_batch,mean=mean,scale=scale,
                train_n=len(train),valid_n=len(valid),input_sha256=digest,
                source=json.loads((PROJECT/"revised_gpu_pilot"/route/"source.json").read_text(encoding="utf-8"))["revision"],
                torch_version=str(torch.__version__),device=torch.cuda.get_device_name())
    save_json(dest/"config.json",config)
    torch.cuda.synchronize()
    load_s=time.perf_counter()-start
    history=[]
    best=float("inf")
    best_epoch=0
    failure=None
    for epoch in range(1,21):
        body.train()
        head.train()
        torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter()
        order=np.random.default_rng(np.random.SeedSequence([0,epoch])).permutation(len(train))
        loss_sum=0.
        for begin in range(0,len(order),effective):
            group=order[begin:begin+effective]
            optimizer.zero_grad(set_to_none=True)
            for k in range(0,len(group),micro):
                ids=group[k:k+micro]
                y=torch.from_numpy(targets[ids]).to("cuda")
                pred=gpu.predict(body,head,tokens(train_text[ids]),pool)[0]
                loss=(pred-y).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite train loss")
                (loss*(len(ids)/len(group))).backward()
                loss_sum+=float(loss.detach())*len(ids)
                del loss,pred
            gb=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            gh=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            if not (gb>0 and gh>0):
                failure=dict(kind="zero_gradient",epoch=epoch,update=1+begin//effective,
                             body_norm=float(gb),head_norm=float(gh),
                             ids=train.iloc[group].sample_id.tolist())
                save_json(dest/"zero_gradient.json",failure)
                print("Rejected update before optimizer.step:",failure,flush=True)
                break
            optimizer.step()
        if failure:
            break
        torch.cuda.synchronize()
        train_s=time.perf_counter()-t
        t=time.perf_counter()
        v=evaluate()
        rmse=float(np.sqrt(np.mean((v-valid_y)**2)))
        mae=float(np.mean(abs(v-valid_y)))
        valid_s=time.perf_counter()-t
        t=time.perf_counter()
        if rmse<best:
            best=rmse
            best_epoch=epoch
            # MoLFormer eval uses the random projections from the last train forward; save its buffers.
            weights = {n:p.detach() for n,p in body.named_parameters() if p.requires_grad} if isq else body.state_dict()
            checkpoint=dict(body=weights,head=head.state_dict(),config=config,epoch=epoch)
            torch.save(checkpoint,dest/"best.tmp")
            os.replace(dest/"best.tmp",dest/"best.pt")
            pd.DataFrame(dict(sample_id=valid.sample_id,y_true=valid_y,prediction=v)).to_csv(dest/"validation.csv",index=False)
            del checkpoint,weights
        history.append(dict(epoch=epoch,train_mse_standardized=loss_sum/len(train),valid_rmse=rmse,valid_mae=mae,
                            train_s=train_s,valid_s=valid_s,save_s=time.perf_counter()-t,
                            gpu_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                            gpu_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
        save_json(dest/"history.json", history)
        print(ds,split,candidate,epoch,"valid",round(rmse,6),"t",round(train_s+valid_s,1),flush=True)
        if epoch-best_epoch>=10:
            break
    # Read the winning weights from disk and reproduce the saved validation predictions.
    if not history:
        save_json(dest/"result.json",dict(**config,status="failed_zero_gradient",epochs=0,
                  best_epoch=None,valid_rmse=None,valid_mae=None,reload_max_abs=None,
                  warm_load_s=load_s,train_s=0,valid_s=0,wall_s=time.perf_counter()-start))
        return
    ck=torch.load(dest/"best.pt",map_location="cpu",weights_only=True)
    assert ck["epoch"]==best_epoch and ck["config"]==config
    if isq:
        with torch.no_grad():
            assert set(ck["body"])=={n for n,p in body.named_parameters() if p.requires_grad}
            for n,p in body.named_parameters():
                if p.requires_grad:
                    p.copy_(ck["body"][n])
    else:
        body.load_state_dict(ck["body"])
    head.load_state_dict(ck["head"])
    del ck
    replay=evaluate()
    reference=pd.read_csv(dest/"validation.csv")
    assert list(reference.sample_id)==list(valid.sample_id)
    err=float(max(abs(replay-reference.prediction)))
    assert err<1e-5,err
    result=dict(**config,status="failed_zero_gradient" if failure else "ok",epochs=len(history),best_epoch=best_epoch,valid_rmse=best,
                valid_mae=history[best_epoch-1]["valid_mae"],warm_load_s=load_s,
                train_s=sum(x["train_s"] for x in history),valid_s=sum(x["valid_s"] for x in history),
                wall_s=time.perf_counter()-start,reload_max_abs=err)
    save_json(dest/"result.json",result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("dataset",choices=SETS)
    p.add_argument("split",choices=SPLITS)
    p.add_argument("candidate",choices=CANDIDATES)
    a=p.parse_args()
    root=ROOT/"runs"/a.dataset/a.split/a.candidate
    root.mkdir(parents=True,exist_ok=True)
    micro=4
    while micro>=1:
        dest=root/f"attempt_{time.time_ns()}"
        oom=False
        try:
            fit(a.dataset,a.split,a.candidate,micro,dest)
        except torch.cuda.OutOfMemoryError as e:
            save_json(dest/"oom.json",dict(micro=micro,error=str(e)))
            oom=True
        if not oom:
            row=json.loads((dest/"result.json").read_text(encoding="utf-8"))
            if row["status"]=="failed_zero_gradient":
                if len(list(root.glob("attempt_*/zero_gradient.json")))<2:
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                marker="failed.json"
            else:
                marker="done.json"
            save_json(root/marker,dict(**row,attempt=dest.name))
            if SEED==3101 and (a.dataset,a.split,a.candidate)==("Lipophilicity","random_scaffold","Q6"):
                from finish import finish
                finish()
            return
        gc.collect()
        torch.cuda.empty_cache()
        micro//=2
    raise RuntimeError("All training microbatches exhausted; review before changing evaluation protocol")


if __name__=="__main__":
    main()
