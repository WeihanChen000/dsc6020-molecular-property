"""Small ESOL GPU training smoke test; test labels are never loaded."""
import argparse,csv,hashlib,json,os,platform,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
P=ROOT.parent
sys.path.insert(0,str(ROOT/'vendor'))
os.environ['HF_MODULES_CACHE']=str(ROOT/'hf_modules')
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import pandas as pd
import torch
from torch import nn
import transformers
from transformers import AutoModel,AutoTokenizer,Qwen3Model,BitsAndBytesConfig
from peft import LoraConfig,PeftModel,get_peft_model,prepare_model_for_kbit_training

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')

def batchrows(ds):
    source=P/f'study_v3/inputs/{ds}/random_group_3101_split.csv'
    if ds=='ESOL' and not source.is_file():
        return pd.read_csv(ROOT/'esol_pilot_inputs.csv')
    df=pd.read_csv(source,usecols=['sample_id','connectivity','partition'])
    if ds=='ESOL':
        allowed=set(df.index[df.partition!='test'])
        df.loc[df.partition!='test','y']=pd.read_csv(source,usecols=['y_true'],skiprows=lambda i:i>0 and i-1 not in allowed).y_true.values
    return df

def load_model(route,checkpoint=None,*,rank=8,alpha=16):
    isq=route=='qwen'
    modelpath=P/'models'/('Qwen3-1.7B' if isq else 'molformer')
    tok=AutoTokenizer.from_pretrained(checkpoint/'tokenizer' if checkpoint else modelpath,trust_remote_code=not isq,
                                      local_files_only=True,padding_side='right')
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    if isq:
        import bitsandbytes as bnb
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16)
        base,info=Qwen3Model.from_pretrained(modelpath,quantization_config=quant,device_map={'':'cuda:0'},
                             dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        assert not info['missing_keys'] and not info['mismatched_keys'],info
        base.config.use_cache=False
        assert any(isinstance(m,bnb.nn.Linear4bit) and m.weight.quant_state is not None for m in base.modules())
        base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=False)
        if checkpoint: model=PeftModel.from_pretrained(base,checkpoint/'adapter',is_trainable=False)
        else: model=get_peft_model(base,LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=0.0,bias='none',target_modules=['q_proj','v_proj']))
        pooling='masked_mean'
    else:
        model,info=AutoModel.from_pretrained(modelpath,trust_remote_code=True,local_files_only=True,
                deterministic_eval=True,use_safetensors=True,output_loading_info=True)
        assert not info['missing_keys'] and not info['mismatched_keys'],info
        model=model.cuda()
        pooling='native'
        if checkpoint:model.load_state_dict(torch.load(checkpoint/'encoder.pt',weights_only=True,map_location='cpu'))
    head=nn.Sequential(nn.Linear(model.config.hidden_size,128),nn.ReLU(),nn.Linear(128,1)).cuda()
    if checkpoint:
        head.load_state_dict(torch.load(checkpoint/'head.pt',weights_only=True,map_location='cpu'))
        saved=json.loads((checkpoint/'config.json').read_text())
        assert saved['pooling']==pooling
    return tok,model,head,pooling

def predict(model,head,tokens,pool):
    output=model(**tokens)
    mask=tokens['attention_mask'].unsqueeze(-1)
    avg=(output.last_hidden_state.float()*mask).sum(1)/mask.sum(1)
    lastindex=tokens['attention_mask'].sum(1)-1
    last=output.last_hidden_state[torch.arange(len(lastindex),device='cuda'),lastindex].float()
    embedding=output.pooler_output if pool=='native' else last if pool=='last' else avg
    return head(embedding.float()).flatten(),avg,last,output

def tensor_hash(named):
    h=hashlib.sha256()
    for n,p in named:
        h.update(n.encode());h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

def run(route,reload):
    isq=route=='qwen'
    dest=ROOT/route;dest.mkdir(exist_ok=True)
    cp=dest/'checkpoint';cp.mkdir(exist_ok=True)
    torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.cuda.reset_peak_memory_stats()
    tok,model,head,pool=load_model(route,cp if reload else None)
    def encode(texts):
        return tok(list(texts),padding=True,add_special_tokens=not isq,truncation=False,return_tensors='pt').to('cuda')
    data=batchrows('ESOL')
    train=data.query("partition=='train'").copy()
    lengths=lambda smiles: [len(x) for x in tok(list(smiles),add_special_tokens=not isq)['input_ids']]
    train['length']=lengths(train.connectivity)
    valid=data.query("partition=='valid'").copy()
    valid['length']=lengths(valid.connectivity)
    valid=valid.sort_values('length',kind='stable').iloc[np.linspace(0,len(valid)-1,8,dtype=int)]
    valid_tokens=encode(valid.connectivity)
    mean=float(train.y.mean()); scale=float(train.y.std(ddof=0))
    if reload:
        cfg=json.loads((cp/'config.json').read_text());assert (mean,scale)==(cfg['mean'],cfg['scale'])
        model.eval();head.eval()
        with torch.inference_mode(): pred=predict(model,head,valid_tokens,pool)[0].cpu().numpy()*cfg['scale']+cfg['mean']
        ref=pd.read_csv(dest/'validation.csv')
        np.testing.assert_array_equal(ref.sample_id,valid.sample_id)
        error=float(np.max(abs(ref.prediction-pred)))
        pd.DataFrame(dict(sample_id=valid.sample_id, saved=ref.prediction.to_numpy(),
                          reloaded=pred.astype('float64'))).to_csv(dest/'reload_predictions.csv',index=False)
        write(dest/'reload.json',dict(max_abs_difference_logS=error,tolerance=1e-5,n=len(pred),pass_reload=error<1e-5))
        assert error<1e-5,error
        return
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    frozen=[(n,p) for n,p in model.named_parameters() if not p.requires_grad]
    assert trainable and all(p.device.type=='cuda' for n,p in trainable)
    if isq:assert all('lora_' in n for n,p in trainable) and frozen
    else:assert not frozen
    def snapshot(pairs):return {n:p.detach().cpu().clone() for n,p in pairs}
    before=snapshot(trainable);head_before=snapshot(head.named_parameters())
    fixed_hash=tensor_hash(frozen) if isq else None
    lr=1e-4 if isq else 3e-5
    optim=torch.optim.AdamW([p for n,p in trainable]+list(head.parameters()),lr=lr)
    micro=2 if isq else 4
    accum=2
    rng=np.random.default_rng(3101)
    poolrows=train.sort_values('length',ascending=False,kind='stable').iloc[:micro*accum]
    normal=train.drop(poolrows.index).iloc[rng.permutation(len(train)-len(poolrows))[:9*micro*accum]]
    ordered=pd.concat([normal,poolrows])
    assert ordered.index.is_unique and len(ordered)==10*micro*accum
    native_difference=None
    log=[]
    for step in range(10):
        g=ordered.iloc[step*micro*accum:(step+1)*micro*accum]
        batches=[(encode(z.connectivity),torch.tensor((z.y.to_numpy()-mean)/scale,device='cuda',dtype=torch.float32))
                 for z in (g.iloc[:micro],g.iloc[micro:])]
        model.train();head.train();optim.zero_grad(set_to_none=True)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        total=0.
        for tokens,labels in batches:
            pred,avg,last,output=predict(model,head,tokens,pool)
            loss=(pred-labels).square().mean()
            assert torch.isfinite(loss)
            (loss/accum).backward()
            total+=loss.detach().item()/accum
            if not isq: native_difference=float((output.pooler_output-avg).abs().max().detach())
            del pred,avg,last,output,loss
        norm_model=torch.nn.utils.clip_grad_norm_([p for n,p in trainable],1.0,error_if_nonfinite=True)
        norm_head=torch.nn.utils.clip_grad_norm_(head.parameters(),1.0,error_if_nonfinite=True)
        assert norm_model>0 and norm_head>0
        optim.step();torch.cuda.synchronize()
        log.append(dict(step=step+1,seconds=time.perf_counter()-start,loss=total,max_tokens=int(g.length.max()),
                        encoder_or_adapter_grad=float(norm_model),head_grad=float(norm_head),
                        allocated_MiB=torch.cuda.max_memory_allocated()/2**20,reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                        sample_ids='|'.join(g.sample_id)))
    pd.DataFrame(log).to_csv(dest/'steps.csv',index=False)
    def changes(params,initial):
        deltas={n:float((p.detach().cpu()-initial[n]).abs().max()) for n,p in params}
        return dict(tensors=len(deltas),changed=sum(v>0 for v in deltas.values()),max_abs_change=max(deltas.values()),
                    max_change_name=max(deltas,key=deltas.get))
    body=changes(trainable,before);hd=changes(head.named_parameters(),head_before)
    assert body['changed'] and hd['changed']
    after_hash=tensor_hash(frozen) if isq else None
    assert fixed_hash==after_hash
    model.eval();head.eval()
    with torch.inference_mode():
        pred,avg,last,output=predict(model,head,valid_tokens,pool)
        val=pred.cpu().numpy()*scale+mean
        # Same shortest molecule, once alone and once padded beside the longest.
        pair=encode(valid.connectivity.iloc[[0,-1]])
        single=encode(valid.connectivity.iloc[:1])
        pad_summary={}
        for pm in (['masked_mean','last'] if isq else ['native','masked_mean']):
            v=predict(model,head,pair,pm)[0][0];u=predict(model,head,single,pm)[0][0]
            pad_summary[pm]=float((u-v).abs()*scale)
    pd.DataFrame(dict(sample_id=valid.sample_id,y=valid.y,prediction=val)).to_csv(dest/'validation.csv',index=False)
    tok.save_pretrained(cp/'tokenizer')
    if isq:model.save_pretrained(cp/'adapter',save_embedding_layers=False)
    else:torch.save(model.state_dict(),cp/'encoder.pt')
    torch.save(head.state_dict(),cp/'head.pt')
    cfg=dict(route=route,mean=mean,scale=scale,pooling=pool,base=str((P/'models'/('Qwen3-1.7B' if isq else 'molformer')).resolve()),
        lr=lr,train_seed=0,split_seed=3101,micro_batch=micro,gradient_accumulation=2,gradient_checkpointing=False)
    write(cp/'config.json',cfg)
    result=dict(**cfg,train_records_seen=len(ordered),optimizer_steps=10,validation_n=len(valid),
                trainable_body_parameters=sum(p.numel() for n,p in trainable),head_parameters=sum(p.numel() for p in head.parameters()),
                body_updates=body,head_updates=hd,frozen_before_sha256=fixed_hash,frozen_after_sha256=after_hash,
                torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0),
                device_MiB=torch.cuda.get_device_properties(0).total_memory/2**20,
                median_step_seconds=statistics.median(r['seconds'] for r in log),
                peak_allocated_MiB=max(r['allocated_MiB'] for r in log),peak_reserved_MiB=max(r['reserved_MiB'] for r in log),
                native_vs_masked_mean_max=native_difference,padded_vs_single_logS=pad_summary)
    write(dest/'summary.json',result)
    print(route,body,result['peak_allocated_MiB'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('route',choices=['molformer','qwen'])
    p.add_argument('--reload',action='store_true')
    a=p.parse_args();run(a.route,a.reload)
