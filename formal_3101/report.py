"""Summarize completed 3101 validation results. Only the complete family gets a winner."""
import json
import math
import os
from pathlib import Path
import pandas as pd
import numpy as np
from protocol import ROOT, PROJECT, SEED, SETS, SPLITS, CANDIDATES


def report():
    rows=[]
    baselines=pd.read_csv(PROJECT/"revised_gpu_pilot/audit/runs.csv",
                usecols=["dataset","split","split_seed","model","train_seed","status","valid_rmse"])
    baselines=baselines.query("split_seed==@SEED and train_seed==0 and status=='reusable'")
    assert len(baselines)==12
    leaders=[]
    for ds in SETS:
        for split in SPLITS:
            expected=pd.read_csv(ROOT/"inputs"/f"{ds}_{split}.csv").query("partition=='valid'")
            result={}
            for candidate in CANDIDATES:
                folder=ROOT/"runs"/ds/split/candidate
                marker=folder/"done.json" if (folder/"done.json").is_file() else folder/"failed.json"
                if not marker.is_file():
                    continue
                row=json.loads(marker.read_text(encoding="utf-8"))
                row.setdefault("status","ok")
                assert (row["dataset"],row["split"],row["split_seed"],row["candidate"],row["train_seed"])==(ds,split,SEED,candidate,0)
                run=folder/row["attempt"]
                if row["epochs"]==0:
                    assert row["status"]=="failed_zero_gradient" and (run/"zero_gradient.json").is_file()
                    rows.append(row)
                    result[candidate]=row
                    continue
                pred=pd.read_csv(run/"validation.csv")
                assert list(pred.sample_id)==list(expected.sample_id)
                np.testing.assert_allclose(pred.y_true,expected.y_true,rtol=0,atol=1e-12)
                diff=pred.prediction.to_numpy()-expected.y_true.to_numpy()
                assert abs(np.sqrt(np.mean(diff**2))-row["valid_rmse"])<1e-8
                hist=json.loads((run/"history.json").read_text(encoding="utf-8"))
                assert 1+np.argmin([h["valid_rmse"] for h in hist])==row["best_epoch"]
                assert 1<=len(hist)<=20 and row["reload_max_abs"]<1e-5
                rows.append(row)
                result[candidate]=row
            tab=baselines[(baselines.dataset==ds)&(baselines.split==split)].set_index("model")
            rec={"dataset":ds,"split":split,"RF":tab.loc["rf","valid_rmse"],"D-MPNN":tab.loc["dmpnn","valid_rmse"]}
            for prefix in ("M","Q"):
                ids=[c for c in CANDIDATES if c.startswith(prefix)]
                if all(c in result for c in ids):
                    order=sorted((c for c in ids if result[c]["status"]=="ok"),key=lambda c:result[c]["valid_rmse"])
                    rec[prefix+"_eligible"]=len(order)
                    if not order:
                        continue
                    win=result[order[0]]
                    rec.update({prefix:order[0], prefix+"_rmse":win["valid_rmse"],
                                prefix+"_epoch":win["best_epoch"],prefix+"_margin":result[order[1]]["valid_rmse"]-win["valid_rmse"] if len(order)>1 else None,
                                prefix+"_eligible":len(order)})
            leaders.append(rec)
    if rows:
        pd.DataFrame(rows).to_csv(ROOT/"候选验证与耗时.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(leaders).to_csv(ROOT/"各划分选模.csv",index=False,encoding="utf-8-sig")
    def cell(r,k):
        if k not in r:return "无合格候选" if r.get(k+"_eligible")==0 else "待完成"
        return f"{r[k]} / {r[k+'_rmse']:.4f} / 第 {r[k+'_epoch']} 轮"
    lines=[f"# {SEED} 验证结果与本机耗时","",
           f"已有 {len(rows)}/48 个候选的最终记录，其中正常完成 {sum(r['status']=='ok' for r in rows)} 个。每个模型在该划分的全部候选都有结果后才从正常完成的候选中选模。现阶段只评估训练区和验证区。","","| 数据集 | 划分 | RF 验证 RMSE | D-MPNN 验证 RMSE | MoLFormer 候选 / 验证 RMSE / 轮次 | Qwen 候选 / 验证 RMSE / 轮次 |",
           "|---|---|---:|---:|---|---|"]
    for r in leaders:
        lines.append(f"| {r['dataset']} | {r['split']} | {r['RF']:.4f} | {r['D-MPNN']:.4f} | {cell(r,'M')} | {cell(r,'Q')} |")
    if rows:
        total=sum(r["wall_s"] for r in rows)
        lines+=["",f"已完成候选的进程内实测累计 {total/3600:.2f} 小时，含模型加载、训练、验证、存盘和最佳权重复算。候选明细另列训练和逐轮验证秒数。队列启动、候选进程导入和中断重试的时间另计。","","| 范围 | 已完成 | 实际训练轮数合计 | 候选进程内累计 |","|---|---:|---:|---:|"]
        for ds in SETS:
            for route in ["molformer","qwen"]:
                group=[r for r in rows if r["dataset"]==ds and r["route"]==route]
                if group:lines.append(f"| {ds} · {route} | {len(group)} | {sum(r['epochs'] for r in group)} | {sum(r['wall_s'] for r in group)/60:.1f} 分钟 |")
    for r in rows:
        folder=ROOT/"runs"/r["dataset"]/r["split"]/r["candidate"]
        failures=list(folder.glob("attempt_*/zero_gradient.json"))
        if failures:
            note="相同协议重试后仍触发零梯度，失败前验证值仅供诊断，本候选不参加选模。" if r["status"]!="ok" else "同种子、同参数重试正常完成，按重试记录选模；首轮失败日志保留。"
            lines+=["",f"{r['dataset']} {r['split']} {r['candidate']} 有 {len(failures)} 次零梯度中止。{note}"]
    if len(rows)==48:
        lines+=["","六份划分的验证名次（按原单位 RMSE 由低到高）：",""]
        for r in leaders:
            scores={"RF":r["RF"],"D-MPNN":r["D-MPNN"]}
            for col,name in [("M_rmse","MoLFormer"),("Q_rmse","Qwen")]:
                if col in r:
                    scores[name]=r[col]
            lines.append(f"- {r['dataset']} {r['split']}：{'、'.join(sorted(scores,key=scores.get))}。")
        lines+=["","同分按预设候选顺序选取，前两名验证 RMSE 差见 M_margin 和 Q_margin。当前表使用选参的验证集；测试集留待所有配置锁定后单独评估。"]
    dest=ROOT/f"给队友的{SEED}验证报告.md"
    tmp=dest.with_suffix(".tmp")
    tmp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    os.replace(tmp,dest)


if __name__=="__main__":
    report()
