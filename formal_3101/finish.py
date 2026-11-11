"""Final team handoff from all 48 checked validation runs. Uses validation fields only."""
import json
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import pandas as pd
from protocol import ROOT, SETS, SPLITS, CANDIDATES
from report import report


def finish():
    report()
    jobs=pd.read_csv(ROOT/"候选验证与耗时.csv")
    picks=pd.read_csv(ROOT/"各划分选模.csv")
    assert len(jobs)==48 and not picks[["M","Q","M_rmse","Q_rmse"]].isna().any().any()
    ordered=list(CANDIDATES)
    assert all(set(parts.candidate)==set(ordered) for _,parts in jobs.groupby(["dataset","split"]))
    q=json.loads((ROOT/"queue.json").read_text(encoding="utf-8"))
    elapsed=(q.get("ended",time.time())-q["started"])/3600
    failed=jobs[jobs.status!="ok"]
    lines=["# 3101 小组讨论稿","",f"六份划分的 48 个候选均已有训练记录，其中正常完成 {48-len(failed)} 个。只在正常完成的候选中按验证 RMSE 选模，完全平局保留候选表靠前的一项。","",
           "| 数据集 | 划分 | MoLFormer 最优候选 / RMSE | Qwen 最优候选 / RMSE | 本划分验证 RMSE 最低的路线 |",
           "|---|---|---|---|---|"]
    for r in picks.to_dict("records"):
        scores={"RF":r["RF"],"D-MPNN":r["D-MPNN"],"MoLFormer":r["M_rmse"],"Qwen":r["Q_rmse"]}
        rank=sorted(scores,key=scores.get)
        lines.append(f"| {r['dataset']} | {r['split']} | {r['M']} / {r['M_rmse']:.4f} | {r['Q']} / {r['Q_rmse']:.4f} | {rank[0]} / {scores[rank[0]]:.4f} |")
    lines+=["","M1 为 MoLFormer 学习率 3e-5，M2 为 1e-4。Qwen 本批入选的 Q1、Q3、Q4、Q5 都用有效 token 均值和学习率 1e-4，rank/alpha 分别为 8/16、16/16、8/32、16/32。全部候选的实际设置见明细。",
            "","MoLFormer 在两份 ESOL 验证集上的误差最低，D-MPNN 在两份 Lipophilicity 验证集上最低。FreeSolv 从 Group Random 换到 Scaffold 时，最低误差路线从 MoLFormer 换成 D-MPNN。"]
    lines+=["","## 同一性质内两种划分的变化","","下列差值为 Scaffold 验证 RMSE 减去 Group Random 验证 RMSE；正值表示骨架留出时误差更高。验证分子可能不同，每项性质单独比较。",""]
    for ds in SETS:
        by=picks[picks.dataset.eq(ds)].set_index("split")
        nums=[float(by.loc["random_scaffold",k]-by.loc["random_group",k]) for k in ["RF","D-MPNN","M_rmse","Q_rmse"]]
        lines.append(f"- {ds}：RF {nums[0]:+.4f}，D-MPNN {nums[1]:+.4f}，MoLFormer {nums[2]:+.4f}，Qwen {nums[3]:+.4f}。")
    lines+=["","## 配置选择与耗时",""]
    for row in failed.itertuples(index=False):
        folder=ROOT/"runs"/row.dataset/row.split/row.candidate
        diagnostics=json.loads((folder/row.attempt/"zero_gradient.json").read_text(encoding="utf-8"))
        lines.append(f"- {row.dataset} {row.split} {row.candidate} 触发零梯度检查，原参数复跑在第 {diagnostics['epoch']} 轮第 {diagnostics['update']} 次更新前中止。主干和头的梯度范数分别为 {diagnostics['body_norm']:.4g}、{diagnostics['head_norm']:.4g}。失败前最佳验证 RMSE 为 {row.valid_rmse:.4f}（第 {row.best_epoch} 轮），仅作诊断，该候选不参加选模。")
    for col,k in [("M","MoLFormer"),("Q","Qwen")]:
        lines.append(f"- {k} 六份划分的入选频数为 "+ "、".join(f"{c} {n} 次" for c,n in picks[col].value_counts().items())
                     +"。各性质内两种划分的前两名差值列在选模表，单位与该性质的目标一致。")
    cap=int((jobs.best_epoch==20).sum())
    lines+=["","Lipophilicity Scaffold 只有 M1 正常完成；其余五份划分的 M1、M2 均有完整候选记录。"]
    lines+=["",f"按每个候选最后一次尝试汇总，共记录 {int(jobs.epochs.sum())} 个完整训练轮次。最佳轮次恰在第 20 轮的候选有 {cap} 个，按预先设定的上限保留。",
            f"首次启动到末项核验约 {elapsed:.2f} 小时，含中断与重试；各候选最后一次尝试的进程内用时合计 {jobs.wall_s.sum()/3600:.2f} 小时，其中已完整记录的训练 {jobs.train_s.sum()/3600:.2f} 小时、逐轮验证 {jobs.valid_s.sum()/3600:.2f} 小时。",
            "", "所有候选均从最佳权重重载复算验证预测，最大逐分子重载误差为 "+f"{jobs.reload_max_abs.max():.3g}（各自原单位）。",
            "", f"若后两批采用同一预算、运行条件与本批相近，可先按约 {2*elapsed:.0f} 小时连续机时讨论排期；早停轮次和重试会影响实际时长。请组里确认是否沿用原候选和失败处理规则，再启动 3102、3103。",
            "", "这批记录只覆盖划分种子 3101，且验证集用于选择候选。测试表现及跨划分种子的稳定性仍待后续独立评估。",
            "", "[逐候选成绩及用时](候选验证与耗时.csv) · [分划分选模及差距](各划分选模.csv)"]
    dest=ROOT/"给队友的3101结论.md"
    dest.write_text("\n".join(lines)+"\n",encoding="utf-8")
    package=ROOT/"3101-验证结果与用时.zip"
    staged=package.with_suffix(".tmp.zip")
    with ZipFile(staged,"w",ZIP_DEFLATED) as archive:
        for path in [dest, ROOT/"给队友的3101验证报告.md", ROOT/"候选验证与耗时.csv",
                     ROOT/"各划分选模.csv",ROOT/"protocol.json"]:
            archive.write(path,path.name)
        for row in jobs.itertuples(index=False):
            folder=ROOT/"runs"/row.dataset/row.split/row.candidate/row.attempt
            for filename in ["config.json","history.json","validation.csv"]:
                archive.write(folder/filename, f"{row.dataset}/{row.split}/{row.candidate}/{filename}")
            for issue in sorted(folder.parent.glob("attempt_*/zero_gradient.json")):
                archive.write(issue, f"{row.dataset}/{row.split}/{row.candidate}/{issue.parent.name}/zero_gradient.json")
    with ZipFile(staged) as archive:
        assert len(archive.namelist())>=5+48*3 and archive.testzip() is None
    staged.replace(package)


if __name__=="__main__":
    finish()
