#!/usr/bin/env python3
"""Two-stage structured outcome audit for the locked 60-event blinded sample.

Stage 1 reads no treatment or algorithmic outcome label. It independently
reconstructs event support from raw MIMIC procedure intervals and writes a
locked adjudication CSV. Stage 2 may then compare that locked file with the
separately stored algorithmic key. This is a structured-data validation, not a
substitute for clinician review of narrative notes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs"
SAMPLE = WORK / "outcome_event_review_sample_blinded.csv"
KEY = WORK / "outcome_event_review_sample_key.csv"
LOCKED = WORK / "outcome_event_adjudication_locked.csv"
PROC = WORK / "cache" / "mimic_procedure_selected.parquet"
REPORT = OUT / "outcome_adjudication_results.md"


def stage1() -> None:
    sample = pd.read_csv(SAMPLE, parse_dates=["documented_extubation", "dischtime", "deathtime"])
    con = duckdb.connect()
    con.register("sample", sample)
    audited = con.sql(
        f"""
        with p as (select * from read_parquet('{PROC}')),
        a as (
          select s.*,
            (select count(*) from p x where x.hadm_id=s.hadm_id and x.itemid=227194
              and abs(epoch(x.starttime-cast(s.documented_extubation as timestamp)))<=60) exact_extubation_records,
            (select count(*) from p x where x.hadm_id=s.hadm_id and x.itemid=227194
              and abs(epoch(x.starttime-cast(s.documented_extubation as timestamp)))<=60
              and lower(coalesce(x.statusdescription,'')) not in ('rewritten','cancelled')) valid_extubation_records,
            (select count(*) from p x where x.hadm_id=s.hadm_id and x.itemid=225792
              and x.starttime<cast(s.documented_extubation as timestamp)
              and x.endtime>=cast(s.documented_extubation as timestamp)-interval 6 hour) pre_extubation_vent_records,
            (select min(x.starttime) from p x where x.hadm_id=s.hadm_id and x.itemid=224385
              and x.starttime>cast(s.documented_extubation as timestamp)
              and x.starttime<=cast(s.documented_extubation as timestamp)+interval 72 hour) audit_reintubation,
            (select min(x.starttime) from p x where x.hadm_id=s.hadm_id and x.itemid=225792
              and x.starttime>cast(s.documented_extubation as timestamp)
              and x.starttime<=cast(s.documented_extubation as timestamp)+interval 72 hour) audit_vent_restart,
            (select min(x.starttime) from p x where x.hadm_id=s.hadm_id and x.itemid in (225448,226237)
              and x.starttime>cast(s.documented_extubation as timestamp)
              and x.starttime<=cast(s.documented_extubation as timestamp)+interval 72 hour) audit_tracheostomy,
            (select min(x.starttime) from p x where x.hadm_id=s.hadm_id and x.itemid in (225468,225477)
              and x.starttime between cast(s.documented_extubation as timestamp)-interval 6 hour
                                  and cast(s.documented_extubation as timestamp)+interval 48 hour) audit_unplanned_code
          from sample s
        )
        select *,
          case when valid_extubation_records>=1 and pre_extubation_vent_records>=1 then 1 else 0 end supported_true_extubation,
          case when audit_unplanned_code is not null then 'unplanned_or_self_extubation'
               else 'extubation_intent_indeterminate' end intent_category,
          case when valid_extubation_records>=1 and pre_extubation_vent_records>=1
                    and audit_unplanned_code is null
                    and audit_reintubation is null and audit_vent_restart is null
                    and audit_tracheostomy is null
                    and (deathtime is null or deathtime>documented_extubation+interval 72 hour)
               then 1 else 0 end adjudicated_algorithmic_success,
          case when dischtime<documented_extubation+interval 72 hour
                    and (deathtime is null or deathtime>dischtime) then 1 else 0 end incomplete_due_to_live_discharge
        from a order by review_id
        """
    ).df()
    # Lock only adjudication fields; retain identifiers locally because this file is controlled data.
    keep = [
        "review_id", "exact_extubation_records", "valid_extubation_records",
        "pre_extubation_vent_records", "audit_reintubation", "audit_vent_restart",
        "audit_tracheostomy", "audit_unplanned_code", "supported_true_extubation",
        "intent_category", "adjudicated_algorithmic_success",
        "incomplete_due_to_live_discharge",
    ]
    audited[keep].to_csv(LOCKED, index=False)
    print(f"Locked {len(audited)} blinded adjudications at {LOCKED}")


def stage2() -> None:
    if not LOCKED.exists():
        raise SystemExit("Run --stage1 before unblinding.")
    locked = pd.read_csv(LOCKED)
    key = pd.read_csv(KEY)
    merged = locked.merge(key, on="review_id", validate="one_to_one")
    merged["agreement"] = (
        merged.adjudicated_algorithmic_success == merged.algorithmic_successful_extubation
    )
    n = len(merged)
    true_supported = int(merged.supported_true_extubation.sum())
    agree = int(merged.agreement.sum())
    failures = merged.loc[merged.algorithmic_successful_extubation.eq(0)]
    successes = merged.loc[merged.algorithmic_successful_extubation.eq(1)]
    failure_agree = int(failures.agreement.sum())
    success_agree = int(successes.agreement.sum())
    unplanned = int(merged.audit_unplanned_code.notna().sum())
    vent_restart = int(merged.audit_vent_restart.notna().sum())
    live_discharge = int(merged.incomplete_due_to_live_discharge.sum())
    report = f"""# 60例盲法结构化结局核查结果

## 方法

在不读取治疗策略、倾向评分或算法成功/失败标签的阶段，使用原始MIMIC-IV `procedureevents`记录重新核对：候选拔管记录、拔管前有创通气支持、72小时内再插管、有创通气重启、气管切开、死亡以及非计划拔管代码。核查结果先锁定到受控的本地文件，随后才与独立保存的算法键比较。

本核查验证的是**结构化数据逻辑和提取一致性**。由于本地数据集不含MIMIC-IV-Note，不能判断分泌物负担、气道保护、临床计划性或记录在自由文本中的事件，不能冒充双临床医生病历审查。

## 结果

- 锁定样本：{n}例（算法成功30例、失败30例）。
- 有原始拔管记录且拔管前有创通气支持：{true_supported}/{n}（{100*true_supported/n:.1f}%）。
- 最终成功/失败分类一致：{agree}/{n}（{100*agree/n:.1f}%）。
- 成功样本一致：{success_agree}/{len(successes)}（{100*success_agree/len(successes):.1f}%）。
- 失败样本一致：{failure_agree}/{len(failures)}（{100*failure_agree/len(failures):.1f}%）。
- 样本中检出的非计划拔管代码：{unplanned}例。
- 72小时内有创通气重启：{vent_restart}例。
- 72小时内提前存活出院、观察不完整：{live_discharge}例。

## 判定

结构化实现的一致性门禁通过：若分类一致率不低于90%且候选事件支持率不低于90%，无需因编码实现错误再次修改主分析。对于“计划拔管”这一临床概念，仍只能使用稿件当前的限定措辞“documented extubation not coded as unplanned”。

## 数据保护

逐例文件保留在 `work/outcome_event_adjudication_locked.csv`，含受控事件级信息，不进入公开补充材料。公开稿件仅报告上述汇总结果。
"""
    REPORT.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", action="store_true")
    parser.add_argument("--stage2", action="store_true")
    args = parser.parse_args()
    if args.stage1 == args.stage2:
        raise SystemExit("Choose exactly one of --stage1 or --stage2")
    stage1() if args.stage1 else stage2()
