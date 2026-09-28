#!/usr/bin/env python3
"""Build and analyse the MIMIC-IV sequential landmark cohort.

The estimand in this implementation is the adjusted association, in the overlap population,
of completing a propofol-to-dexmedetomidine transition during a 6-hour strategy
assessment window versus continuing propofol throughout that window. Follow-up
starts at the end of the window. This aligns completed strategy classification
with the start of outcome follow-up but remains vulnerable to residual clinical-
readiness and treatment-intent confounding.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
CACHE = WORK / "cache"
OUT = ROOT / "outputs" / "primary_analysis"


def data_root_from_env(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(
            f"Set {variable} to the extracted MIMIC-IV 3.1 directory before running."
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"{variable} is not a directory: {root}")
    return root


MIMIC = data_root_from_env("MIMIC_IV_ROOT")

CHART_ITEMS = {
    220045: "heart_rate",
    220052: "map",
    220181: "map",
    220277: "spo2",
    220339: "peep",
    223835: "fio2",
    224688: "rr_set",
    224689: "rr_spontaneous",
    224690: "rr_total",
    224686: "tidal_volume_spontaneous",
    224701: "psv_level",
    223849: "ventilator_mode",
    229314: "ventilator_mode_hamilton",
    224715: "sbt_started",
    224716: "sbt_stopped",
    224717: "sbt_successful",
    228096: "rass",
}
LAB_ITEMS = {
    50813: "lactate",
    50912: "creatinine",
    51265: "platelets",
    50885: "bilirubin",
    51300: "wbc",
    51301: "wbc",
    51222: "hemoglobin",
}
PROP = 222168
DEX = (225150, 229420)
VASO = (221289, 221749, 221906, 222315, 229617, 229630, 229631, 229632)
MIDAZOLAM = (221668,)
OPIOID = (221744, 221833, 225154, 225942, 225972)
KETAMINE = (221712, 227211)
INPUT_ITEMS = tuple(sorted({PROP, *DEX, *VASO, *MIDAZOLAM, *OPIOID, *KETAMINE}))
PROC_ITEMS = (225792, 224385, 227194, 225468, 225477, 225448, 226237)
PYTHON_RANDOM_SEED = 20260924


def qpath(path: Path) -> str:
    return str(path).replace("'", "''")


def sql_list(values: Iterable[int]) -> str:
    return ",".join(str(int(x)) for x in values)


def copy_if_missing(con: duckdb.DuckDBPyConnection, target: Path, query: str) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building {target.name} ...", flush=True)
    con.execute(
        f"copy ({query}) to '{qpath(target)}' (format parquet, compression zstd)"
    )
    print(f"Built {target.name}", flush=True)


def build_source_caches(con: duckdb.DuckDBPyConnection) -> dict[str, Path]:
    files = {
        "input": CACHE / "mimic_input_selected.parquet",
        "proc": CACHE / "mimic_procedure_selected.parquet",
        "chart": CACHE / "mimic_chart_readiness_v2.parquet",
        "lab": CACHE / "mimic_lab_selected.parquet",
        "static": CACHE / "mimic_static_v3_discharge.parquet",
    }
    copy_if_missing(
        con,
        files["input"],
        f"""
        select cast(subject_id as bigint) subject_id,cast(hadm_id as bigint) hadm_id,
          cast(stay_id as bigint) stay_id,cast(starttime as timestamp) starttime,
          cast(endtime as timestamp) endtime,cast(itemid as bigint) itemid,
          try_cast(rate as double) rate,rateuom,try_cast(patientweight as double) patientweight,
          statusdescription,ordercategorydescription
        from read_csv_auto('{qpath(MIMIC / 'icu/inputevents.csv.gz')}',all_varchar=true,strict_mode=false)
        where cast(itemid as bigint) in ({sql_list(INPUT_ITEMS)})
        """,
    )
    copy_if_missing(
        con,
        files["proc"],
        f"""
        select cast(subject_id as bigint) subject_id,cast(hadm_id as bigint) hadm_id,
          cast(stay_id as bigint) stay_id,cast(starttime as timestamp) starttime,
          cast(endtime as timestamp) endtime,cast(itemid as bigint) itemid,
          statusdescription
        from read_csv_auto('{qpath(MIMIC / 'icu/procedureevents.csv.gz')}',all_varchar=true,strict_mode=false)
        where cast(itemid as bigint) in ({sql_list(PROC_ITEMS)})
        """,
    )
    copy_if_missing(
        con,
        files["chart"],
        f"""
        select cast(subject_id as bigint) subject_id,cast(hadm_id as bigint) hadm_id,
          cast(stay_id as bigint) stay_id,cast(charttime as timestamp) charttime,
          cast(itemid as bigint) itemid,try_cast(valuenum as double) valuenum,value
        from read_csv_auto('{qpath(MIMIC / 'icu/chartevents.csv.gz')}',all_varchar=true,strict_mode=false)
        where cast(itemid as bigint) in ({sql_list(CHART_ITEMS)})
          and coalesce(value,'') <> 'Error'
        """,
    )
    copy_if_missing(
        con,
        files["lab"],
        f"""
        select cast(subject_id as bigint) subject_id,cast(hadm_id as bigint) hadm_id,
          cast(charttime as timestamp) charttime,cast(itemid as bigint) itemid,
          try_cast(valuenum as double) valuenum
        from read_csv_auto('{qpath(MIMIC / 'hosp/labevents.csv.gz')}',all_varchar=true,strict_mode=false,ignore_errors=true)
        where cast(itemid as bigint) in ({sql_list(LAB_ITEMS)})
          and try_cast(valuenum as double) is not null
        """,
    )
    copy_if_missing(
        con,
        files["static"],
        f"""
        with dx as (
          select cast(hadm_id as bigint) hadm_id,
            max(case when (icd_version='10' and (icd_code like 'A40%' or icd_code like 'A41%' or icd_code like 'R652%'))
                          or (icd_version='9' and (icd_code like '038%' or icd_code in ('99591','99592','78552'))) then 1 else 0 end) dx_sepsis,
            max(case when (icd_version='10' and (icd_code like 'I6%' or icd_code like 'G%' or icd_code like 'S06%'))
                          or (icd_version='9' and (icd_code between '430' and '43899' or icd_code like '85%')) then 1 else 0 end) dx_neuro,
            max(case when (icd_version='10' and (icd_code like 'J1%' or icd_code like 'J4%' or icd_code like 'J8%' or icd_code like 'J96%'))
                          or (icd_version='9' and (icd_code between '480' and '49699' or icd_code like '518%')) then 1 else 0 end) dx_respiratory,
            max(case when (icd_version='10' and icd_code between 'I20' and 'I5099')
                          or (icd_version='9' and icd_code between '410' and '42999') then 1 else 0 end) dx_cardiac,
            max(case when (icd_version='10' and (icd_code like 'N17%' or icd_code like 'N18%' or icd_code like 'N19%'))
                          or (icd_version='9' and icd_code between '584' and '58699') then 1 else 0 end) dx_renal,
            max(case when (icd_version='10' and icd_code between 'K70' and 'K7799')
                          or (icd_version='9' and icd_code between '570' and '57399') then 1 else 0 end) dx_liver,
            max(case when (icd_version='10' and icd_code like 'C%')
                          or (icd_version='9' and icd_code between '140' and '20999') then 1 else 0 end) dx_malignancy
          from read_csv_auto('{qpath(MIMIC / 'hosp/diagnoses_icd.csv.gz')}',all_varchar=true,strict_mode=false)
          group by 1
        )
        select cast(i.subject_id as bigint) subject_id,cast(i.hadm_id as bigint) hadm_id,
          cast(i.stay_id as bigint) stay_id,cast(i.intime as timestamp) intime,
          cast(i.outtime as timestamp) outtime,i.first_careunit,i.last_careunit,
          p.gender,cast(p.anchor_age as double)+greatest(0,year(cast(a.admittime as timestamp))-cast(p.anchor_year as integer)) age,
          p.anchor_year_group,a.admission_type,a.admission_location,a.race,
          try_cast(a.deathtime as timestamp) deathtime,try_cast(a.dischtime as timestamp) dischtime,
          a.discharge_location,cast(a.hospital_expire_flag as integer) hospital_expire_flag,
          coalesce(dx.dx_sepsis,0) dx_sepsis,coalesce(dx.dx_neuro,0) dx_neuro,
          coalesce(dx.dx_respiratory,0) dx_respiratory,coalesce(dx.dx_cardiac,0) dx_cardiac,
          coalesce(dx.dx_renal,0) dx_renal,coalesce(dx.dx_liver,0) dx_liver,
          coalesce(dx.dx_malignancy,0) dx_malignancy
        from read_csv_auto('{qpath(MIMIC / 'icu/icustays.csv.gz')}',all_varchar=true,strict_mode=false) i
        join read_csv_auto('{qpath(MIMIC / 'hosp/patients.csv.gz')}',all_varchar=true,strict_mode=false) p
          on cast(p.subject_id as bigint)=cast(i.subject_id as bigint)
        join read_csv_auto('{qpath(MIMIC / 'hosp/admissions.csv.gz')}',all_varchar=true,strict_mode=false) a
          on cast(a.hadm_id as bigint)=cast(i.hadm_id as bigint)
        left join dx on dx.hadm_id=cast(i.hadm_id as bigint)
        """,
    )
    return files


def build_trials(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> Path:
    target = CACHE / "mimic_landmark_trials_v11_strategy_outcome_qc.parquet"
    if target.exists():
        return target
    print("Building sequential landmark trial table ...", flush=True)
    con.execute(
        f"""
        create or replace temp view inp as select * from read_parquet('{qpath(files['input'])}');
        create or replace temp view proc as select * from read_parquet('{qpath(files['proc'])}');
        create or replace temp view chart as select * from read_parquet('{qpath(files['chart'])}');
        create or replace temp view lab as select * from read_parquet('{qpath(files['lab'])}');
        create or replace temp view stat as select * from read_parquet('{qpath(files['static'])}');

        create or replace temp view sed_raw as
        select stay_id,starttime,endtime,rate,rateuom,patientweight,
          case when itemid={PROP} then 'propofol' else 'dex' end agent
        from inp where itemid in ({PROP},{sql_list(DEX)})
          and coalesce(statusdescription,'')<>'Rewritten' and endtime>starttime;

        create or replace temp view sed_flag as
        select *,case when starttime>coalesce(max(endtime) over (
          partition by stay_id,agent order by starttime,endtime
          rows between unbounded preceding and 1 preceding
        ),timestamp '1900-01-01')+interval 2 hour then 1 else 0 end new_episode
        from sed_raw;

        create or replace temp view sed_episode as
        select stay_id,agent,min(starttime) episode_start,max(endtime) episode_end,
          median(rate) filter(where rate is not null) median_rate,
          median(patientweight) filter(where patientweight between 20 and 400) patientweight
        from (select *,sum(new_episode) over(partition by stay_id,agent order by starttime,endtime) episode_id
              from sed_flag) x
        group by stay_id,agent,episode_id;

        create or replace temp view eligible_precoverage as
        select row_number() over() trial_id,p.stay_id,s.subject_id,s.hadm_id,
          p.episode_start prop_start,p.episode_end prop_end,
          (select median(x.rate) from inp x where x.stay_id=p.stay_id and x.itemid={PROP}
             and x.starttime<p.episode_start+h*interval 1 hour
             and x.endtime>p.episode_start+(h-6)*interval 1 hour
             and coalesce(x.statusdescription,'')<>'Rewritten'
             and x.rateuom='mcg/kg/min' and x.rate>0 and x.rate<=200) prop_rate,
          (select median(x.patientweight) from inp x where x.stay_id=p.stay_id and x.itemid={PROP}
             and x.starttime<p.episode_start+h*interval 1 hour
             and x.endtime>p.episode_start+(h-6)*interval 1 hour
             and coalesce(x.statusdescription,'')<>'Rewritten'
             and x.patientweight between 20 and 400) patientweight,
          h landmark_hour,
          p.episode_start+h*interval 1 hour landmark_time,
          p.episode_start+(h+6)*interval 1 hour grace_end,
          s.intime,s.outtime,s.first_careunit,s.gender,s.age,s.anchor_year_group,
          s.admission_type,s.admission_location,s.race,s.deathtime,s.dischtime,s.discharge_location,
          s.dx_sepsis,s.dx_neuro,s.dx_respiratory,s.dx_cardiac,s.dx_renal,
          s.dx_liver,s.dx_malignancy,
          (select min(d.episode_start) from sed_episode d
           where d.stay_id=p.stay_id and d.agent='dex'
             and d.episode_start>=p.episode_start+h*interval 1 hour
             and d.episode_start<p.episode_start+(h+6)*interval 1 hour
             and d.episode_start<=p.episode_end+interval 2 hour
             and p.episode_end<=d.episode_start+interval 6 hour
             and p.episode_end<=p.episode_start+(h+6)*interval 1 hour) dex_start,
          case
            when exists(select 1 from sed_episode d where d.stay_id=p.stay_id and d.agent='dex'
              and d.episode_start>=p.episode_start+h*interval 1 hour
              and d.episode_start<p.episode_start+(h+6)*interval 1 hour
              and d.episode_start<=p.episode_end+interval 2 hour
              and p.episode_end<=d.episode_start+interval 6 hour
              and p.episode_end<=p.episode_start+(h+6)*interval 1 hour) then 1
            when p.episode_end>=p.episode_start+(h+6)*interval 1 hour
              and not exists(select 1 from sed_episode d where d.stay_id=p.stay_id and d.agent='dex'
                             and d.episode_start<p.episode_start+(h+6)*interval 1 hour) then 0
            else null end treatment,
          case when exists(select 1 from inp x where x.stay_id=p.stay_id and x.itemid={PROP}
                     and x.starttime<=p.episode_start+h*interval 1 hour
                     and x.endtime>=p.episode_start+h*interval 1 hour
                     and coalesce(x.statusdescription,'')<>'Rewritten') then 1 else 0 end raw_prop_at_landmark,
          case when exists(select 1 from inp x where x.stay_id=p.stay_id and x.itemid={PROP}
                     and x.starttime<=p.episode_start+(h+6)*interval 1 hour
                     and x.endtime>=p.episode_start+(h+6)*interval 1 hour
                     and coalesce(x.statusdescription,'')<>'Rewritten') then 1 else 0 end raw_prop_at_grace_end
          ,case when exists(select 1 from inp x where x.stay_id=p.stay_id and x.itemid in ({sql_list(DEX)})
                     and x.starttime<=p.episode_start+(h+6)*interval 1 hour
                     and x.endtime>=p.episode_start+(h+6)*interval 1 hour
                     and coalesce(x.statusdescription,'')<>'Rewritten') then 1 else 0 end raw_dex_at_grace_end
          ,case when not exists(
               select 1 from inp nxt
               where nxt.stay_id=p.stay_id and nxt.itemid={PROP}
                 and nxt.starttime<p.episode_start+(h+6)*interval 1 hour
                 and nxt.endtime>p.episode_start+h*interval 1 hour
                 and coalesce(nxt.statusdescription,'')<>'Rewritten'
                 and greatest(nxt.starttime,p.episode_start+h*interval 1 hour) > coalesce((
                   select max(least(prev.endtime,p.episode_start+(h+6)*interval 1 hour))
                   from inp prev where prev.stay_id=p.stay_id and prev.itemid={PROP}
                     and prev.starttime<p.episode_start+(h+6)*interval 1 hour
                     and prev.endtime>p.episode_start+h*interval 1 hour
                     and coalesce(prev.statusdescription,'')<>'Rewritten'
                     and greatest(prev.starttime,p.episode_start+h*interval 1 hour)
                         < greatest(nxt.starttime,p.episode_start+h*interval 1 hour)
                 ),p.episode_start+h*interval 1 hour)
             ) then 1 else 0 end raw_prop_continuous_window
          ,case when exists(select 1 from proc v where v.stay_id=p.stay_id and v.itemid=225792
                     and p.episode_start+(h+6)*interval 1 hour between v.starttime and v.endtime)
                 then 1 else 0 end invasive_vent_at_grace_end
        from sed_episode p join stat s using(stay_id) cross join range(6,49,6) r(h)
        where p.agent='propofol' and p.episode_end>=p.episode_start+h*interval 1 hour
          and s.age>=18 and p.episode_start+(h+6)*interval 1 hour<s.outtime
          and not exists(select 1 from sed_episode d where d.stay_id=p.stay_id and d.agent='dex'
                         and d.episode_start<p.episode_start+h*interval 1 hour)
          and exists(select 1 from proc v where v.stay_id=p.stay_id and v.itemid=225792
                     and p.episode_start+h*interval 1 hour between v.starttime and v.endtime)
          and not exists(select 1 from proc z where z.stay_id=p.stay_id and z.itemid in (225448,226237)
                         and z.starttime<=p.episode_start+h*interval 1 hour)
          and not exists(select 1 from proc e where e.stay_id=p.stay_id and e.itemid in (227194,225468,225477)
                         and e.starttime between p.episode_start+h*interval 1 hour
                                             and p.episode_start+(h+6)*interval 1 hour)
          and (s.deathtime is null or s.deathtime>p.episode_start+(h+6)*interval 1 hour);

        create or replace temp view eligible as
        select * from eligible_precoverage
        where raw_prop_at_landmark=1 and invasive_vent_at_grace_end=1
          and ((treatment=1 and raw_dex_at_grace_end=1)
               or (treatment=0 and raw_prop_at_grace_end=1 and raw_prop_continuous_window=1));

        create or replace temp view assigned as
        select * from eligible where treatment is not null;

        create or replace temp view with_outcome as
        select a.*,
          (select min(e.starttime) from proc e where e.hadm_id=a.hadm_id and e.itemid=227194
             and e.starttime>a.grace_end and e.starttime<=a.grace_end+interval 48 hour) documented_extubation,
          (select arg_min(e.stay_id,e.starttime) from proc e where e.hadm_id=a.hadm_id and e.itemid=227194
             and e.starttime>a.grace_end and e.starttime<=a.grace_end+interval 48 hour) documented_extubation_stay_id,
          (select min(e.starttime) from proc e where e.hadm_id=a.hadm_id and e.itemid in (225468,225477)
             and e.starttime>a.grace_end and e.starttime<=a.grace_end+interval 48 hour) unplanned_extubation,
          (select min(x.starttime) from inp x where x.hadm_id=a.hadm_id and x.itemid in ({sql_list(DEX)})
             and x.starttime>a.grace_end and x.starttime<=a.grace_end+interval 48 hour
             and coalesce(x.statusdescription,'')<>'Rewritten') later_dex_48h
        from assigned a;

        copy (
        select o.*,
          (select min(i.starttime) from proc i where i.hadm_id=o.hadm_id and i.itemid=224385
             and i.starttime>documented_extubation and i.starttime<=documented_extubation+interval 72 hour) reintubation_72h_time,
          (select min(v.starttime) from proc v where v.hadm_id=o.hadm_id and v.itemid=225792
             and v.starttime>documented_extubation and v.starttime<=documented_extubation+interval 72 hour) invasive_vent_restart_72h_time,
          (select min(z.starttime) from proc z where z.hadm_id=o.hadm_id and z.itemid in (225448,226237)
             and z.starttime>documented_extubation and z.starttime<=documented_extubation+interval 72 hour) tracheostomy_72h_time,
          case when documented_extubation is not null
            and unplanned_extubation is null
            and not exists(select 1 from proc i where i.hadm_id=o.hadm_id and i.itemid=224385
                           and i.starttime>documented_extubation and i.starttime<=documented_extubation+interval 72 hour)
            and not exists(select 1 from proc v where v.hadm_id=o.hadm_id and v.itemid=225792
                           and v.starttime>documented_extubation and v.starttime<=documented_extubation+interval 72 hour)
            and not exists(select 1 from proc z where z.hadm_id=o.hadm_id and z.itemid in (225448,226237)
                           and z.starttime>documented_extubation and z.starttime<=documented_extubation+interval 72 hour)
            and (o.deathtime is null or o.deathtime>documented_extubation+interval 72 hour)
            then 1 else 0 end successful_extubation,
          case when documented_extubation is not null
            and unplanned_extubation is null
            and not exists(select 1 from proc i where i.hadm_id=o.hadm_id and i.itemid=224385
                           and i.starttime>documented_extubation and i.starttime<=documented_extubation+interval 72 hour)
            and not exists(select 1 from proc v where v.hadm_id=o.hadm_id and v.itemid=225792
                           and v.starttime>documented_extubation and v.starttime<=documented_extubation+interval 72 hour)
            and not exists(select 1 from proc z where z.hadm_id=o.hadm_id and z.itemid in (225448,226237)
                           and z.starttime>documented_extubation and z.starttime<=documented_extubation+interval 72 hour)
            and (o.deathtime is null or o.deathtime>documented_extubation+interval 72 hour)
            and o.dischtime>=documented_extubation+interval 72 hour
            then 1 else 0 end successful_extubation_full_72h_observation,
          case when documented_extubation is not null and o.dischtime<documented_extubation+interval 72 hour
                    and (o.deathtime is null or o.deathtime>o.dischtime)
            then 1 else 0 end live_discharge_before_72h,
          case when documented_extubation is not null and exists(
              select 1 from proc i where i.hadm_id=o.hadm_id and i.stay_id<>o.documented_extubation_stay_id and i.itemid=224385
              and i.starttime>documented_extubation and i.starttime<=documented_extubation+interval 72 hour
            ) then 1 else 0 end cross_stay_reintubation_72h,
          case when o.deathtime is not null and o.deathtime>o.grace_end
                 and o.deathtime<=o.grace_end+interval 48 hour then 1 else 0 end death_48h,
          case when (select count(distinct c.charttime) from chart c where c.stay_id=o.stay_id and c.itemid=220045
                           and c.valuenum between 20 and 49.999 and c.charttime>o.landmark_time-interval 6 hour
                           and c.charttime<=o.landmark_time)>=2 then 1 else 0 end baseline_bradycardia,
          case when (select count(distinct c.charttime) from chart c where c.hadm_id=o.hadm_id and c.itemid=220045
                           and c.valuenum between 20 and 49.999 and c.charttime>o.grace_end
                           and c.charttime<=o.grace_end+interval 24 hour)>=2 then 1 else 0 end bradycardia_24h,
          (select count(distinct c.charttime) from chart c where c.hadm_id=o.hadm_id and c.itemid=220045
                           and c.valuenum between 20 and 250 and c.charttime>o.grace_end
                           and c.charttime<=o.grace_end+interval 24 hour) hr_measurements_24h,
          case when (select count(distinct c.charttime) from chart c where c.hadm_id=o.hadm_id and c.itemid=220045
                           and c.valuenum between 20 and 39.999 and c.charttime>o.grace_end
                           and c.charttime<=o.grace_end+interval 24 hour)>=2 then 1 else 0 end severe_bradycardia_24h,
          case when (select count(distinct c.charttime) from chart c where c.stay_id=o.stay_id and c.itemid in (220052,220181)
                           and c.valuenum between 20 and 64.999 and c.charttime>o.landmark_time-interval 6 hour
                           and c.charttime<=o.landmark_time)>=2 then 1 else 0 end baseline_hypotension,
          case when (select count(distinct c.charttime) from chart c where c.hadm_id=o.hadm_id and c.itemid in (220052,220181)
                           and c.valuenum between 20 and 64.999 and c.charttime>o.grace_end
                           and c.charttime<=o.grace_end+interval 24 hour)>=2 then 1 else 0 end hypotension_24h,
          (select count(distinct c.charttime) from chart c where c.hadm_id=o.hadm_id and c.itemid in (220052,220181)
                           and c.valuenum between 20 and 200 and c.charttime>o.grace_end
                           and c.charttime<=o.grace_end+interval 24 hour) map_measurements_24h,
          case when (select count(distinct c.charttime) from chart c where c.hadm_id=o.hadm_id and c.itemid in (220052,220181)
                           and c.valuenum between 20 and 54.999 and c.charttime>o.grace_end
                           and c.charttime<=o.grace_end+interval 24 hour)>=2 then 1 else 0 end severe_hypotension_24h,
          case when
            (select count(distinct case
               when x.itemid in (221289,229617) then 'epinephrine'
               when x.itemid in (221749,229630,229631,229632) then 'phenylephrine'
               when x.itemid=221906 then 'norepinephrine'
               when x.itemid=222315 then 'vasopressin' end)
             from inp x where x.hadm_id=o.hadm_id and x.itemid in ({sql_list(VASO)})
               and x.starttime<o.grace_end+interval 24 hour and x.endtime>o.grace_end
               and coalesce(x.statusdescription,'')<>'Rewritten')
            >
            (select count(distinct case
               when x.itemid in (221289,229617) then 'epinephrine'
               when x.itemid in (221749,229630,229631,229632) then 'phenylephrine'
               when x.itemid=221906 then 'norepinephrine'
               when x.itemid=222315 then 'vasopressin' end)
             from inp x where x.stay_id=o.stay_id and x.itemid in ({sql_list(VASO)})
               and x.starttime<o.grace_end and x.endtime>=o.grace_end
               and coalesce(x.statusdescription,'')<>'Rewritten')
            then 1 else 0 end new_or_additional_vasopressor_24h,
          (select median(c.valuenum) from chart c where c.stay_id=o.stay_id and c.itemid in (220052,220181)
             and c.valuenum between 20 and 200 and c.charttime>o.landmark_time-interval 6 hour and c.charttime<=o.landmark_time) map_6h,
          (select median(c.valuenum) from chart c where c.stay_id=o.stay_id and c.itemid=220045
             and c.valuenum between 20 and 250 and c.charttime>o.landmark_time-interval 6 hour and c.charttime<=o.landmark_time) hr_6h,
          (select median(c.valuenum) from chart c where c.stay_id=o.stay_id and c.itemid=220277
             and c.valuenum between 40 and 100 and c.charttime>o.landmark_time-interval 6 hour and c.charttime<=o.landmark_time) spo2_6h,
          (select arg_max(c.valuenum,c.charttime) from chart c where c.stay_id=o.stay_id and c.itemid=228096
             and c.valuenum between -5 and 4 and c.charttime>o.landmark_time-interval 12 hour and c.charttime<=o.landmark_time) rass_last,
          (select arg_min(c.valuenum,c.charttime) from chart c where c.stay_id=o.stay_id and c.itemid=228096
             and c.valuenum between -5 and 4 and c.charttime>o.landmark_time-interval 12 hour and c.charttime<=o.landmark_time) rass_first_12h,
          (select arg_max(c.valuenum,c.charttime) from chart c where c.stay_id=o.stay_id and c.itemid=220339
             and c.valuenum between 0 and 30 and c.charttime>o.landmark_time-interval 6 hour and c.charttime<=o.landmark_time) peep_last,
          (select arg_max(case when c.valuenum<=1 then c.valuenum*100 else c.valuenum end,c.charttime)
             from chart c where c.stay_id=o.stay_id and c.itemid=223835
             and c.valuenum between 0.2 and 100 and c.charttime>o.landmark_time-interval 6 hour and c.charttime<=o.landmark_time) fio2_last,
          (select arg_max(c.valuenum,c.charttime) from chart c where c.stay_id=o.stay_id and c.itemid in (224688,224690)
             and c.valuenum between 0 and 80 and c.charttime>o.landmark_time-interval 6 hour and c.charttime<=o.landmark_time) rr_last,
          (select arg_max(c.valuenum,c.charttime) from chart c where c.stay_id=o.stay_id and c.itemid=224689
             and c.valuenum between 1 and 80 and c.charttime>o.landmark_time-interval 6 hour and c.charttime<=o.landmark_time) spontaneous_rr_last,
          (select arg_max(c.valuenum,c.charttime) from chart c where c.stay_id=o.stay_id and c.itemid=224686
             and c.valuenum between 50 and 2000 and c.charttime>o.landmark_time-interval 6 hour and c.charttime<=o.landmark_time) spontaneous_tidal_volume_last,
          (select arg_max(c.valuenum,c.charttime) from chart c where c.stay_id=o.stay_id and c.itemid=224701
             and c.valuenum between 0 and 40 and c.charttime>o.landmark_time-interval 6 hour and c.charttime<=o.landmark_time) psv_level_last,
          (select arg_max(c.value,c.charttime) from chart c where c.stay_id=o.stay_id and c.itemid in (223849,229314)
             and c.value is not null and c.charttime>o.landmark_time-interval 6 hour and c.charttime<=o.landmark_time) ventilator_mode_last,
          case when exists(select 1 from chart c where c.stay_id=o.stay_id and c.itemid=224715
             and lower(c.value)='yes' and c.charttime>o.landmark_time-interval 24 hour and c.charttime<=o.landmark_time)
             then 1 else 0 end sbt_started_24h,
          case when exists(select 1 from chart c where c.stay_id=o.stay_id and c.itemid=224717
             and lower(c.value)='yes' and c.charttime>o.landmark_time-interval 24 hour and c.charttime<=o.landmark_time)
             then 1 else 0 end sbt_successful_24h,
          (select arg_max(l.valuenum,l.charttime) from lab l where l.hadm_id=o.hadm_id and l.itemid=50813
             and l.valuenum between 0 and 30 and l.charttime>o.landmark_time-interval 24 hour and l.charttime<=o.landmark_time) lactate_last,
          (select arg_max(l.valuenum,l.charttime) from lab l where l.hadm_id=o.hadm_id and l.itemid=50912
             and l.valuenum between 0.1 and 30 and l.charttime>o.landmark_time-interval 24 hour and l.charttime<=o.landmark_time) creatinine_last,
          (select arg_max(l.valuenum,l.charttime) from lab l where l.hadm_id=o.hadm_id and l.itemid=51265
             and l.valuenum between 1 and 2000 and l.charttime>o.landmark_time-interval 24 hour and l.charttime<=o.landmark_time) platelets_last,
          (select arg_max(l.valuenum,l.charttime) from lab l where l.hadm_id=o.hadm_id and l.itemid=50885
             and l.valuenum between 0 and 80 and l.charttime>o.landmark_time-interval 24 hour and l.charttime<=o.landmark_time) bilirubin_last,
          (select arg_max(l.valuenum,l.charttime) from lab l where l.hadm_id=o.hadm_id and l.itemid in (51300,51301)
             and l.valuenum between 0.1 and 200 and l.charttime>o.landmark_time-interval 24 hour and l.charttime<=o.landmark_time) wbc_last,
          (select arg_max(l.valuenum,l.charttime) from lab l where l.hadm_id=o.hadm_id and l.itemid=51222
             and l.valuenum between 2 and 25 and l.charttime>o.landmark_time-interval 24 hour and l.charttime<=o.landmark_time) hemoglobin_last,
          case when exists(select 1 from inp x where x.stay_id=o.stay_id and x.itemid in ({sql_list(VASO)})
            and x.starttime<o.landmark_time and x.endtime>o.landmark_time-interval 6 hour
            and coalesce(x.statusdescription,'')<>'Rewritten') then 1 else 0 end vasopressor_6h,
          case when exists(select 1 from inp x where x.stay_id=o.stay_id and x.itemid in ({sql_list(OPIOID)})
            and x.starttime<o.landmark_time and x.endtime>o.landmark_time-interval 6 hour
            and coalesce(x.statusdescription,'')<>'Rewritten') then 1 else 0 end opioid_6h,
          case when exists(select 1 from inp x where x.stay_id=o.stay_id and x.itemid in ({sql_list(MIDAZOLAM)})
            and x.starttime<o.landmark_time and x.endtime>o.landmark_time-interval 6 hour
            and coalesce(x.statusdescription,'')<>'Rewritten') then 1 else 0 end midazolam_6h,
          case when exists(select 1 from inp x where x.stay_id=o.stay_id and x.itemid in ({sql_list(KETAMINE)})
            and x.starttime<o.landmark_time and x.endtime>o.landmark_time-interval 6 hour
            and coalesce(x.statusdescription,'')<>'Rewritten') then 1 else 0 end ketamine_6h,
          date_diff('hour',o.intime,o.landmark_time) hours_since_icu
        from with_outcome o
        ) to '{qpath(target)}' (format parquet,compression zstd)
        ;

        copy (
          with stages as (
            select 1 stage_order,'Landmark eligible and event-free through window' stage,
              count(*) trial_instances,count(distinct subject_id) unique_subjects,count(distinct stay_id) unique_stays
            from eligible_precoverage
            union all
            select 2,'Strict-window strategy classifiable',count(*),count(distinct subject_id),count(distinct stay_id)
            from eligible_precoverage where treatment is not null
            union all
            select 3,'Raw propofol record covers landmark',count(*),count(distinct subject_id),count(distinct stay_id)
            from eligible_precoverage where treatment is not null and raw_prop_at_landmark=1
            union all
            select 4,'Invasive ventilation covers window end',count(*),count(distinct subject_id),count(distinct stay_id)
            from eligible_precoverage where treatment is not null and raw_prop_at_landmark=1 and invasive_vent_at_grace_end=1
            union all
            select 5,'Strategy drug covers window end',count(*),count(distinct subject_id),count(distinct stay_id)
            from eligible_precoverage where treatment is not null and raw_prop_at_landmark=1 and invasive_vent_at_grace_end=1
              and ((treatment=1 and raw_dex_at_grace_end=1) or (treatment=0 and raw_prop_at_grace_end=1))
            union all
            select 6,'Final v11 analysis cohort',count(*),count(distinct subject_id),count(distinct stay_id)
            from assigned
            union all
            select 7,'Transition strategy',count(*),count(distinct subject_id),count(distinct stay_id)
            from assigned where treatment=1
            union all
            select 8,'Continuation strategy',count(*),count(distinct subject_id),count(distinct stay_id)
            from assigned where treatment=0
          ) select * from stages order by stage_order
        ) to '{qpath(OUT / 'table0_flow_counts.csv')}' (header,format csv)
        """
    )
    print("Built sequential landmark trial table", flush=True)
    return target


NUMERIC = [
    "age", "patientweight", "hours_since_icu", "prop_rate",
    "map_6h", "hr_6h", "spo2_6h", "rass_last", "peep_last", "fio2_last",
    "rass_change_12h", "rr_last", "spontaneous_rr_last",
    "spontaneous_tidal_volume_last", "psv_level_last", "rsbi_proxy",
    "lactate_last", "creatinine_last", "platelets_last",
    "bilirubin_last", "wbc_last", "hemoglobin_last",
]
BINARY = [
    "vasopressor_6h", "opioid_6h", "midazolam_6h", "ketamine_6h",
    "sbt_started_24h", "sbt_successful_24h",
]
CATEGORICAL = [
    "landmark_factor", "gender", "first_careunit", "anchor_year_group",
    "admission_type", "admission_location", "race_group", "ventilator_mode_group",
]


def prepare_model(df: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    usable_num = [c for c in NUMERIC if df[c].notna().mean() >= 0.20]
    dropped = [c for c in NUMERIC if c not in usable_num]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True)),
         ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25))]
    )
    pre = ColumnTransformer(
        [("num", numeric_pipe, usable_num + BINARY),
         ("cat", categorical_pipe, CATEGORICAL)],
        remainder="drop",
    )
    model = Pipeline(
        [("pre", pre),
         ("logit", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))]
    )
    return model, usable_num, dropped


def prepare_flexible_model(df: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    """Alternative propensity model with restricted cubic-style spline terms.

    This is a model-specification sensitivity analysis, not a replacement for
    the primary prespecified linear-logit propensity model.
    """
    usable_num = [c for c in NUMERIC if df[c].notna().mean() >= 0.20]
    dropped = [c for c in NUMERIC if c not in usable_num]
    spline_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
            ("scale", StandardScaler()),
        ]
    )
    missing_pipe = Pipeline(
        [("missing", MissingIndicator(features="missing-only", error_on_new=False))]
    )
    binary_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=25)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num_spline", spline_pipe, usable_num),
            ("num_missing", missing_pipe, usable_num),
            ("binary", binary_pipe, BINARY),
            ("cat", categorical_pipe, CATEGORICAL),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("pre", pre),
            ("logit", LogisticRegression(C=0.5, max_iter=3000, solver="lbfgs")),
        ]
    )
    return model, usable_num, dropped


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else math.nan


def smd_numeric(x: pd.Series, treatment: pd.Series, weights: np.ndarray) -> float:
    a = treatment.to_numpy() == 1
    xv = pd.to_numeric(x, errors="coerce").to_numpy(float)
    med = np.nanmedian(xv)
    xv = np.where(np.isfinite(xv), xv, med)
    m1 = weighted_mean(xv[a], weights[a]); m0 = weighted_mean(xv[~a], weights[~a])
    v1 = weighted_mean((xv[a] - m1) ** 2, weights[a])
    v0 = weighted_mean((xv[~a] - m0) ** 2, weights[~a])
    den = math.sqrt(max((v1 + v0) / 2, 1e-12))
    return (m1 - m0) / den


def balance_table(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    rows = []
    ones = np.ones(len(df), dtype=float)
    for col in NUMERIC + BINARY:
        rows.append({
            "variable": col,
            "missing_percent": 100 * df[col].isna().mean(),
            "smd_unweighted": smd_numeric(df[col], df.treatment, ones),
            "smd_weighted": smd_numeric(df[col], df.treatment, weights),
        })
        if df[col].isna().any():
            missing_indicator = df[col].isna().astype(float)
            rows.append({
                "variable": f"{col}: missing",
                "missing_percent": 0.0,
                "smd_unweighted": smd_numeric(missing_indicator, df.treatment, ones),
                "smd_weighted": smd_numeric(missing_indicator, df.treatment, weights),
            })
    for col in CATEGORICAL:
        values = df[col].fillna("Missing").astype(str)
        for level in sorted(values.unique()):
            indicator = values.eq(level).astype(float)
            rows.append({
                "variable": f"{col}: {level}",
                "missing_percent": 0.0,
                "smd_unweighted": smd_numeric(indicator, df.treatment, ones),
                "smd_weighted": smd_numeric(indicator, df.treatment, weights),
            })
    return pd.DataFrame(rows).sort_values("smd_weighted", key=lambda s: s.abs(), ascending=False)


def weighted_sd(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if ok.sum() < 2:
        return math.nan
    mean = np.average(x[ok], weights=w[ok])
    return float(np.sqrt(np.average(np.square(x[ok] - mean), weights=w[ok])))


def descriptive_table(df: pd.DataFrame, balance: pd.DataFrame) -> pd.DataFrame:
    """Create a conventional unweighted and overlap-weighted baseline table."""
    rows: list[dict] = []
    balance_lookup = balance.set_index("variable")[["smd_unweighted", "smd_weighted"]].to_dict("index")
    treatment = df.treatment.to_numpy(int)
    weights = df.overlap_weight.to_numpy(float)

    def add_row(variable: str, level: str, control: str, treated: str,
                weighted_control: str, weighted_treated: str,
                missing_control: float = 0.0, missing_treated: float = 0.0,
                balance_key: str | None = None) -> None:
        key = balance_key or (variable if not level else f"{variable}: {level}")
        smd = balance_lookup.get(key, {})
        rows.append({
            "variable": variable,
            "level": level,
            "continuation_unweighted": control,
            "transition_unweighted": treated,
            "continuation_overlap_weighted": weighted_control,
            "transition_overlap_weighted": weighted_treated,
            "missing_control_percent": missing_control,
            "missing_treated_percent": missing_treated,
            "smd_unweighted": smd.get("smd_unweighted", math.nan),
            "smd_weighted": smd.get("smd_weighted", math.nan),
        })

    for col in NUMERIC:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        summaries = []
        weighted_summaries = []
        missing = []
        for group in (0, 1):
            mask = treatment == group
            observed = mask & np.isfinite(values)
            mean = float(np.nanmean(values[mask]))
            sd = float(np.nanstd(values[mask], ddof=1))
            summaries.append(f"{mean:.2f} ({sd:.2f})")
            weighted_mean_value = weighted_mean(values[observed], weights[observed])
            weighted_sd_value = weighted_sd(values[observed], weights[observed])
            weighted_summaries.append(f"{weighted_mean_value:.2f} ({weighted_sd_value:.2f})")
            missing.append(100 * float((mask & ~np.isfinite(values)).sum()) / float(mask.sum()))
        add_row(col, "", summaries[0], summaries[1], weighted_summaries[0], weighted_summaries[1], missing[0], missing[1])

    for col in BINARY:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        summaries = []
        weighted_summaries = []
        for group in (0, 1):
            mask = treatment == group
            event_count = int(np.nansum(values[mask]))
            summaries.append(f"{event_count:,}/{int(mask.sum()):,} ({100*np.nanmean(values[mask]):.1f}%)")
            weighted_summaries.append(f"{100*weighted_mean(values[mask], weights[mask]):.1f}%")
        add_row(col, "Yes", summaries[0], summaries[1], weighted_summaries[0], weighted_summaries[1], balance_key=col)

    for col in CATEGORICAL:
        values = df[col].fillna("Missing").astype(str)
        for level in sorted(values.unique()):
            indicator = values.eq(level).to_numpy(float)
            summaries = []
            weighted_summaries = []
            for group in (0, 1):
                mask = treatment == group
                event_count = int(indicator[mask].sum())
                summaries.append(f"{event_count:,}/{int(mask.sum()):,} ({100*indicator[mask].mean():.1f}%)")
                weighted_summaries.append(f"{100*weighted_mean(indicator[mask], weights[mask]):.1f}%")
            add_row(col, level, summaries[0], summaries[1], weighted_summaries[0], weighted_summaries[1])
    return pd.DataFrame(rows)


def create_flow_figure() -> None:
    flow_path = OUT / "table0_flow_counts.csv"
    if not flow_path.exists():
        return
    flow = pd.read_csv(flow_path).set_index("stage")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis("off")
    ax.set_ylim(-0.08, 1.0)

    def label(stage: str) -> str:
        row = flow.loc[stage]
        return (
            f"{stage}\n{int(row.trial_instances):,} trial instances; "
            f"{int(row.unique_subjects):,} patients; {int(row.unique_stays):,} ICU stays"
        )

    main_stages = [
        "Landmark eligible and event-free through window",
        "Strict-window strategy classifiable",
        "Raw propofol record covers landmark",
        "Invasive ventilation covers window end",
        "Strategy drug covers window end",
        "Final v11 analysis cohort",
    ]
    ys = [0.94, 0.78, 0.62, 0.46, 0.30, 0.14]
    for stage, y in zip(main_stages, ys):
        ax.text(0.5, y, label(stage), ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef3f8", edgecolor="#4c72b0"))
    for y1, y2 in zip(ys[:-1], ys[1:]):
        ax.annotate("", xy=(0.5, y2 + 0.07), xytext=(0.5, y1 - 0.07),
                    arrowprops=dict(arrowstyle="->", color="#4c72b0", lw=1.5))
    stage_counts = [int(flow.loc[stage].trial_instances) for stage in main_stages]
    exclusions = [
        f"Not strategy-classifiable: {stage_counts[0]-stage_counts[1]:,}",
        f"No raw propofol coverage at landmark: {stage_counts[1]-stage_counts[2]:,}",
        f"No invasive-ventilation coverage at window end: {stage_counts[2]-stage_counts[3]:,}",
        f"Strategy drug not covering window end: {stage_counts[3]-stage_counts[4]:,}",
        f"Continuation with internal propofol gap: {stage_counts[4]-stage_counts[5]:,}",
    ]
    for text_value, y1, y2 in zip(exclusions, ys[:-1], ys[1:]):
        ax.text(0.72, (y1 + y2) / 2, text_value, ha="left", va="center", fontsize=8.5, color="#555555")
    for stage, x in [("Transition strategy", 0.25), ("Continuation strategy", 0.75)]:
        row = flow.loc[stage]
        text_value = f"{stage}\n{int(row.trial_instances):,} trial instances"
        ax.text(x, -0.02, text_value, ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f6efe8", edgecolor="#dd8452"))
        ax.annotate("", xy=(x, 0.04), xytext=(0.5, 0.09),
                    arrowprops=dict(arrowstyle="->", color="#dd8452", lw=1.3))
    fig.tight_layout()
    fig.savefig(OUT / "figure0_cohort_flow.png", dpi=220)
    plt.close(fig)


def group_risks(df: pd.DataFrame, outcome: str, multiplier: np.ndarray | None = None) -> tuple[float, float, float, float]:
    y = df[outcome].to_numpy(float)
    t = df.treatment.to_numpy(int)
    w = df.overlap_weight.to_numpy(float).copy()
    if multiplier is not None:
        w *= multiplier
    r0 = weighted_mean(y[t == 0], w[t == 0])
    r1 = weighted_mean(y[t == 1], w[t == 1])
    return r0, r1, r1 - r0, r1 / r0 if r0 > 0 else math.nan


def cluster_multiplier_ci(df: pd.DataFrame, outcome: str, reps: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(PYTHON_RANDOM_SEED)
    subjects, inv = np.unique(df.subject_id.to_numpy(), return_inverse=True)
    estimates = np.empty((reps, 2), dtype=float)
    for b in range(reps):
        cluster_w = rng.exponential(1.0, size=len(subjects))
        _, _, rd, rr = group_risks(df, outcome, cluster_w[inv])
        estimates[b] = (rd, rr)
    return {
        "rd_low": float(np.nanquantile(estimates[:, 0], 0.025)),
        "rd_high": float(np.nanquantile(estimates[:, 0], 0.975)),
        "rr_low": float(np.nanquantile(estimates[:, 1], 0.025)),
        "rr_high": float(np.nanquantile(estimates[:, 1], 0.975)),
    }


def effective_n(weights: np.ndarray) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def full_cluster_resample_bootstrap(
    work: pd.DataFrame,
    reps: int = 300,
) -> dict[str, float]:
    """Patient-cluster bootstrap with the complete PS pipeline refitted.

    Subjects are sampled with replacement. Every row belonging to a sampled
    subject is repeated together, so within-patient dependence and the number
    of eligible sequential trials are preserved. Variable-availability
    screening, imputation, scaling, encoding, and treatment-model estimation
    are all repeated inside each bootstrap sample.
    """
    subject_codes, subjects = pd.factorize(work.subject_id, sort=True)
    n_subjects = len(subjects)
    rng = np.random.default_rng(PYTHON_RANDOM_SEED + 17)
    estimates = np.empty((reps, 2), dtype=float)
    row_numbers = np.arange(len(work))
    for b in range(reps):
        sampled = rng.integers(0, n_subjects, size=n_subjects)
        counts = np.bincount(sampled, minlength=n_subjects)
        boot_idx = np.repeat(row_numbers, counts[subject_codes])
        boot = work.iloc[boot_idx].copy()
        model, usable_num, _ = prepare_model(boot)
        predictors = usable_num + BINARY + CATEGORICAL
        model.fit(boot[predictors], boot.treatment)
        treatment = boot.treatment.to_numpy(int)
        outcome = boot.successful_extubation.to_numpy(float)
        ps = np.clip(model.predict_proba(boot[predictors])[:, 1], 1e-5, 1 - 1e-5)
        overlap = np.where(treatment == 1, 1 - ps, ps)
        r0 = weighted_mean(outcome[treatment == 0], overlap[treatment == 0])
        r1 = weighted_mean(outcome[treatment == 1], overlap[treatment == 1])
        estimates[b] = (r1 - r0, r1 / r0 if r0 > 0 else np.nan)
        if (b + 1) % 50 == 0:
            print(f"Completed {b + 1}/{reps} full cluster bootstrap replicates", flush=True)
    return {
        "rd_low": float(np.nanquantile(estimates[:, 0], 0.025)),
        "rd_high": float(np.nanquantile(estimates[:, 0], 0.975)),
        "rr_low": float(np.nanquantile(estimates[:, 1], 0.025)),
        "rr_high": float(np.nanquantile(estimates[:, 1], 0.975)),
    }


def fit_analysis(df: pd.DataFrame, label: str, refit: bool = False) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    model, usable_num, dropped = prepare_model(work)
    predictors = usable_num + BINARY + CATEGORICAL
    model.fit(work[predictors], work.treatment)
    ps = model.predict_proba(work[predictors])[:, 1]
    ps = np.clip(ps, 1e-5, 1 - 1e-5)
    work["ps"] = ps
    work["overlap_weight"] = np.where(work.treatment.eq(1), 1 - ps, ps)
    balance = balance_table(work, work.overlap_weight.to_numpy())
    r0, r1, rd, rr = group_risks(work, "successful_extubation")
    ci = (
        full_cluster_resample_bootstrap(work)
        if refit else cluster_multiplier_ci(work, "successful_extubation")
    )
    treated = work.treatment.eq(1).to_numpy()
    result = {
        "analysis": label,
        "trials": int(len(work)),
        "unique_subjects": int(work.subject_id.nunique()),
        "unique_stays": int(work.stay_id.nunique()),
        "treated_trials": int(treated.sum()),
        "control_trials": int((~treated).sum()),
        "events_treated": int(work.loc[treated, "successful_extubation"].sum()),
        "events_control": int(work.loc[~treated, "successful_extubation"].sum()),
        "weighted_risk_control": r0,
        "weighted_risk_treated": r1,
        "risk_difference": rd,
        "rd_ci_low": ci["rd_low"],
        "rd_ci_high": ci["rd_high"],
        "risk_ratio": rr,
        "rr_ci_low": ci["rr_low"],
        "rr_ci_high": ci["rr_high"],
        "ess_treated": effective_n(work.loc[treated, "overlap_weight"].to_numpy()),
        "ess_control": effective_n(work.loc[~treated, "overlap_weight"].to_numpy()),
        "max_abs_smd_unweighted": float(balance.smd_unweighted.abs().max()),
        "max_abs_smd_weighted": float(balance.smd_weighted.abs().max()),
        "ps_treated_p01": float(np.quantile(ps[treated], 0.01)),
        "ps_treated_p99": float(np.quantile(ps[treated], 0.99)),
        "ps_control_p01": float(np.quantile(ps[~treated], 0.01)),
        "ps_control_p99": float(np.quantile(ps[~treated], 0.99)),
        "dropped_high_missing_covariates": ";".join(dropped),
    }
    return result, balance, work


def fit_flexible_ps_sensitivity(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Fit a spline-logit propensity model and report conditional-weight uncertainty."""
    work = df.copy()
    model, usable_num, dropped = prepare_flexible_model(work)
    predictors = usable_num + BINARY + CATEGORICAL
    model.fit(work[predictors], work.treatment)
    ps = np.clip(model.predict_proba(work[predictors])[:, 1], 1e-5, 1 - 1e-5)
    work["ps"] = ps
    work["overlap_weight"] = np.where(work.treatment.eq(1), 1 - ps, ps)
    balance = balance_table(work, work.overlap_weight.to_numpy())
    r0, r1, rd, rr = group_risks(work, "successful_extubation")
    ci = cluster_multiplier_ci(work, "successful_extubation", reps=1000)
    treated = work.treatment.eq(1).to_numpy()
    result = {
        "analysis": "Alternative spline-logit propensity model",
        "trials": int(len(work)),
        "unique_subjects": int(work.subject_id.nunique()),
        "treated_trials": int(treated.sum()),
        "control_trials": int((~treated).sum()),
        "weighted_risk_control": r0,
        "weighted_risk_treated": r1,
        "risk_difference": rd,
        "rd_ci_low": ci["rd_low"],
        "rd_ci_high": ci["rd_high"],
        "risk_ratio": rr,
        "rr_ci_low": ci["rr_low"],
        "rr_ci_high": ci["rr_high"],
        "ess_treated": effective_n(work.loc[treated, "overlap_weight"].to_numpy()),
        "ess_control": effective_n(work.loc[~treated, "overlap_weight"].to_numpy()),
        "max_abs_smd_weighted": float(balance.smd_weighted.abs().max()),
        "ps_treated_p01": float(np.quantile(ps[treated], 0.01)),
        "ps_treated_p99": float(np.quantile(ps[treated], 0.99)),
        "ps_control_p01": float(np.quantile(ps[~treated], 0.01)),
        "ps_control_p99": float(np.quantile(ps[~treated], 0.99)),
        "interval_note": "1000 subject-cluster multiplier replicates conditional on fitted spline-logit weights",
        "dropped_high_missing_covariates": ";".join(dropped),
    }
    return result, balance


def create_plots(primary_work: pd.DataFrame, primary_balance: pd.DataFrame, effects: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    top = primary_balance.assign(abs_pre=lambda x: x.smd_unweighted.abs()).nlargest(18, "abs_pre").sort_values("abs_pre")
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    y = np.arange(len(top))
    ax.scatter(top.smd_unweighted, y, label="Unweighted", color="#c44e52", s=34)
    ax.scatter(top.smd_weighted, y, label="Overlap weighted", color="#4c72b0", s=34)
    ax.axvline(-0.1, color="grey", linestyle="--", linewidth=1)
    ax.axvline(0.1, color="grey", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y, top.variable)
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Covariate balance")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figure1_love_plot.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for value, color, name in [(0, "#4c72b0", "Continue propofol"), (1, "#dd8452", "Transition")]:
        vals = primary_work.loc[primary_work.treatment.eq(value), "ps"]
        ax.hist(vals, bins=40, density=True, alpha=0.45, color=color, label=name)
    ax.set_xlabel("Estimated probability of transition")
    ax.set_ylabel("Density")
    ax.set_title("Propensity-score overlap")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figure2_propensity_overlap.png", dpi=220)
    plt.close(fig)

    plot_df = effects.iloc[::-1].reset_index(drop=True)
    figure_labels = {
        "Primary: all sequential trials": "Primary",
        "Low ventilator settings: PEEP<=8 and FiO2<=50": "PEEP ≤8 and FiO2 ≤50%",
        "Enhanced readiness: low settings plus successful SBT or spontaneous mode with PSV<=10": "Enhanced readiness",
        "Documented successful SBT in prior 24 h": "Successful SBT in prior 24 h",
        "Excluding cardiovascular ICU": "Excluding cardiovascular ICU",
        "First eligible trial per stay": "First eligible trial per stay",
        "Early landmarks: <=24 h": "Landmarks ≤24 h",
        "Complete key respiratory covariates": "Complete PEEP and FiO2",
        "Complete readiness documentation: PEEP, FiO2, RASS, spontaneous RR, tidal volume, and PSV": "Complete six-variable readiness data",
        "Dex initiated within 1 h vs continued propofol": "Dex initiated within 1 h",
        "Dex initiated within 3 h vs continued propofol": "Dex initiated within 3 h",
        "Propofol stopped within 2 h after dex initiation": "Propofol stopped within 2 h of dex",
        "No later dexmedetomidine among continuation instances during the 48-h outcome window": "No later dex in continuation group (48 h)",
    }
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    y = np.arange(len(plot_df))
    point = plot_df.risk_difference * 100
    low = plot_df.rd_ci_low * 100
    high = plot_df.rd_ci_high * 100
    ax.errorbar(point, y, xerr=[point - low, high - point], fmt="o", color="#4c72b0", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y, plot_df.analysis.map(figure_labels).fillna(plot_df.analysis))
    ax.set_xlabel("Stable documented-extubation risk difference (percentage points)")
    ax.set_title("Primary and sensitivity analyses")
    fig.tight_layout()
    fig.savefig(OUT / "figure3_effect_forest.png", dpi=220)
    plt.close(fig)


def create_landmark_plot(effects: pd.DataFrame) -> None:
    if effects.empty:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = effects.landmark_hour.to_numpy()
    point = effects.risk_difference.to_numpy() * 100
    low = effects.rd_ci_low.to_numpy() * 100
    high = effects.rd_ci_high.to_numpy() * 100
    ax.errorbar(x, point, yerr=[point - low, high - point], fmt="o-", color="#4c72b0", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Hours after propofol episode start")
    ax.set_ylabel("Risk difference (percentage points)")
    ax.set_title("Stable documented extubation by sequential landmark")
    fig.tight_layout()
    fig.savefig(OUT / "figure4_landmark_effects.png", dpi=220)
    plt.close(fig)


def write_report(effects: pd.DataFrame, missing: pd.DataFrame, primary: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    def pct(x: float) -> str:
        return f"{100*x:.1f}%"
    report = f"""# MIMIC-IV 序贯 landmark 调整后关联分析

## 主结果

主分析包含 {primary['trials']:,} 个序贯试验实例（{primary['unique_subjects']:,} 名患者、{primary['unique_stays']:,} 个 ICU 住院次），其中转换策略 {primary['treated_trials']:,} 个、继续丙泊酚 {primary['control_trials']:,} 个。

重叠加权后，继续丙泊酚组 48 小时内稳定成功拔管风险为 {pct(primary['weighted_risk_control'])}，转换组为 {pct(primary['weighted_risk_treated'])}。风险差为 {100*primary['risk_difference']:.2f} 个百分点（95% 区间 {100*primary['rd_ci_low']:.2f} 至 {100*primary['rd_ci_high']:.2f}），风险比为 {primary['risk_ratio']:.2f}（95% 区间 {primary['rr_ci_low']:.2f} 至 {primary['rr_ci_high']:.2f}）。

## 诊断

- 加权前最大绝对 SMD：{primary['max_abs_smd_unweighted']:.3f}。
- 加权后最大绝对 SMD：{primary['max_abs_smd_weighted']:.3f}。
- 转换组有效样本量：{primary['ess_treated']:.1f}；继续组有效样本量：{primary['ess_control']:.1f}。
- 转换组倾向评分 1%–99%：{primary['ps_treated_p01']:.4f}–{primary['ps_treated_p99']:.4f}。
- 继续组倾向评分 1%–99%：{primary['ps_control_p01']:.4f}–{primary['ps_control_p99']:.4f}。

## 解释边界

这是严格 6 小时策略评估窗完成后的序贯 landmark 估计；转换组要求窗口末仍有原始右美托咪定记录覆盖，继续组要求整个窗口无间断原始丙泊酚记录覆盖。主分析置信区间采用 300 次患者整簇有放回重抽样，并在每次重复中重新执行变量筛选、插补、缩放、编码和倾向模型拟合；敏感性分析区间以已估计权重为条件。

该结果只能解释为目标重叠人群中的调整后关联。主策略允许窗口后治疗变化；排除后续右美托咪定的分析使用了未来治疗信息，仅是交叉用药诊断。即使敏感性分析方向一致，也不能排除未记录的撤机准备度、气道保护能力和临床治疗意图造成的残余混杂，不能升级为强因果表述。

## 敏感性分析

详见 `table3_sensitivity_effects.csv`。低呼吸机支持分析要求时间零前 PEEP ≤8 cmH2O 且 FiO2 ≤50%；增强准备度分析进一步要求成功 SBT 或低水平压力支持；“首次试验”每个 ICU 住院次只保留首个合格决策时点。
"""
    (OUT / "analysis_report.md").write_text(report, encoding="utf-8")


def secondary_outcomes(primary_work: pd.DataFrame) -> pd.DataFrame:
    work = primary_work.copy()
    work["documented_extubation_48h"] = (
        work.documented_extubation.notna() & work.unplanned_extubation.isna()
    ).astype(int)
    work["unplanned_extubation_48h"] = work.unplanned_extubation.notna().astype(int)
    work["new_bradycardia_24h"] = np.where(
        work.hr_measurements_24h.ge(2),
        (work.bradycardia_24h.eq(1) & work.baseline_bradycardia.eq(0)).astype(float),
        np.nan,
    )
    work["observed_severe_bradycardia_24h"] = np.where(
        work.hr_measurements_24h.ge(2), work.severe_bradycardia_24h.astype(float), np.nan
    )
    work["new_hypotension_24h"] = np.where(
        work.map_measurements_24h.ge(2),
        (work.hypotension_24h.eq(1) & work.baseline_hypotension.eq(0)).astype(float),
        np.nan,
    )
    work["observed_severe_hypotension_24h"] = np.where(
        work.map_measurements_24h.ge(2), work.severe_hypotension_24h.astype(float), np.nan
    )
    rows = []
    for outcome, label in [
        ("documented_extubation_48h", "Documented extubation not coded as unplanned within 48 h"),
        ("unplanned_extubation_48h", "Unplanned extubation within 48 h"),
        ("death_48h", "Death within 48 h"),
        ("successful_extubation_full_72h_observation", "Stable documented extubation with full 72-h in-hospital observation"),
        ("new_bradycardia_24h", "New bradycardia within 24 h (>=2 HR readings <50/min)"),
        ("observed_severe_bradycardia_24h", "Severe bradycardia within 24 h (>=2 HR readings <40/min)"),
        ("new_hypotension_24h", "New hypotension within 24 h (>=2 MAP readings <65 mm Hg)"),
        ("observed_severe_hypotension_24h", "Severe hypotension within 24 h (>=2 MAP readings <55 mm Hg)"),
        ("new_or_additional_vasopressor_24h", "New or additional vasopressor agent within 24 h"),
    ]:
        r0, r1, rd, rr = group_risks(work, outcome)
        ci = cluster_multiplier_ci(work, outcome)
        rows.append({
            "outcome": label,
            "observed_control_trials": int(work.loc[work.treatment.eq(0), outcome].notna().sum()),
            "observed_treated_trials": int(work.loc[work.treatment.eq(1), outcome].notna().sum()),
            "events_control": int(work.loc[work.treatment.eq(0), outcome].sum(skipna=True)),
            "events_treated": int(work.loc[work.treatment.eq(1), outcome].sum(skipna=True)),
            "weighted_risk_control": r0,
            "weighted_risk_treated": r1,
            "risk_difference": rd,
            "rd_ci_low": ci["rd_low"],
            "rd_ci_high": ci["rd_high"],
            "risk_ratio": rr,
            "rr_ci_low": ci["rr_low"],
            "rr_ci_high": ci["rr_high"],
        })
    return pd.DataFrame(rows)


def write_outcome_event_audit(df: pd.DataFrame) -> None:
    """Write aggregate and sampled audits of hospital-level event sequences."""
    events = df.loc[df.documented_extubation.notna()].copy()
    events = events.sort_values(["hadm_id", "documented_extubation", "landmark_time"])
    events = events.drop_duplicates(["hadm_id", "documented_extubation"], keep="first")
    death_time = pd.to_datetime(events.deathtime)
    ext_time = pd.to_datetime(events.documented_extubation)
    events["death_within_72h"] = (
        death_time.notna() & death_time.gt(ext_time) & death_time.le(ext_time + pd.Timedelta(hours=72))
    ).astype(int)
    events["reintubation_within_72h"] = events.reintubation_72h_time.notna().astype(int)
    events["invasive_vent_restart_within_72h"] = events.invasive_vent_restart_72h_time.notna().astype(int)
    events["tracheostomy_within_72h"] = events.tracheostomy_72h_time.notna().astype(int)
    rows = [
        ("unique_documented_extubation_events", len(events)),
        ("events_with_unplanned_extubation_code", int(events.unplanned_extubation.notna().sum())),
        ("events_not_coded_as_unplanned", int(events.unplanned_extubation.isna().sum())),
        ("stable_success_events", int(events.successful_extubation.sum())),
        ("stable_success_events_with_unplanned_code", int(
            (events.successful_extubation.eq(1) & events.unplanned_extubation.notna()).sum()
        )),
        ("stable_success_events_with_vent_restart", int(
            (events.successful_extubation.eq(1) & events.invasive_vent_restart_72h_time.notna()).sum()
        )),
        ("reintubation_within_72h", int(events.reintubation_within_72h.sum())),
        ("invasive_vent_restart_within_72h", int(events.invasive_vent_restart_within_72h.sum())),
        ("vent_restart_without_reintubation_code", int(
            (events.invasive_vent_restart_72h_time.notna() & events.reintubation_72h_time.isna()).sum()
        )),
        ("tracheostomy_within_72h", int(events.tracheostomy_within_72h.sum())),
        ("death_within_72h", int(events.death_within_72h.sum())),
        ("cross_icu_stay_reintubation_within_72h", int(events.cross_stay_reintubation_72h.sum())),
        ("live_hospital_discharge_before_72h", int(events.live_discharge_before_72h.sum())),
        ("stable_success_with_full_72h_in_hospital", int(events.successful_extubation_full_72h_observation.sum())),
    ]
    pd.DataFrame(rows, columns=["audit_metric", "count"]).to_csv(
        OUT / "table6_outcome_event_audit.csv", index=False
    )
    blinded_columns = [
        "review_id",
        "subject_id", "hadm_id", "stay_id", "documented_extubation_stay_id",
        "documented_extubation", "reintubation_72h_time", "invasive_vent_restart_72h_time",
        "tracheostomy_72h_time",
        "deathtime", "dischtime", "cross_stay_reintubation_72h",
    ]
    success_events = events.loc[events.successful_extubation.eq(1)]
    failure_events = events.loc[events.successful_extubation.eq(0)]
    sample = pd.concat(
        [
            success_events.sample(n=min(30, len(success_events)), random_state=PYTHON_RANDOM_SEED),
            failure_events.sample(n=min(30, len(failure_events)), random_state=PYTHON_RANDOM_SEED + 1),
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=PYTHON_RANDOM_SEED + 2)
    sample = sample.reset_index(drop=True)
    sample.insert(0, "review_id", [f"OE-{i:03d}" for i in range(1, len(sample) + 1)])
    sample[blinded_columns].to_csv(WORK / "outcome_event_review_sample_blinded.csv", index=False)
    sample[["review_id", "successful_extubation", "unplanned_extubation"]].rename(
        columns={"successful_extubation": "algorithmic_successful_extubation"}
    ).to_csv(WORK / "outcome_event_review_sample_key.csv", index=False)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(WORK / "mimic_landmark.duckdb"))
    con.execute("set threads=4")
    con.execute("set preserve_insertion_order=false")
    files = build_source_caches(con)
    trials_path = build_trials(con, files)
    df = con.execute(f"select * from read_parquet('{qpath(trials_path)}')").fetchdf()
    df["treatment"] = df.treatment.astype(int)
    df["successful_extubation"] = df.successful_extubation.astype(int)
    df["landmark_factor"] = df.landmark_hour.astype(int).astype(str) + " h"
    race_upper = df.race.fillna("Unknown").astype(str).str.upper()
    df["race_group"] = np.select(
        [race_upper.str.contains("WHITE"), race_upper.str.contains("BLACK"),
         race_upper.str.contains("ASIAN"), race_upper.str.contains("HISPANIC|LATINO")],
        ["White", "Black", "Asian", "Hispanic/Latino"], default="Other/Unknown"
    )
    mode_upper = df.ventilator_mode_last.fillna("").astype(str).str.upper()
    controlled_pattern = "SIMV|CMV|ASSIST|PRVC|PCV|APV|P-CMV|ASV|MMV|APRV|PRES/AC"
    spontaneous_pattern = "CPAP|PSV|SPONT|SBT|PPS|(?:^|[^A-Z])VS(?:[^A-Z]|$)"
    df["ventilator_mode_group"] = np.select(
        [
            mode_upper.eq(""),
            mode_upper.str.contains("STANDBY|AMBIENT", regex=True),
            mode_upper.str.contains(controlled_pattern, regex=True),
            mode_upper.str.contains(spontaneous_pattern, regex=True),
        ],
        ["Missing", "Standby/ambient", "Controlled/assisted", "Spontaneous/pressure support"],
        default="Other",
    )
    df["rass_change_12h"] = df.rass_last - df.rass_first_12h
    df["rsbi_proxy"] = np.where(
        df.spontaneous_rr_last.gt(0) & df.spontaneous_tidal_volume_last.gt(50),
        df.spontaneous_rr_last / (df.spontaneous_tidal_volume_last / 1000.0),
        np.nan,
    )
    df = df.sort_values(["stay_id", "landmark_time"]).reset_index(drop=True)

    missing = pd.DataFrame({
        "variable": NUMERIC + BINARY + CATEGORICAL,
        "missing_percent": [100 * df[c].isna().mean() for c in NUMERIC + BINARY + CATEGORICAL],
    }).sort_values("missing_percent", ascending=False)
    missing.to_csv(OUT / "table1_missingness.csv", index=False)

    analyses: list[tuple[str, pd.DataFrame]] = [("Primary: all sequential trials", df)]
    ready = df.peep_last.le(8) & df.fio2_last.le(50)
    analyses.append(("Low ventilator settings: PEEP<=8 and FiO2<=50", df.loc[ready].copy()))
    enhanced_ready = ready & (
        df.sbt_successful_24h.eq(1)
        | (
            df.ventilator_mode_group.eq("Spontaneous/pressure support")
            & df.psv_level_last.le(10)
        )
    )
    analyses.append(("Enhanced readiness: low settings plus successful SBT or spontaneous mode with PSV<=10", df.loc[enhanced_ready].copy()))
    analyses.append(("Documented successful SBT in prior 24 h", df.loc[df.sbt_successful_24h.eq(1)].copy()))
    analyses.append((
        "Excluding cardiovascular ICU",
        df.loc[~df.first_careunit.eq("Cardiac Vascular Intensive Care Unit (CVICU)")].copy(),
    ))
    first = df.sort_values("landmark_time").drop_duplicates("stay_id", keep="first")
    analyses.append(("First eligible trial per stay", first.copy()))
    analyses.append(("Early landmarks: <=24 h", df.loc[df.landmark_hour.le(24)].copy()))
    analyses.append(("Complete key respiratory covariates", df.loc[df.peep_last.notna() & df.fio2_last.notna()].copy()))
    complete_readiness = (
        df.peep_last.notna()
        & df.fio2_last.notna()
        & df.rass_last.notna()
        & df.spontaneous_rr_last.notna()
        & df.spontaneous_tidal_volume_last.notna()
        & df.psv_level_last.notna()
    )
    analyses.append((
        "Complete readiness documentation: PEEP, FiO2, RASS, spontaneous RR, tidal volume, and PSV",
        df.loc[complete_readiness].copy(),
    ))
    controls = df.treatment.eq(0)
    dex_delay = (pd.to_datetime(df.dex_start) - pd.to_datetime(df.landmark_time)).dt.total_seconds() / 3600
    analyses.append(("Dex initiated within 1 h vs continued propofol",
                     df.loc[controls | (df.treatment.eq(1) & dex_delay.le(1))].copy()))
    analyses.append(("Dex initiated within 3 h vs continued propofol",
                     df.loc[controls | (df.treatment.eq(1) & dex_delay.le(3))].copy()))
    overlap_after_dex = (pd.to_datetime(df.prop_end) - pd.to_datetime(df.dex_start)).dt.total_seconds() / 3600
    analyses.append(("Propofol stopped within 2 h after dex initiation",
                     df.loc[controls | (df.treatment.eq(1) & overlap_after_dex.le(2))].copy()))
    analyses.append((
        "No later dexmedetomidine among continuation instances during the 48-h outcome window",
        df.loc[df.treatment.eq(1) | (controls & df.later_dex_48h.isna())].copy(),
    ))

    results = []
    primary_balance = None
    primary_work = None
    for i, (label, subset) in enumerate(analyses):
        if len(subset) < 200 or subset.treatment.sum() < 30:
            continue
        print(f"Fitting {label}: n={len(subset):,}", flush=True)
        result, balance, weighted = fit_analysis(subset, label, refit=(i == 0))
        results.append(result)
        if i == 0:
            primary_balance = balance
            primary_work = weighted
            balance.to_csv(OUT / "table2_covariate_balance.csv", index=False)
            descriptive_table(weighted, balance).to_csv(OUT / "table7_descriptive_baseline.csv", index=False)
            weighted[["trial_id", "subject_id", "hadm_id", "stay_id", "landmark_time", "treatment", "successful_extubation", "ps", "overlap_weight"]].to_csv(
                OUT / "analysis_cohort_weights.csv", index=False
            )

    effects = pd.DataFrame(results)
    effects.to_csv(OUT / "table3_sensitivity_effects.csv", index=False)

    print("Fitting alternative spline-logit propensity model", flush=True)
    flexible_result, flexible_balance = fit_flexible_ps_sensitivity(df)
    pd.DataFrame([flexible_result]).to_csv(OUT / "table8_model_specification_sensitivity.csv", index=False)
    flexible_balance.to_csv(OUT / "table9_model_specification_balance.csv", index=False)

    landmark_results = []
    for hour, subset in df.groupby("landmark_hour", sort=True):
        if len(subset) >= 200 and subset.treatment.sum() >= 30:
            result, _, _ = fit_analysis(subset.copy(), f"Landmark {int(hour)} h")
            result["landmark_hour"] = int(hour)
            landmark_results.append(result)
    landmark_effects = pd.DataFrame(landmark_results)
    landmark_effects.to_csv(OUT / "table4_landmark_effects.csv", index=False)
    assert primary_balance is not None and primary_work is not None
    secondary_outcomes(primary_work).to_csv(OUT / "table5_secondary_outcomes.csv", index=False)
    write_outcome_event_audit(df)
    create_plots(primary_work, primary_balance, effects)
    create_landmark_plot(landmark_effects)
    create_flow_figure()
    write_report(effects, missing, results[0])
    metadata = {
        "random_seed": PYTHON_RANDOM_SEED,
        "source": "MIMIC-IV 3.1",
        "follow_up_origin": "end of 6-hour strategy assessment window",
        "outcome": "documented extubation not coded as unplanned within 48 hours without hospital-admission-level reintubation code, invasive-ventilation restart, tracheostomy, or death within 72 hours; live hospital discharge ends in-hospital ascertainment without observed failure",
        "weighting": "overlap weighting from logistic propensity score",
        "model_specification_sensitivity": "Alternative regularized spline-logit propensity model with 4 knots for eligible continuous covariates; uncertainty conditional on fitted weights.",
        "bootstrap": "Primary analysis: 300 subject_id-cluster nonparametric resamples with full refitting of variable screening, imputation, scaling, encoding, and propensity model; sensitivity analyses: 1000 subject_id-cluster multiplier replicates conditional on fitted weights.",
        "treatment_window": "Half-open [landmark, landmark+6h) dexmedetomidine initiation interval; propofol discontinuation and raw dexmedetomidine coverage required at landmark+6h for transition. Continuation requires gap-free raw propofol coverage across the full 6-hour window. Later treatment changes are allowed in the primary point-strategy estimand.",
        "diagnosis_covariates": "Hospital-discharge ICD diagnosis domains excluded from the treatment model and balance estimand to prevent future-information leakage.",
        "safety_outcomes": "Post hoc safety analysis over 24 hours after follow-up origin: hemodynamic risks are estimated only for trials with >=2 valid post-origin measurements; new bradycardia requires >=2 distinct HR readings <50/min with no analogous baseline event; new hypotension requires >=2 distinct MAP readings <65 mm Hg with no analogous baseline event; severe thresholds are HR <40/min and MAP <55 mm Hg; vasopressor escalation is a new or additional agent class.",
        "source_parse_note": "Malformed labevents CSV records were skipped with DuckDB ignore_errors=true; selected target analytes were otherwise retained.",
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(effects.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
