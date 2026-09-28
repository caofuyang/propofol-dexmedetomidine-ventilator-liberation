#!/usr/bin/env python3
"""External validation of the propofol-to-dexmedetomidine landmark study in eICU."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
CACHE = WORK / "cache"
OUT = ROOT / "outputs" / "external_validation"


def data_root_from_env(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise SystemExit(
            f"Set {variable} to the extracted eICU-CRD 2.0 directory before running."
        )
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"{variable} is not a directory: {root}")
    return root


EICU = data_root_from_env("EICU_CRD_ROOT")
O2_CACHE = CACHE / "eicu_o2_device.parquet"
SEED = 20260924


def qp(path: Path) -> str:
    return str(path).replace("'", "''")


def copy_if_missing(con: duckdb.DuckDBPyConnection, path: Path, query: str) -> None:
    if path.exists():
        return
    print(f"Building {path.name} ...", flush=True)
    con.execute(f"copy ({query}) to '{qp(path)}' (format parquet,compression zstd)")
    print(f"Built {path.name}", flush=True)


def build_caches(con: duckdb.DuckDBPyConnection) -> dict[str, Path]:
    files = {
        "inf": CACHE / "eicu_infusions_selected.parquet",
        "patient": CACHE / "eicu_patient_static.parquet",
        "resp": CACHE / "eicu_resp_settings.parquet",
        "base": CACHE / "eicu_landmark_base.parquet",
        "vital": CACHE / "eicu_vitals_for_trials.parquet",
        "trials": CACHE / "eicu_landmark_trials.parquet",
    }
    terms = (
        "propofol|dexmedetomidine|precedex|norepinephrine|levophed|epinephrine|"
        "phenylephrine|vasopressin|dopamine|fentanyl|morphine|hydromorphone|"
        "remifentanil|midazolam|versed|ketamine"
    )
    copy_if_missing(
        con, files["inf"],
        f"""select cast(patientunitstayid as bigint) patientunitstayid,
          cast(infusionoffset as integer) infusionoffset,lower(drugname) drugname,
          try_cast(drugrate as double) drugrate
        from read_csv_auto('{qp(EICU / 'infusionDrug.csv.gz')}',all_varchar=true,strict_mode=false)
        where regexp_matches(lower(drugname),'{terms}') and try_cast(infusionoffset as integer) is not null""",
    )
    copy_if_missing(
        con, files["patient"],
        f"""with ap as (
          select cast(patientunitstayid as bigint) patientunitstayid,
            try_cast(apachescore as double) apachescore,
            try_cast(acutephysiologyscore as double) acutephysiologyscore
          from read_csv_auto('{qp(EICU / 'apachePatientResult.csv.gz')}',all_varchar=true,strict_mode=false)
          where apacheversion='IVa'
        )
        select cast(p.patientunitstayid as bigint) patientunitstayid,
          cast(p.hospitalid as integer) hospitalid,p.gender,
          case when p.age like '>%' then 90 else try_cast(p.age as double) end age,
          p.ethnicity,p.unittype,p.unitadmitsource,try_cast(p.admissionweight as double) admissionweight,
          try_cast(p.unitdischargeoffset as integer) unitdischargeoffset,
          lower(coalesce(p.unitdischargestatus,'')) unitdischargestatus,
          ap.apachescore,ap.acutephysiologyscore
        from read_csv_auto('{qp(EICU / 'patient.csv.gz')}',all_varchar=true,strict_mode=false) p
        left join ap on ap.patientunitstayid=cast(p.patientunitstayid as bigint)""",
    )
    copy_if_missing(
        con, files["resp"],
        f"""select cast(patientunitstayid as bigint) patientunitstayid,
          cast(respchartoffset as integer) chartoffset,respchartvaluelabel,
          try_cast(respchartvalue as double) setting_value
        from read_csv_auto('{qp(EICU / 'respiratoryCharting.csv.gz')}',all_varchar=true,strict_mode=false)
        where respchartvaluelabel in ('FiO2','FIO2 (%)','Set Fraction of Inspired Oxygen (FIO2)','PEEP','PEEP/CPAP')
          and try_cast(respchartvalue as double) is not null""",
    )
    if not files["base"].exists():
        print("Building eICU landmark base ...", flush=True)
        con.execute(
            f"""
            create or replace temp view inf as select * from read_parquet('{qp(files['inf'])}');
            create or replace temp view pat as select * from read_parquet('{qp(files['patient'])}');
            create or replace temp view o2 as
            select *,case
              when regexp_matches(device,'ventilator|endotracheal|(^|[^a-z])ett([^a-z]|$)') and not regexp_matches(device,'home ventilator') then 'invasive'
              when regexp_matches(device,'nasal cannula|room air|high flow|non-rebreather|simple mask|face mask|venturi|aerosol mask|bipap|bi-pap|^cpap$') then 'noninvasive'
              else 'other' end device_class
            from read_parquet('{qp(O2_CACHE)}');
            create or replace temp view sed as
            select patientunitstayid,infusionoffset,
              case when regexp_matches(drugname,'propofol') then 'propofol' else 'dex' end agent
            from inf where regexp_matches(drugname,'propofol|dexmedetomidine|precedex');
            create or replace temp view sf as
            select *,case when infusionoffset-lag(infusionoffset) over(
              partition by patientunitstayid,agent order by infusionoffset)>120
              or lag(infusionoffset) over(partition by patientunitstayid,agent order by infusionoffset) is null
              then 1 else 0 end new_episode from sed;
            create or replace temp view ep as
            select patientunitstayid,agent,min(infusionoffset) episode_start,max(infusionoffset) episode_last
            from (select *,sum(new_episode) over(partition by patientunitstayid,agent order by infusionoffset) episode_id from sf)
            group by patientunitstayid,agent,episode_id;
            create or replace temp view hospital_coverage as
            with prop as (select distinct patientunitstayid from sed where agent='propofol'),
            ox as (select distinct patientunitstayid from o2)
            select p.hospitalid,count(*) prop_stays,count(ox.patientunitstayid) o2_stays,
              count(ox.patientunitstayid)*1.0/count(*) o2_coverage
            from prop join pat p using(patientunitstayid) left join ox using(patientunitstayid)
            group by p.hospitalid;
            copy (
            select row_number() over() trial_id,p.patientunitstayid,p.episode_start prop_start,
              p.episode_last prop_last,h landmark_minute,p.episode_start+h landmark_offset,
              p.episode_start+h+360 grace_end,
              s.hospitalid,s.gender,s.age,s.ethnicity,s.unittype,s.unitadmitsource,
              s.admissionweight,s.unitdischargeoffset,s.unitdischargestatus,
              s.apachescore,s.acutephysiologyscore,hc.prop_stays,hc.o2_stays,hc.o2_coverage,
              case when hc.o2_coverage>=0.50 and hc.prop_stays>=20 then 1 else 0 end coverage_hospital,
              (select min(d.episode_start) from ep d where d.patientunitstayid=p.patientunitstayid and d.agent='dex'
                and d.episode_start between p.episode_start+h and p.episode_start+h+360
                and d.episode_start<=p.episode_last+120 and p.episode_last<=d.episode_start+360) dex_start,
              case
                when exists(select 1 from ep d where d.patientunitstayid=p.patientunitstayid and d.agent='dex'
                  and d.episode_start between p.episode_start+h and p.episode_start+h+360
                  and d.episode_start<=p.episode_last+120 and p.episode_last<=d.episode_start+360) then 1
                when p.episode_last>=p.episode_start+h+360
                  and not exists(select 1 from ep d where d.patientunitstayid=p.patientunitstayid and d.agent='dex'
                                 and d.episode_start<p.episode_start+h+360) then 0
                else null end treatment
            from ep p join pat s using(patientunitstayid) join hospital_coverage hc using(hospitalid)
              cross join range(360,2881,360) r(h)
            where p.agent='propofol' and p.episode_last>=p.episode_start+h
              and s.age>=18 and p.episode_start+h+360<s.unitdischargeoffset
              and not exists(select 1 from ep d where d.patientunitstayid=p.patientunitstayid and d.agent='dex'
                             and d.episode_start<p.episode_start+h)
              and exists(select 1 from o2 x where x.patientunitstayid=p.patientunitstayid and x.device_class='invasive'
                         and x.chartoffset between p.episode_start+h-360 and p.episode_start+h+60)
              and not exists(select 1 from o2 x where x.patientunitstayid=p.patientunitstayid and x.device_class='noninvasive'
                         and x.chartoffset between p.episode_start+h and p.episode_start+h+360)
              and not (s.unitdischargestatus='expired' and s.unitdischargeoffset<=p.episode_start+h+360)
            ) to '{qp(files['base'])}' (format parquet,compression zstd)
            """
        )
        print("Built eICU landmark base", flush=True)
    copy_if_missing(
        con, files["vital"],
        f"""select cast(v.patientunitstayid as bigint) patientunitstayid,
          cast(v.observationoffset as integer) observationoffset,
          try_cast(v.sao2 as double) sao2,try_cast(v.heartrate as double) heartrate,
          try_cast(v.respiration as double) respiration,try_cast(v.systemicmean as double) systemicmean
        from read_csv_auto('{qp(EICU / 'vitalPeriodic.csv.gz')}',all_varchar=true,strict_mode=false) v
        where exists(select 1 from read_parquet('{qp(files['base'])}') b
                     where b.patientunitstayid=cast(v.patientunitstayid as bigint))""",
    )
    if not files["trials"].exists():
        print("Building eICU covariate/outcome table ...", flush=True)
        con.execute(
            f"""
            create or replace temp view b as select * from read_parquet('{qp(files['base'])}') where treatment is not null;
            create or replace temp view inf as select * from read_parquet('{qp(files['inf'])}');
            create or replace temp view vit as select * from read_parquet('{qp(files['vital'])}');
            create or replace temp view resp as select * from read_parquet('{qp(files['resp'])}');
            create or replace temp view o2 as select *,case
              when regexp_matches(device,'ventilator|endotracheal|(^|[^a-z])ett([^a-z]|$)') and not regexp_matches(device,'home ventilator') then 'invasive'
              when regexp_matches(device,'nasal cannula|room air|high flow|non-rebreather|simple mask|face mask|venturi|aerosol mask|bipap|bi-pap|^cpap$') then 'noninvasive'
              else 'other' end device_class
            from read_parquet('{qp(O2_CACHE)}');
            create or replace temp view outcome as
            select b.*,(select min(x.chartoffset) from o2 x where x.patientunitstayid=b.patientunitstayid
              and x.device_class='noninvasive' and x.chartoffset>b.grace_end and x.chartoffset<=b.grace_end+2880
              and exists(select 1 from o2 i where i.patientunitstayid=x.patientunitstayid and i.device_class='invasive'
                         and i.chartoffset between x.chartoffset-720 and x.chartoffset)) proxy_extubation
            from b;
            copy (select o.*,
              case when proxy_extubation is not null
                and not exists(select 1 from o2 i where i.patientunitstayid=o.patientunitstayid and i.device_class='invasive'
                               and i.chartoffset>proxy_extubation and i.chartoffset<=proxy_extubation+2880)
                and not (o.unitdischargestatus='expired' and o.unitdischargeoffset>proxy_extubation
                         and o.unitdischargeoffset<=proxy_extubation+4320) then 1 else 0 end successful_extubation,
              (select median(v.systemicmean) from vit v where v.patientunitstayid=o.patientunitstayid
                and v.systemicmean between 20 and 200 and v.observationoffset>o.landmark_offset-360 and v.observationoffset<=o.landmark_offset) map_6h,
              (select median(v.heartrate) from vit v where v.patientunitstayid=o.patientunitstayid
                and v.heartrate between 20 and 250 and v.observationoffset>o.landmark_offset-360 and v.observationoffset<=o.landmark_offset) hr_6h,
              (select median(v.sao2) from vit v where v.patientunitstayid=o.patientunitstayid
                and v.sao2 between 40 and 100 and v.observationoffset>o.landmark_offset-360 and v.observationoffset<=o.landmark_offset) spo2_6h,
              (select arg_max(case when r.setting_value<=1 then r.setting_value*100 else r.setting_value end,r.chartoffset) from resp r
                where r.patientunitstayid=o.patientunitstayid and r.respchartvaluelabel in ('FiO2','FIO2 (%)','Set Fraction of Inspired Oxygen (FIO2)')
                  and r.setting_value between 0.2 and 100 and r.chartoffset>o.landmark_offset-360 and r.chartoffset<=o.landmark_offset) fio2_last,
              (select arg_max(r.setting_value,r.chartoffset) from resp r where r.patientunitstayid=o.patientunitstayid
                and r.respchartvaluelabel in ('PEEP','PEEP/CPAP') and r.setting_value between 0 and 30
                and r.chartoffset>o.landmark_offset-360 and r.chartoffset<=o.landmark_offset) peep_last,
              case when exists(select 1 from inf x where x.patientunitstayid=o.patientunitstayid
                and regexp_matches(x.drugname,'norepinephrine|levophed|epinephrine|phenylephrine|vasopressin|dopamine')
                and x.infusionoffset between o.landmark_offset-360 and o.landmark_offset) then 1 else 0 end vasopressor_6h,
              case when exists(select 1 from inf x where x.patientunitstayid=o.patientunitstayid
                and regexp_matches(x.drugname,'fentanyl|morphine|hydromorphone|remifentanil')
                and x.infusionoffset between o.landmark_offset-360 and o.landmark_offset) then 1 else 0 end opioid_6h,
              case when exists(select 1 from inf x where x.patientunitstayid=o.patientunitstayid
                and regexp_matches(x.drugname,'midazolam|versed')
                and x.infusionoffset between o.landmark_offset-360 and o.landmark_offset) then 1 else 0 end midazolam_6h
            from outcome o) to '{qp(files['trials'])}' (format parquet,compression zstd)
            """
        )
        print("Built eICU covariate/outcome table", flush=True)
    return files


NUM = ["age", "admissionweight", "apachescore", "acutephysiologyscore", "map_6h", "hr_6h", "spo2_6h", "fio2_last", "peep_last"]
BIN = ["vasopressor_6h", "opioid_6h", "midazolam_6h"]
CAT = ["landmark_factor", "gender", "unittype", "hospital_factor"]


def wmean(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[ok], weights=w[ok])) if ok.any() else math.nan


def smd(x: pd.Series, t: np.ndarray, w: np.ndarray) -> float:
    xv = pd.to_numeric(x, errors="coerce").to_numpy(float)
    xv = np.where(np.isfinite(xv), xv, np.nanmedian(xv))
    m1=wmean(xv[t],w[t]); m0=wmean(xv[~t],w[~t])
    v1=wmean((xv[t]-m1)**2,w[t]); v0=wmean((xv[~t]-m0)**2,w[~t])
    return (m1-m0)/math.sqrt(max((v1+v0)/2,1e-12))


def fit(df: pd.DataFrame, label: str) -> tuple[dict, pd.DataFrame]:
    use_num=[c for c in NUM if df[c].notna().mean()>=.20]
    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),("std",StandardScaler())]),use_num+BIN),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore",min_frequency=10))]),CAT),
    ])
    # Weak regularization is retained for numerical stability while allowing
    # sparse hospital/landmark indicators to achieve covariate balance.
    model=Pipeline([("pre",pre),("logit",LogisticRegression(C=10.0,max_iter=2000))])
    cols=use_num+BIN+CAT
    model.fit(df[cols],df.treatment)
    ps=np.clip(model.predict_proba(df[cols])[:,1],1e-5,1-1e-5)
    work=df.copy(); work["ps"]=ps; work["overlap_weight"]=np.where(work.treatment.eq(1),1-ps,ps)
    t=work.treatment.to_numpy(int)==1; w=work.overlap_weight.to_numpy(); y=work.successful_extubation.to_numpy(float)
    r0=wmean(y[~t],w[~t]); r1=wmean(y[t],w[t]); rd=r1-r0; rr=r1/r0 if r0>0 else math.nan
    stays,inv=np.unique(work.patientunitstayid.to_numpy(),return_inverse=True); rng=np.random.default_rng(SEED)
    boot=[]
    for _ in range(1000):
        bw=w*rng.exponential(1,size=len(stays))[inv]
        b0=wmean(y[~t],bw[~t]); b1=wmean(y[t],bw[t]); boot.append((b1-b0,b1/b0 if b0>0 else np.nan))
    boot=np.asarray(boot)
    balances=[]; one=np.ones(len(work))
    for col in NUM+BIN:
        balances.append((col,smd(work[col],t,one),smd(work[col],t,w)))
        if work[col].isna().any():
            z=work[col].isna().astype(float); balances.append((col+": missing",smd(z,t,one),smd(z,t,w)))
    for col in CAT:
        s=work[col].fillna("Missing").astype(str)
        for lev in sorted(s.unique()):
            z=s.eq(lev).astype(float); balances.append((f"{col}: {lev}",smd(z,t,one),smd(z,t,w)))
    bal=pd.DataFrame(balances,columns=["variable","smd_unweighted","smd_weighted"])
    result={
        "analysis":label,"trials":len(work),"unique_stays":work.patientunitstayid.nunique(),
        "treated_trials":int(t.sum()),"control_trials":int((~t).sum()),
        "events_treated":int(y[t].sum()),"events_control":int(y[~t].sum()),
        "weighted_risk_control":r0,"weighted_risk_treated":r1,"risk_difference":rd,
        "rd_ci_low":float(np.nanquantile(boot[:,0],.025)),"rd_ci_high":float(np.nanquantile(boot[:,0],.975)),
        "risk_ratio":rr,"rr_ci_low":float(np.nanquantile(boot[:,1],.025)),"rr_ci_high":float(np.nanquantile(boot[:,1],.975)),
        "ess_treated":float(w[t].sum()**2/(w[t]**2).sum()),"ess_control":float(w[~t].sum()**2/(w[~t]**2).sum()),
        "max_abs_smd_weighted":float(bal.smd_weighted.abs().max()),
    }
    return result,bal.sort_values("smd_weighted",key=lambda x:x.abs(),ascending=False)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(str(WORK/"eicu_external.duckdb")); con.execute("set threads=4"); con.execute("set preserve_insertion_order=false")
    files=build_caches(con)
    df=con.execute(f"select * from read_parquet('{qp(files['trials'])}')").fetchdf()
    df["treatment"]=df.treatment.astype(int); df["successful_extubation"]=df.successful_extubation.astype(int)
    df["landmark_factor"]=(df.landmark_minute/60).astype(int).astype(str)+" h"
    df["hospital_factor"]=df.hospitalid.astype(str)
    hospital_treated=df.loc[df.coverage_hospital.eq(1)].groupby("hospitalid").treatment.sum()
    support_hospitals=set(hospital_treated[hospital_treated.ge(5)].index.tolist())
    coverage=(df[["hospitalid","prop_stays","o2_stays","o2_coverage","coverage_hospital"]]
              .drop_duplicates().sort_values(["coverage_hospital","o2_coverage"],ascending=False))
    coverage.to_csv(OUT/"table1_hospital_coverage.csv",index=False)
    analyses=[("All hospitals with observable invasive device",df),
              ("Hospitals with >=50% O2-device coverage",df.loc[df.coverage_hospital.eq(1)].copy()),
              ("Common-support hospitals with >=5 treated trials",
               df.loc[df.coverage_hospital.eq(1) & df.hospitalid.isin(support_hospitals)].copy())]
    rows=[]; restricted_bal=None; common_bal=None
    for label,sub in analyses:
        if len(sub)>=200 and sub.treatment.sum()>=30:
            print(f"Fitting {label}: n={len(sub)}, treated={sub.treatment.sum()}",flush=True)
            result,bal=fit(sub,label); rows.append(result)
            if "50%" in label: restricted_bal=bal
            if "Common-support" in label: common_bal=bal
    effects=pd.DataFrame(rows); effects.to_csv(OUT/"table2_external_effects.csv",index=False)
    if restricted_bal is not None: restricted_bal.to_csv(OUT/"table3_restricted_balance.csv",index=False)
    if common_bal is not None: common_bal.to_csv(OUT/"table4_common_support_balance.csv",index=False)
    fig,ax=plt.subplots(figsize=(10.8,4.5))
    if not effects.empty:
        y=np.arange(len(effects)); pt=effects.risk_difference.to_numpy()*100; lo=effects.rd_ci_low.to_numpy()*100; hi=effects.rd_ci_high.to_numpy()*100
        ax.errorbar(pt,y,xerr=[pt-lo,hi-pt],fmt="o",capsize=3,color="#55a868"); ax.set_yticks(y,effects.analysis); ax.axvline(0,color="black",lw=.8)
    ax.set_xlabel("Proxy-extubation risk difference (percentage points)"); ax.set_title("eICU external validation")
    fig.tight_layout(); fig.savefig(OUT/"figure1_external_effects.png",dpi=220); plt.close(fig)
    metadata={"outcome":"O2 device transition proxy; no invasive redocumentation for 48 h and no death for 72 h",
              "hospital_restriction":"at least 20 propofol stays and O2-device observability >=50%",
              "random_seed":SEED,"warning":"Proxy outcome is not equivalent to explicit planned extubation in MIMIC-IV."}
    (OUT/"external_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(effects.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
