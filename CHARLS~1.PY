#!/usr/bin/env python3
"""Reproducible CHARLS source-data analysis with transparent cohort diagnostics.

IMPORTANT: Questionnaire-based mappings below are *provisional*, and must be
validated against the original author's data dictionary and final manuscript.
Never claim exact reproduction solely because counts happen to match.

Input: official CHARLS2015r.zip and CHARLS2020r.zip, kept outside public GitHub.
Output: aggregate results only; no row-level data or respondent identifiers.
"""
from __future__ import annotations
import argparse
import io
import json
import math
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, chi2

ARCHIVED_R7_REFERENCE = {'eligible':1487, 'observed':1183, 'final':682, 'events':149}
BASELINE_CHECKPOINTS = {'eligible':1484, 'observed':1182, 'final':676, 'events':145}


def read_dta(archive: Path, year: int, module: str, columns: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as z:
        member = f'CHARLS{year}r/{module}.dta'
        if member not in z.namelist():
            raise FileNotFoundError(f'{member} missing from {archive}')
        df = pd.read_stata(io.BytesIO(z.read(member)), columns=columns,
                           convert_categoricals=False)
    if df.ID.duplicated().any():
        raise ValueError(f'Duplicate individual IDs in {year} {module}')
    return df


def yes(x):
    return x.eq(1)


def disease2015(df: pd.DataFrame, num: int, conservative: bool = True) -> pd.Series:
    """Three-state 2015 disease reconstruction, preserving unverified prior positives.

    Prior-wave affirmative flag and correction branch must be used together.
    A missing verification of a prior affirmative disease is UNKNOWN for
    eligibility (lung and heart), and is positive for hypertension/diabetes.
    """
    z = df[f'zda007_{num}_']
    verify = df[f'da007_w2_1_{num}_']
    since = df[f'da007_w2_2_{num}_']
    current = df[f'da007_{num}_']
    positive = (z.eq(1) & verify.eq(1)) | since.eq(1) | current.eq(1)
    corrected = z.eq(1) & verify.eq(2)
    negative = (z.isna() | corrected) & (since.eq(2) | current.eq(2) | verify.eq(1))
    # An unanswered correction after prior-positive is never considered negative.
    unknown_prior = z.eq(1) & verify.isna() & ~positive
    out = np.where(positive, 1., np.where(negative, 0., np.nan))
    if not conservative:
        out = np.where(unknown_prior, 1., out)
    return pd.Series(out, index=df.index)


def official_disease2015(df: pd.DataFrame, num: int) -> pd.Series:
    """2015 CHARLS Followup Questionnaire p.55, physician-report routes.

    The value of verification = 2 depends on whether the PRELOADED
    previous record is positive or negative.  An unanswered branch remains
    unknown. This audit deliberately does not fold self-reported knowledge
    ZDA008 into physician diagnosis, because that changes the estimand.
    """
    prior = df[f'zda007_{num}_'].eq(1)
    verify = df[f'da007_w2_1_{num}_']
    new_diagnosis = df[f'da007_w2_2_{num}_']
    current = df[f'da007_{num}_']
    positive = (current.eq(1) | (prior & verify.eq(1)) |
                (~prior & verify.eq(2)) | new_diagnosis.eq(1))
    negative = (current.eq(2) | (prior & verify.eq(2) & new_diagnosis.eq(2)) |
                (~prior & verify.eq(1) & new_diagnosis.eq(2)))
    conflict = positive & negative
    return pd.Series(np.select([positive & ~conflict, negative & ~conflict],
                               [1., 0.], default=np.nan), index=df.index)


def disease2020(df: pd.DataFrame, num: int) -> pd.Series:
    """2020 better/worse/same retains disease; 'never had' corrects history.

    Explicit 2020 new diagnosis yes/no is honored; skipped/unknown branches
    are not assumed to be negative without questionnaire support.
    """
    prior = df[f'zdisease_{num}_']
    verify = df[f'da002_{num}_']
    newly = df[f'da003_{num}_']
    positive = (prior.eq(1) & verify.isin([1, 2, 3])) | newly.eq(1)
    negative = (prior.eq(1) & verify.eq(99)) | (prior.isna() & newly.eq(2))
    return pd.Series(np.where(positive, 1., np.where(negative, 0., np.nan)), index=df.index)

def robust_poisson(data: pd.DataFrame, xcols: list[str], outcome: str, cluster: str,
                   weights: str | None = None):
    m = data[[outcome, cluster]+xcols+([weights] if weights else [])].dropna().copy()
    if len(m) < 30: return [], None
    X = sm.add_constant(m[xcols].astype(float), has_constant='add')
    fit = sm.GLM(m[outcome].astype(float), X, family=sm.families.Poisson(),
                 freq_weights=m[weights].astype(float) if weights else None).fit(
                 cov_type='cluster', cov_kwds={'groups': m[cluster]})
    rows=[]
    for k in xcols:
        beta=float(fit.params[k]);se=float(fit.bse[k])
        rows.append(dict(outcome=outcome, term=k, n=len(m), events=int(m[outcome].sum()),
                         rr=float(np.exp(beta)),ci_low=float(np.exp(beta-1.96*se)),
                         ci_high=float(np.exp(beta+1.96*se)),p=float(fit.pvalues[k])))
    return rows,fit


def main(args):
    a15=Path(args.data_dir)/'CHARLS2015r.zip'
    a20=Path(args.data_dir)/'CHARLS2020r.zip'
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    h15_cols=['ID','householdID','communityID']+[f'{v}{i}_' for i in [1,3,5,7]
          for v in ['zda007_','da007_','da007_w2_1_','da007_w2_2_']]+[
              'zda059','da059','da061_w3','da061','da067',
              'zda008_1_','zda008_5_',
              'da014s1','da014s2','da014s3','da014s4',
              'da010_5_s1','da010_5_s2','da010_5_s3','da010_5_s4']
    h15=read_dta(a15,2015,'Health_Status_and_Functioning',h15_cols)
    demo=read_dta(a15,2015,'Demographic_Background',[
        'ID','ba004_w3_1','ba004_w3_2','ba002_1','ba002_2','ba000_w2_3','be001'])
    sample15=read_dta(a15,2015,'Sample_Infor',['ID','iyear','imonth'])
    blood=read_dta(a15,2015,'Blood',[
        'ID','bl_fasting','bl_hbalc','bl_tg','bl_glu','bl_crp','bl_top_coding_tg','Blood_weight','bl_hgb','bl_crea'])
    bio=read_dta(a15,2015,'Biomarker',['ID','qi002','ql002'])
    h20_cols=['ID']+[f'{v}{i}_' for i in [5,7] for v in ['zdisease_','da002_','da003_']]
    h20=read_dta(a20,2020,'Health_Status_and_Functioning',h20_cols)
    s20=read_dta(a20,2020,'Sample_Infor',['ID','died'])
    df=(h15.merge(demo,on='ID',how='left',validate='one_to_one')
          .merge(sample15,on='ID',how='left',validate='one_to_one')
          .merge(blood,on='ID',how='left',validate='one_to_one')
          .merge(bio,on='ID',how='left',validate='one_to_one')
          .merge(h20,on='ID',how='left',validate='one_to_one',suffixes=('','_20'))
          .merge(s20,on='ID',how='left',validate='one_to_one'))
    birthyear = df.ba004_w3_1.combine_first(df.ba002_1)
    birthmonth = df.ba004_w3_2.combine_first(df.ba002_2)
    df['age'] = pd.to_numeric(df.iyear, errors='coerce') - pd.to_numeric(birthyear, errors='coerce') - (pd.to_numeric(df.imonth,errors='coerce') < pd.to_numeric(birthmonth,errors='coerce')).astype(float)
    df['female']=np.where(df.ba000_w2_3.isin([1,2]), df.ba000_w2_3.eq(2).astype(float),np.nan)
    # The questionnaire distinguishes prior/current smoking pathways.
    # Where current status is absent we leave it unknown rather than invent it.
    # 2015 questionnaire p.70: DA059 is EVER smoked, NOT current smoking.
    # DA061 = current state: 1 still, 2 quit, 3 never.
    # DA061_W3 is LAST WAVE state only and MUST NOT override 2015 DA061.
    # ZDA059 is prior-ever history, never a current smoking response.
    current_state = pd.to_numeric(df.da061, errors='coerce')
    explicit_never = df.da059.eq(2)
    smoking_conflict = explicit_never & current_state.isin([1, 2])
    df['smoking_status'] = np.select([
        current_state.eq(1) & ~smoking_conflict,
        current_state.eq(2) & ~smoking_conflict,
        (current_state.eq(3) | (explicit_never & current_state.isna())) & ~smoking_conflict
    ], [2.,1.,0.],default=np.nan)
    # Save original V22 classification only for aggregate auditing.
    legacy_current = df.da059.eq(1)
    legacy_ever = df.zda059.eq(1) | df.da061_w3.eq(1) | df.da061.eq(1)
    df['smoking_status_v22_legacy'] = np.where(legacy_current,2., np.where(legacy_ever,1.,
        np.where(df.zda059.eq(2) | df.da059.eq(2),0.,np.nan)))
    df['drinking']=np.where(df.da067.isin([1,2,3]),df.da067.isin([1,2]).astype(float),np.nan)
    df['married']=np.where(df.be001.isin([1,2,3,4,5,6,7]),df.be001.isin([1,2]).astype(float),np.nan)
    # Routing evidence table: ZDA008 informs questionnaire preload, not doctor diagnosis.
    routing=[]
    for disease,number in [('hypertension',1),('lung',5)]:
        physician=df[f'zda007_{number}_'].eq(1)
        awareness=df[f'zda008_{number}_'].eq(1)
        verify=df[f'da007_w2_1_{number}_']
        for name,mask in [('physician_only',physician & ~awareness),
                          ('awareness_only',~physician & awareness),
                          ('both',physician & awareness),
                          ('neither_flag',~physician & ~awareness)]:
            for code in [1,2,'other_or_missing']:
                selection=(verify.eq(code) if isinstance(code,int) else ~verify.isin([1,2])) & mask
                routing.append({'disease':disease,'preload_source':name,'verification_code':code,
                                'n':int(selection.sum())})
    pd.DataFrame(routing).to_csv(out/'preload_routing_aggregate.csv',index=False)
    df['Biomarker_weight']=pd.to_numeric(df['Blood_weight'],errors='coerce')
    df['lung15']=official_disease2015(df,5)
    df['heart15']=official_disease2015(df,7)
    df['heart20']=disease2020(df,7)
    df['lung20']=disease2020(df,5)
    ht=pd.to_numeric(df.qi002,errors='coerce');wt=pd.to_numeric(df.ql002,errors='coerce')
    df['bmi']=wt/(ht/100)**2
    df.loc[(~ht.between(100,220))|(~wt.between(20,300)),'bmi']=np.nan
    df['hba1c']=pd.to_numeric(df.bl_hbalc,errors='coerce')
    df['tg']=pd.to_numeric(df.bl_tg,errors='coerce')
    df['glu']=pd.to_numeric(df.bl_glu,errors='coerce')
    df['crp']=pd.to_numeric(df.bl_crp,errors='coerce')
    # WHO sex-specific hemoglobin thresholds (g/dL) and 2021 CKD-EPI creatinine equation.
    # These are sensitivity definitions, not assumptions about unavailable medication data.
    df['hemoglobin_gdl']=pd.to_numeric(df.bl_hgb,errors='coerce')
    df['creatinine_mgdl']=pd.to_numeric(df.bl_crea,errors='coerce')
    df['anemia']=np.where(df.hemoglobin_gdl.gt(0) & df.female.notna(),
                          np.where(df.female.eq(1),df.hemoglobin_gdl.lt(12),df.hemoglobin_gdl.lt(13)).astype(float),np.nan)
    female=df.female.eq(1)
    kappa=np.where(female,0.7,0.9)
    alpha=np.where(female,-0.241,-0.302)
    creat=df.creatinine_mgdl.to_numpy(dtype=float)
    ratio=creat/kappa
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        egfr=142*np.minimum(ratio,1)**alpha*np.maximum(ratio,1)**(-1.2)*0.9938**df.age.to_numpy(dtype=float)*np.where(female,1.012,1.)
    df['egfr_2021']=np.where((creat>0)&df.age.notna()&df.female.notna(),egfr,np.nan)
    df['fasting']=df.bl_fasting.eq(1)
    df['tyg']=np.log(df.tg*df.glu/2)
    # CTI definition needs confirmation against final manuscript methods.
    df['cti']=0.412*np.log(df.crp)+df.tyg
    df.loc[(df.crp<=0)|(df.tg<=0)|(df.glu<=0),'cti']=np.nan
    df['high_a1c']=np.where(df.hba1c.notna(),(df.hba1c>=6.5).astype(float),np.nan)
    df['hba_middle']=np.where(df.hba1c.notna(),((df.hba1c>=5.7)&(df.hba1c<6.5)).astype(float),np.nan)
    df['hba_high_vs_low']=np.where(df.hba1c.notna(),(df.hba1c>=6.5).astype(float),np.nan)
    df['diabetes15']=disease2015(df,3,conservative=False)
    df['hypertension15']=official_disease2015(df,1)
    df.loc[df.hypertension15.isna() & df.zda007_1_.eq(1), 'hypertension15']=1.
    # No dyslipidemia in core model
    df['heart_new'] = np.where(df.heart20.isin([0,1]),df.heart20,np.nan)
    df['strict_outcome']=np.where(df.heart_new==1,np.where(df.lung20==1,1.,np.where(df.lung20==0,0.,np.nan)),
                                  np.where(df.heart_new==0,0.,np.nan))
    flow=[]
    def stage(name,d):
        flow.append({'stage':name,'n':int(len(d)),'events_observed':int(d.heart_new.sum(skipna=True))})
    cohort=df.loc[(df.age.ge(45)&df.lung15.eq(1)&df.heart15.eq(0))].copy();stage('2015 aged >=45, lung disease, no baseline heart disease',cohort)
    cohort=cohort.loc[cohort.heart_new.notna()].copy();stage('2020 heart-disease response observed',cohort)
    cohort=cohort.loc[cohort.hba1c.notna()].copy();stage('HbA1c observed',cohort)
    cohort=cohort.loc[cohort.fasting].copy();stage('Fasting blood draw',cohort)
    cohort=cohort.loc[cohort.cti.notna()].copy();stage('Positive CTI components',cohort)
    # Audit only aggregated transitions in the 703 pre-complete-case cohort.
    smoke_map={0.:'never',1.:'former',2.:'current'}
    tmp=cohort[['smoking_status_v22_legacy','smoking_status','heart_new','da059','da061','da061_w3','zda059']].copy()
    tmp['old']=tmp.smoking_status_v22_legacy.map(smoke_map).fillna('unknown')
    tmp['new']=tmp.smoking_status.map(smoke_map).fillna('unknown')
    tmp.groupby(['old','new'],dropna=False).agg(n=('heart_new','size'),events=('heart_new','sum')).reset_index().to_csv(out/'V28_smoking_transition.csv',index=False)
    for field in ['da059','da061','da061_w3','zda059']:
        d=tmp.copy(); d['value']=d[field].astype('string').fillna('NA')
        d.groupby(['value','old','new'],dropna=False).agg(n=('heart_new','size'),events=('heart_new','sum')).reset_index().assign(field=field).to_csv(out/f'V28_{field}_transition.csv',index=False)
    covars=['age','female','bmi','drinking','married','hypertension15']
    # Prevent missing smoking from being silently classified as never-smoker.
    # This is a conservative diagnostic change; original smoking branching still requires independent verification.
    smoking_missing_before=int(cohort.smoking_status.isna().sum())
    cohort=cohort.loc[cohort.smoking_status.notna()].copy()
    cohort=pd.get_dummies(cohort,columns=['smoking_status'],prefix='smoke',dtype=float)
    for col in ['smoke_1.0','smoke_2.0']:
        if col not in cohort: cohort[col]=0.
    covars += ['smoke_1.0','smoke_2.0']
    audit_columns = ['age','female','bmi','drinking','married','hypertension15','communityID']
    pd.DataFrame([{'variable':x,'missing':int(cohort[x].isna().sum()),'nonmissing':int(cohort[x].notna().sum())} for x in audit_columns]).to_csv(out/'covariate_missingness_before_completecase.csv', index=False)
    # Privacy-preserving aggregate diagnostics: no ID exported.
    import itertools
    audit_cov = covars + ['communityID']
    flags=pd.DataFrame({v:cohort[v].isna() for v in audit_cov},index=cohort.index)
    flags['missing_total']=flags.sum(axis=1)
    summary=[]
    for v in audit_cov:
        s=cohort.loc[flags[v]]
        summary.append(dict(variable=v,missing_n=int(len(s)),events=int(s.heart_new.sum()),exclusively_missing=int((flags[v]&flags.missing_total.eq(1)).sum())))
    pd.DataFrame(summary).to_csv(out/'v6_missing_covariates.csv',index=False)
    pattern=flags[audit_cov].astype(int).astype(str).agg(''.join,axis=1)
    pd.DataFrame({'pattern':pattern,'event':cohort.heart_new}).groupby('pattern').agg(n=('event','size'),events=('event','sum')).reset_index().sort_values('n',ascending=False).to_csv(out/'v6_missingness_patterns.csv',index=False)
    candidate=[]
    for v in ['age','female','bmi','smoking_status','drinking','married','hypertension15']:
        if v in cohort:
            dat=cohort[v].value_counts(dropna=False)
            candidate.extend([dict(field=v,value=str(k),n=int(n)) for k,n in dat.items()])
    pd.DataFrame(candidate).to_csv(out/'v6_covariate_values.csv',index=False)
    # Validation summary of fixed changes, no row-level detail.
    pd.DataFrame([{'n_before_completecase':len(cohort),'events_before_completecase':int(cohort.heart_new.sum()),'n_after':int((flags.missing_total==0).sum()),'events_after':int(cohort.loc[flags.missing_total==0,'heart_new'].sum()),'n_missing_any':int((flags.missing_total>0).sum())}]).to_csv(out/'v6_final_gate.csv',index=False)
    # Investigate missing hypertension and invalid BMI with only grouped counts.
    hyp_fields=['zda007_1_','da007_w2_1_1_','da007_w2_2_1_','da007_1_']
    hyp=cohort.loc[cohort.hypertension15.isna()].copy()
    for col in hyp_fields:
        hyp[col]=hyp[col].astype('string').fillna('NA')
    hyp.groupby(hyp_fields,dropna=False).agg(n=('heart_new','size'),events=('heart_new','sum')).reset_index().to_csv(out/'v6_hypertension_unknown_patterns.csv',index=False)
    ht=pd.to_numeric(cohort.qi002,errors='coerce'); wt=pd.to_numeric(cohort.ql002,errors='coerce')
    reason=np.select([ht.isna(),~ht.between(100,220),wt.isna(),~wt.between(20,300)],['height_missing','height_invalid','weight_missing','weight_invalid'],default='valid')
    pd.DataFrame({'reason':reason,'event':cohort.heart_new}).groupby('reason').agg(n=('event','size'),events=('event','sum')).reset_index().to_csv(out/'v6_bmi_reasons.csv',index=False)
    pd.DataFrame({'height':ht.where(cohort.bmi.isna()).astype('string').fillna('NA'),'weight':wt.where(cohort.bmi.isna()).astype('string').fillna('NA'),'event':cohort.heart_new}).loc[cohort.bmi.isna()].groupby(['height','weight']).agg(n=('event','size'),events=('event','sum')).reset_index().to_csv(out/'v6_bmi_invalid_values.csv',index=False)
    # Exploratory scenarios only: these change assumptions for previously unclassified hypertension.
    # Never interpret either replacement as officially confirmed disease status.
    initial_missing_hyp = int(cohort.hypertension15.isna().sum())
    initial_missing_hyp_events = int(cohort.loc[cohort.hypertension15.isna(), 'heart_new'].sum())
    if args.hypertension_scenario == 'unknown_as_negative':
        cohort['hypertension15'] = cohort['hypertension15'].fillna(0.)
    elif args.hypertension_scenario == 'unknown_as_positive':
        cohort['hypertension15'] = cohort['hypertension15'].fillna(1.)
    pd.DataFrame([{'scenario':args.hypertension_scenario,'hypertension_unknown_n':initial_missing_hyp,'hypertension_unknown_events':initial_missing_hyp_events}]).to_csv(out/'v10_scenario_metadata.csv',index=False)
    cohort=cohort.dropna(subset=covars+['communityID']).copy();stage('Complete chosen covariates',cohort)
    cohort['hba_z']=(cohort.hba1c-cohort.hba1c.mean())/cohort.hba1c.std(ddof=1)
    cohort['cti_z']=(cohort.cti-cohort.cti.mean())/cohort.cti.std(ddof=1)
    cohort['cti_q']=pd.qcut(cohort.cti,4,labels=False,duplicates='drop')
    flow_df=pd.DataFrame(flow);flow_df.to_csv(out/'raw_selection_flow.csv',index=False)
    results=[]
    covariate_definitions={'age':'2015 interview year/month versus birth year/month','female':'2015 sex code 2',
          'bmi':'2015 measured weight / measured height squared','smoking_status':'2015 DA061 still/quit/never, DA059=2 never when DA061 missing; ZA and W3 historical only',
          'drinking':'2015 da067 in (1,2)','married':'2015 be001 in (1,2)',
          'hypertension15':'2015 disease item 1'}
    if len(cohort)>30 and cohort.heart_new.nunique()==2:
        specifications=[('HbA1c three-category',['hba_middle','hba_high_vs_low']),('HbA1c binary',['high_a1c']),('HbA1c per SD',['hba_z']),
            ('CTI per SD',['cti_z']),('Joint binary + CTI',['high_a1c','cti_z']),
            ('Same-scale joint',['hba_z','cti_z'])]
        for y in ['heart_new','strict_outcome']:
            for name,predictors in specifications:
                try:
                    rr,fit=robust_poisson(cohort,predictors+covars,y,'communityID')
                    results.extend([dict(model=name,**r) for r in rr])
                    if name=='Same-scale joint' and fit is not None:
                        delta=fit.params['hba_z']-fit.params['cti_z']
                        cov=fit.cov_params(); var=cov.loc['hba_z','hba_z']+cov.loc['cti_z','cti_z']-2*cov.loc['hba_z','cti_z']
                        results.append(dict(model='Wald contrast',outcome=y,term='hba_z minus cti_z',
                                            n=len(cohort),events=int(cohort[y].sum()),rr=np.nan,
                                            ci_low=np.nan,ci_high=np.nan,p=float(chi2.sf(delta*delta/var,1))))
                except Exception as e:
                    results.append(dict(model=name,outcome=y,term='MODEL_ERROR: '+str(e)[:160]))
    # Sensitivity restrictions are evaluated in the reconstructed cohort, NOT the archived cohort.
    # Standardization remains fixed to the unrestricted reconstructed cohort.
    restrictions={
        'full_reconstructed_cohort':pd.Series(True,index=cohort.index),
        'exclude_diagnosed_diabetes':cohort.diabetes15.eq(0),
        'CRP_at_most_10_mg_L':cohort.crp.le(10),
        'TG_below_500_mg_dL':cohort.tg.lt(500),
        'no_anemia':cohort.anemia.eq(0),
        'eGFR_at_least_60':cohort.egfr_2021.ge(60),
    }
    sensitivity=[]
    for label,mask in restrictions.items():
        subset=cohort.loc[mask].copy()
        for predictor in ['high_a1c','cti_z']:
            try:
                rr,_=robust_poisson(subset,[predictor]+covars,'heart_new','communityID')
                for r in rr:
                    if r['term']==predictor:sensitivity.append(dict(restriction=label,**r))
            except Exception as exc:
                sensitivity.append(dict(restriction=label,term=predictor,error=str(exc)[:200]))
    pd.DataFrame(sensitivity).to_csv(out/'raw_sensitivity_models.csv',index=False)
    # Distinguish the archived manuscript values from actual calculated outputs.
    pd.DataFrame(results).to_csv(out/'raw_main_models.csv',index=False)
    # Extended analysis is diagnostic only; preserve baseline sample and outcomes exactly.
    try:
        from extended import produce
        produce(df,cohort,covars,out,robust_poisson,flow)
        from v22_audit import produce_v22
        produce_v22(df,cohort,covars,out,robust_poisson)
    except Exception as exc:
        (out/'extended_analysis_error.txt').write_text(repr(exc))
        print('WARNING: extended analyses incomplete:',repr(exc))
    # No confidential participant-level information is written.
    checkpoints={r['stage']:r['n'] for r in flow}
    actual={'eligible':flow[0]['n'],'observed':flow[1]['n'],'final':flow[-1]['n'],
            'events':flow[-1]['events_observed']}
    audit={'smoking_missing_prior_to_completecase':smoking_missing_before,'reference_from_archived_R7':ARCHIVED_R7_REFERENCE,'frozen_V22_reference':BASELINE_CHECKPOINTS,'actual_from_raw_script':actual,
           'matches':{k:(actual[k]==v) for k,v in BASELINE_CHECKPOINTS.items()},
           'method_warnings':['2020 da002 never-had is code 99, verified against Stata value labels; 2015 disease-branch reconstruction remains under audit.',
               'CTI formula and primary core covariates follow Manuscript Methods; sensitivity analyses are not all implemented.',
               'If counts differ, do not alter the source data or thresholds to force agreement.',
               'Anemia (<12 women, <13 men g/dL) and 2021 CKD-EPI eGFR (>=60) were added as transparent sensitivity definitions.'],
           'covariate_definitions':covariate_definitions}
    (out/'reproducibility_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding='utf-8')
    print(flow_df.to_string(index=False));print(json.dumps(audit['matches'],indent=2))
    if not all(audit['matches'].values()):
        print('WARNING: raw analysis DOES NOT match all archived checkpoints; see reproducibility_audit.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True,help='Directory containing CHARLS2015r.zip and CHARLS2020r.zip')
    parser.add_argument('--out',default='local_results',help='Output directory (do not push individual-level files)')
    parser.add_argument('--hypertension-scenario', choices=['exclude_unknown','unknown_as_negative','unknown_as_positive'], default='exclude_unknown')
    main(parser.parse_args())
