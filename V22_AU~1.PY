"""V22 standalone aggregate treatment and selection-weight audit.
No individual-level records saved. Outputs are sensitivity analyses only.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

def output_model(data,exposure,covars,name):
    sub=data.dropna(subset=['heart_new','communityID',exposure]+covars).reset_index(drop=True)
    if not len(sub):return {'model':name,'term':exposure,'n':0,'events':0,'error':'empty'}
    X=sm.add_constant(sub[[exposure]+covars].astype(float),has_constant='add')
    try:
        fit=sm.GLM(sub.heart_new.astype(float),X,family=sm.families.Poisson()).fit(cov_type='cluster',cov_kwds={'groups':sub.communityID})
        b,se=fit.params[exposure],fit.bse[exposure]
        return {'model':name,'term':exposure,'n':len(sub),'events':int(sub.heart_new.sum()),'rr':np.exp(b),'ci_low':np.exp(b-1.96*se),'ci_high':np.exp(b+1.96*se),'p':fit.pvalues[exposure]}
    except Exception as e:return {'model':name,'term':exposure,'n':len(sub),'events':int(sub.heart_new.sum()),'error':str(e)[:180]}

def classify(d, prefix):
    cols=[prefix+str(i) for i in (1,2,3,4)]
    v=d[cols].apply(pd.to_numeric,errors='coerce')
    checked=pd.concat([(v[cols[i-1]]==i).rename(str(i)) for i in range(1,5)],axis=1)
    therapy=checked[['1','2','3']].any(axis=1);none=checked['4']
    status=np.select([therapy & none,therapy,none],['contradictory','treated','explicit_none'],default='unasked_or_missing')
    return pd.Series(status,index=d.index),pd.Series(np.where(status=='treated',1.,np.where(status=='explicit_none',0.,np.nan)),index=d.index)

def produce_v22(df,cohort,covars,out,robust_poisson):
    out=Path(out);c=cohort.copy()
    rows=[];modelrows=[]
    for label,prefix in [('diabetes','da014s'),('lung','da010_5_s')]:
        status,indicator=classify(c,prefix)
        c[label+'_tx_explicit']=indicator
        for group,sub in c.groupby(status,dropna=False):
            rows.append({'treatment':label,'status':group,'n':len(sub),'events':int(sub.heart_new.sum())})
        for exposure in ['high_a1c','cti_z']:
            z=c.copy();z['tx']=indicator
            modelrows.append(output_model(z,exposure,covars+['tx'],label+'_explicit_only'))
            if label=='diabetes':
                z['tx']=np.where(z.diabetes15.eq(0),0.,indicator)
                modelrows.append(output_model(z,exposure,covars+['tx'],label+'_non_diabetic_assumed_none'))
    for exposure in ['high_a1c','cti_z']:
        modelrows.append(output_model(c,exposure,covars+['diabetes_tx_explicit','lung_tx_explicit'],'both_explicit_only'))
    pd.DataFrame(rows).to_csv(out/'V22_treatment_response_status.csv',index=False)
    pd.DataFrame(modelrows).to_csv(out/'V22_treatment_models.csv',index=False)
    base=df[(df.age>=45)&df.lung15.eq(1)&df.heart15.eq(0)].copy()
    base['smoke_1.0']=base.smoking_status.eq(1).astype(float)
    base['smoke_2.0']=base.smoking_status.eq(2).astype(float)
    predictors=['age','female','bmi','drinking','married','hypertension15','smoke_1.0','smoke_2.0']
    target=base.dropna(subset=predictors+['communityID']).copy()
    target['outcome_observed']=target.heart_new.notna().astype(int)
    target['in_final']=target.ID.isin(c.ID).astype(int)
    m1=LogisticRegression(max_iter=4000).fit(target[predictors],target.outcome_observed)
    target['p1']=m1.predict_proba(target[predictors])[:,1]
    seen=target[target.outcome_observed==1].copy()
    m2=LogisticRegression(max_iter=4000).fit(seen[predictors],seen.in_final)
    seen['p2']=m2.predict_proba(seen[predictors])[:,1]
    selected=seen[seen.in_final==1].copy()
    ids=selected.ID
    chosen=c[c.ID.isin(ids)].copy()
    info=[];weighted_rows=[]
    for bound in [0.01,0.025,0.05]:
        p1=np.clip(selected.p1.to_numpy(),bound,1-bound);p2=np.clip(selected.p2.to_numpy(),bound,1-bound)
        weights=1/(p1*p2)
        for trunc in [None,(0.01,0.99)]:
            w=weights.copy()
            if trunc is not None:w=np.clip(w,*np.quantile(w,trunc))
            w=w/w.mean()
            wdf=chosen.merge(pd.DataFrame({'ID':ids,'audit_weight':w}),on='ID',validate='one_to_one')
            n_eff=float((w.sum()**2)/(w**2).sum())
            spec=f'prob_clip_{bound}_trim_{trunc}'
            info.append({'scenario':spec,'baseline_n':len(base),'target_n':len(target),'observed_n':len(seen),'selected_n':len(selected),'selected_events':int(wdf.heart_new.sum()),'effective_n':n_eff,'wt_min':float(w.min()),'wt_max':float(w.max()),'p1_min':float(p1.min()),'p2_min':float(p2.min()),'stage1_positive':int(target.outcome_observed.sum()),'stage2_positive':int(seen.in_final.sum())})
            for var in ['high_a1c','cti_z']:
                X=sm.add_constant(wdf[[var]+covars].astype(float),has_constant='add')
                try:
                    fit=sm.GEE(wdf.heart_new,X,groups=wdf.communityID,family=sm.families.Poisson(),weights=wdf.audit_weight).fit()
                    b,se=fit.params[var],fit.bse[var]
                    weighted_rows.append({'scenario':spec,'term':var,'n':len(wdf),'events':int(wdf.heart_new.sum()),'rr':np.exp(b),'ci_low':np.exp(b-1.96*se),'ci_high':np.exp(b+1.96*se),'p':fit.pvalues[var]})
                except Exception as e:weighted_rows.append({'scenario':spec,'term':var,'error':str(e)[:100]})
    pd.DataFrame(info).to_csv(out/'V22_IPSW_diagnostics.csv',index=False)
    pd.DataFrame(weighted_rows).to_csv(out/'V22_IPSW_sensitivity.csv',index=False)
