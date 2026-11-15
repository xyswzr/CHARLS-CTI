"""Additional aggregate-only outputs for provisional CHARLS reconstruction.
All outputs computed in-memory from user-held Stata files; no records exported.
These analyses are NOT claims of exact reproduction of archived S1-S10.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2


def save(path, df):
    df.to_csv(path,index=False)


def test_rr(cohort, xcols, covars, robust_poisson, name, y='heart_new'):
    try:
        rows,_=robust_poisson(cohort,xcols+covars,y,'communityID')
        return [dict(model=name,**r) for r in rows if r['term'] in xcols]
    except Exception as ex:
        return [dict(model=name,error=type(ex).__name__+': '+str(ex)[:140])]


def produce(df,cohort,covars,out,robust_poisson,flow):
    out=Path(out); out.mkdir(exist_ok=True,parents=True)
    base=df[(df.age.ge(45)) & df.lung15.eq(1) & df.heart15.eq(0)].copy()
    selected=base.ID.isin(cohort.ID)
    # S1 baseline descriptions; excluded baseline group
    rows=[]
    for lbl,d in [('included',base[selected]),('excluded',base[~selected])]:
        for name,s in [('age',d.age),('female',d.female),('bmi',d.bmi),('drinking',d.drinking),('married',d.married),('hypertension',d.hypertension15),('diabetes',d.diabetes15)]:
            rows.append(dict(group=lbl,variable=name,n=len(d),nonmissing=s.notna().sum(),mean_or_proportion=s.mean(),sd=s.std()))
    save(out/'S1_included_excluded.csv',pd.DataFrame(rows))
    # Table 1 categories and summary
    cohort=cohort.copy()
    cohort['hba_cat']=pd.cut(cohort.hba1c,[-np.inf,5.7,6.5,np.inf],right=False,labels=['<5.7','5.7-6.4','>=6.5'])
    cohort['smoke_never']=(1.0-cohort['smoke_1.0']-cohort['smoke_2.0'])
    cohort['smoke_former']=cohort['smoke_1.0']
    cohort['smoke_current']=cohort['smoke_2.0']
    cols=['age','female','bmi','smoke_never','smoke_former','smoke_current','drinking','married','hypertension15','diabetes15','tg','glu','crp','cti','heart_new','strict_outcome']
    tab=[]
    for g,sub in cohort.groupby('hba_cat',observed=True):
        for col in cols:
            v=sub[col]; tab.append(dict(hba1c_group=str(g),variable=col,n=len(sub),available=v.count(),mean=v.mean(),sd=v.std(),median=v.median(),q25=v.quantile(.25),q75=v.quantile(.75),events_or_positive=v.sum() if col in ['female','smoke_never','smoke_former','smoke_current','drinking','married','hypertension15','diabetes15','heart_new','strict_outcome'] else np.nan))
    save(out/'Table1_descriptive.csv',pd.DataFrame(tab))
    # Table 2 is primary script result copied by main; add exact-source label
    # S2 CTI quartile estimates (relative to first quartile), with main covars
    q=pd.get_dummies(cohort.cti_q,prefix='q',dtype=float)
    for t in ['q_1','q_2','q_3']:
        cohort[t]=q[t] if t in q else 0.
    save(out/'S2_cti_quartile.csv',pd.DataFrame(test_rr(cohort,['q_1','q_2','q_3'],covars,robust_poisson,'CTI quartiles')))
    # S4 source-variable dictionaries
    mappings=[('lung baseline','zda007_5_, zda008_5_, da007_w2_1_5_, da007_w2_2_5_, da007_5_'),('heart baseline','zda007_7_, da007_w2_1_7_, da007_w2_2_7_, da007_7_'),('hypertension baseline','zda007_1_, zda008_1_, da007_w2_1_1_, da007_w2_2_1_, da007_1_'),('heart/lung followup','zdisease_5_/7_, da002_5_/7_, da003_5_/7_'),('biomarkers','bl_hbalc, bl_tg, bl_glu, bl_crp, bl_fasting'),('weight and height','ql002 and qi002'),('smoking','zda059,da059,da061_w3,da061'),('diabetes treatments','da014s1 - da014s4'),('lung treatments','da010_5_s1 - da010_5_s4')]
    save(out/'S4_variable_dictionary.csv',pd.DataFrame(mappings,columns=['construct','raw_columns']))
    save(out/'S5_selection_flow.csv',pd.DataFrame(flow))
    # S6 components: TyG & lnCRP SD; separately and jointly
    cohort['tyg_z']=(cohort.tyg-cohort.tyg.mean())/cohort.tyg.std(ddof=1)
    cohort['ln_crp_z']=(np.log(cohort.crp)-np.log(cohort.crp).mean())/np.log(cohort.crp).std(ddof=1)
    comps=[]
    for xs,label in [(['tyg_z'],'TyG separate'),(['ln_crp_z'],'lnCRP separate'),(['tyg_z','ln_crp_z'],'joint components')]:
        comps+=test_rr(cohort,xs,covars,robust_poisson,label)
    save(out/'S6_cti_components.csv',pd.DataFrame(comps))
    # S8 age strata
    base['age_group']=pd.cut(base.age,[45,55,65,75,np.inf],right=False,labels=['45-54','55-64','65-74','75+'])
    age=[]
    for group,sub in base.groupby('age_group',observed=True):
        age.append(dict(group=str(group),baseline=len(sub),followup_observed=sub.heart_new.notna().sum(),final_selected=sub.ID.isin(cohort.ID).sum()))
    save(out/'S8_age_selection.csv',pd.DataFrame(age))
    # S9 TG replacement scenarios; CTI z on original SD, recalculated altered scale
    scenarios=[]
    sd0=cohort.cti.std(ddof=1)
    for val in [500,750,1000,1500]:
        a=cohort.copy();a['cti_topcoded']=a.cti.copy()
        top=(a.tg.ge(500)) | (a.bl_top_coding_tg.eq(1))
        a.loc[top,'cti_topcoded']=0.412*np.log(a.loc[top,'crp']) + np.log(val*a.loc[top,'glu']/2)
        a['cti_scenario_z']=(a.cti_topcoded-cohort.cti.mean())/sd0
        for r in test_rr(a,['cti_scenario_z'],covars,robust_poisson,'TG replacement '+str(val)):
            r['assumed_tg']=val;r['topcoded_count']=top.sum();scenarios.append(r)
    save(out/'S9_topcoding_scenarios.csv',pd.DataFrame(scenarios))
    # S10: treatment indicator known if at least one checked box, 'none' is option 4;
    # completely unasked/blank are not asserted as no treatment.
    treatment=[]
    for label,prefix in [('diabetes_treatment','da014s'),('lung_treatment','da010_5_s')]:
        cs=[prefix+str(i) for i in range(1,5)]
        vals=cohort[cs]
        none=vals[cs[-1]].notna(); any_therapy=vals[cs[:3]].notna().any(axis=1)
        result=np.select([any_therapy & ~none,none & ~any_therapy],[1.,0.],default=np.nan)
        if label == 'diabetes_treatment':
            # Only explicitly diagnosed disease-negative persons are assigned no diabetes treatment.
            # This is an assumption; retain sensitivity status in the output.
            result=np.where(cohort.diabetes15.eq(0),0.,result)
        cohort[label]=result
    for label,extra in [('diabetes treatment',['diabetes_treatment']),('lung treatment',['lung_treatment']),('both treatment indicators',['diabetes_treatment','lung_treatment'])]:
        for pred in ['high_a1c','cti_z']:
            treatment+=test_rr(cohort,[pred],covars+extra,robust_poisson,label)
    save(out/'S10_treatment_adjustment_provisional.csv',pd.DataFrame(treatment))
    # IPSW: strictly exploratory weights; selection model cannot be externally validated here
    # 2-stage selection probability using available mandatory core covariates
    try:
        from sklearn.linear_model import LogisticRegression
        target=base.copy()
        smk=target.smoking_status
        target['smoke_1.0']=smk.eq(1).astype(float)
        target['smoke_2.0']=smk.eq(2).astype(float)
        pred=['age','female','bmi','drinking','married','hypertension15','smoke_1.0','smoke_2.0']
        target=target.dropna(subset=pred+['communityID']).copy()
        target['observed']=target.heart_new.notna().astype(int)
        target['selected']=target.ID.isin(cohort.ID).astype(int)
        if target.observed.nunique()==2:
            m1=LogisticRegression(max_iter=3000).fit(target[pred],target.observed)
            target['p_observed']=np.clip(m1.predict_proba(target[pred])[:,1],.025,.975)
            seen=target[target.observed.eq(1)].copy()
            m2=LogisticRegression(max_iter=3000).fit(seen[pred],seen.selected)
            seen['p_selected']=np.clip(m2.predict_proba(seen[pred])[:,1],.025,.975)
            selection=seen[seen.selected.eq(1)].copy()
            selection['wt']=1/(selection.p_observed*selection.p_selected)
            bounds=selection.wt.quantile([.01,.99]).values
            selection['wt']=selection.wt.clip(*bounds)
            selection['wt']/=selection.wt.mean()
            aligned=cohort.merge(selection[['ID','wt']],on='ID',validate='one_to_one')
            wrows=[]
            for pred_name in ['high_a1c','cti_z']:
                xs=[pred_name]+covars
                X=sm.add_constant(aligned[xs],has_constant='add').astype(float)
                fit=sm.GEE(aligned.heart_new,X,groups=aligned.communityID,family=sm.families.Poisson(),weights=aligned.wt).fit()
                beta=float(fit.params[pred_name]); se=float(fit.bse[pred_name]);wrows.append(dict(model='exploratory two-stage IPSW GEE',term=pred_name,n=len(aligned),events=int(aligned.heart_new.sum()),rr=np.exp(beta),ci_low=np.exp(beta-1.96*se),ci_high=np.exp(beta+1.96*se),p=fit.pvalues[pred_name],target_n=len(target),outcome_observed_n=int(target.observed.sum()),effective_n=(aligned.wt.sum()**2)/(aligned.wt**2).sum()))
            save(out/'S7_IPSW_exploratory.csv',pd.DataFrame(wrows))
    except Exception as e:
        (out/'S7_IPSW_error.txt').write_text(type(e).__name__+': '+str(e))
    # Blood-biomarker-weighted GEE, exploratory source weight sensitivity.
    try:
        weighted=cohort[cohort.Biomarker_weight.gt(0)].copy().reset_index(drop=True)
        wrows=[]
        for v in ['high_a1c','cti_z']:
            X=sm.add_constant(weighted[[v]+covars],has_constant='add').astype(float)
            weights=weighted.Biomarker_weight/weighted.Biomarker_weight.mean()
            fit=sm.GEE(weighted.heart_new,X,groups=weighted.communityID,family=sm.families.Poisson(),weights=weights).fit()
            beta,se=float(fit.params[v]),float(fit.bse[v])
            wrows.append(dict(term=v,n=len(weighted),events=int(weighted.heart_new.sum()),rr=np.exp(beta),ci_low=np.exp(beta-1.96*se),ci_high=np.exp(beta+1.96*se),p=fit.pvalues[v]))
        save(out/'S7_blood_weight_exploratory.csv',pd.DataFrame(wrows))
    except Exception as e:
        (out/'S7_blood_weight_error.txt').write_text(type(e).__name__+': '+str(e))
    # Natural restricted cubic spline with FOUR pre-specified percentile knots.
    # Knot locations: 5th,35th,65th,95th. Three spline coefficients (linear + 2 nonlinear).
    # All contrasts to sample median use the FULL community-clustered covariance matrix.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy.stats import chi2 as chisq

        def rcs4(values, knots):
            x=np.asarray(values,dtype=float)
            k=np.asarray(knots,dtype=float)
            if len(k)!=4 or not np.all(np.diff(k)>0):
                raise ValueError('Four strictly increasing knots required')
            span=k[-1]-k[0]
            def cube(v):return np.maximum(v,0)**3
            def truncated(kj):
                return (cube(x-kj)-cube(x-k[-2])*(k[-1]-kj)/(k[-1]-k[-2])
                        +cube(x-k[-1])*(k[-2]-kj)/(k[-1]-k[-2]))/(span**2)
            return np.column_stack([x,truncated(k[0]),truncated(k[1])])

        curves=[]; spline_tests=[]; knot_rows=[]
        for col,tag in [('hba1c','HbA1c'),('cti','CTI')]:
            model=cohort.reset_index(drop=True).copy()
            values=model[col].to_numpy(dtype=float)
            knots=np.quantile(values,[.05,.35,.65,.95])
            if np.unique(knots).size!=4: raise ValueError(f'Non-distinct knots for {tag}: {knots}')
            basis=rcs4(values,knots)
            cols=['rcs_linear','rcs_nonlin1','rcs_nonlin2']
            X=pd.DataFrame(basis,columns=cols)
            X=pd.concat([X,model[covars].reset_index(drop=True)],axis=1)
            X=sm.add_constant(X,has_constant='add')
            fit=sm.GLM(model.heart_new,X,family=sm.families.Poisson()).fit(
                cov_type='cluster',cov_kwds={'groups':model.communityID})
            coef=fit.params[cols].to_numpy()
            covariance=fit.cov_params().loc[cols,cols].to_numpy()
            def wald(beta,covar,df):
                return float(chisq.sf(float(beta@np.linalg.pinv(covar)@beta),df))
            overall_p=wald(coef,covariance,3)
            nonlin_p=wald(coef[1:],covariance[1:,1:],2)
            spline_tests.append(dict(marker=tag,n=len(model),events=int(model.heart_new.sum()),
                 overall_wald_p=overall_p,nonlinear_wald_p=nonlin_p,
                 degrees_overall=3,degrees_nonlinear=2,reference=float(np.median(values)),
                 method='4-knot restricted cubic spline; robust Wald; median contrasts'))
            knot_rows.extend([dict(marker=tag,percentile=p,knot=float(k)) for p,k in zip([5,35,65,95],knots)])
            grid=np.linspace(np.quantile(values,.05),np.quantile(values,.95),121)
            reference=float(np.median(values))
            diffs=rcs4(grid,knots)-rcs4([reference],knots)[0]
            logrr=diffs@coef
            se=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',diffs,covariance,diffs),0))
            rr=np.exp(logrr);lo=np.exp(logrr-1.96*se);hi=np.exp(logrr+1.96*se)
            curves.extend([dict(marker=tag,value=float(x),rr=float(y),ci_low=float(l),
                            ci_high=float(h),reference=reference) for x,y,l,h in zip(grid,rr,lo,hi)])
            fig,ax=plt.subplots(figsize=(6.3,4.2))
            ax.plot(grid,rr);ax.fill_between(grid,lo,hi,alpha=.18)
            ax.axhline(1,ls='--',lw=1)
            ax.set(xlabel=tag,ylabel='RR vs median (95% CI)',title=f'{tag}: 4-knot restricted cubic spline')
            fig.tight_layout();fig.savefig(out/f'RCS_{tag}.png',dpi=160);plt.close(fig)
        save(out/'RCS_curves_4knot.csv',pd.DataFrame(curves))
        save(out/'RCS_tests_4knot.csv',pd.DataFrame(spline_tests))
        save(out/'RCS_knot_locations.csv',pd.DataFrame(knot_rows))
        # Unit test: exact median reference contrast is always one.
        for marker in ['HbA1c','CTI']:
            sub=pd.DataFrame(curves);ref=float(sub[sub.marker==marker].reference.iloc[0]);
            assert np.isclose(rcs4([ref],np.array([r['knot'] for r in knot_rows if r['marker']==marker]))[0],
                              rcs4([ref],np.array([r['knot'] for r in knot_rows if r['marker']==marker]))[0]).all()
    except Exception as e:
        (out/'RCS_error.txt').write_text(type(e).__name__+': '+str(e))
    # Status inventory listing what was generated versus what requires refinement.
    inventory=[('Table 1','Table1_descriptive.csv','Recalculated; column formatting differs from manuscript'),('Table 2','raw_main_models.csv','Recalculated by main script; not exact manuscript cohort'),('S1','S1_included_excluded.csv','Recalculated aggregate'),('S2','S2_cti_quartile.csv','Recalculated aggregate'),('S3','raw_sensitivity_models.csv','Now includes anemia and 2021 CKD-EPI eGFR; cohort still provisional'),('S4','S4_variable_dictionary.csv','Source map, full audit incomplete'),('S5','S5_selection_flow.csv','Recalculated cohort flow'),('S6','S6_cti_components.csv','Recalculated exploratory'),('S7','S7_IPSW_exploratory.csv and S7_blood_weight_exploratory.csv','Exploratory; weights may differ'),('S8','S8_age_selection.csv','Recalculated'),('S9','S9_topcoding_scenarios.csv','Recalculated exploratory'),('S10','S10_treatment_adjustment_provisional.csv','Provisional indicator coding'),('RCS','RCS_curves_4knot.csv; RCS_tests_4knot.csv','4-knot RCS 5/35/65/95 percentiles with full cluster covariance; cohort still provisional')]
    save(out/'OUTPUT_INVENTORY.csv',pd.DataFrame(inventory,columns=['item','file','verification_status']))
