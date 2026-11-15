#!/usr/bin/env python3
import argparse, pandas as pd, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--results',default='local_results');a=p.parse_args();r=Path(a.results)
f=pd.read_csv(r/'raw_selection_flow.csv');m=pd.read_csv(r/'raw_main_models.csv');
expected=[1484,1182,825,705,703,676]
actual=f.n.astype(int).tolist();assert actual==expected,(actual,expected)
assert int(f.events_observed.iloc[-1])==145
for model,term,target in [('HbA1c binary','high_a1c',1.700308),('CTI per SD','cti_z',1.176610)]:
 z=m[(m.model==model)&(m.term==term)&(m.outcome=='heart_new')].iloc[0];assert abs(z.rr-target)<0.0005,(model,z.rr,target)
print('PASS: V28 corrected smoking coding; V22 cohort maintained; V28 adjusted models matched')
