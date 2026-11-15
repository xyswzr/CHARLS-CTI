# CHARLS CTI — V22 cohort, V28 smoking-corrected analysis

**This package supersedes the previously supplied V22 GitHub package.** The V22 selection algorithm remains fixed (676 participants; 145 outcomes), but covariate smoking status was corrected against the official 2015 questionnaire. Current smoking is determined by DA061; DA059 reports ever smoked; DA061_W3 references the preceding interview.

Run locally with authorized CHARLS 2015 and 2020 raw ZIP archives (do not upload these to GitHub):

```bash
pip install -r requirements.txt
python src/charls_raw_reanalysis.py --data-dir /path/to/CHARLS_archives --out local_results
python check_reproduction.py --results local_results
```

Public `results/` contains aggregate outputs only. Raw and participant-level records must not be redistributed. See V28_SMOKING_CORRECTION.md and VERSION_DECISION.md. Other classification assumptions remain operational choices rather than externally adjudicated physician diagnoses.
