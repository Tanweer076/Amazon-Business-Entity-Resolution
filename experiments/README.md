# Experiments: how the pipeline got to 0.96

Each folder holds the notebook exactly as it was run, except that private Drive links were replaced with placeholders. `docs/running_log.md` records every run's outputs and how they were interpreted.

| Version | Folder | What changed | Evidence behind the change | Result |
|---|---|---|---|---|
| **v2** | `v2/` | Multi-view preprocessing (original script, Latin, core, compact, legal form, numbers). Country-partitioned TF-IDF blocking (top-50 ∪ name-only top-10, DF cap 0.5%). 49 features. LightGBM. Many-to-one resolution. | Exploration: 0 cross-country links; strictly many-to-one; token recall 95.6% (address) and 85.8% (name). Rule gate: none of the optional normalisation rules helped. | Holdout 0.950 (train density). **LB 0.933** (Colab, 197 min) |
| **v3** | `v3/` | Density matching (the training pool made as dense as the test pool). 6 house-number features. 4 name/address frequency features. 200k training entities. Saved test probabilities plus re-thresholding. | The test pool is 23% denser than the training pool. Errors showed dropped or off-by-one house numbers and generic names. More training entities helped (+0.003 on the mini-world). | Holdout 0.954 at test density. **LB 0.929** |
| v3.1 | `v3_1/` | Country treated as an open set of labels: no US/India allow-lists; reporting weights taken from the data; every test label is scored. | Problem statement: country is an open set and every test entity must appear. | Same predictions as v3 |
| **v3.2** | `../notebooks/` | The 4 frequency features removed (55 features). Everything else from v3.1 kept. | v3's frequency features were rates per 100k records of the scored population. The test population is 2–5× smaller than the training one, so the inputs shifted on test. The holdout (same population as training) could not see this. | Holdout 0.954. **LB 0.96** |

`v3/rethreshold_cell.py` is a standalone cell that re-thresholds saved test probabilities in seconds instead of rerunning the three-hour inference. `v3/results_v3.json` is the full v3 run log.
