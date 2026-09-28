# Results of the v3.2 run

| File | Content |
|---|---|
| `results_v3_2.json` | The full machine-readable run log written by the notebook: configuration, rule-gate table, Phase A and Phase B metrics, blocking recall, feature importance, error examples, density matching, and test statistics per country and per threshold. |
| `validation_metrics.csv` | F0.5 (overall, India, US), precision, recall, singleton accuracy and predictions per entity for the mini-world, the full-pool validation set and the untouched holdout. |
| `holdout_threshold_curve.csv` | Holdout metrics at thresholds 0.70–0.90. |
| `blocking_recall.csv` | Candidates per entity, recall at 10/20/30/50, union recall, and the ceiling F0.5 of a perfect matcher. |
| `test_per_country.csv` | Test entities, pool size, candidate pairs, predicted matches per entity and minutes, per country. |
| `threshold_variants.csv` | Test prediction profile of each threshold variant written by the notebook. |
| `feature_importance_gain.csv` | LightGBM gain importance of all 55 features. |
| `density_matching.csv` | Training vs test pool density per country, and the distractor records added. |
| `leave_one_country_out.csv` | Transfer to an unseen country (US→India, India→US). |
| `switch_gate.csv` | Effect of each optional preprocessing rule, one at a time, on the mini-world. |

The validation and holdout numbers come from 20k-entity samples of the training data. The **public leaderboard score is 0.94**.
