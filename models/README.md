# Trained matcher: `matcher_lgb.txt`

This is the LightGBM binary classifier from the v3.2 run (public leaderboard 0.96). It is a text model file that `lightgbm.Booster` can load directly.

- **Trees:** 1,000.
- **Parameters:** 127 leaves, learning rate 0.08, `min_data_in_leaf` 200, feature and bagging fraction 0.8, L2 regularisation 1.0, seed 42.
- **Training data:** 10,584,961 candidate pairs (651,504 positive) from 200k training entities, retrieved from the density-matched full training pool.
- **Output:** the probability that a (Source 1, candidate) pair is the same business.
- **Decision rule (applied outside the model):** accept pairs with p ≥ **0.75**, give each Source 2/3 record to at most one Source 1 entity (the one with the highest probability), and keep at most 15 matches per entity.

The model was trained on an unnamed matrix, so the file lists its features as `Column_i`. The order is the notebook's `FEATS` list with `CONTEXT_FEATURES=False`. It was verified against the gain importances saved in `results/results_v3_2.json`, which match to within 4×10⁻⁸ relative difference.

```python
import lightgbm as lgb
booster = lgb.Booster(model_file="models/matcher_lgb.txt")
p = booster.predict(X)          # X: float32 array, columns in the order below (NaN allowed)
```

Features are computed by `pair_features()` in `notebooks/er_pipeline_v3_2.ipynb` (section 3.3).

| # | Model column | Feature |
|--:|---|---|
| 0 | `Column_0` | `n_core_ratio` |
| 1 | `Column_1` | `n_core_tset` |
| 2 | `Column_2` | `n_core_tsort` |
| 3 | `Column_3` | `n_core_partial` |
| 4 | `Column_4` | `n_core_jw` |
| 5 | `Column_5` | `n_latin_ratio` |
| 6 | `Column_6` | `n_latin_tset` |
| 7 | `Column_7` | `n_compact_ratio` |
| 8 | `Column_8` | `n_compact_partial` |
| 9 | `Column_9` | `n_light_ratio` |
| 10 | `Column_10` | `n_core_exact` |
| 11 | `Column_11` | `n_first_eq` |
| 12 | `Column_12` | `legal_same` |
| 13 | `Column_13` | `legal_conflict` |
| 14 | `Column_14` | `alias_best` |
| 15 | `Column_15` | `init_match` |
| 16 | `Column_16` | `a_ratio` |
| 17 | `Column_17` | `a_tset` |
| 18 | `Column_18` | `a_partial` |
| 19 | `Column_19` | `a_plain_tset` |
| 20 | `Column_20` | `num_tset` |
| 21 | `Column_21` | `num_exact` |
| 22 | `Column_22` | `q_has_num` |
| 23 | `Column_23` | `c_has_num` |
| 24 | `Column_24` | `postal_eq` |
| 25 | `Column_25` | `postal_conflict` |
| 26 | `Column_26` | `unit_eq` |
| 27 | `Column_27` | `unit_conflict` |
| 28 | `Column_28` | `q_ncomp` |
| 29 | `Column_29` | `c_ncomp` |
| 30 | `Column_30` | `c_addr_missing` |
| 31 | `Column_31` | `c_fixed` |
| 32 | `Column_32` | `q_deva` |
| 33 | `Column_33` | `c_deva` |
| 34 | `Column_34` | `cross_script` |
| 35 | `Column_35` | `q_ntok` |
| 36 | `Column_36` | `c_ntok` |
| 37 | `Column_37` | `len_ratio` |
| 38 | `Column_38` | `is_s3` |
| 39 | `Column_39` | `hn_both` |
| 40 | `Column_40` | `hn_eq` |
| 41 | `Column_41` | `hn_absdiff` |
| 42 | `Column_42` | `hn_reldiff` |
| 43 | `Column_43` | `hn_lev` |
| 44 | `Column_44` | `hn_trunc` |
| 45 | `Column_45` | `blk` |
| 46 | `Column_46` | `ncos` |
| 47 | `Column_47` | `acos` |
| 48 | `Column_48` | `rank` |
| 49 | `Column_49` | `in_name` |
| 50 | `Column_50` | `n_cand` |
| 51 | `Column_51` | `blk_rel` |
| 52 | `Column_52` | `blk_gap` |
| 53 | `Column_53` | `name_gap` |
| 54 | `Column_54` | `addr_gap` |
