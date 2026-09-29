# Business Entity Resolution — Running Log

**Goal:** understand the data, then build and validate a matching pipeline for ML Challenge 2026 (Business Entity Resolution).
**Environment:** Google Colab, one cell at a time; outputs recorded here with their interpretation.
**Data source:** the official challenge package (not redistributed in this repository).
**Last updated:** 28 Sep 2026 — final model v3.2, public leaderboard 0.96.

---

# Part 1 — Exploratory data analysis

## Cell 1 — Download and inventory
Download succeeded. macOS junk (`__MACOSX/`, `._*`, `.DS_Store`) is ignorable. Real content under `<hash>_student_resource/student_resource/`:

| File | Size | Role |
|---|---|---|
| dataset/train/train_source1.tsv | 210 MB | train S1 (reference) |
| dataset/train/train_source2.tsv | 489 MB | train S2 |
| dataset/train/train_source3.tsv | 504 MB | train S3 |
| dataset/train/train_ground_truth.tsv | 127 MB | train labels |
| dataset/test/test_source1.tsv | 175 MB | test S1 |
| dataset/test/test_source2.tsv | 509 MB | test S2 |
| dataset/test/test_source3.tsv | 506 MB | test S3 |
| README.md / Documentation_template.md / utils/validate_submission.py | small | task, write-up template, validator |

## Cell 2 — Task definition (README, template, validator)
- For each S1 record (deduplicated reference), find all matching S2/S3 records; an S1 may match 0, 1 or many.
- Source files: `entity_id` (prefix S1-/S2-/S3-), `business_name`, `business_address`, `country`. Train = US + India; **test adds France** (open-set country).
- GT: `source1_entity_id`, `matched_entity_ids` (comma list, empty for singletons). All files are TSV (commas inside fields).
- Outputs: `matching_results.tsv` (scored) and `candidate_pairs.tsv` (the exact candidate set fed to the model; matches must be a subset). One row per test S1.
- Metric: **macro F0.5** per S1 (precision weighted 2×), singletons included (empty prediction = 1.0, any match = 0.0).
- Constraints: final model MIT/Apache, ≤ 8B params; no external data, APIs or geocoding.
- Validator docstring: test set ≈ 1.7M S1 entities.

## Cell 3 — Header peek
All files genuinely tab-separated with documented columns. Noise observed: Devanagari names (`राम मार्केटिंग प्राइवेट लिमिटेड`), French accents, leading junk (`-- `, `<< `, `B+ `), typos (COIMATORE, Léarning, JEN ZAY), domains as names (wilfordhancock.com), legal-suffix variety (Inc/LLC/Pvt Ltd/Sarl/प्राइवेट लिमिटेड), S2 all-caps addresses, reordered addresses (state first), full vs abbreviated states, empty addresses. entity_id numeric part is variable length.

## Cell 4 — Row counts

| Split | S1 | S2 | S3 | GT |
|---|--:|--:|--:|--:|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | 2,206,821 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | — |

S2/S3 are ~2.3× S1. Naive all-pairs ≈ 2.3×10¹³ comparisons → blocking mandatory; memory discipline needed on Colab.

## Cell 5 — Ground-truth distribution
- 0 duplicate S1 rows. **Singletons 123,247 (5.58%)**; 94.42% have ≥1 match.
- 7,638,365 matched ids: S2 48.36% / S3 51.64%. Mean 3.46 per S1 (3.67 among matched), **max 11**.
- Distribution: 0:123,247 · 1:119,157 · 2:375,212 · 3:530,841 · 4:484,115 · 5:321,957 · 6:164,868 · 7:63,968 · 8:18,680 · 9:4,205 · 10:534 · 11:37.
- Implication: an all-empty prediction scores ≈0.056 — the score is earned by finding matches; singletons still reward caution.

## Cell 6 — Integrity
train_source1 ids unique, all `S1-`; the GT S1 id set is **identical** to the train_source1 id set.

## Cell 7 — Cardinality and orphans
- Every matched id appears in exactly one S1 list (max multiplicity 1) → **strictly many-to-one**; disjoint clusters.
- Orphans (never matched): S2 1,340,997 (26.64%), S3 1,340,857 (25.37%); every matched id exists in its file.

## Cell 8 — Country, missingness, malformed rows
- 0 malformed rows in all six files. `business_name` never empty. `business_address` empty only in S2/S3 (train ~3.3%, test ~2.7%); never both empty.

| File | US | India | France |
|---|--:|--:|--:|
| train S1/S2/S3 | ~59.9–60.0% | ~40.0–40.1% | — |
| test S1 | 38.27% | 46.75% | 14.98% |
| test S2/S3 | ~38.3% | ~47.3% | ~14.4% |

Train→test covariate shift; France unseen and ~15% of test.

## Cell 9 — Are matches within-country? (41,671 sampled matched S1, 153,075 pairs)
**100.000% same-country, 0 cross-country** → country is a safe hard partition for blocking.

## Cell 10 — Separability, true vs random same-country pairs

| Feature | TRUE mean | TRUE p10 | RANDOM mean | RANDOM p90 |
|---|--:|--:|--:|--:|
| name token Jaccard | 0.617 | 0.000 | 0.029 | 0.143 |
| name trigram Jaccard | 0.622 | 0.172 | 0.026 | 0.097 |
| address token Jaccard | 0.598 | 0.250 | 0.015 | 0.067 |
| exact name (casefold) | 0.108 | — | 0.000 | 0.000 |

Strong separation, but ≥10% of true pairs share no name token (cross-script / heavy noise); exact names rare.
(Preparation: `\w+` tokens on casefolded text; trigrams on casefolded text without spaces; exact = casefold+strip equality.)

## Cell 11 — Blocking-key feasibility (pair recall on the Cell 9 sample)

| Key | Pair recall |
|---|--:|
| shared address token | 95.61% |
| shared name trigram | 91.54% |
| shared name token | 85.77% |
| name prefix 4 / first name token / prefix 6 | 76.02% / 72.69% / 71.81% |
| shared postal code | 4.98% |
| country only | 100% |

Postal codes present in India 1.0% / US 11.3% of addresses → not a blocker (can still be an optional feature).

*Interim EDA Word report was produced from Cells 1–8.*

---

# Part 2 — Exploratory pipeline experiments (India, full train pool)

## Cells A / A2 / A3 — Normaliser
- v1 (NFKC + unidecode + casefold + punctuation→space + legal suffixes + state map). Two bugs found on samples: country-blind state map (US `TN` → tamil nadu) and romanised Devanagari suffixes not stripped (`praaivett limittedd`).
- A2: country-aware US/India state maps + expanded suffix list → verified (TN→tennessee, DL→delhi, suffixes dropped, French `sci` dropped).
- A3: restored the normaliser functions after a variable-name clash in C2.

## Cell B — Inverted-index token blocker (4,133,346 India S2+S3; 950,421 tokens; 294 tokens with DF > 0.5% dropped)
Pair recall: name-only 74.27%, address-only 92.70%, **union 98.66%**; 0 queries without candidates; but ~27,837 candidates per query (median 25,754, p90 50,299, max 124,464) — unusable without ranking. 288 s.

## Cell C — IDF-sum ranking
recall@20/50/100/200/500 = 85.87 / 88.36 / 89.88 / 91.32 / 92.90% — length-biased.

## Cell C2 — Cosine ranking variants (recall@100 / @200 / @500)
full cosine 92.44 / 93.17 / 95.09 · **weighted sum of channel cosines 92.97 / 94.09 / 95.33** · two-channel split-K union 89.81 / 91.76 / 93.88. Plateau ~95%; pure-Python scoring took 650 s.

## Cell D — Character n-gram name blocker (char_wb 3–4, vocab 200,147)
name recall@50/100/200 = 59.20 / 64.23 / 66.90% — weaker than name tokens → char n-grams belong in the matcher, not the blocker.

## Cells E / E2 — Labelled pairs from the locked blocker (4,727 queries)
E's query-normalisation bug dropped recall to 83.99%. **E2 (candidate-normalised, address-dominated blend): recall@200 94.09%**, 944,760 pairs, 16,320 positives (1.73%). (Cell F, the first matcher, was not run — superseded by the notebook.)

---

# Part 3 — Design decisions

- **Preprocessing log adopted as the trusted specification** (built on executed EDA batches A/B/C/D/E/F/G). Base proposals implemented directly; additional views stored as columns for the matcher; check-before-enabling items become on/off switches.
- **Run-once, multi-view preprocessing:** raw fields are kept; views are added (light/original-script, Latin, core, compact, legal forms, numeric, flags). Models choose views; switch changes produce a new preprocessing version.
- **Blocking vs scoring preprocessing:** blocking collapses variants (recall), the matcher keeps distinctions as evidence (precision), e.g. `Green & Sons` vs `Green & Co` → same core name, different legal forms.
- **Validation first:** split by S1 cluster, singletons included, stratified by country × match-count bucket; evaluation end-to-end against the full pool; per-country F0.5 plus test-mix weighting (India 0.5499 / US 0.4501); leave-one-country-out as a France proxy; an untouched holdout.
- **Mini-world** for fast iteration: cluster sampling (whole clusters + orphans at the same rate), never per-file random sampling (that would fake singletons).

## Cell V (revised) — split output
train 2,006,821 / val 100,000 / holdout 100,000 / mini_val 20,000 (nested in val). Identical mix in all four: US 59.98 / India 40.02; buckets b0 5.58, b1 5.40, b2-3 41.06, b4-5 36.53, b6+ 11.43. Positive pairs: val 346,037, holdout 346,188, mini_val 69,300. Eval clusters claimed by a train S1: **0**. Saved `splits_v1.tsv` (+json). Drive copy declined (not needed).

## Cell M (revised) — mini-world output
220,000 S1 (20k mini-val + 200k train; frac 0.0997); S2 502,386 (orphans 133,707), S3 526,354 (orphans 132,979); 1,248,740 records. Fidelity: singletons 5.64% (full 5.58), mean matches 3.464 (3.461), orphans 26.61 / 25.26% (26.64 / 25.37), country ~60/40 everywhere.

## Cell R — train/ and val/ folders
`mini_v1/train/` and `mini_v1/val/` each with `<split>_source1/2/3.tsv` and ground truth; S2/S3 is the same shared pool in both folders (val queries must search the whole pool, as test does). Zipped as `mini_v1_train.zip` and `mini_v1_val.zip` for the user's Drive folder.

---

# Part 4 — End-to-end pipeline notebook (`er_pipeline.ipynb`)

## What it does
| Section | Content |
|---|---|
| 0 | install + config (all switches and parameters in one dict) |
| 1 | downloads the user's Drive folder (either zip layout), optional splits file |
| 2 | preprocessing per the log: `name_light`, `name_latin` (Devanagari → IAST + schwa deletion), `name_core`, `name_compact`, `name_legal`, `name_first`, `addr_latin` (country-aware state + street abbreviations), `addr_num` (zero-stripped), `addr_empty`, `deva`; regression gallery printed; Parquet output, all cores |
| 3 | blocker, 39 pairwise features, LightGBM, many-to-one resolution, exact macro-F0.5, threshold search, blocking report, error examples |
| A | mini-world: automatic frequency-cap choice (A.0), train 100k / validate 20k (A.1), leave-one-country-out (A.2), optional ablations (A.3) |
| B | full data: preprocess train pool + test + holdout sample (B.1); retrain on full-pool candidates and calibrate the threshold on mini-val vs full pool, holdout check (B.2); stream the test set per country incl. France (B.3); official validator (B.4) |
| 4 | README / requirements / filled Documentation.md; zip; downloads |

## Key choices and the evidence behind them
- **Country partition** — Cell 9 (0 cross-country links).
- **Blocking = IDF-weighted blend of name and address cosine (candidate-normalised), top-K_MAIN=50 ∪ name-only top-K_NAME=10** — reproduces C2/E2's best ranking (94.09% @200 on the full India pool); the name channel targets empty-address targets (3.8–4.9% of matched records per EDA batch C) and garbled addresses.
- **Postal codes not used for blocking** — Cell 11 (5% recall; 1% / 11% presence).
- **Devanagari transliteration via IAST with schwa deletion** — `राम मार्केटिंग प्राइवेट लिमिटेड` → `ram marketing praivet limited` (plain unidecode gave `raam maarketting praaivett limittedd`); candra-o (ॉ) and conjunct-final vowels handled.
- **Features on several views side by side** (full, core, compact, original script, numeric, legal-form agreement/conflict) — blocking-vs-scoring principle; gap from the unrun Cell F closed.
- **No country feature** — the model must apply unchanged to France.
- **Many-to-one resolution** — Cell 7.
- **Threshold tuned for test-mix-weighted macro-F0.5** — Cell 8 shift.
- **Final model trained on full-pool candidates** — mini-pool negatives are 10× easier; K and threshold must be recalibrated at full scale (decision recorded with the mini-world).

## Testing done before delivery (sandbox: 2 CPUs, 7 GB RAM)
- Synthetic challenge world with the same formats and structure: clusters with the real match-count distribution, ~26% orphans, Devanagari/mixed-script names, typos, junk prefixes, suffix swaps, abbreviations, reordered/missing/`NULL` addresses, zero-padded numbers, France only in test; zips built with the Cell V/M/R logic; the **real validator script**.
- Full runs (small world and a 1/9-scale world: 200k test S1, 935k test pool, 1.17M train pool): **all cells pass, validator PASS**, both zip layouts. Executed the actual `.ipynb` in a Jupyter kernel: 21 code cells, 0 errors.
- Measured throughput: preprocessing ≈ 49k rows/s; features ≈ 100k pairs/s; peak RSS 2.3 GB at 1/9 scale.
- Blocking benchmark at real density (4.1M pool, 950k tokens, DF cap 20,666): scipy matmul + sort ≈ 10 ms/query (too slow, 2–3 h for test) → switched to `sparse_dot_topn` (identical top-50, 2 threads) ≈ 3.5 ms/query at ~2× real density → est. 1.5–2 ms/query real.
- Bugs found and fixed through testing: DF-cap floor made mini vs full caps non-proportional (auto-cap picked 0.1% and lost 14 recall points on the larger pool) → floor 5, candidates {0.2%, 0.5%}; empty vocabulary crash when every token is common → graceful fallback; challenge-folder search could walk a mounted Drive → depth-limited; full-GT load for holdout (~1 GB) → filtered load; error-example lookup → streamed.
- Synthetic scores are **not** indicative of real performance (synthetic vocabulary is tiny; matcher is near-perfect there).

## Expected runtime on free Colab (estimates)
Phase A ~15 min · Phase B preprocessing ~10 min + final model ~20–25 min · test ~45–75 min (blocking dominates) · total ≈ 1.5–2 h. Peak memory estimate 5–6 GB (Colab ~12.7 GB).

## What to paste back after running
A.0 cap table, A.1 blocking report + threshold table + feature importance + error examples, A.2 LOCO lines, B.2 blocking report + metrics + holdout line, B.3 per-country test lines, B.4 validator result, and `results.json`.

## Open items / next levers
1. Cross-script (Devanagari ↔ Latin) names remain the largest known gap — a multilingual character model or learned transliteration (≤8B, permissive licence).
2. K_MAIN 50 → 100 if blocking recall is the limit (ablation provided; costs ~2× test candidates).
3. Per-country thresholds once France behaviour is known; test-time resolution already uses all test S1.
4. Run the switch ablations (A.3) if time allows; switches whose removal does not hurt can be simplified.

---

# Part 5 — Notebook v2: alignment with the Preprocessing Plan (P01–P17)

**Trigger:** user asked whether everything in `Amazon_Preprocessing_Plan.docx` is included; the Drive link is a plain folder (TSVs), not zips.

**Finding on v1:** base items were implemented, but several plan items were missing or partial, and four "check before enabling" rules (FTFY, PLACEHOLDER_EMPTY, STATE_EXPAND, STREET_ABBREV) plus IAST transliteration were ON by default — contrary to the plan.

**v2 changes:**
- Input: downloads a plain Drive folder of TSVs (`train/`, `val/` or flat) — zips still accepted; `splits_v1.tsv` optional.
- Check-before-enabling rules start OFF (plan setting). **Phase A gate (plan Q12):** each rule is switched on alone on the mini-world; adopted only if the test-mix-weighted macro-F0.5 gain on the same 20k val S1 is > 0 and > 2 × paired SE; adopted rules are re-checked together. Gate uses 30k training queries per run.
- New views/flags: `addr_missing` (0 present · 1 raw empty · 2 placeholder · 3 emptied by normalisation), `name_empty`, `fixed` (ftfy changed text), `addr_plain` (unmapped address), `addr_postal` (zeros kept), `addr_unit`, `addr_ncomp`, `name_alias` (d.b.a./a.k.a./t/a), `name_init`; ordinals added to the US street map; control characters removed; dangling "and" trimmed from core names.
- New features (49 total): alias similarity, initials vs short names, unmapped-address token-set, postal and unit agreement/conflict, component counts, missing-address type, ftfy flag.
- P14: preprocessing version now hashes switches + `PP_CODE_VERSION` + dictionaries; every preprocessed file's row count is verified against its TSV.
- Tests: plain-folder run end-to-end incl. gate → validator PASS; notebook executed in a Jupyter kernel, 21 code cells, 0 errors. (On synthetic data the gate adopts nothing — no mojibake/placeholders there; real data will decide.)
- Runtime: Phase A ≈ 40 min with the gate (set `RUN_SWITCH_GATE=False` to skip; conservative settings are then used).

## Plan compliance (v2)

| Plan | Status |
|---|---|
| P01 preserve records, strings | Done — raw fields kept, strings/leading zeros, row counts verified per file |
| P02 missingness | Done — missing type 0/1/2/3 + name-empty flag; placeholder conversion gated, whole field only, embedded NULL kept |
| P03 encoding repair | Gated (starts OFF); only on damaged-looking strings; changed flag stored |
| P04 Unicode / invisible | Done — NFC original-script view keeps Indic marks and ZWJ/ZWNJ; NFKC only in the derived Latin view; control chars removed (not classified) |
| P05 case | Done — casefold; country only trimmed, never inferred from addresses |
| P06 domains | Done — full text kept; compact view drops domain endings only when a domain pattern is present |
| P07 punctuation / numbers | Done — separators → space, digits and alphanumerics kept (12/3 ≠ 123); `&` kept in the original-script view |
| P08 accent folding | Done — inside `name_latin` (one derived view with P09); original script kept separately |
| P09 transliteration | Gated — unidecode (plan baseline) vs IAST + schwa deletion |
| P10 abbreviations | Gated (state, street, ordinals); unmapped `addr_plain` kept; `st` never mapped in US/India |
| P11 reduced name | Done — explicit legal list, full name kept, legal forms kept as evidence, empty equality never a match |
| P12 address signals | Done — postal candidates (zeros kept), unit IDs, component count; none required for a match |
| P13 token views | Done — order kept; sets / sorted tokens only inside scorers |
| P14 versioning | Done — switches + code version + dictionaries; regression gallery; results.json; pinned requirements |
| P15 numeric view | Done as a feature only (zeros stripped in `addr_num`, kept in `addr_postal`); slashes still split into separate numbers |
| P16 alias / compact / initials | Done — no spell correction, no guessed expansions |
| P17 retrieval filtering | Done — per-country, per-field DF on the exact pool; never edits stored text; cap chosen by sweep |

**Not covered (documentation-only or later):** Q01/Q11/Q13/Q17 (recovering earlier batch code/output lines), email-specific handling (Q05, `@` treated as punctuation), classification of invisible characters (P04), French rules cannot be validated (no labels — plan's stated limit).

---

# Part 6 — First real-data run (notebook v2, Colab)

## Cell 1.2 — inputs
Plain Drive folder read correctly: train S1 200,000 · val S1 20,000 · S2 502,386 · S3 526,354 · GT 200,000 / 20,000 · splits_v1.tsv found (holdout check enabled). All counts match Cells V/M/R.

## Cell A.0 — preprocessing-rule gate (mini-world, 30k training queries, paired on the same 20k val S1)
Mini preprocessing ≈ 50 s per run (1.25M rows, all verified). Pools: India 412,255 (name vocab 89,868 / addr 123,788 after cap), US 616,485 (148,023 / 115,422).

**Baseline (plan conservative settings):** F0.5 0.9712 · test-mix 0.9687 · India 0.9613 · US 0.9777 · mean P 0.9852 · mean R 0.9427 · singleton accuracy 0.9615 · threshold 0.75 · 3.29 predictions/S1.
Blocking: 50.9 candidates/S1 · recall@10 0.947 · @20 0.959 · @50 0.9688 · union 0.9690 · name-only-channel hits 15 · all links found for 91.0% of S1 · ceiling F0.5 (perfect matcher) 0.9882.
Top features (gain): blk, rank, num_tset, a_tset, n_latin_ratio, a_partial, n_core_partial, acos, n_compact_ratio.

| Rule switched on alone | Test-mix F0.5 | Δ vs baseline | paired SE | Blocking recall | Adopted |
|---|--:|--:|--:|--:|---|
| FTFY | 0.9687 | +0.0000 | 0.0003 | 0.9690 | no |
| PLACEHOLDER_EMPTY | 0.9687 | +0.0000 | 0.0000 | 0.9690 | no (identical run → no whole-field placeholders in mini data) |
| STATE_EXPAND | 0.9689 | +0.0002 | 0.0004 | 0.9688 | no |
| STREET_ABBREV | 0.9683 | −0.0004 | 0.0004 | 0.9692 | no |
| DEVA_TRANSLIT=indic | 0.9693 | +0.0006 | 0.0004 | 0.9699 | no (1.5 SE; India F0.5 0.9613 → 0.9627, recall +0.0009) |

**Interpretation:** none of the plan's check-before-enabling rules gives a gain beyond noise, so the plan's conservative settings stay (preprocessing version pp_17b494c2). The matcher already absorbs most surface variation through multi-view features. IAST transliteration is the only rule with a consistent positive direction (India F0.5, blocking recall, links found) but it did not reach the 2-SE bar; it stays a candidate for a later full-pool check. Street abbreviations slightly hurt.

**Error budget (mini):** ceiling 0.9882 vs achieved 0.9712 → matcher loses ≈0.017, blocking ≈0.012. About 3.9% of singletons receive a false match at the chosen threshold. Many-to-one resolution adds only +0.0001 on val (competition among 20k val S1 only; larger effect expected on the full test set).

## Cell A.1 — blocking frequency cap
| cap | recall union | cand/S1 | ms/query (mini) |
|---|--:|--:|--:|
| 0.002 | 0.942 | 49.4 | 0.084 |
| 0.005 | 0.969 | 50.9 | 0.175 |
Chosen 0.005: the cheaper cap loses 2.7 recall points (well beyond the 0.5-point tolerance). Full-pool cost ≈ 10× mini ≈ 1.75 ms/query → ≈ 50 min of blocking for 1.73M test S1.

**Timing observed:** gate run ≈ 3.3 min each (whole gate ≈ 20 min); features ≈ 70k pairs/s; LightGBM ≈ 65 s for 1.37M pairs.

## Cell A.2 — final mini run (100k training queries, plan conservative preprocessing)
Training pairs 5.09M (India 2.07M, US 3.02M); LightGBM 411 trees on 4.58M pairs (302,036 positive), 329 s. Phase A total 30.1 min.

**Result (threshold 0.75):** F0.5 **0.9737** · test-mix **0.9715** · India 0.9647 · US 0.9797 · mean P 0.9868 · mean R 0.9456 · singleton accuracy 0.9696 · 3.30 predictions/S1. Without many-to-one resolution 0.9736.
vs the 30k-query gate baseline: +0.0025 F0.5 / +0.0028 test-mix → more training queries help.
Threshold curve is flat between 0.65 and 0.80 (test-mix 0.9711–0.9715) → threshold choice is robust.
Error budget: ceiling 0.9882 → matcher gap 0.0145, blocking gap 0.0118.
Top features: blk, rank, num_tset, a_tset, n_latin_ratio, n_core_partial, a_partial, acos, n_compact_ratio, n_core_jw.

**Error examples (mini-val) — patterns:**
- *Near-identical name, different house number* → false merges (Nexic Mfa LLC 233 vs nexicyn mfa llc 246C Short Bark Rd; Dode Chase Gulf Lab 4430 vs 4443 Homan Ave) — yet a true match also differs by a digit typo (Interstate Innovative 12268 vs 22268, missed at p=0.64). Number disagreement is ambiguous evidence.
- *Same address, unrelated name* → both false merges (Thrissur Com vs Sonalika Consultancy; Marc Technologies vs Chalapathi; Josepha Wang Pacific vs WestpearlCom) and true matches (Delk Cloud of Phoenix vs "Nylazeph - 3935100673" same unit; Design Trading vs Irijax Co). Shared premises vs DBA-style unrelated names are indistinguishable from the pair alone.
- *Generic/duplicated name, empty candidate address* → both false merges (Shiva Team Private; Industrial Trading Private Limited) and misses (Surgical Optimal; Emerald Associates). Name-only evidence is weak when the same name exists for several S1.
- Embedded placeholder seen in real data: `08 AVE H, <NULL>, LEWISTOOWN, IL` (also digit loss 708→08 and typo) — the angle-bracket form is not in the placeholder list; embedded placeholders are kept anyway (P02).

**Improvement ideas for v3 (after the first submission):** label-free context features computed per country — how many S1 share the same core name (q_name_dup), how many S1 share the same address (q_addr_dup, business-centre signal), how many pool records share the candidate's core name; number-similarity beyond token sets (edit distance of leading house numbers). Test-time many-to-one resolution (all 1.73M S1 compete) should already remove part of the "same address / same name, belongs to another S1" false merges that val cannot see.

## Cell A.3 — leave-one-country-out (France proxy; threshold chosen on the source country)
| Train → score | F0.5 on target | In-country model | Drop | Threshold carried over |
|---|--:|--:|--:|--:|
| US → India | 0.9248 | 0.9647 | −0.040 | 0.725 |
| India → US | 0.9648 | 0.9797 | −0.015 | 0.60 |

Models: US-only 324 trees on 2.72M pairs; India-only 281 trees on 1.86M pairs.
**Interpretation:** transfer to an unseen country works but costs 1.5–4 points; the asymmetry says India-specific patterns (Devanagari names, landmark-style addresses) cannot be learned from US data, while US patterns are mostly covered by India data. France (Latin script, structured addresses, French vocabulary) is closer to the US case, so a France drop of roughly 1.5–3 points is a reasonable expectation; at 15% of test that costs ≈0.002–0.005 of the overall score. Carried-over thresholds stayed in the flat region, so a single global threshold is acceptable. Note: French street abbreviations (R./Rue, Av./Avenue) are inactive because the gate kept STREET_ABBREV off on US/India evidence; their effect on France cannot be validated (no labels).

# Phase B (full data)

## Cell B.1 — challenge download + full preprocessing (pp_17b494c2)
Download ≈ 1.5 min. Every row count verified and identical to the EDA counts: train S2 5,034,616 (211 s) · train S3 5,285,603 (220 s) · test S1 1,732,544 (70 s) · test S2 4,887,273 (215 s) · test S3 5,082,316 (223 s) · holdout sample 20,000 S1. Total ≈ 17 min for 22M records (≈ 24k rows/s on Colab — half the sandbox rate; Colab CPUs are slower). Mini S1 files reused from Phase A (same version).

## Cell B.2 — final matcher on full-pool candidates; threshold calibration; holdout
Index: India pool 4,133,346 (name vocab 520,625 / addr 429,520 after cap, 75 s); US pool 6,186,873 (962,178 / 348,756, 84 s).
Blocking speed on the full pool (Colab): India 3.1 ms/query (40,116 in 124 s), **US 5.7 ms/query** (59,884 in 341 s) — slower than the 1.75 ms estimate. Features ≈ 60k pairs/s.
Training: 5.27M pairs; LightGBM 770 trees on 4.74M pairs (294,359 positive), 568 s.

**Blocking (val, full pool):** 52.7 candidates/S1 · recall@10 0.906 · @20 0.927 · @50 0.9434 · union **0.9445** (name-only channel adds 75 links) · all links found for 84.1% of S1 · ceiling F0.5 0.9786. (Mini: 0.969 / 0.9882 → the 10× denser pool costs ≈2.5 recall points.)

**Validation (mini-val S1 vs full pool), threshold 0.75:** F0.5 0.9471 · test-mix 0.9427 · India 0.9296 · US 0.9587 · mean P 0.9753 · mean R 0.8914 · singleton accuracy 0.9239 · 3.13 predictions/S1. Threshold curve flat 0.65–0.85 (test-mix 0.9406–0.9427).

**HOLDOUT (20k untouched S1, threshold fixed from val): F0.5 0.9500 · test-mix 0.9450 · India 0.9299 · US 0.9634 · P 0.9778 · R 0.8907 · singleton accuracy 0.9495.** Holdout ≥ val → no optimism from threshold tuning; this is the number to report for US+India.

**Error budget (full pool):** ceiling 0.9786 → matcher gap ≈ 0.032, blocking gap ≈ 0.021. Recall (0.89) is now the weaker side.
Feature importance shifted: rank ≫ blk, num_tset, n_latin_ratio, addr_gap, n_compact_ratio, a_tset.

**Error patterns (full pool):**
- *Missed true matches with corrupted house numbers* — digit dropped (11102→1110, 6101→610, 2010→010, 2200→220) or off by 1–2 (2520→2518, 80→79), often with an OCR-like name error (`8IG BISTRO`). The model rejects them (p 0.02–0.6) because number tokens disagree.
- *False merges with near-identical names at nearby but different numbers* — 5158 vs 5167a Gunn Rd, Door 50-81-24/2 vs /9, No.1/374-A1 vs 6-1/374-A8, same street number in a different city. These look like branches/neighbours.
- *Same premises, different business* (ZC Solutions vs Hitech; North Systems vs नॉर्थ एक्सपोर्ट्स) and *unrelated-name true matches* (Mae Chandler Patriot Rate LLC vs "Ectoectobelo (ID: 12357)") remain ambiguous from the pair alone.
- Embedded placeholders again (`NULL`, earlier `<NULL>`).

**Estimated leaderboard:** holdout test-mix 0.945 covers US+India (85% of test); France expected 1.5–3 points lower (A.3); test-time many-to-one resolution (all 1.73M S1 compete) should recover part of the same-premises false merges → expected ≈ 0.935–0.945.

**Evidence-based v3 improvements (priority order):**
1. House-number features: leading-number equality, absolute / relative numeric difference, digit-string edit distance, "one number is a truncation (prefix/suffix) of the other" — directly targets both the missed digit-drop/off-by-one matches and the nearby-number false merges.
2. Context features without labels: count of S1 sharing the core name / the address (per country), count of pool records sharing the candidate's core name.
3. Larger K only if blocking stays the limit (costs test time: US blocking already 5.7 ms/query).

**Test-run time estimate (revised):** blocking ≈ 42 min India + ≈ 63 min US + ≈ 10 min France, features ≈ 25 min, prediction (770 trees) ≈ 15–25 min → ≈ 2–2.5 h.

---

# Part 7 — Notebook v2.1: runs on Colab or Kaggle

User asked whether the notebook runs on Kaggle. v2 was Colab-bound (writes to `/content/work`, searches `/content` for the challenge data, unaware of read-only `/kaggle/input` and `/kaggle/working`). **No modelling change** — only environment handling:
- Auto-detects Kaggle; work folder `/tmp/er_work` (large scratch disk), outputs `/kaggle/working/output` (Output tab); Colab unchanged (`/content/work`). Prints free disk and warns below 25 GB.
- Finds attached Kaggle datasets automatically: the mini-world (by `val_source1.tsv` / zips) and the challenge folder (by `dataset/test/test_source1.tsv`), whether uploaded as two datasets or one; the official `dataset/train|test` files are never mistaken for mini files. Falls back to the Drive links when nothing is attached.
- Nothing is written into `/kaggle/input`; validator skipped with a warning if `utils/validate_submission.py` is absent; submission zip saved in the output folder.
- Tests: simulated Kaggle (two datasets and one combined dataset), Colab mode regression, and the `.ipynb` executed in a Jupyter kernel in Kaggle mode — 21 cells, 0 errors, validator PASS, results identical to v2.

## Cell B.3 — test inference (Colab run of v2, 197 min)
| Country | Test S1 | Pool (S2+S3) | Candidate pairs | per S1 | Minutes |
|---|--:|--:|--:|--:|--:|
| France | 259,452 | 1,434,993 | 14,018,032 | 54.0 | 17.8 |
| India | 809,986 | 4,717,565 | 44,014,968 | 54.3 | 106.3 |
| US | 663,106 | 3,817,031 | 34,066,160 | 51.4 | 68.7 |
| **Total** | **1,732,544** | 9,969,589 | **92,099,160** | 53.2 | **197.2** |

Candidate file has one row for every test S1 (1,732,544). **Predicted matches: 5,319,907 pairs; 1,611,358 S1 with ≥1 match; empty 6.99%** (≈121k S1). Sanity vs expectations: 3.07 predictions per S1 (val 3.13; true mean 3.46 → precision-first, as intended); empty share 6.99% vs 5.58% true singleton rate in train → ~1.4% of S1 get no prediction despite having matches (consistent with val recall 0.89 and France's unseen patterns). Throughput on Colab: ≈ 7.9 ms per test S1 end-to-end (blocking + 53 candidates' features + 770-tree prediction).

## Results saved (after a Colab kernel restart; runtime reattached)
Files on disk: candidate_pairs.tsv 1,209.4 MB · matching_results.tsv 91.0 MB · matcher_lgb.txt 10.7 MB · results.json.
Official validator: **PASS** — matching_results 1,732,544 rows (121,186 empty, 1,611,358 non-empty); candidate_pairs 1,732,544 rows (241 empty, 1,732,303 non-empty).
Zips: matching_results.zip 39 MB (downloaded) · submission_outputs.zip 550 MB. Copied to Google Drive `MyDrive/er_challenge/run_v2/` (both zips + results.json).
Note: only 241 test S1 (0.014%) received no candidates at all.

## Leaderboard — first submission (v2, Colab run)
**Public leaderboard F0.5 = 0.933** (26 Sep 2026, 22:39 IST). Estimate before submission was 0.935–0.945 → at the low end.
Inference used the **full official test set** (1,732,544 S1; downloaded by cell B.1 from the challenge folder link), not the mini-world; the validator confirmed all 1,732,544 required S1.
Rough decomposition (assumes US/India behave like the holdout, test-mix 0.945, 85% of test): 0.933 ≈ 0.85·0.945 + 0.15·F → **France ≈ 0.86–0.87**, i.e. ~8 points below US/India — larger than the 1.5–3-point LOCO estimate. France (unseen, 15% of test) looks like the biggest single lever: +0.01 overall if France reached ≈0.93. (Caveat: public leaderboard is a subset; the decomposition is approximate.)
Next: per-country diagnostics on the test predictions (empty share, predictions per S1) to see whether France is under-matched (recall) or over-matched (precision).

## Test-prediction diagnostics per country (no labels)
| Country | S1 | Predicted per S1 | Empty share | ≥6 predicted | S3 share |
|---|--:|--:|--:|--:|--:|
| France | 259,452 | 2.91 | 8.97% | 6.76% | 0.530 |
| India | 809,986 | 3.03 | 7.12% | 7.33% | 0.510 |
| US | 663,106 | 3.19 | 6.06% | 8.15% | 0.517 |
Reference: train truth 3.46 per S1, 5.58% singletons, S3 share 0.516; val predictions 3.13 per S1.
France predicted-count distribution: 0:23,281 · 1:30,922 · 2:53,530 · 3:60,055 · 4:46,962 · 5:27,156 · 6:12,076 · 7:4,125 · 8:1,085 · 9:231 · 10:22 · 11:7 — same shape as the train truth (peak at 3, max 11).

**Interpretation:** France is only moderately more conservative (−9% predictions vs US, +3 points empty) — it is **not** being badly under-matched. The earlier inference "France ≈ 0.86" relied on assuming US/India score on test as on the holdout; that assumption is weak (see next finding), so it is withdrawn.

**Key finding — the test pool is 23% denser than the train pool:** pool records per S1 = 5.75 in test (France 5.53, India 5.82, US 5.76) vs 4.68 in train (India 4.68, US 4.67). If test S1 have ~3.46 true matches like train, **≈40% of test S2/S3 records are orphans vs 26% in train** → more distractors per S1 in every country. The same density effect cost 2.4 points from mini to full (0.9715 → 0.945 test-mix); a further +23% density plausibly explains most of the holdout→leaderboard gap (0.945 → 0.933) without a France-specific failure.

**Implications for v3:** (1) the threshold (tuned at train density) is probably slightly too permissive for test → try higher thresholds; (2) B.3 must **save test probabilities** so re-thresholding takes a minute instead of a 3-hour rerun; (3) features that separate near-duplicate distractors (house-number similarity, shared-name / shared-address context) matter more on the denser test set.

---

# Part 8 — Notebook v3 (for the final 24 h; Kaggle run)

**Context:** deadline 27 Sep 2026 21:00 IST; 12 of 15 submissions left; v2 leaderboard 0.933.

**Changes (evidence → change):**
1. *Test pool 23% denser than train (5.75 vs 4.68 per S1)* → **density matching** (`DENSITY_MATCH`): per country, sampled test S2/S3 records (ids disjoint from train → guaranteed non-matches) are added to the training pool until densities match; the final model is trained, its threshold tuned and the holdout scored at test-like density. Expected additions ≈ India 21% of its test pool (~1.0M), US 38% (~1.4M).
2. *B.2 errors: dropped digits / off-by-one true matches vs nearby-number false merges* → **house-number features**: hn_both, hn_eq, hn_absdiff (log), hn_reldiff, hn_lev (digit edit distance), hn_trunc (edit distance = length difference → digits dropped). House number = first number in `addr_plain`, leading zeros stripped (derived at load time; preprocessing unchanged, version pp_17b494c2).
3. *Generic names / shared premises ambiguous* → **context features** (label-free): q_name_dup, q_addr_dup (frequency among all S1 of the country, per 100k), c_name_dup, c_addr_dup (frequency in the candidate pool). Populations: Phase A = mini S1; Phase B = full train S1; test = full test S1.
4. *Re-thresholding required a 3-h rerun* → B.3 saves `test_scores.parquet` (p ≥ 0.05); **B.5** writes `variants/matching_results_thr{0.70…0.90 + val-optimal}.tsv` with per-country predictions/S1; B.2 prints the **holdout F0.5 at each variant threshold** to pick submissions.
5. More training data (+0.003 from 30k→100k in Phase A) → **FULL_TRAIN_QUERIES = 200k** (all mini-train S1).
6. Known real-data answers reused: gate skipped (v2 table reported in RESULTS / documentation), DF cap fixed at 0.005.
Features: 49 → 59. B.3 writes kept pairs straight to Parquet (lower peak memory than v2's in-memory lists that likely caused the Colab crash).

**Tests (sandbox):** synthetic world with a 25%-denser test pool, Kaggle-mode — density matching computed and applied, holdout threshold curve printed, test scores saved, 6 variant files written, validator PASS; `.ipynb` executed in a Jupyter kernel: 22 code cells, 0 errors.

**Expected Kaggle runtime (4 CPUs):** Phase A ≈ 12 min · B.1 ≈ 12 min · B.2 ≈ 30 min · B.3 ≈ 110–120 min · B.5 + packaging ≈ 8 min → ≈ 3 h.

**Submission plan:** (1) main file (val-optimal threshold at test density); (2) the variant whose holdout F0.5 is highest if different; (3) one stricter variant (+0.05) as a density check. Keep ≥ 5 submissions in reserve.

---

# Part 9 — Notebook v3.1: country as an open set (compliance with the problem statement)

**Rule (problem statement):** treat country as an open set of string labels; never hard-code, filter or one-hot the pipeline to {US, India}; every test entity (France included) must be in the submission.

**Audit of v3 (already compliant in behaviour):** country is used only as a partition key read from the data (`countries_in`); the matcher has no country feature (no one-hot); B.3 loops over every label in test S1 and writes an empty row for a label with no pool; the v2 submission had all 1,732,544 S1 (France 259,452) and the validator passed.

**Tightened in v3.1 (no change to the predictions for US / India / France):**
- `UNIT_COUNTRIES = {"US","India"}` allowlist → `UNIT_SKIP = {"France"}` (unit extraction is generic; France skipped because "ste/st" = Sainte/Saint). Unseen labels get generic defaults; postal extraction only where a pattern is known.
- `TEST_WEIGHTS` hard-coded {India, US} → `"auto"`: each labelled country's share of the test S1, computed in B.1 (real data: India 0.5499 / US 0.4501, same as before). B.1 prints a train/test country table and lists labelled vs test-only labels.
- A.3 leave-one-country-out loops over whatever labelled countries exist (train on all others).
- B.3 asserts that every test S1 appears exactly once in both files and prints per-label counts.
- Gallery adds an unseen-label case (Germany).

**Test:** synthetic world with two extra test-only labels (Germany: S1 + pool; Atlantis: S1 with no pool records). All cells run (22 code cells, 0 errors in a Jupyter kernel); Germany scored with the same model (1.09 predictions/S1), Atlantis gets empty rows; completeness assert passes; validator PASS.
**The Kaggle v3 run in progress is unaffected; v3.1 reproduces its predictions** (Phase B does not depend on Phase A; preprocessing is identical for the three real labels). Submit v3.1 as the code file.

## Leaderboard — v3 main file (Kaggle run): **0.929** (27 Sep 2026, 11:30 IST)
Below v2 (0.933) by 0.004. Density matching + house-number/context features did not improve the leaderboard as submitted. Next: standalone re-threshold cell (`rethreshold_cell.py`) reads the v3 `test_scores.parquet` + `results.json`, prints the v3 chosen threshold and holdout curve, and writes zipped matching files for thresholds 0.50–0.90 with predictions/S1 and empty share per country, compared with the v2 file's profile (3.07 per S1, 6.99% empty). Verified on synthetic output: identical to the pipeline's file at the same threshold.

## v3 results.json (Kaggle) — diagnosis of the 0.929
- Density matching added India 1.01M / US 1.43M test records (train density 4.68 → 5.82 / 4.67 → 5.76).
- Final model: 10.58M training pairs (651,504 positive), threshold 0.725. Val test-mix 0.9535; **holdout 0.9542** (India 0.9392, US 0.9726) at test-like density vs v2 holdout 0.9450 at train density → the v3 model is better in-distribution. Holdout curve flat 0.70–0.80 (0.9534–0.9542), so re-thresholding cannot recover the gap.
- Test: 3.11 predictions/S1 (France 2.85, India 3.09, US 3.24), empty 6.93% — profile close to v2.
- LOCO (mini): US→India 0.9218 (v2 0.9248), India→US 0.9731 (v2 0.9648).
- **Cause (most likely):** the four context features are rates per 100k records of the population (`count × 1e5 / N`). N differs between training and test: US S1 1.32M train vs 663k test (×2), France 259k (≈×3.4 vs India train); US pool 7.6M (train + density extra) vs 3.8M test. On test, every name/address looks 2–5× more common than it would in training → shifted inputs to features with gain rank 16–26. India's sizes are similar (883k vs 810k S1; 5.1M vs 4.7M pool), consistent with the loss concentrating in US/France. Holdout cannot see this (same population as training).

## Notebook v3.2
- `CONTEXT_FEATURES=False`: the four population-dependent features are removed (55 features). House-number features, density matching, 200k training S1 and all v3.1 open-set changes kept.
- Outputs are zipped directly in the output folder: `matching_results.zip` (main) and `matching_results_thr{0.700…0.900}.zip` (each contains `matching_results.tsv`), so no text-tab download problem.
- Fixed an edge case in `error_examples` (empty id list → Arrow null type).
- Tests: synthetic world incl. unseen labels, Jupyter kernel: 22 code cells, 0 errors, validator PASS, all zips correct.

## Leaderboard — v3.2 (Kaggle run): **0.96**
v3.2 = v3.1 with the four population-dependent context features removed (55 features); density matching, house-number features and 200k training S1 kept.
- Final model: LightGBM, 1,000 trees, 10,584,961 training pairs (651,504 positive), threshold 0.75.
- Validation (mini-val vs full density-matched pool): F0.5 0.9525 · test-mix 0.9480 · India 0.9346 · US 0.9645.
- **Holdout (20k untouched S1): F0.5 0.9541 · test-mix 0.9488 · India 0.9331 · US 0.9681 · P 0.9798 · R 0.8977 · singleton accuracy 0.9566.**
- Test: 1,732,544 S1, 92.1M candidate pairs, 5,342,337 predicted links (3.08 per S1; France 2.90, India 3.04, US 3.21), 7.0% empty; 142 min; official validator PASS.
- **Public leaderboard 0.96** (v2 0.933, v3 0.929): removing the population-scaled context features fixed the v3 regression, and the density-matched training with house-number features now carries over to the test set.
