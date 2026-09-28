# RE-THRESHOLD CELL — standalone: needs only the v3 run's output (test_scores.parquet, results.json)
# and the challenge data attached as inputs. Takes about 1-2 minutes. No retraining.
import os, glob, json, csv, zipfile, time
import numpy as np, pandas as pd, pyarrow.parquet as pq

IN_ROOT = "/kaggle/input"                     # where the attached inputs live
OUT_DIR = "/kaggle/working/rethreshold"       # zips to download appear here (Output tab / file browser)
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
MAX_PRED = 15
os.makedirs(OUT_DIR, exist_ok=True)

def find(pattern, must=None):
    hits = [h for h in glob.glob(os.path.join(IN_ROOT, "**", pattern), recursive=True) if must is None or must in h]
    return hits[0] if hits else None

scores_p = find("test_scores.parquet")
s1_p = find("test_source1.tsv", must=os.sep + "test" + os.sep)
res_p = find("results.json")
print("scores :", scores_p); print("test S1:", s1_p); print("results:", res_p)
assert scores_p and s1_p, "attach the v3 notebook output AND the challenge dataset (Add Input), then re-run"

# 1) what the v3 run chose, and its holdout curve
if res_p:
    R = json.load(open(res_p))
    B = R.get("phaseB", {})
    print("\nv3 chosen threshold:", B.get("threshold"))
    print("val   :", {k: round(v, 4) for k, v in (B.get("metrics") or {}).items() if isinstance(v, float)})
    ho = B.get("holdout") or {}
    print("holdout:", {k: round(v, 4) for k, v in ho.items() if isinstance(v, float)})
    if ho.get("threshold_curve"):
        print(pd.DataFrame(ho["threshold_curve"])[["thr", "F05_testmix", "mean_P", "mean_R", "singleton_acc", "pred_per_S1"]]
              .to_string(index=False))
    print("density match:", R.get("density_match"))
    print("test (main file):", {k: v for k, v in (R.get("test") or {}).items() if k != "per_country"})

# 2) load test probabilities and the full test S1 list (every S1 must appear, empty or not)
t0 = time.time()
t = pq.read_table(scores_p)
kq = t.column("q").to_numpy(zero_copy_only=False); kc = t.column("c").to_numpy(zero_copy_only=False)
kp = t.column("p").to_numpy(zero_copy_only=False).astype(np.float32)
del t
s1 = pd.read_csv(s1_p, sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=csv.QUOTE_NONE,
                 usecols=["entity_id", "country"])
ids, ctry = s1["entity_id"].to_numpy(), s1["country"].str.strip().to_numpy()
code = {q: i for i, q in enumerate(ids)}
qc = np.fromiter((code[q] for q in kq), dtype=np.int64, count=len(kq))
cc, _ = pd.factorize(kc)
print(f"\nloaded {len(kp):,} scored pairs for {len(ids):,} test S1 in {time.time()-t0:.0f}s")

def decide(thr):
    """p >= thr, then each S2/S3 record goes to its highest-p S1 (many-to-one), then cap per S1."""
    idx = np.flatnonzero(kp >= thr)
    o = idx[np.lexsort((-kp[idx], cc[idx]))]
    idx = o[np.r_[True, cc[o][1:] != cc[o][:-1]]]
    o = idx[np.lexsort((-kp[idx], qc[idx]))]
    qs = qc[o]; start = np.r_[True, qs[1:] != qs[:-1]]
    gs = np.maximum.accumulate(np.where(start, np.arange(len(o)), 0))
    return o[(np.arange(len(o)) - gs) < MAX_PRED]

rows = []
for thr in THRESHOLDS:
    idx = decide(thr)
    n = np.bincount(qc[idx], minlength=len(ids))
    order = np.argsort(qc[idx], kind="stable")
    q_sorted, c_sorted = qc[idx][order], kc[idx][order]
    bounds = np.searchsorted(q_sorted, np.arange(len(ids) + 1))
    name = f"matching_results_thr{thr:.2f}"
    tsv = os.path.join(OUT_DIR, "matching_results.tsv")
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for i, q in enumerate(ids):
            f.write(f"{q}\t{','.join(c_sorted[bounds[i]:bounds[i+1]])}\n")
    with zipfile.ZipFile(os.path.join(OUT_DIR, name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tsv, "matching_results.tsv")          # the file inside is always named matching_results.tsv
    row = dict(thr=thr, pred_per_S1=round(n.mean(), 3), empty_share=round((n == 0).mean(), 4))
    for c in sorted(set(ctry)):
        m = ctry == c
        row[f"pred_{c}"] = round(n[m].mean(), 3); row[f"empty_{c}"] = round((n[m] == 0).mean(), 4)
    rows.append(row)
os.remove(tsv)
print("\nReference: v2 file (leaderboard 0.933) had pred_per_S1 3.07 and empty_share 0.0699 "
      "(France 2.91 / 8.97%, India 3.03 / 7.12%, US 3.19 / 6.06%).")
print(pd.DataFrame(rows).to_string(index=False))
print("\nzips written to", OUT_DIR, ":", sorted(os.listdir(OUT_DIR)))
