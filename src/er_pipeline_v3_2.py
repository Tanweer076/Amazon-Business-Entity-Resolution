# Business Entity Resolution pipeline v3.2 — plain-Python export of notebooks/er_pipeline_v3_2.ipynb
# (generated for code browsing; the notebook is the source of truth. Cells are separated by "# %%".)

# %% [markdown]
# # Business Entity Resolution — end-to-end pipeline
#
# **ML Challenge 2026.** For every Source 1 entity, find all matching Source 2 / Source 3 records.
# Metric: macro F0.5 per S1 entity (singletons included).
#
# **Runs on Colab or Kaggle (CPU is enough).** On Kaggle: attach the challenge data and the mini-world as Datasets
# (found automatically), turn **Internet on** (for `pip install`), then **Save Version → Save & Run All**; results
# appear in the Output tab. Without attached datasets it downloads from the Drive links below instead.
#
# **How to run on Colab:**
# 1. Set `DATA_FOLDER_URL` in the config cell to your Google Drive folder (shared as *Anyone with the link*)
#    holding the mini-world files: `train/` and `val/` sub-folders with `train_source1.tsv` … `val_ground_truth.tsv`
#    (zips also work). Add `splits_v1.tsv` to the folder for the holdout check (optional).
# 2. **Runtime → Run all.** Phase A (mini-world, ~40 min incl. the preprocessing gate) builds and validates the pipeline.
#    Phase B (full data, ~1.5–2 h) trains the final model on the full candidate pool, calibrates the
#    threshold, runs the test set and writes both submission files.
# 3. Outputs land in `/content/work/output/`: `matching_results.tsv`, `candidate_pairs.tsv`,
#    `results.json`, a filled `Documentation.md`, plus `README.md` / `requirements.txt` for the code folder.
#
# **Pipeline:** preprocessing (multi-view, per the preprocessing plan P01–P17; "check before enabling" rules
# adopted only if they improve validation F0.5) → country-partitioned blocking
# (IDF-weighted name+address cosine top-K ∪ name-only top-K) → pairwise similarity features →
# LightGBM matcher → many-to-one resolution → F0.5-tuned threshold.
#
# Every cell prints what it did. Paste the outputs back for interpretation.

# %%
# 0.1 Install dependencies (all MIT/BSD/Apache licensed; no external data or APIs used)
# (notebook shell command) !pip -q install unidecode rapidfuzz lightgbm indic-transliteration ftfy gdown sparse_dot_topn

# %%
# 0.2 Imports and configuration
import os, re, csv, gc, json, time, glob, zipfile, shutil, hashlib, unicodedata, random, sys
from functools import lru_cache
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from unidecode import unidecode
import ftfy
import lightgbm as lgb
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.process import cpdist
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from sparse_dot_topn import sp_matmul_topn

CONFIG = dict(
    # ---- data ----
    DATA_FOLDER_URL="PASTE_YOUR_DRIVE_FOLDER_LINK_HERE",   # Drive folder with the mini-world TSVs (or zips)
    CHALLENGE_FOLDER_URL="PASTE_YOUR_CHALLENGE_FOLDER_LINK_HERE",
    WORK="auto",           # auto: /content/work on Colab, /tmp/er_work on Kaggle (large scratch disk)
    OUT_DIR="auto",        # auto: <WORK>/output on Colab, /kaggle/working/output on Kaggle (Output tab)
    INPUT_DIR="",          # optional: folder holding the mini-world files; on Kaggle attached datasets are found automatically
    CHALLENGE_DIR="",      # optional: path to .../student_resource; on Kaggle attached datasets are found automatically
    KAGGLE_INPUT="/kaggle/input",
    # ---- preprocessing switches: the plan's "check before enabling" rules (P02, P03, P09, P10) ----
    # Start at the plan's conservative setting; the Phase A gate (A.0) enables each one only if it improves
    # mini-validation F0.5 (plan Q12). Set RUN_SWITCH_GATE=False to keep these values as they are.
    FTFY=False,                # P03 repair mojibake (only on strings that look damaged); try: True
    PLACEHOLDER_EMPTY=False,   # P02 whole-field NULL / NAN / N/A address -> empty (embedded ones kept); try: True
    STATE_EXPAND=False,        # P10 state abbreviation -> full name (whole address component only); try: True
    STREET_ABBREV=False,       # P10 country-aware street-type abbreviations; try: True
    DEVA_TRANSLIT="unidecode", # P09 Devanagari -> Latin: "unidecode" (plan baseline) or "indic" (IAST + schwa deletion)
    SWITCHES_TRY=dict(FTFY=True, PLACEHOLDER_EMPTY=True, STATE_EXPAND=True, STREET_ABBREV=True, DEVA_TRANSLIT="indic"),
    RUN_SWITCH_GATE=False,     # v3: already decided on real data in run v2 (no rule adopted) -> KNOWN_GATE below
    GATE_TRAIN_QUERIES=30_000, # smaller training sample for the gate runs (comparisons stay paired on the same val S1)
    # ---- blocking ----
    DF_MAX_FRAC=0.005,         # tokens in more than this share of a country's pool are not blocking keys
    AUTO_DF_CAP=False,         # v3: already chosen on real data in run v2 (0.002 lost 2.7 recall points -> 0.005)
    DF_CAP_CANDIDATES=[0.002, 0.005],
    DF_CAP_TOLERANCE=0.005,    # accept up to 0.5 points lower recall for a faster cap
    W_NAME=0.5,                # blend of name and address cosine (address dominates in practice)
    K_MAIN=50,                 # top-K by blended score
    K_NAME=10,                 # plus top-K by name-only cosine (rescues empty/garbled-address matches)
    BLOCK_BATCH=2048,
    # ---- matcher ----
    N_TRAIN_QUERIES=100_000,   # Phase A: S1 training queries sampled from the mini train split
    FULL_TRAIN_QUERIES=200_000,# Phase B: training queries for the final model (all 200k mini-train S1)
    CONTEXT_FEATURES=False,    # v3.2: OFF. The name/address frequency features (q_/c_ name_dup, addr_dup) are rates per
                               # 100k records of a population whose size differs between train and test (US test S1 is
                               # half of train, France a quarter) -> shifted on test; v3 (with them) fell 0.933 -> 0.929.
    DENSITY_MATCH=True,        # Phase B: add sampled test S2/S3 records to the training pool so every country has
                               # the test set's pool density (test ids never match train S1 -> pure distractors)
    VARIANT_THRESHOLDS=[0.70, 0.75, 0.80, 0.85, 0.90],   # B.5 writes one matching file per threshold
    SEED=42,
    LGB_ROUNDS=1000,
    # ---- decision ----
    THR_GRID=[round(x, 3) for x in np.arange(0.10, 0.951, 0.025)],
    MAX_PRED=15,               # safety cap on matches per S1 (labels never exceed 11)
    P_FLOOR=0.05,              # test pairs below this probability are dropped before resolution
    TEST_WEIGHTS="auto",       # reporting weights for validation F0.5: "auto" = each LABELLED country's share of the
                               # test S1 (computed from the data in B.1; before that, the val set's own mix).
                               # Countries are an open set of strings: never filtered, never a model feature.
    TEST_QUERY_BATCH=20_000,
    RUN_HOLDOUT=True,          # Phase B holdout check (needs splits_v1.tsv in the Drive folder)
    # real-data gate result from run v2 (mini-world, paired on 20k val S1), reported when the gate is skipped
    KNOWN_GATE=[dict(rule="baseline (plan conservative settings)", F05_testmix=0.9687, delta=0.0, se=0.0, recall=0.9690, adopted="-"),
                dict(rule="FTFY=True", F05_testmix=0.9687, delta=0.0000, se=0.0003, recall=0.9690, adopted="no"),
                dict(rule="PLACEHOLDER_EMPTY=True", F05_testmix=0.9687, delta=0.0000, se=0.0000, recall=0.9690, adopted="no"),
                dict(rule="STATE_EXPAND=True", F05_testmix=0.9689, delta=0.0002, se=0.0004, recall=0.9688, adopted="no"),
                dict(rule="STREET_ABBREV=True", F05_testmix=0.9683, delta=-0.0004, se=0.0004, recall=0.9692, adopted="no"),
                dict(rule="DEVA_TRANSLIT=indic", F05_testmix=0.9693, delta=0.0006, se=0.0004, recall=0.9699, adopted="no")],
)
if os.environ.get("ER_CONFIG_JSON"):          # local testing hook
    CONFIG.update(json.loads(os.environ["ER_CONFIG_JSON"]))

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
ON_KAGGLE = os.path.isdir(CONFIG["KAGGLE_INPUT"])
if CONFIG["WORK"] == "auto":
    CONFIG["WORK"] = "/tmp/er_work" if ON_KAGGLE else "/content/work"
if CONFIG["OUT_DIR"] == "auto":
    CONFIG["OUT_DIR"] = "/kaggle/working/output" if ON_KAGGLE else os.path.join(CONFIG["WORK"], "output")
WORK, OUT = CONFIG["WORK"], CONFIG["OUT_DIR"]
os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
RESULTS = {"config": {k: v for k, v in CONFIG.items() if "URL" not in k}}

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def save_results():
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)

free_gb = shutil.disk_usage(WORK).free / 1e9
log(f"python {sys.version.split()[0]} | pandas {pd.__version__} | lightgbm {lgb.__version__} | "
    f"cpus {os.cpu_count()} | colab {IN_COLAB} | kaggle {ON_KAGGLE} | work {WORK} ({free_gb:.0f} GB free) | output {OUT}")
if free_gb < 25:
    log("WARNING: less than 25 GB free for the work folder; the full run needs about 15-20 GB")

# %% [markdown]
# ## 1. Data
# Downloads your mini-world Drive folder (train / val S1, shared S2 / S3 pool, ground truth). The folder can hold
# the TSV files directly (`train/`, `val/` sub-folders or flat) or the zips; `splits_v1.tsv` is optional (holdout).

# %%
# 1.1 Data helpers
def read_tsv(path, **kw):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                       quoting=csv.QUOTE_NONE, **kw)

def gdown_folder(url, out):
    import gdown
    os.makedirs(out, exist_ok=True)
    gdown.download_folder(url, output=out, quiet=True, use_cookies=False, remaining_ok=True)

def _find(roots, name):
    roots = [roots] if isinstance(roots, str) else roots
    for root in roots:
        hits = sorted(h for h in glob.glob(os.path.join(root, "**", name), recursive=True)
                      if "__MACOSX" not in h and "/dataset/train/" not in h and "/dataset/test/" not in h)
        if hits:
            return hits[0]
    return None

def _kaggle_mini_dir():
    """An attached Kaggle dataset that holds the mini-world (val_source1.tsv, splits file or zips)."""
    if not ON_KAGGLE:
        return None
    for name in ("val_source1.tsv", "mini_v1_val.zip", "mini_world_v1.zip"):
        hits = [h for h in glob.glob(os.path.join(CONFIG["KAGGLE_INPUT"], "**", name), recursive=True)
                if "/dataset/" not in h]
        if hits:
            rel = os.path.relpath(hits[0], CONFIG["KAGGLE_INPUT"]).split(os.sep)
            return os.path.join(CONFIG["KAGGLE_INPUT"], rel[0])      # the dataset root
    return None

def get_mini_inputs():
    inp = CONFIG["INPUT_DIR"] or _kaggle_mini_dir() or os.path.join(WORK, "inputs")
    have = lambda: glob.glob(os.path.join(inp, "**", "*.tsv"), recursive=True) + \
                   glob.glob(os.path.join(inp, "**", "*.zip"), recursive=True)
    if not have():
        assert "PASTE" not in CONFIG["DATA_FOLDER_URL"], "Set CONFIG['DATA_FOLDER_URL'] to your Drive folder link."
        log("downloading mini-world folder from Drive ...")
        gdown_folder(CONFIG["DATA_FOLDER_URL"], inp)
    assert have(), f"No .tsv or .zip files found in {inp} — is the Drive folder shared as 'Anyone with the link'?"
    ext = os.path.join(WORK, "inputs_extracted")
    zips = glob.glob(os.path.join(inp, "**", "*.zip"), recursive=True)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(ext)
    log("found:", ", ".join(sorted({os.path.relpath(f, inp) for f in have()})))
    roots = [inp, ext]
    splits = _find(roots, "splits_v1.tsv")
    if _find(roots, "val_source1.tsv"):                      # train/ + val/ layout (Cell R)
        paths = dict(train_s1=_find(roots, "train_source1.tsv"), val_s1=_find(roots, "val_source1.tsv"),
                     s2=_find(roots, "train_source2.tsv") or _find(roots, "val_source2.tsv"),
                     s3=_find(roots, "train_source3.tsv") or _find(roots, "val_source3.tsv"),
                     train_gt=_find(roots, "train_ground_truth.tsv"), val_gt=_find(roots, "val_ground_truth.tsv"))
    elif splits and _find(roots, "train_source1.tsv"):       # combined mini + splits file (Cell M)
        mv = read_tsv(splits, usecols=["entity_id", "mini_val"])
        val_ids = set(mv.loc[mv.mini_val == "1", "entity_id"])
        d = os.path.join(WORK, "mini_v1"); os.makedirs(d, exist_ok=True)
        paths = dict(s2=_find(roots, "train_source2.tsv"), s3=_find(roots, "train_source3.tsv"))
        for kind, src, key_col in (("s1", "train_source1.tsv", "entity_id"),
                                   ("gt", "train_ground_truth.tsv", "source1_entity_id")):
            df = read_tsv(_find(roots, src))
            is_val = df[key_col].isin(val_ids)
            for role, part in (("train", df[~is_val]), ("val", df[is_val])):
                p = os.path.join(d, f"{role}_{kind}.tsv")
                part.to_csv(p, sep="\t", index=False, quoting=csv.QUOTE_NONE)
                paths[f"{role}_{kind}"] = p
    else:
        raise FileNotFoundError("Unrecognised layout: need val_source1.tsv (+ train/val files), "
                                "or splits_v1.tsv + a combined train_source1.tsv")
    for k, v in paths.items():
        assert v and os.path.exists(v), f"missing {k}"
    return paths, splits

def get_challenge_dir():
    if CONFIG["CHALLENGE_DIR"] and os.path.exists(CONFIG["CHALLENGE_DIR"]):
        return CONFIG["CHALLENGE_DIR"]
    def search():
        marker = os.path.join("dataset", "test", "test_source1.tsv")
        pats = [os.path.join(WORK, "challenge", "**", marker)]
        if ON_KAGGLE:
            pats.append(os.path.join(CONFIG["KAGGLE_INPUT"], "**", marker))
        pats += [os.path.join("/content", *(["*"] * d), marker) for d in (1, 2, 3)]
        for pat in pats:
            for hit in sorted(glob.glob(pat, recursive="**" in pat)):
                base = os.path.dirname(os.path.dirname(os.path.dirname(hit)))
                if "__MACOSX" not in base and "/drive/" not in base and \
                        os.path.exists(os.path.join(base, "dataset", "train", "train_source2.tsv")):
                    return base
        return None
    d = search()
    if d is None:
        log("downloading challenge data (~2.5 GB) ...")
        gdown_folder(CONFIG["CHALLENGE_FOLDER_URL"], os.path.join(WORK, "challenge"))
        d = search()
    assert d, "challenge folder not found"
    return d

def load_gt(path, keep=None):
    gt = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, _, rest = line.rstrip("\n").partition("\t")
            if keep is not None and s1 not in keep:
                continue
            rest = rest.strip()
            gt[s1] = frozenset(rest.split(",")) if rest else frozenset()
    return gt

# %%
# 1.2 Fetch the mini-world
MINI_PATHS, SPLITS_PATH = get_mini_inputs()
for k, v in MINI_PATHS.items():
    with open(v, encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1
    log(f"{k:9} {n:>10,} rows   {v}")
log("splits file:", SPLITS_PATH or "not provided (holdout check will be skipped)")

# %% [markdown]
# ## 2. Preprocessing (implements the preprocessing plan P01–P17)
# Raw fields are kept (P01); every other column is an additional *view* of the same record, never a replacement.
#
# | column | plan | used by | what it is |
# |---|---|---|---|
# | `name_raw`, `addr_raw` | P01 | error analysis | exactly as read (strings, leading zeros kept) |
# | `name_light` | P04/P05 | matcher | NFC + casefold, original script and Indic marks kept |
# | `name_latin` | P07/P08/P09 | matcher | accents folded + Devanagari romanised, punctuation → space, digits kept |
# | `name_core` | P11 | blocker + matcher | `name_latin` minus legal-form tokens (explicit list); never empty |
# | `name_legal` | P11 | matcher | canonical legal forms — kept as evidence (`Green & Sons` ≠ `Green & Co`) |
# | `name_compact` | P06/P16 | matcher | core name without spaces / domain endings: `krebsview.com` ≈ `Krebs View` |
# | `name_alias` | P16 | matcher | text after an explicit d.b.a. / a.k.a. / t/a marker (full name also kept) |
# | `name_init` | P16 | matcher | initials, compared only against very short names (`CC`) |
# | `addr_plain` | P10 | matcher | romanised address with **no** abbreviation mapping |
# | `addr_latin` | P10 | blocker + matcher | as `addr_plain` plus the enabled state / street mappings |
# | `addr_num` | P15 | matcher | all-digit tokens with leading zeros stripped (`00530` ≈ `530`) |
# | `addr_postal` | P12/P15 | matcher | postal-code candidate, zeros **kept**, stored as string |
# | `addr_unit` | P12 | matcher | unit / suite / shop / office identifiers (`# C6` ≈ `Unit C6`) |
# | `addr_missing` | P02 | matcher | 0 present · 1 empty in the raw file · 2 whole-field placeholder · 3 empty after normalisation |
# | `addr_ncomp`, `name_ntok` | P12 | matcher | address component count (partial-address signal), core-name token count |
# | `fixed`, `name_empty`, `deva` | P03/P02 | matcher / audit | ftfy changed the text · name normalised to empty · Devanagari script |
#
# Retrieval-only token filtering (document frequency) lives in the blocker and never edits these columns (P17).

# %%
# 2.1 Normalisation functions
PP_CODE_VERSION = "2026-09-26.2"      # bump whenever this cell changes (P14): part of the preprocessing version
ZW_RE = re.compile("[​⁠﻿­]")            # removed in the original-script view (ZWJ/ZWNJ kept)
ZW_ALL_RE = re.compile("[​‌‍⁠﻿­]")
CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
MOJIBAKE_RE = re.compile("Ã.|Â.|â€|à¤|à¥")
DEVA_RE = re.compile(r"[ऀ-ॿ]")
DEVA_WORD_RE = re.compile(r"[ऀ-ॿ]+")
DEVA_CONS_RE = re.compile(r"[क-हक़-य़]")
NONALNUM_RE = re.compile(r"[^0-9a-z]+")
DOMAIN_RE = re.compile(r"(www\.)|(\.(com|net|org|in|co|fr|us|info|biz|io)\b)", re.I)
DBA_RE = re.compile(r"\b(?:d\.?\s?b\.?\s?a\.?|a\.?\s?k\.?\s?a\.?|t/a|trading as|doing business as)(?=\s|$)", re.I)
UNIT_RE = re.compile(r"(?:#\s*|\b(?:unit|apt|apartment|suite|ste|flat|shop|office|room|rm|plot|block)\b\.?\s*"
                     r"(?:no\.?\s*)?[:#-]?\s*)([a-z0-9][a-z0-9/-]*)", re.I)
# Country is an open set of string labels. The look-ups below are OPTIONAL per-label refinements; any other label
# (an unseen country) is processed with the generic defaults, and every record is kept whatever its label.
POSTAL_RE = {"US": re.compile(r"\b(\d{5})(?:-\d{4})?\b"), "India": re.compile(r"\b(\d{3}\s?\d{3})\b"),
             "France": re.compile(r"\b(\d{5})\b")}          # default for other labels: no postal extraction
UNIT_SKIP = {"France"}     # unit extraction is generic; skipped where "ste"/"st" mean Sainte/Saint, not Suite
TLDS = {"www", "com", "net", "org", "in", "co", "fr", "us", "info", "biz", "io"}
INIT_STOP = {"and", "the", "of", "a", "an", "de", "la", "le", "et", "des", "du"}
PLACEHOLDERS = {"null", "nan", "na", "n/a", "none", "nil", "-", "--", "unknown", "not available"}
COMMA = " zzcommazz "

LEGAL_CANON = {
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "kanpani": "co", "kampani": "co",
    "ltd": "ltd", "limited": "ltd", "li": "ltd",
    "pvt": "pvt", "private": "pvt", "praivet": "pvt", "pra": "pvt", "prayvet": "pvt",
    "llc": "llc", "llp": "llp", "elaelapi": "llp", "elelpi": "llp", "lp": "lp", "plc": "plc",
    "pllc": "pllc", "gmbh": "gmbh", "srl": "srl", "bv": "bv", "ag": "ag", "pte": "pte",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl",
    "snc": "snc", "selarl": "selarl", "scp": "scp", "scop": "scop",
    "praaivett": "pvt", "limittedd": "ltd", "praa": "pvt", "limitedd": "ltd",   # unidecode Devanagari forms
}
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand",
    "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "pb": "punjab", "rj": "rajasthan",
    "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana", "ts": "telangana", "tr": "tripura",
    "up": "uttar pradesh", "uk": "uttarakhand", "wb": "west bengal", "dl": "delhi",
    "jk": "jammu and kashmir", "ch": "chandigarh", "py": "puducherry",
}
STATE_MAPS = {"US": US_STATES, "India": IN_STATES}
US_STREET = {   # "st" is never mapped (Street vs Saint, plan P10); "fl" only as floor inside non-state components
    "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "hwy": "highway", "pkwy": "parkway", "cir": "circle", "sq": "square", "pl": "place",
    "ter": "terrace", "trl": "trail", "fwy": "freeway", "expy": "expressway", "apt": "apartment",
    "ste": "suite", "bldg": "building", "flr": "floor", "fl": "floor", "hts": "heights", "jct": "junction",
    "mt": "mount", "ft": "fort", "n": "north", "s": "south", "e": "east", "w": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth", "5th": "fifth",
    "6th": "sixth", "7th": "seventh", "8th": "eighth", "9th": "ninth", "10th": "tenth",
}
IN_STREET = {
    "rd": "road", "nr": "near", "opp": "opposite", "bldg": "building", "flr": "floor", "apt": "apartment",
    "soc": "society", "ngr": "nagar", "mkt": "market", "sec": "sector", "dist": "district",
    "tq": "taluka", "tal": "taluka", "ind": "industrial", "estt": "estate",
}
FR_STREET = {
    "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard", "blvd": "boulevard",
    "pl": "place", "all": "allee", "imp": "impasse", "ch": "chemin", "chem": "chemin", "rte": "route",
    "fbg": "faubourg", "sq": "square", "qu": "quai", "crs": "cours", "st": "saint", "ste": "sainte",
    "pte": "porte", "prom": "promenade", "res": "residence", "zi": "zone industrielle",
}
STREET_MAPS = {"US": US_STREET, "India": IN_STREET, "France": FR_STREET}

@lru_cache(maxsize=1_000_000)
def deva_word(w):
    """Devanagari word -> Latin: IAST, nasals -> n, sibilants -> sh, final schwa deleted, ph -> f."""
    w = w.replace("ॉ", "ो").replace("ऑ", "ओ")        # candra-o (English 'o')
    keep_final_a = len(w) >= 3 and DEVA_CONS_RE.fullmatch(w[-1]) is not None and w[-2] == "्"
    s = transliterate(w, sanscript.DEVANAGARI, sanscript.IAST)
    for a, b in (("ṃ", "n"), ("ṅ", "n"), ("ñ", "n"), ("ṇ", "n"), ("m̐", "n"), ("ś", "sh"), ("ṣ", "sh"), ("ḥ", "h")):
        s = s.replace(a, b)
    if len(s) > 2 and s.endswith("a") and not s.endswith("ā") and not keep_final_a:
        s = s[:-1]
    return unidecode(s).lower().replace("ph", "f")

def base_clean(s):
    """P03 + P04: control characters removed; ftfy only on strings that look damaged. Returns (text, changed)."""
    s = CTRL_RE.sub(" ", s)
    if CONFIG["FTFY"] and MOJIBAKE_RE.search(s):
        fixed = ftfy.fix_text(s)
        return fixed, int(fixed != s)
    return s, 0

def to_latin(s):
    """P07/P08/P09 derived view: NFKC, Devanagari romanised, accents folded, & -> and, punctuation -> space."""
    s = unicodedata.normalize("NFKC", ZW_ALL_RE.sub("", s))
    if CONFIG["DEVA_TRANSLIT"] == "indic" and DEVA_RE.search(s):
        s = DEVA_WORD_RE.sub(lambda m: " " + deva_word(m.group(0)) + " ", s)
    s = unidecode(s).lower().replace("&", " and ")
    return " ".join(NONALNUM_RE.sub(" ", s).split())

def proc_name(raw):
    s, fixed = base_clean(raw.strip())
    light = " ".join(unicodedata.normalize("NFC", ZW_RE.sub("", s)).casefold().split())
    latin = to_latin(s)
    toks = latin.split()
    core_t = [t for t in toks if t not in LEGAL_CANON]
    while core_t and core_t[-1] in ("and", "et"):          # "cherry & co" -> "cherry", not "cherry and"
        core_t.pop()
    while core_t and core_t[0] in ("and", "et"):
        core_t.pop(0)
    legal = " ".join(sorted({LEGAL_CANON[t] for t in toks if t in LEGAL_CANON}))
    if not core_t:
        core_t = toks
    core = " ".join(core_t)
    comp_t = [t for t in core_t if t not in TLDS] if DOMAIN_RE.search(s) else core_t
    compact = "".join(comp_t or core_t)
    parts = DBA_RE.split(s, maxsplit=1)
    alias = " ".join(t for t in to_latin(parts[1]).split() if t not in LEGAL_CANON) if len(parts) > 1 else ""
    init = "".join(t[0] for t in toks if t not in INIT_STOP)
    return dict(name_light=light, name_latin=latin, name_core=core, name_compact=compact, name_legal=legal,
                name_first=core_t[0] if core_t else "", name_alias=alias, name_init=init,
                name_ntok=len(core_t), name_empty=int(not latin), deva=int(bool(DEVA_RE.search(s))), fixed=fixed)

def proc_addr(raw, country):
    s = raw.strip()
    out = dict(addr_plain="", addr_latin="", addr_num="", addr_postal="", addr_unit="", addr_ncomp=0,
               addr_missing=0, addr_fixed=0)
    if not s:
        out["addr_missing"] = 1
        return out
    if CONFIG["PLACEHOLDER_EMPTY"] and s.casefold() in PLACEHOLDERS:      # whole field only; embedded kept
        out["addr_missing"] = 2
        return out
    s, out["addr_fixed"] = base_clean(s)
    out["addr_ncomp"] = sum(1 for c in s.split(",") if c.strip())
    m = POSTAL_RE.get(country)
    if m:
        hits = m.findall(s)
        out["addr_postal"] = hits[-1].replace(" ", "") if hits else ""
    if country not in UNIT_SKIP:
        units = {u.lower().replace("-", "").replace("/", "") for u in UNIT_RE.findall(s)}
        out["addr_unit"] = " ".join(sorted(u.lstrip("0") or "0" if u.isdigit() else u for u in units if u))
    lat = to_latin(s.replace(",", COMMA))
    smap = STATE_MAPS.get(country, {}) if CONFIG["STATE_EXPAND"] else {}
    stmap = STREET_MAPS.get(country, {}) if CONFIG["STREET_ABBREV"] else {}
    plain, mapped = [], []
    for comp in lat.split("zzcommazz"):
        t = comp.split()
        if not t:
            continue
        plain.extend(t)
        if t[0] in smap and all(x.isdigit() for x in t[1:]):     # "TN" or "TN 37814" = a state component
            t = smap[t[0]].split() + t[1:]
        elif stmap:
            t = [stmap.get(x, x) for x in t]
        mapped.extend(t)
    out["addr_plain"], out["addr_latin"] = " ".join(plain), " ".join(mapped)
    out["addr_num"] = " ".join(sorted({x.lstrip("0") or "0" for x in plain if x.isdigit()}))
    if not out["addr_latin"]:
        out["addr_missing"] = 3
    return out

PP_STR = ["name_light", "name_latin", "name_core", "name_compact", "name_legal", "name_first", "name_alias",
          "name_init", "addr_plain", "addr_latin", "addr_num", "addr_postal", "addr_unit"]
PP_NUM = {"name_ntok": pa.int16(), "name_empty": pa.int8(), "deva": pa.int8(), "fixed": pa.int8(),
          "addr_ncomp": pa.int16(), "addr_missing": pa.int8(), "addr_empty": pa.int8()}

def _pp_chunk(args):
    names, addrs, ctrys = args
    cols = {k: [] for k in PP_STR + list(PP_NUM)}
    for n, a, c in zip(names, addrs, ctrys):
        rec = proc_name(n)
        ad = proc_addr(a, c)
        rec.update(ad)
        rec["fixed"] = int(rec["fixed"] or ad["addr_fixed"])
        rec["addr_empty"] = int(ad["addr_missing"] != 0)
        for k in cols:
            cols[k].append(rec[k])
    return cols

PP_KEYS = ["FTFY", "PLACEHOLDER_EMPTY", "STATE_EXPAND", "STREET_ABBREV", "DEVA_TRANSLIT"]

def pp_version():
    """P14: switches + code version + dictionaries -> a short tag used for every preprocessed folder."""
    tag = json.dumps({"switches": {k: CONFIG[k] for k in PP_KEYS}, "code": PP_CODE_VERSION,
                      "dicts": [LEGAL_CANON, STATE_MAPS, STREET_MAPS, sorted(PLACEHOLDERS)]}, sort_keys=True)
    return "pp_" + hashlib.md5(tag.encode()).hexdigest()[:8]

def _count_rows(path):
    n, last = 0, b"\n"
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 22), b""):
            n += buf.count(b"\n"); last = buf[-1:]
    return n + (0 if last == b"\n" else 1) - 1

def preprocess_tsv(src, dst, chunksize=400_000, sub=25_000):
    """TSV -> Parquet with all views. Skips work if dst exists. Uses all CPU cores. Verifies row count (P01)."""
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    t0, n, writer = time.time(), 0, None
    schema = pa.schema([("entity_id", pa.string()), ("country", pa.string()), ("name_raw", pa.string()),
                        ("addr_raw", pa.string())] + [(c, pa.string()) for c in PP_STR]
                       + [(c, t) for c, t in PP_NUM.items()])
    tmp = dst + ".tmp"
    with Pool(os.cpu_count()) as pool:
        for chunk in read_tsv(src, chunksize=chunksize):
            ids = chunk["entity_id"].tolist(); names = chunk["business_name"].tolist()
            addrs = chunk["business_address"].tolist(); ctry = [c.strip() for c in chunk["country"].tolist()]
            jobs = [(names[i:i + sub], addrs[i:i + sub], ctry[i:i + sub]) for i in range(0, len(ids), sub)]
            cols = {k: [] for k in PP_STR + list(PP_NUM)}
            for part in pool.imap(_pp_chunk, jobs):
                for k in cols:
                    cols[k].extend(part[k])
            data = {"entity_id": ids, "country": ctry, "name_raw": names, "addr_raw": addrs, **cols}
            table = pa.table({f.name: pa.array(data[f.name], type=f.type) for f in schema})
            if writer is None:
                writer = pq.ParquetWriter(tmp, schema)
            writer.write_table(table)
            n += len(ids)
    if writer is not None:
        writer.close()
    else:
        pq.write_table(schema.empty_table(), tmp)
    expected = _count_rows(src)
    assert n == expected, f"row count mismatch for {src}: wrote {n:,}, file has {expected:,}"
    os.replace(tmp, dst)
    log(f"preprocessed {n:,} rows (verified)  {os.path.basename(src)} -> {os.path.basename(dst)}  ({time.time()-t0:.0f}s)")
    return dst

# %%
# 2.2 Regression gallery (plan P14): original field, derived views and flags on known cases
GALLERY = [
    ("A DIAMOND  SHORT LLC", "126-B New Line Road, 3rd FL, MORRISTOWN, TN", "US"),
    ("wilfordhancock.com", "Mack Rd, Haltom City, Texas", "US"),
    ("Krebs View Inc", "1795 Westchester Drive, Unit C6, High Point, NC", "US"),
    ("krebsview.com", "1795 WESTCHESTER DR # C6, HIGH POINT, NC", "US"),
    ("-- Holloway Peak Inc Seafood", "00530 RAMOS DRIVE, MORGANTON, NC 02865", "US"),
    ("Vantagexylo D.B.A. Trinity Lutheran Church", "321 Coal Bend, NULL, Delaware, Ohio", "US"),
    ("Cherry & Co Co", "12 Elm St, Tyler, TX", "US"),
    ("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India"),
    ("आनंद फाउंडेशन प्राइवेट लिमिटेड", "L-1318/38 GROUND FLOOR SANGAM VIHAR, DL", "India"),
    ("PVT. INDUSTRIES PMB LEASING LIMITED", "SURAT, OFFICE NO. 03038, NR RUSHABH PETROL PUMP, RING RD, Gujarat 395 003", "India"),
    ("SCI Ptit Àmicale", "18 RUE JEN ZAY, Dunkerque, Nord 59140", "France"),
    ("Marina Ecole France Sarl", "63 R. DE DIEPPE, LILLE, Hauts-de-France", "France"),
    ("CafÃ© Bleu SARL", "NULL", "France"),
    ("---", "  ", "US"),
    ("Müller Bäckerei GmbH", "Hauptstraße 12, Whg. 3, 10115 Berlin", "Germany"),   # unseen label: generic defaults
]
for n, a, c in GALLERY:
    r, d = proc_name(n), proc_addr(a, c)
    print(f"[{c}] name {n!r} | addr {a!r}")
    print(f"   latin={r['name_latin']!r} core={r['name_core']!r} legal={r['name_legal']!r} compact={r['name_compact']!r}"
          f" alias={r['name_alias']!r} init={r['name_init']!r} deva={r['deva']} fixed={r['fixed']} name_empty={r['name_empty']}")
    print(f"   addr_plain={d['addr_plain']!r}\n   addr_latin={d['addr_latin']!r} num={d['addr_num']!r} "
          f"postal={d['addr_postal']!r} unit={d['addr_unit']!r} ncomp={d['addr_ncomp']} missing={d['addr_missing']}")
log("preprocessing version:", pp_version(), {k: CONFIG[k] for k in PP_KEYS})

# %% [markdown]
# ## 3. Blocking, features, matcher, evaluation (shared by Phase A and Phase B)

# %%
# 3.1 Columnar tables (pyarrow keeps millions of strings compact in RAM)
LOAD_COLS = ["entity_id", "country"] + PP_STR + list(PP_NUM)

class Table:
    def __init__(self, paths, country=None):
        paths = [paths] if isinstance(paths, str) else list(paths)
        filt = [("country", "=", country)] if country is not None else None
        parts = [pq.read_table(p, columns=LOAD_COLS, filters=filt) for p in paths]
        t = pa.concat_tables(parts) if len(parts) > 1 else parts[0]
        self.n = t.num_rows
        self.cols = {c: t.column(c).combine_chunks() for c in LOAD_COLS}
        self._num, self._len = {}, {}
    def take(self, col, idx):
        return self.cols[col].take(pa.array(idx, type=pa.int64())).to_numpy(zero_copy_only=False)
    def num(self, col):
        if col not in self._num:
            self._num[col] = self.cols[col].to_numpy(zero_copy_only=False)
        return self._num[col]
    def strlen(self, col):
        if col not in self._len:
            self._len[col] = pc.utf8_length(self.cols[col]).to_numpy(zero_copy_only=False).astype(np.int32)
        return self._len[col]
    def ids(self):
        return self.cols["entity_id"].to_numpy(zero_copy_only=False)
    def hnum(self):
        """First number of the address (leading zeros stripped): (string array, float array with NaN)."""
        if "_hn" not in self._num:
            e = pc.extract_regex(self.cols["addr_plain"], r"(?:^|\s)0*(?P<h>\d+)")
            h = pc.fill_null(pc.struct_field(e, "h"), "")
            v = pc.cast(pc.if_else(pc.equal(h, ""), None, h), pa.float64()).to_numpy(zero_copy_only=False)
            self._num["_hn"] = (h, np.asarray(v, dtype=np.float64))
            self._len["_hn"] = pc.utf8_length(h).to_numpy(zero_copy_only=False).astype(np.int32)
        return self._num["_hn"]
    def take_hn(self, idx):
        return self.hnum()[0].take(pa.array(idx, type=pa.int64())).to_numpy(zero_copy_only=False)
    def add_col(self, name, values):
        self.cols[name] = pa.array(values)
        self._num.pop(name, None)
    def subset(self, mask):
        new = object.__new__(Table)
        m = pa.array(mask)
        new.cols = {c: a.filter(m) for c, a in self.cols.items()}
        new.n = len(new.cols["entity_id"]); new._num, new._len = {}, {}
        return new

def add_dup_counts(tab, prefix, pop=None):
    """How common a record's core name / address is in its population (per 100k records); NaN for empty values.
    pop=None: the table itself is the population (candidate pools, full test S1)."""
    src = pop if pop is not None else tab
    for col, out in (("name_core", f"{prefix}name_dup"), ("addr_latin", f"{prefix}addr_dup")):
        vals = src.cols[col].to_numpy(zero_copy_only=False)
        codes, uniq = pd.factorize(vals)
        counts = np.bincount(codes, minlength=len(uniq)).astype(np.float64)
        if pop is None:
            rec = counts[codes]
            own = vals
        else:
            own = tab.cols[col].to_numpy(zero_copy_only=False)
            pos = pd.Index(uniq).get_indexer(own)
            rec = np.where(pos >= 0, counts[np.maximum(pos, 0)], 1.0)
        rate = rec * 1e5 / max(src.n, 1)
        rate[np.asarray([not v for v in own])] = np.nan
        tab.add_col(out, rate.astype(np.float32))

def countries_in(paths):
    paths = [paths] if isinstance(paths, str) else paths
    s = set()
    for p in paths:
        s |= set(pq.read_table(p, columns=["country"]).column("country").unique().to_pylist())
    return sorted(s)

# %%
# 3.2 Blocker: IDF-weighted name+address cosine top-K  ∪  name-only cosine top-K
#     (multithreaded top-N sparse products; the full score rows are never materialised)
NTHREADS = max(1, os.cpu_count() or 1)

def _topn_arrays(C):
    counts = np.diff(C.indptr)
    r = np.repeat(np.arange(C.shape[0], dtype=np.int64), counts)
    rank = (np.arange(C.nnz) - np.repeat(C.indptr[:-1], counts)).astype(np.int16)
    return r, C.indices.astype(np.int64), rank

def _rowdot(Q, P, r, c):
    return np.asarray(Q[r].multiply(P[c]).sum(axis=1)).ravel().astype(np.float32)

class Blocker:
    def __init__(self, pool, df_max_frac, w_name):
        t0 = time.time()
        self.N, self.w = pool.n, w_name
        self.cv_n, self.keep_n, self.idf_n, self.Pn = self._fit(pool.cols["name_core"], df_max_frac)
        self.cv_a, self.keep_a, self.idf_a, self.Pa = self._fit(pool.cols["addr_latin"], df_max_frac)
        self.PnT = self.Pn.T.tocsr()
        self.PT = sp.vstack([self.PnT * w_name, self.Pa.T.tocsr() * (1 - w_name)]).tocsr().astype(np.float32)
        gc.collect()
        self.stats = dict(pool=self.N, name_vocab=len(self.keep_n), addr_vocab=len(self.keep_a),
                          nnz=int(self.PT.nnz), secs=round(time.time() - t0, 1))
    def _fit(self, arr, frac):
        docs = arr.to_numpy(zero_copy_only=False)
        cv = CountVectorizer(binary=True, lowercase=False, token_pattern=r"(?u)\b\w\w+\b", dtype=np.float32)
        try:
            X = cv.fit_transform(docs)
        except ValueError:                      # no tokens at all in this channel (e.g. every address empty)
            cv = CountVectorizer(vocabulary={"__none__": 0}, binary=True, lowercase=False, dtype=np.float32)
            X = cv.fit_transform(docs)
        del docs
        df = np.bincount(X.indices, minlength=X.shape[1])
        cap = max(5, int(frac * X.shape[0]))
        keep = np.flatnonzero((df > 0) & (df <= cap))
        if keep.size == 0:                      # every token is common: fall back to the rarest tokens
            keep = np.flatnonzero(df > 0)
            keep = keep[np.argsort(df[keep])[:max(1, keep.size // 10)]] if keep.size else np.array([0])
        X = X[:, keep].tocsr()
        idf = np.log(X.shape[0] / np.maximum(df[keep], 1)).astype(np.float32)
        X.data = idf[X.indices]
        X = normalize(X, norm="l2", copy=False).astype(np.float32)
        return cv, keep, idf, X
    def _qvec(self, cv, keep, idf, arr):
        Xq = cv.transform(arr.to_numpy(zero_copy_only=False))[:, keep].tocsr()
        Xq.data = idf[Xq.indices]
        return Xq.astype(np.float32)
    def query(self, qtab, k_main, k_name, batch=2048):
        Qn = self._qvec(self.cv_n, self.keep_n, self.idf_n, qtab.cols["name_core"])
        Qa = self._qvec(self.cv_a, self.keep_a, self.idf_a, qtab.cols["addr_latin"])
        Q = sp.hstack([Qn, Qa]).tocsr()
        N, w = self.N, self.w
        out = {k: [] for k in ("q", "c", "blk", "ncos", "acos", "rank", "in_name")}
        for s in range(0, qtab.n, batch):
            e = min(s + batch, qtab.n)
            Cm = sp_matmul_topn(Q[s:e], self.PT, top_n=k_main, sort=True, n_threads=NTHREADS)
            rm, cm, km = _topn_arrays(Cm)
            if k_name > 0:
                Cn = sp_matmul_topn(Qn[s:e], self.PnT, top_n=k_name, sort=True, n_threads=NTHREADS)
                rn, cn, _ = _topn_arrays(Cn)
            else:
                rn = cn = np.zeros(0, np.int64)
            km_keys, kn_keys = rm * N + cm, rn * N + cn
            keys = np.union1d(km_keys, kn_keys)
            if len(keys) == 0:
                continue
            r, c = keys // N, keys % N
            rank = np.full(len(keys), k_main, np.int16)
            rank[np.searchsorted(keys, km_keys)] = km
            ncos = _rowdot(Qn[s:e], self.Pn, r, c)
            acos = _rowdot(Qa[s:e], self.Pa, r, c)
            out["q"].append(r + s); out["c"].append(c); out["ncos"].append(ncos); out["acos"].append(acos)
            out["blk"].append((w * ncos + (1 - w) * acos).astype(np.float32)); out["rank"].append(rank)
            out["in_name"].append(np.isin(keys, kn_keys).astype(np.int8))
        return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}

# %%
# 3.3 Pairwise features (computed per query-aligned chunk, multithreaded C++ string scorers)
FEATS = ["n_core_ratio", "n_core_tset", "n_core_tsort", "n_core_partial", "n_core_jw",
         "n_latin_ratio", "n_latin_tset", "n_compact_ratio", "n_compact_partial", "n_light_ratio",
         "n_core_exact", "n_first_eq", "legal_same", "legal_conflict", "alias_best", "init_match",
         "a_ratio", "a_tset", "a_partial", "a_plain_tset", "num_tset", "num_exact", "q_has_num", "c_has_num",
         "postal_eq", "postal_conflict", "unit_eq", "unit_conflict", "q_ncomp", "c_ncomp",
         "c_addr_missing", "c_fixed", "q_deva", "c_deva", "cross_script", "q_ntok", "c_ntok", "len_ratio", "is_s3",
         "hn_both", "hn_eq", "hn_absdiff", "hn_reldiff", "hn_lev", "hn_trunc",
         "q_name_dup", "q_addr_dup", "c_name_dup", "c_addr_dup",
         "blk", "ncos", "acos", "rank", "in_name",
         "n_cand", "blk_rel", "blk_gap", "name_gap", "addr_gap"]

if not CONFIG["CONTEXT_FEATURES"]:
    FEATS = [f for f in FEATS if f not in ("q_name_dup", "q_addr_dup", "c_name_dup", "c_addr_dup")]

def _cp(a, b, scorer, empty):
    v = cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    v[empty] = np.nan
    return v

def _cp_where(a, b, scorer, valid):
    """Score only the rows where `valid` (rare views such as aliases); NaN elsewhere."""
    v = np.full(len(a), np.nan, np.float32)
    if valid.any():
        v[valid] = cpdist(a[valid], b[valid], scorer=scorer, workers=-1, dtype=np.float32)
    return v

def _eq_conflict(a, b, m):
    return ((a == b) & ~m).astype(np.float32), ((a != b) & ~m).astype(np.float32)

def query_chunks(q, size):
    """Chunk boundaries that never split one query's candidates (q must be sorted)."""
    s, n = 0, len(q)
    while s < n:
        e = min(s + size, n)
        if e < n:
            e = int(np.searchsorted(q, q[e], side="left"))
            if e <= s:
                e = int(np.searchsorted(q, q[s], side="right"))
        yield s, e
        s = e

def pair_features(P, qtab, ptab, pool_ids, chunk=600_000):
    frames = []
    for s, e in query_chunks(P["q"], chunk):
        qi, ci = P["q"][s:e], P["c"][s:e]
        f = {}
        def both(col):
            return qtab.take(col, qi), ptab.take(col, ci)
        def empty(col):
            return (qtab.strlen(col)[qi] == 0) | (ptab.strlen(col)[ci] == 0)
        a, b = both("name_core"); m = empty("name_core")
        f["n_core_ratio"] = _cp(a, b, fuzz.ratio, m); f["n_core_tset"] = _cp(a, b, fuzz.token_set_ratio, m)
        f["n_core_tsort"] = _cp(a, b, fuzz.token_sort_ratio, m); f["n_core_partial"] = _cp(a, b, fuzz.partial_ratio, m)
        f["n_core_jw"] = _cp(a, b, JaroWinkler.normalized_similarity, m)
        f["n_core_exact"] = ((a == b) & ~m).astype(np.float32)
        a, b = both("name_latin"); m = empty("name_latin")
        f["n_latin_ratio"] = _cp(a, b, fuzz.ratio, m); f["n_latin_tset"] = _cp(a, b, fuzz.token_set_ratio, m)
        a, b = both("name_compact"); m = empty("name_compact")
        f["n_compact_ratio"] = _cp(a, b, fuzz.ratio, m); f["n_compact_partial"] = _cp(a, b, fuzz.partial_ratio, m)
        a, b = both("name_light"); m = empty("name_light")
        f["n_light_ratio"] = _cp(a, b, fuzz.ratio, m)
        a, b = both("name_first"); m = empty("name_first")
        f["n_first_eq"] = ((a == b) & ~m).astype(np.float32)
        a, b = both("name_legal"); m = empty("name_legal")
        f["legal_same"], f["legal_conflict"] = _eq_conflict(a, b, m)
        # P16 aliases: query core vs candidate alias, and the reverse
        qc_core, cc_core = both("name_core"); qa, ca = both("name_alias")
        v1 = _cp_where(qc_core, ca, fuzz.token_set_ratio, ptab.strlen("name_alias")[ci] > 0)
        v2 = _cp_where(qa, cc_core, fuzz.token_set_ratio, qtab.strlen("name_alias")[qi] > 0)
        f["alias_best"] = np.fmax(v1, v2)
        # P16 initials: only against very short names (abbreviations such as "CC")
        qi_init, ci_init = both("name_init"); qcomp, ccomp = both("name_compact")
        v1 = _cp_where(qi_init, ccomp, fuzz.ratio, (ptab.strlen("name_compact")[ci] <= 5) & (qtab.strlen("name_init")[qi] >= 2))
        v2 = _cp_where(ci_init, qcomp, fuzz.ratio, (qtab.strlen("name_compact")[qi] <= 5) & (ptab.strlen("name_init")[ci] >= 2))
        f["init_match"] = np.fmax(v1, v2)
        a, b = both("addr_latin"); m = empty("addr_latin")
        f["a_ratio"] = _cp(a, b, fuzz.ratio, m); f["a_tset"] = _cp(a, b, fuzz.token_set_ratio, m)
        f["a_partial"] = _cp(a, b, fuzz.partial_ratio, m)
        a, b = both("addr_plain"); m = empty("addr_plain")
        f["a_plain_tset"] = _cp(a, b, fuzz.token_set_ratio, m)
        a, b = both("addr_num"); m = empty("addr_num")
        f["num_tset"] = _cp(a, b, fuzz.token_set_ratio, m); f["num_exact"] = ((a == b) & ~m).astype(np.float32)
        f["q_has_num"] = (qtab.strlen("addr_num")[qi] > 0).astype(np.float32)
        f["c_has_num"] = (ptab.strlen("addr_num")[ci] > 0).astype(np.float32)
        a, b = both("addr_postal"); m = empty("addr_postal")
        f["postal_eq"], f["postal_conflict"] = _eq_conflict(a, b, m)
        a, b = both("addr_unit"); m = empty("addr_unit")
        f["unit_eq"], f["unit_conflict"] = _eq_conflict(a, b, m)
        f["q_ncomp"] = qtab.num("addr_ncomp")[qi].astype(np.float32)
        f["c_ncomp"] = ptab.num("addr_ncomp")[ci].astype(np.float32)
        f["c_addr_missing"] = ptab.num("addr_missing")[ci].astype(np.float32)
        f["c_fixed"] = ptab.num("fixed")[ci].astype(np.float32)
        # house numbers: equality, numeric distance, digit edit distance, "digit dropped" (11102 vs 1110)
        hq, hc = qtab.hnum()[1][qi], ptab.hnum()[1][ci]
        both = ~np.isnan(hq) & ~np.isnan(hc)
        f["hn_both"] = both.astype(np.float32)
        f["hn_eq"] = np.where(both, (hq == hc), np.nan).astype(np.float32)
        dd = np.abs(hq - hc)
        f["hn_absdiff"] = np.where(both, np.log1p(dd), np.nan).astype(np.float32)
        f["hn_reldiff"] = np.where(both, dd / np.maximum(np.maximum(hq, hc), 1), np.nan).astype(np.float32)
        lev = _cp(qtab.take_hn(qi), ptab.take_hn(ci), Levenshtein.distance, ~both)
        lq, lc = qtab._len["_hn"][qi], ptab._len["_hn"][ci]
        f["hn_lev"] = lev
        f["hn_trunc"] = np.where(both, (lev == np.abs(lq - lc)) & (lq != lc), np.nan).astype(np.float32)
        # context: how common the name / address is among S1 (query side) and in the pool (candidate side)
        for side, tab, idx in (("q", qtab, qi), ("c", ptab, ci)):
            for k in ("name_dup", "addr_dup"):
                col = f"{side}_{k}"
                f[col] = tab.num(col)[idx].astype(np.float32) if col in tab.cols else np.full(len(idx), np.nan, np.float32)
        qd, cd = qtab.num("deva")[qi], ptab.num("deva")[ci]
        f["q_deva"], f["c_deva"] = qd.astype(np.float32), cd.astype(np.float32)
        f["cross_script"] = (qd != cd).astype(np.float32)
        qn, cn = qtab.num("name_ntok")[qi].astype(np.float32), ptab.num("name_ntok")[ci].astype(np.float32)
        f["q_ntok"], f["c_ntok"] = qn, cn
        ql, cl = qtab.strlen("name_core")[qi].astype(np.float32), ptab.strlen("name_core")[ci].astype(np.float32)
        f["len_ratio"] = np.minimum(ql, cl) / np.maximum(np.maximum(ql, cl), 1)
        f["is_s3"] = np.char.startswith(pool_ids[ci].astype(str), "S3-").astype(np.float32)
        for k in ("blk", "ncos", "acos", "rank", "in_name"):
            f[k] = P[k][s:e].astype(np.float32)
        df = pd.DataFrame(f)
        g = pd.Series(qi).groupby(qi)
        df["n_cand"] = g.transform("size").to_numpy(np.float32)
        qmax = df.groupby(qi)["blk"].transform("max").to_numpy()
        df["blk_rel"] = df["blk"] / np.maximum(qmax, 1e-6); df["blk_gap"] = qmax - df["blk"]
        df["name_gap"] = df.groupby(qi)["n_core_tset"].transform("max") - df["n_core_tset"]
        df["addr_gap"] = df.groupby(qi)["a_tset"].transform("max") - df["a_tset"]
        frames.append(df[FEATS].astype(np.float32))
    if not frames:
        return pd.DataFrame(columns=FEATS, dtype=np.float32)
    return pd.concat(frames, ignore_index=True)

# %%
# 3.4 Labels, training, decision rule and exact macro-F0.5
def label_pairs(q_ids, c_ids, gt):
    empty = frozenset()
    return np.fromiter((c in gt.get(q, empty) for q, c in zip(q_ids, c_ids)), dtype=np.int8, count=len(q_ids))

LGB_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.08, num_leaves=127,
                  min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, seed=42)

def train_matcher(X, y, q_ids, rounds):
    uq = np.unique(q_ids)
    rng = np.random.default_rng(CONFIG["SEED"])
    es_q = set(rng.choice(uq, size=max(1, len(uq) // 10), replace=False).tolist())
    es = np.fromiter((q in es_q for q in q_ids), dtype=bool, count=len(q_ids))
    dtr = lgb.Dataset(X[~es], label=y[~es], free_raw_data=True)
    dva = lgb.Dataset(X[es], label=y[es], reference=dtr, free_raw_data=True)
    t0 = time.time()
    model = lgb.train(LGB_PARAMS, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])
    log(f"trained {model.best_iteration} trees on {int((~es).sum()):,} pairs "
        f"({int(y[~es].sum()):,} positive) in {time.time()-t0:.0f}s")
    return model

def decide(qc, cc, p, thr, resolve=True, max_pred=None):
    """Keep pairs with p >= thr; each pool record goes to at most one S1 (many-to-one); cap per S1."""
    m = p >= thr
    idx = np.flatnonzero(m)
    if resolve and len(idx):
        o = idx[np.lexsort((-p[idx], cc[idx]))]
        first = np.r_[True, cc[o][1:] != cc[o][:-1]]
        idx = o[first]
    if max_pred and len(idx):
        o = idx[np.lexsort((-p[idx], qc[idx]))]
        qs = qc[o]
        start = np.r_[True, qs[1:] != qs[:-1]]
        grp_start = np.maximum.accumulate(np.where(start, np.arange(len(o)), 0))
        idx = o[(np.arange(len(o)) - grp_start) < max_pred]
    return idx

def f05_scores(tp, npred, tsize):
    f = np.zeros(len(tp), np.float64)
    sing = tsize == 0
    f[sing] = (npred[sing] == 0)
    ok = (~sing) & (tp > 0)
    P = tp[ok] / npred[ok]; R = tp[ok] / tsize[ok]
    f[ok] = 1.25 * P * R / (0.25 * P + R)
    return f

TEST_W = None   # set in B.1 from the test S1 counts when CONFIG["TEST_WEIGHTS"] == "auto"

def test_weights():
    w = CONFIG["TEST_WEIGHTS"]
    return w if isinstance(w, dict) else TEST_W

def evaluate(E, idx):
    """E: eval bundle (per-pair qc, y; per-entity tsize, country). idx: kept pair indices."""
    nE = len(E["tsize"])
    tp = np.bincount(E["qc"][idx], weights=E["y"][idx], minlength=nE)
    npred = np.bincount(E["qc"][idx], minlength=nE)
    f = f05_scores(tp, npred, E["tsize"])
    res = {"F05": float(f.mean()), "n": int(nE)}
    for c in np.unique(E["country"]):
        res[f"F05_{c}"] = float(f[E["country"] == c].mean())
    w = test_weights() or {}
    present = [c for c in w if f"F05_{c}" in res]
    res["F05_testmix"] = float(sum(w[c] * res[f"F05_{c}"] for c in present) / sum(w[c] for c in present)) if present else res["F05"]
    nz = E["tsize"] > 0
    res["mean_P"] = float(np.mean(np.where(npred[nz] > 0, tp[nz] / np.maximum(npred[nz], 1), 0)))
    res["mean_R"] = float(np.mean(tp[nz] / E["tsize"][nz]))
    res["singleton_acc"] = float(np.mean(npred[~nz] == 0)) if (~nz).any() else None
    res["pred_per_S1"] = float(npred.mean())
    return res, f

def make_eval_bundle(q_ids_pairs, y, eval_ids, eval_country, gt):
    code = {q: i for i, q in enumerate(eval_ids)}
    return dict(qc=np.fromiter((code[q] for q in q_ids_pairs), dtype=np.int64, count=len(q_ids_pairs)),
                y=y.astype(np.float64),
                tsize=np.array([len(gt[q]) for q in eval_ids], dtype=np.float64),
                country=np.asarray(eval_country, dtype=object), ids=np.asarray(eval_ids, dtype=object))

def tune_threshold(E, cc, p, resolve=True):
    rows = []
    for thr in CONFIG["THR_GRID"]:
        r, _ = evaluate(E, decide(E["qc"], cc, p, thr, resolve, CONFIG["MAX_PRED"]))
        rows.append({"thr": thr, **r})
    tab = pd.DataFrame(rows)
    best = tab.loc[tab["F05_testmix"].idxmax()]
    return float(best["thr"]), tab

def blocking_report(E, rank, in_name, k_main):
    """Recall of the true pairs that exist, by candidate channel and cut-off."""
    total = E["tsize"].sum()
    pos = E["y"] > 0
    rep = {"true_pairs": int(total), "cand_pairs": int(len(rank)),
           "cand_per_S1": round(len(rank) / len(E["tsize"]), 1)}
    for k in sorted({10, 20, 30, k_main}):
        if k <= k_main:
            rep[f"recall_main@{k}"] = round(float((pos & (rank < k)).sum() / total), 4)
    rep["recall_union"] = round(float(pos.sum() / total), 4)
    rep["name_channel_only_hits"] = int((pos & (rank >= k_main)).sum())
    found = np.bincount(E["qc"], weights=E["y"], minlength=len(E["tsize"]))
    nz = E["tsize"] > 0
    rep["S1_all_found"] = round(float(np.mean(found[nz] == E["tsize"][nz])), 4)
    rep["ceiling_F05"] = round(float(f05_scores(found, found, E["tsize"]).mean()), 4)   # perfect matcher
    return rep

# %%
# 3.5 One stage runner: block every query set against a pool, per country, then build features + labels
def run_stage(pool_paths, query_sets, gt, tag, pop_paths=None):
    """query_sets: {name: parquet path of S1 queries}. pop_paths: the full S1 population the queries come from
    (for the name / address frequency features). Returns {name: dict(X, y, q_ids, c_ids, rank, in_name)}."""
    out = {name: {k: [] for k in ("X", "y", "q_ids", "c_ids", "rank", "in_name")} for name in query_sets}
    all_ctry = sorted(set(countries_in(list(query_sets.values()))) & set(countries_in(pool_paths)))
    stats = {}
    for ctry in all_ctry:
        pool = Table(pool_paths, ctry)
        add_dup_counts(pool, "c_")
        pop = Table(pop_paths, ctry) if pop_paths else None
        blk = Blocker(pool, CONFIG["DF_MAX_FRAC"], CONFIG["W_NAME"])
        pool_ids = pool.ids()
        stats[ctry] = blk.stats
        log(f"[{tag}] {ctry}: pool {pool.n:,}  vocab name {blk.stats['name_vocab']:,} / addr "
            f"{blk.stats['addr_vocab']:,}  index {blk.stats['secs']}s")
        for name, qpath in query_sets.items():
            qt = Table(qpath, ctry)
            if qt.n == 0:
                continue
            add_dup_counts(qt, "q_", pop)
            t0 = time.time()
            P = blk.query(qt, CONFIG["K_MAIN"], CONFIG["K_NAME"], CONFIG["BLOCK_BATCH"])
            t1 = time.time()
            X = pair_features(P, qt, pool, pool_ids)
            q_ids = qt.ids()[P["q"]]; c_ids = pool_ids[P["c"]]
            d = out[name]
            d["X"].append(X); d["y"].append(label_pairs(q_ids, c_ids, gt)); d["q_ids"].append(q_ids)
            d["c_ids"].append(c_ids); d["rank"].append(P["rank"]); d["in_name"].append(P["in_name"])
            log(f"[{tag}] {ctry} {name}: {qt.n:,} S1 -> {len(q_ids):,} pairs  "
                f"(block {t1-t0:.0f}s, features {time.time()-t1:.0f}s)")
        del blk, pool, pop; gc.collect()
    for name, d in out.items():
        for k in d:
            d[k] = (pd.concat(d[k], ignore_index=True) if k == "X" else np.concatenate(d[k])) if d[k] else \
                   (pd.DataFrame(columns=FEATS, dtype=np.float32) if k == "X" else np.zeros(0))
    return out, stats

def s1_table_info(path):
    t = pq.read_table(path, columns=["entity_id", "country"])
    return t.column("entity_id").to_pylist(), t.column("country").to_pylist()

def sample_queries(src_parquet, dst_parquet, n, seed):
    if os.path.exists(dst_parquet):
        return dst_parquet
    t = pq.read_table(src_parquet)
    if t.num_rows > n:
        idx = np.sort(np.random.default_rng(seed).choice(t.num_rows, size=n, replace=False))
        t = t.take(pa.array(idx))
    pq.write_table(t, dst_parquet)
    return dst_parquet

def fit_and_validate(stage, gt, tag):
    tr, va = stage["train"], stage["val"]
    model = train_matcher(tr["X"].to_numpy(np.float32), tr["y"], tr["q_ids"], CONFIG["LGB_ROUNDS"])
    p = model.predict(va["X"].to_numpy(np.float32), num_iteration=model.best_iteration)
    val_ids, val_ctry = s1_table_info(stage["val_path"])
    E = make_eval_bundle(va["q_ids"], va["y"], val_ids, val_ctry, gt)
    cc = pd.factorize(va["c_ids"])[0]
    blk_rep = blocking_report(E, va["rank"], va["in_name"], CONFIG["K_MAIN"])
    thr, tab = tune_threshold(E, cc, p, resolve=True)
    best, f_ent = evaluate(E, decide(E["qc"], cc, p, thr, True, CONFIG["MAX_PRED"]))
    no_res, _ = evaluate(E, decide(E["qc"], cc, p, thr, False, CONFIG["MAX_PRED"]))
    imp = pd.Series(model.feature_importance("gain"), index=FEATS).sort_values(ascending=False)
    log(f"[{tag}] blocking: {blk_rep}")
    log(f"[{tag}] best threshold {thr}:  " + "  ".join(f"{k}={v:.4f}" for k, v in best.items() if isinstance(v, float)))
    log(f"[{tag}] without many-to-one resolution: F05={no_res['F05']:.4f}  testmix={no_res['F05_testmix']:.4f}")
    print(tab[["thr", "F05", "F05_testmix", "mean_P", "mean_R", "singleton_acc", "pred_per_S1"]]
          .iloc[::2].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("top features (gain):", ", ".join(f"{k}={v:.0f}" for k, v in imp.head(12).items()))
    return dict(model=model, p=p, E=E, cc=cc, thr=thr, metrics=best, no_resolution=no_res, blocking=blk_rep,
                thr_table=tab, importance=imp, f_entity=f_ent)

def error_examples(stage, R, pool_paths, n=8, seed=0):
    """Print false merges and missed matches (normalised views) for the write-up."""
    va, E, p, thr = stage["val"], R["E"], R["p"], R["thr"]
    kept = np.zeros(len(p), bool); kept[decide(E["qc"], R["cc"], p, thr, True, CONFIG["MAX_PRED"])] = True
    rng = np.random.default_rng(seed)
    fp = np.flatnonzero(kept & (va["y"] == 0)); fn = np.flatnonzero(~kept & (va["y"] == 1))
    pick = lambda a: a[rng.choice(len(a), size=min(n, len(a)), replace=False)] if len(a) else a
    fp, fn = pick(fp), pick(fn)
    need = pa.array(sorted(set(va["q_ids"][np.r_[fp, fn]]) | set(va["c_ids"][np.r_[fp, fn]])), type=pa.string())
    look = {}
    for path in [stage["val_path"]] + list(pool_paths):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=["entity_id", "name_raw", "addr_raw"]):
            hit = batch.filter(pc.is_in(batch.column(0), value_set=need))
            look.update({r["entity_id"]: (r["name_raw"], r["addr_raw"]) for r in hit.to_pylist()})
    ex = {"false_merges": [], "missed": []}
    for kind, arr in (("false_merges", fp), ("missed", fn)):
        print(f"\n--- {kind} ---")
        for i in arr:
            q, c = va["q_ids"][i], va["c_ids"][i]
            row = {"p": round(float(p[i]), 3), "S1": look.get(q), "cand": look.get(c), "rank": int(va["rank"][i])}
            ex[kind].append(row)
            print(f"p={row['p']:.3f} rank={row['rank']:>2}  S1={row['S1']}\n{'':22}cand={row['cand']}")
    return ex

# %% [markdown]
# ## Phase A — mini-world (fast iteration)
# Queries: sampled mini-train S1 (training) + the 20k mini-val S1 (evaluation). Pool: the shared mini S2/S3 pool.
# Scores here are optimistic (10× fewer distractors), but comparisons between settings are reliable: every
# comparison is paired on the same 20k val S1. Phase B repeats the final setting against the full pool.
#
# * **A.0 gate (plan Q12):** the preprocessing rules marked *check before enabling* (P02 placeholders, P03 ftfy,
#   P09 transliteration, P10 state / street abbreviations) start OFF. Each is switched on alone; it is adopted only if
#   the test-mix-weighted macro-F0.5 gain on the same val S1 is positive and larger than twice its paired standard error.
# * **A.1** frequency cap for blocking keys (P17) · **A.2** final mini run · **A.3** leave-one-country-out.

# %%
# A.0 Helpers and the preprocessing-rule gate
def preprocess_mini():
    ppd = os.path.join(WORK, pp_version(), "mini")
    pp = {k: preprocess_tsv(MINI_PATHS[src], os.path.join(ppd, f"{k}.parquet"))
          for k, src in (("train_s1", "train_s1"), ("val_s1", "val_s1"), ("pool_s2", "s2"), ("pool_s3", "s3"))}
    pp["train_q"] = sample_queries(pp["train_s1"], os.path.join(ppd, f"train_q_{CONFIG['N_TRAIN_QUERIES']}.parquet"),
                                   CONFIG["N_TRAIN_QUERIES"], CONFIG["SEED"])
    gt = {**load_gt(MINI_PATHS["train_gt"]), **load_gt(MINI_PATHS["val_gt"])}
    return pp, gt

def run_mini(overrides=None, tag="mini", errors=True):
    saved = dict(CONFIG)
    CONFIG.update(overrides or {})
    try:
        pp, gt = preprocess_mini()
        pools = [pp["pool_s2"], pp["pool_s3"]]
        stage, stats = run_stage(pools, {"train": pp["train_q"], "val": pp["val_s1"]}, gt, tag,
                                 pop_paths=[pp["train_s1"], pp["val_s1"]])
        stage["val_path"] = pp["val_s1"]
        R = fit_and_validate(stage, gt, tag)
        R["examples"] = error_examples(stage, R, pools) if errors else None
        R["stage"], R["pool_stats"], R["pools"], R["train_q_path"] = stage, stats, pools, pp["train_q"]
        return R
    finally:
        CONFIG.clear(); CONFIG.update(saved)

def paired_delta(R0, R1):
    """Test-mix-weighted macro-F0.5 difference on the same val S1, and its paired standard error."""
    E0, E1 = R0["E"], R1["E"]
    assert (E0["ids"] == E1["ids"]).all(), "gate runs must evaluate the same val S1 in the same order"
    d = R1["f_entity"] - R0["f_entity"]
    w = test_weights() or {c: float((E0["country"] == c).mean()) for c in np.unique(E0["country"])}
    present = [c for c in w if (E0["country"] == c).any()] or list(np.unique(E0["country"]))
    tot = sum(w.get(c, 1.0) for c in present)
    delta = var = 0.0
    for c in present:
        m = E0["country"] == c
        wc = w.get(c, 1.0) / tot
        delta += wc * d[m].mean()
        var += (wc / m.sum()) ** 2 * ((d[m] - d[m].mean()) ** 2).sum()
    return float(delta), float(np.sqrt(var))

t_a = time.time()
GATE_ROWS = []
if CONFIG["RUN_SWITCH_GATE"]:
    g = {"N_TRAIN_QUERIES": CONFIG["GATE_TRAIN_QUERIES"]}
    base = run_mini(g, tag="gate: baseline", errors=False)
    GATE_ROWS.append(dict(rule="baseline (plan conservative settings)", F05_testmix=base["metrics"]["F05_testmix"],
                          delta=0.0, se=0.0, recall=base["blocking"]["recall_union"], adopted="-"))
    adopted, gains = {}, {}
    for k, v in CONFIG["SWITCHES_TRY"].items():
        if CONFIG[k] == v:
            continue
        R = run_mini({**g, k: v}, tag=f"gate: {k}={v}", errors=False)
        d, se = paired_delta(base, R)
        ok = d > 0 and d > 2 * se
        GATE_ROWS.append(dict(rule=f"{k}={v}", F05_testmix=R["metrics"]["F05_testmix"], delta=d, se=se,
                              recall=R["blocking"]["recall_union"], adopted="yes" if ok else "no"))
        if ok:
            adopted[k], gains[k] = v, d
        del R; gc.collect()
    if len(adopted) > 1:                     # confirm the adopted rules also help together
        R = run_mini({**g, **adopted}, tag="gate: combined", errors=False)
        d, se = paired_delta(base, R)
        GATE_ROWS.append(dict(rule="combined: " + ", ".join(f"{k}={v}" for k, v in adopted.items()),
                              F05_testmix=R["metrics"]["F05_testmix"], delta=d, se=se,
                              recall=R["blocking"]["recall_union"], adopted="yes" if d >= max(gains.values()) - 2 * se else "no"))
        if GATE_ROWS[-1]["adopted"] == "no":  # together worse than the best single rule: keep only that one
            best = max(gains, key=gains.get)
            adopted = {best: adopted[best]}
        del R; gc.collect()
    del base; gc.collect()
    CONFIG.update(adopted)
    print(pd.DataFrame(GATE_ROWS).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log("adopted rules:", adopted or "none (plan conservative settings kept)")
else:
    GATE_ROWS = [dict(r, note="from run v2 (real data)") for r in CONFIG.get("KNOWN_GATE", [])]
    log("switch gate skipped; using the configured switches (real-data gate result from run v2 is reported)")
RESULTS["switch_gate"] = GATE_ROWS
RESULTS["config"].update({k: CONFIG[k] for k in PP_KEYS})
log("preprocessing switches in use:", {k: CONFIG[k] for k in PP_KEYS}, "->", pp_version())
save_results()

# %%
# A.1 Blocking frequency cap (P17): the cheapest cap within tolerance of the best recall, on mini-val
def blocking_sweep(pp, gt, fracs):
    pools = [pp["pool_s2"], pp["pool_s3"]]
    rows = []
    for frac in fracs:
        hit = tot = ncand = nq = 0; secs = 0.0
        for ctry in sorted(set(countries_in(pp["val_s1"])) & set(countries_in(pools))):
            pool, qt = Table(pools, ctry), Table(pp["val_s1"], ctry)
            b = Blocker(pool, frac, CONFIG["W_NAME"])
            t0 = time.time(); P = b.query(qt, CONFIG["K_MAIN"], CONFIG["K_NAME"], CONFIG["BLOCK_BATCH"]); secs += time.time() - t0
            y = label_pairs(qt.ids()[P["q"]], pool.ids()[P["c"]], gt)
            hit += int(y.sum()); tot += sum(len(gt[q]) for q in qt.ids()); ncand += len(y); nq += qt.n
            del b, pool; gc.collect()
        rows.append(dict(df_cap=frac, recall_union=round(hit / max(tot, 1), 4), cand_per_S1=round(ncand / max(nq, 1), 1),
                         ms_per_query_mini=round(1000 * secs / max(nq, 1), 3)))
    return pd.DataFrame(rows)

MINI_PP, MINI_GT = preprocess_mini()
if CONFIG["AUTO_DF_CAP"]:
    sweep = blocking_sweep(MINI_PP, MINI_GT, CONFIG["DF_CAP_CANDIDATES"])
    best = sweep["recall_union"].max()
    ok = sweep[sweep["recall_union"] >= best - CONFIG["DF_CAP_TOLERANCE"]]
    CONFIG["DF_MAX_FRAC"] = float(ok["df_cap"].min())
    print(sweep.to_string(index=False))
    log(f"chosen DF cap {CONFIG['DF_MAX_FRAC']} (full-pool time per query is roughly 10x the mini figure)")
    RESULTS["df_cap_sweep"] = sweep.to_dict("records")
RESULTS["config"]["DF_MAX_FRAC"] = CONFIG["DF_MAX_FRAC"]

# %%
# A.2 Final mini run with the adopted rules: train, validate, error examples
MINI = run_mini()
RESULTS["phaseA"] = dict(metrics=MINI["metrics"], threshold=MINI["thr"], blocking=MINI["blocking"],
                         no_resolution=MINI["no_resolution"], pool_stats=MINI["pool_stats"],
                         importance=MINI["importance"].round(0).to_dict(), examples=MINI["examples"],
                         minutes=round((time.time() - t_a) / 60, 1))
save_results()
log(f"Phase A done in {(time.time()-t_a)/60:.1f} min")

# %%
# A.3 Leave-one-country-out (proxy for an unseen country such as France): for every labelled country, train on
#     all the OTHER labelled countries and score it (countries come from the data, none are hard-coded)
def loco(R):
    tr, E = R["stage"]["train"], R["E"]
    tr_map = dict(zip(*s1_table_info(R["train_q_path"])))
    tr_ctry = np.array([tr_map[q] for q in tr["q_ids"]], dtype=object)
    out = {}
    labelled = sorted(set(tr_ctry) & set(E["country"]))
    for dst in labelled:
        others = [c for c in labelled if c != dst]
        if not others:
            continue
        src = "+".join(others)
        m = np.isin(tr_ctry, others)
        if m.sum() == 0 or (E["country"] == dst).sum() == 0:
            continue
        model = train_matcher(tr["X"].to_numpy(np.float32)[m], tr["y"][m], tr["q_ids"][m], CONFIG["LGB_ROUNDS"])
        p = model.predict(R["stage"]["val"]["X"].to_numpy(np.float32), num_iteration=model.best_iteration)
        # threshold chosen on the SOURCE country's val entities, applied to the target country
        src_mask_ent = np.isin(E["country"], others)
        best_thr, best_f = None, -1
        for thr in CONFIG["THR_GRID"]:
            r, f = evaluate(E, decide(E["qc"], R["cc"], p, thr, True, CONFIG["MAX_PRED"]))
            fs = f[src_mask_ent].mean()
            if fs > best_f:
                best_f, best_thr, f_all = fs, thr, f
        out[f"{src}->{dst}"] = dict(thr=best_thr, F05_target=float(f_all[E["country"] == dst].mean()),
                                    F05_target_in_country_model=R["metrics"].get(f"F05_{dst}"))
        log(f"LOCO {src}->{dst}: F05 on {dst} = {out[f'{src}->{dst}']['F05_target']:.4f} "
            f"(in-country model: {R['metrics'].get(f'F05_{dst}'):.4f}, thr {best_thr})")
    return out

RESULTS["phaseA"]["loco"] = loco(MINI)
save_results()

# %% [markdown]
# ## Phase B — full data: final model, calibration, test inference, submission
# The matcher is retrained on candidates retrieved from the **full** training pool, made as dense as the test pool
# by adding sampled test S2/S3 records as distractors (density matching). The threshold is tuned on the mini-val S1
# against that pool and checked on an untouched holdout; the test set is processed per country (France included,
# no country-specific model features); test probabilities are saved and B.5 writes one file per threshold.

# %%
# B.1 Challenge data + full preprocessing (train S1 + pool, test S1/S2/S3, holdout S1) + density matching
BASE = get_challenge_dir()
log("challenge dir:", BASE)
PPD = os.path.join(WORK, pp_version())
FULL = {}
for key, rel in (("train_s1", "dataset/train/train_source1.tsv"),
                 ("train_s2", "dataset/train/train_source2.tsv"), ("train_s3", "dataset/train/train_source3.tsv"),
                 ("test_s1", "dataset/test/test_source1.tsv"), ("test_s2", "dataset/test/test_source2.tsv"),
                 ("test_s3", "dataset/test/test_source3.tsv")):
    FULL[key] = preprocess_tsv(os.path.join(BASE, rel), os.path.join(PPD, "full", f"{key}.parquet"))
MINI_PP = {k: preprocess_tsv(MINI_PATHS[src], os.path.join(PPD, "mini", f"{k}.parquet"))
           for k, src in (("train_s1", "train_s1"), ("val_s1", "val_s1"))}

# Country labels found in the data (open set). Labelled = present in train S1; the others are scored with the
# same model and rules (no country feature), and every test S1 appears in the submission whatever its label.
_c_tr = pd.Series(pq.read_table(FULL["train_s1"], columns=["country"]).column("country").to_pylist()).value_counts()
_c_te = pd.Series(pq.read_table(FULL["test_s1"], columns=["country"]).column("country").to_pylist()).value_counts()
print(pd.DataFrame({"train_S1": _c_tr, "test_S1": _c_te}).fillna(0).astype(int).to_string())
_lab = [c for c in _c_te.index if c in _c_tr.index]
if CONFIG["TEST_WEIGHTS"] == "auto" and _lab:
    TEST_W = {c: round(float(_c_te[c] / _c_te[_lab].sum()), 4) for c in _lab}
log("labelled countries:", sorted(_c_tr.index), "| test-only (unlabelled) countries:",
    sorted(set(_c_te.index) - set(_c_tr.index)), "| validation weights:", test_weights())
RESULTS["countries"] = dict(train_S1=_c_tr.to_dict(), test_S1=_c_te.to_dict(), weights=test_weights())

HOLDOUT = None
if CONFIG["RUN_HOLDOUT"] and SPLITS_PATH:
    sp_df = read_tsv(SPLITS_PATH, usecols=["entity_id", "split"])
    ho_ids = sp_df.loc[sp_df.split == "holdout", "entity_id"]
    ho_ids = set(ho_ids.sample(n=min(20_000, len(ho_ids)), random_state=CONFIG["SEED"]))
    ho_tsv = os.path.join(WORK, "holdout_s1.tsv")
    if not os.path.exists(ho_tsv):
        s1 = read_tsv(os.path.join(BASE, "dataset/train/train_source1.tsv"))
        s1[s1.entity_id.isin(ho_ids)].to_csv(ho_tsv, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    HOLDOUT = preprocess_tsv(ho_tsv, os.path.join(PPD, "full", "holdout_s1.parquet"))
    log(f"holdout sample: {len(ho_ids):,} S1")

def country_counts(paths):
    return pd.concat([pq.read_table(p, columns=["country"]).to_pandas() for p in paths])["country"].value_counts()

# Density matching: the test pool has more S2/S3 records per S1 than the training pool (5.75 vs 4.68 in run v2).
# Sampled test records (never a match for a training S1) are added as distractors per country, so the final model
# is trained, calibrated and checked at the density it will face on the test set.
EXTRA = None
if CONFIG["DENSITY_MATCH"]:
    EXTRA = os.path.join(PPD, "full", "density_extra.parquet")
    tr_s1, tr_pool = country_counts([FULL["train_s1"]]), country_counts([FULL["train_s2"], FULL["train_s3"]])
    te_s1, te_pool = country_counts([FULL["test_s1"]]), country_counts([FULL["test_s2"], FULL["test_s3"]])
    rows, writer, rng = [], None, np.random.default_rng(CONFIG["SEED"])
    for ctry in sorted(set(tr_s1.index) & set(tr_pool.index) & set(te_s1.index) & set(te_pool.index)):
        d_tr, d_te = tr_pool[ctry] / tr_s1[ctry], te_pool[ctry] / te_s1[ctry]
        frac = float(np.clip((d_te - d_tr) * tr_s1[ctry] / te_pool[ctry], 0, 1))
        added = 0
        if frac > 0 and not os.path.exists(EXTRA):
            for path in (FULL["test_s2"], FULL["test_s3"]):
                t = pq.read_table(path, filters=[("country", "=", ctry)])
                t = t.filter(pa.array(rng.random(t.num_rows) < frac))
                if writer is None:
                    writer = pq.ParquetWriter(EXTRA + ".tmp", t.schema)
                writer.write_table(t); added += t.num_rows
                del t; gc.collect()
        rows.append(dict(country=ctry, train_density=round(d_tr, 3), test_density=round(d_te, 3),
                         sampled_fraction=round(frac, 3), added=added))
    if writer is not None:
        writer.close(); os.replace(EXTRA + ".tmp", EXTRA)
    if not os.path.exists(EXTRA):
        EXTRA = None
    print(pd.DataFrame(rows).to_string(index=False))
    RESULTS["density_match"] = rows
    save_results()

# %%
# B.2 Final matcher on density-matched full-pool candidates; threshold calibrated on mini-val; holdout check
t_b = time.time()
gt_full = {**load_gt(MINI_PATHS["train_gt"]), **load_gt(MINI_PATHS["val_gt"])}
if HOLDOUT:
    gt_full.update(load_gt(os.path.join(BASE, "dataset/train/train_ground_truth.tsv"),
                           keep=set(s1_table_info(HOLDOUT)[0])))
tr_q_full = sample_queries(MINI_PP["train_s1"],
                           os.path.join(PPD, "mini", f"train_q_{CONFIG['FULL_TRAIN_QUERIES']}.parquet"),
                           CONFIG["FULL_TRAIN_QUERIES"], CONFIG["SEED"])
qsets = {"train": tr_q_full, "val": MINI_PP["val_s1"]}
if HOLDOUT:
    qsets["holdout"] = HOLDOUT
POOL_FULL = [FULL["train_s2"], FULL["train_s3"]] + ([EXTRA] if EXTRA else [])
STAGE_B, stats_b = run_stage(POOL_FULL, qsets, gt_full, "full", pop_paths=[FULL["train_s1"]])
STAGE_B["val_path"] = MINI_PP["val_s1"]
FINAL = fit_and_validate(STAGE_B, gt_full, "full")
FINAL["examples"] = error_examples(STAGE_B, FINAL, POOL_FULL)
MODEL, THR = FINAL["model"], FINAL["thr"]
MODEL.save_model(os.path.join(OUT, "matcher_lgb.txt"), num_iteration=MODEL.best_iteration)

ho_res = None
if HOLDOUT:
    ho = STAGE_B["holdout"]
    p_ho = MODEL.predict(ho["X"].to_numpy(np.float32), num_iteration=MODEL.best_iteration)
    ho_ids, ho_ctry = s1_table_info(HOLDOUT)
    E_ho = make_eval_bundle(ho["q_ids"], ho["y"], ho_ids, ho_ctry, gt_full)
    cc_ho = pd.factorize(ho["c_ids"])[0]
    ho_res, _ = evaluate(E_ho, decide(E_ho["qc"], cc_ho, p_ho, THR, True, CONFIG["MAX_PRED"]))
    ho_res["blocking"] = blocking_report(E_ho, ho["rank"], ho["in_name"], CONFIG["K_MAIN"])
    log("HOLDOUT (untouched, threshold fixed from val): " +
        "  ".join(f"{k}={v:.4f}" for k, v in ho_res.items() if isinstance(v, float)))
    ho_curve = []
    for t in sorted(set(CONFIG["VARIANT_THRESHOLDS"]) | {THR}):
        r, _ = evaluate(E_ho, decide(E_ho["qc"], cc_ho, p_ho, t, True, CONFIG["MAX_PRED"]))
        ho_curve.append({"thr": t, **{k: round(v, 4) for k, v in r.items() if isinstance(v, float)}})
    print("holdout at the variant thresholds (for choosing leaderboard submissions):")
    print(pd.DataFrame(ho_curve)[["thr", "F05", "F05_testmix", "mean_P", "mean_R", "singleton_acc"]].to_string(index=False))
    ho_res["threshold_curve"] = ho_curve

RESULTS["phaseB"] = dict(metrics=FINAL["metrics"], threshold=THR, blocking=FINAL["blocking"],
                         no_resolution=FINAL["no_resolution"], pool_stats=stats_b, holdout=ho_res,
                         importance=FINAL["importance"].round(0).to_dict(), examples=FINAL["examples"],
                         train_pairs=int(len(STAGE_B["train"]["y"])),
                         train_positive=int(STAGE_B["train"]["y"].sum()),
                         minutes=round((time.time() - t_b) / 60, 1))
save_results()
del STAGE_B; gc.collect()

# %%
# B.3 Test inference: every country in test S1 (France included), streamed in query batches.
#     Probabilities (p >= P_FLOOR) are saved, so other thresholds can be written in seconds (B.5).
t_t = time.time()
cand_path, match_path = os.path.join(OUT, "candidate_pairs.tsv"), os.path.join(OUT, "matching_results.tsv")
scores_path = os.path.join(OUT, "test_scores.parquet")
test_ids_all, test_ctry_all = s1_table_info(FULL["test_s1"])
n_cand_total, cand_rows, sw = 0, 0, None
SCORE_SCHEMA = pa.schema([("q", pa.string()), ("c", pa.string()), ("p", pa.float32())])
pool_test = [FULL["test_s2"], FULL["test_s3"]]
pool_ctry = set(countries_in(pool_test))
test_stats = {}
with open(cand_path, "w", encoding="utf-8") as fc:
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    for ctry in countries_in(FULL["test_s1"]):
        qt_all = Table(FULL["test_s1"], ctry)
        q_all_ids = qt_all.ids()
        if ctry not in pool_ctry:
            for q in q_all_ids:
                fc.write(f"{q}\t\n")
            cand_rows += len(q_all_ids)
            continue
        add_dup_counts(qt_all, "q_")                     # population = all test S1 of this country
        pool = Table(pool_test, ctry)
        add_dup_counts(pool, "c_")
        pool_ids = pool.ids()
        blk = Blocker(pool, CONFIG["DF_MAX_FRAC"], CONFIG["W_NAME"])
        log(f"[test] {ctry}: {qt_all.n:,} S1 vs pool {pool.n:,}  (index {blk.stats['secs']}s)")
        c0, n0 = time.time(), n_cand_total
        for s in range(0, qt_all.n, CONFIG["TEST_QUERY_BATCH"]):
            e = min(s + CONFIG["TEST_QUERY_BATCH"], qt_all.n)
            m = np.zeros(qt_all.n, bool); m[s:e] = True
            qt = qt_all.subset(m)
            P = blk.query(qt, CONFIG["K_MAIN"], CONFIG["K_NAME"], CONFIG["BLOCK_BATCH"])
            q_ids = qt.ids()
            if len(P["q"]):
                X = pair_features(P, qt, pool, pool_ids)
                p = MODEL.predict(X.to_numpy(np.float32), num_iteration=MODEL.best_iteration)
                keep = p >= CONFIG["P_FLOOR"]
                tb = pa.table({"q": pa.array(q_ids[P["q"][keep]], type=pa.string()),
                               "c": pa.array(pool_ids[P["c"][keep]], type=pa.string()),
                               "p": pa.array(p[keep].astype(np.float32))}, schema=SCORE_SCHEMA)
                if sw is None:
                    sw = pq.ParquetWriter(scores_path, SCORE_SCHEMA)
                sw.write_table(tb)
                del X, tb
                bounds = np.searchsorted(P["q"], np.arange(qt.n + 1))
            else:
                bounds = np.zeros(qt.n + 1, dtype=np.int64)
            c_strs = pool_ids[P["c"]] if len(P["q"]) else np.array([], dtype=object)
            for i, q in enumerate(q_ids):
                fc.write(f"{q}\t{','.join(c_strs[bounds[i]:bounds[i+1]])}\n")
            n_cand_total += len(P["q"]); cand_rows += qt.n
            done = e / qt_all.n
            log(f"[test] {ctry}: {e:,}/{qt_all.n:,} S1  ({done:.0%}, {time.time()-c0:.0f}s elapsed, "
                f"~{(time.time()-c0)*(1-done)/max(done,1e-9)/60:.0f} min left)")
        test_stats[ctry] = dict(S1=int(qt_all.n), pool=int(pool.n), cand_pairs=int(n_cand_total - n0),
                                minutes=round((time.time() - c0) / 60, 1))
        del blk, pool, qt_all; gc.collect()
if sw is None:
    pq.write_table(SCORE_SCHEMA.empty_table(), scores_path)
else:
    sw.close()

def write_matching(thr, path):
    """Matching file for one threshold from the saved test probabilities (many-to-one resolution + cap)."""
    t = pq.read_table(scores_path)
    kq = t.column("q").to_numpy(zero_copy_only=False); kc = t.column("c").to_numpy(zero_copy_only=False)
    kp = t.column("p").to_numpy(zero_copy_only=False)
    del t
    qc_codes, _ = pd.factorize(kq); cc_codes, _ = pd.factorize(kc)
    idx = decide(qc_codes, cc_codes, kp, thr, True, CONFIG["MAX_PRED"])
    pred = pd.DataFrame({"q": kq[idx], "c": kc[idx]}).groupby("q")["c"].apply(lambda x: ",".join(x)).to_dict()
    n_by_ctry, s1_by_ctry = Counter(), Counter(test_ctry_all)
    with open(path, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for q, c in zip(test_ids_all, test_ctry_all):
            v = pred.get(q, "")
            fm.write(f"{q}\t{v}\n")
            n_by_ctry[c] += v.count(",") + 1 if v else 0
    n_match = sum(n_by_ctry.values())
    return dict(thr=thr, matched_pairs=n_match, S1_with_matches=len(pred),
                pred_per_S1=round(n_match / len(test_ids_all), 4),
                empty_share=round(1 - len(pred) / len(test_ids_all), 4),
                **{f"pred_per_S1_{c}": round(n_by_ctry[c] / s1_by_ctry[c], 4) for c in sorted(s1_by_ctry)})

main_stats = write_matching(THR, match_path)
# completeness: every test S1 (every country label, incl. ones never seen in training) has exactly one row in both files
_m = read_tsv(match_path, usecols=["source1_entity_id"])["source1_entity_id"]
_c = read_tsv(cand_path, usecols=["source1_entity_id"])["source1_entity_id"]
assert len(_m) == len(_c) == len(test_ids_all) and _m.is_unique and _c.is_unique \
       and set(_m) == set(test_ids_all) == set(_c), "a test S1 is missing or duplicated in the submission files"
log("every test S1 present once in both files:", dict(Counter(test_ctry_all)))
del _m, _c
RESULTS["test"] = dict(S1=len(test_ids_all), candidate_rows=cand_rows, candidate_pairs=n_cand_total,
                       threshold=THR, **{k: v for k, v in main_stats.items() if k != "thr"},
                       per_country=test_stats, minutes=round((time.time() - t_t) / 60, 1))
save_results()
log(f"test done: {RESULTS['test']}")

# %%
# B.5 Threshold variants for the leaderboard (seconds each, from the saved test probabilities)
# Each file is zipped straight into the output folder (ready to upload; the file inside is matching_results.tsv).
def zip_matching(tsv, zpath):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tsv, arcname="matching_results.tsv")

zip_matching(match_path, os.path.join(OUT, "matching_results.zip"))           # main file, zipped
var_rows, tmp_tsv = [], os.path.join(WORK, "variant_tmp.tsv")
for thr in sorted(set(CONFIG["VARIANT_THRESHOLDS"]) | {THR}):
    var_rows.append(write_matching(thr, tmp_tsv))
    zip_matching(tmp_tsv, os.path.join(OUT, f"matching_results_thr{thr:.3f}.zip"))
os.remove(tmp_tsv)
var_df = pd.DataFrame(var_rows)
print(var_df.to_string(index=False))
print(f"main file (matching_results.zip) uses the validation-optimal threshold {THR}; "
      f"variant zips: matching_results_thr*.zip in {OUT}")
RESULTS["variants"] = var_rows
save_results()

# %%
# B.4 Official validator
import subprocess
if os.path.exists(os.path.join(BASE, "utils", "validate_submission.py")):
    res = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", match_path,
                          "--candidate", cand_path, "--test-dir", "dataset/test"],
                         cwd=BASE, capture_output=True, text=True)
    print(res.stdout[-3000:], res.stderr[-2000:])
    RESULTS["validator"] = "PASS" if res.returncode == 0 else "FAIL"
else:
    log("utils/validate_submission.py not found next to the dataset — validator skipped (run it locally before submitting)")
    RESULTS["validator"] = "skipped"
save_results()

# %% [markdown]
# ## 4. Submission package, documentation, downloads

# %%
# 4.1 README / requirements for code/business_entity_resolution, and the filled methodology document
import importlib.metadata as md
def ver(pkg):
    try:
        return md.version(pkg)
    except Exception:
        return "unknown"
REQ = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "lightgbm", "rapidfuzz", "unidecode",
       "indic-transliteration", "ftfy", "gdown", "sparse_dot_topn"]
with open(os.path.join(OUT, "requirements.txt"), "w") as f:
    f.write("\n".join(f"{p}=={ver(p)}" for p in REQ) + "\n")
with open(os.path.join(OUT, "README.md"), "w") as f:
    f.write(f"""# Business Entity Resolution — reproduction

Pipeline: `src/er_pipeline.ipynb` (Google Colab, CPU runtime). Python {sys.version.split()[0]}.

1. `pip install -r requirements.txt`
2. Open `src/er_pipeline.ipynb`, set `DATA_FOLDER_URL` (mini-world zips: train/val S1 split, shared S2/S3
   pool, ground truth) and, if needed, `CHALLENGE_DIR` (the official `student_resource` folder).
3. Run all cells. Phase A validates on the mini-world; Phase B trains the final LightGBM matcher on
   full-pool candidates, calibrates the threshold, processes the test set and writes
   `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then runs `utils/validate_submission.py`.

Data flow: raw TSV → preprocessing (multi-view Parquet) → country-partitioned blocking
(IDF-weighted name+address cosine top-{CONFIG['K_MAIN']} ∪ name-only top-{CONFIG['K_NAME']}) → pairwise
features → LightGBM → many-to-one resolution → threshold {THR}.
No external data, APIs or pretrained models are used; the matcher is a LightGBM model (MIT).
""")

def fmt(d, keys):
    return ", ".join(f"{k} {d[k]:.4f}" if isinstance(d.get(k), float) else f"{k} {d.get(k)}" for k in keys if k in d)
A, B = RESULTS.get("phaseA", {}), RESULTS.get("phaseB", {})
bm, T = B.get("metrics", {}), RESULTS.get("test", {})
gate_lines = "\n".join(f"  - {r['rule']}: Δ testmix F0.5 {r['delta']:+.4f} (±{2*r['se']:.4f}), adopted {r['adopted']}"
                       for r in RESULTS.get("switch_gate", [])) or "  - gate not run; switches as configured"
fp_lines = "\n".join(f"  - S1 `{e['S1']}` vs `{e['cand']}` (p={e['p']})" for e in (B.get("examples") or {}).get("false_merges", [])[:5])
fn_lines = "\n".join(f"  - S1 `{e['S1']}` vs `{e['cand']}` (p={e['p']}, rank {e['rank']})" for e in (B.get("examples") or {}).get("missed", [])[:5])
imp_top = ", ".join(list(B.get("importance", {}).keys())[:10])
doc = f"""# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]
**Team Members:** [List all team members]
**Submission Date:** {time.strftime('%Y-%m-%d')}

---

## 1. Executive Summary
A three-stage pipeline: multi-view text normalisation (with Devanagari transliteration and country-aware address
rules), country-partitioned blocking that blends IDF-weighted name and address cosine similarity, and a LightGBM
matcher on {len(FEATS)} pairwise features, followed by many-to-one resolution and an F0.5-tuned threshold.
Validation F0.5 (full pool, test-mix weighted): **{bm.get('F05_testmix', float('nan')):.4f}**.

---

## 2. Methodology

### 2.1 Problem Analysis
- 2.2M / 5.0M / 5.3M train records (S1/S2/S3); test 1.7M / 4.9M / 5.1M. Clean TSV structure (0 malformed rows);
  names never empty; S2/S3 addresses empty in ~3% of records (and in 3.8–4.9% of *matched* records vs 0.2–0.3% of
  orphans).
- Labels are strictly **many-to-one** (every S2/S3 record belongs to at most one S1); 5.58% of S1 are singletons;
  mean 3.46 matches (max 11), split evenly between S2 and S3; ~26% of S2/S3 are orphans (unmatched distractors).
- **All matches are within-country** (0 cross-country links in 7.6M) → country is a safe partition.
- Test shifts the mix (India 47% / US 38%) and adds **France (15%)**, unseen in training.
- Noise: Devanagari vs Latin names (true cross-script pairs share almost no tokens), leading junk (`--`, `<<`),
  typos, legal-suffix variation (Pvt/Private/प्राइवेट, LLC, SARL), domains as names, casing differences,
  reordered / abbreviated / partial addresses, zero-padded numbers. Postal codes are present in only ~1% (India) /
  11% (US) of addresses, so they are not used for blocking.

### 2.2 Solution Strategy
**Approach Type:** Blocking + Classifier (with constrained many-to-one resolution)
**Core Innovation:** multi-view preprocessing that keeps raw fields and adds named views (blocking uses collapsed
views for recall; the matcher sees full, core, compact, alias, initials, original-script, numeric, postal and unit
views side by side for precision); preprocessing rules with uncertain value are gated by validation (each enabled
only if it raises test-mix macro-F0.5 beyond twice its paired standard error); optional Devanagari→Latin
transliteration with schwa deletion (`राम मार्केटिंग प्राइवेट लिमिटेड` → `ram marketing praivet limited`); a name-only
candidate channel that rescues matches with missing or garbled addresses; and precision-oriented resolution that
assigns each S2/S3 record to at most one S1.

Preprocessing gate results (mini-validation, paired on the same S1):
{gate_lines}

Density matching: the test pool holds ~23% more S2/S3 records per S1 than the training pool (5.75 vs 4.68), i.e.
more distractors. Sampled test S2/S3 records (never a match for a training S1) are added to the training pool per
country until the densities match, so the final model is trained, its threshold tuned and the holdout scored at
test-like density. Test probabilities are saved so alternative thresholds can be written without re-running.

Validation: S1 entities split by cluster (train 2.0M / val 100k / holdout 100k), stratified by country ×
match-count bucket, singletons included; a 10% cluster-sampled "mini-world" (200k train + 20k val S1 with their
full clusters and orphans at the same rate) is used for fast iteration; final numbers use the full candidate pool.

---

## 3. Candidate Generation (Blocking)
- **Blocking keys used:** per country, TF-IDF (binary, idf = ln N/df) over core-name tokens and address tokens;
  tokens in more than {CONFIG['DF_MAX_FRAC']:.1%} of a country's pool are not used as keys. Candidate score =
  {CONFIG['W_NAME']}·name cosine + {1-CONFIG['W_NAME']}·address cosine (candidate-normalised); top-{CONFIG['K_MAIN']}
  by this score ∪ top-{CONFIG['K_NAME']} by name cosine alone, computed as multithreaded top-N sparse matrix
  products (sparse_dot_topn) so full score rows are never materialised. The frequency cap was chosen on
  mini-validation as the cheapest cap within {CONFIG['DF_CAP_TOLERANCE']:.1%} of the best blocking recall.
- **Candidate pairs generated:** {T.get('candidate_pairs', 'n/a'):,} for {T.get('S1', 'n/a'):,} test S1
  (≈{(T.get('candidate_pairs', 0) / max(T.get('S1', 1), 1)):.1f} per S1).
- **How you ensured true matches were not lost:** recall measured on held-out S1 against the full pool:
  {fmt(B.get('blocking', {}), [k for k in B.get('blocking', {}) if k.startswith('recall') or k in ('S1_all_found', 'ceiling_F05', 'cand_per_S1')])}.
  Country partitioning loses no training link.

---

## 4. Matching Model

**Features used ({len(FEATS)}):**
- Name features: ratio / token-set / token-sort / partial / Jaro-Winkler on core names; ratio and token-set on
  full romanised names; compact-name ratio and partial ratio (domains, spacing); original-script ratio; exact
  core-name and first-token agreement; legal-form agreement and conflict; d.b.a. alias similarity; initials vs
  short names; token counts and length ratio.
- Address features: ratio, token-set and partial ratio on normalised addresses; token-set on the unmapped address;
  numeric-token agreement with leading zeros stripped; postal-code candidate agreement / conflict (zeros kept);
  unit identifier agreement / conflict; component counts (partial addresses); missing-address type (empty /
  placeholder / emptied by normalisation); encoding-repair flag; house-number agreement (equality, numeric and
  relative difference, digit edit distance, "digit dropped" flag).
- Other: blocking scores (blended, name and address cosine, rank, name-channel flag), per-S1 context (candidate
  count, gap to the best candidate's score / name / address similarity), script flags (Devanagari, cross-script),
  source flag (S2/S3). No country feature, so the model applies unchanged to France.

**Model type:** LightGBM gradient-boosted trees (binary logloss, early stopping on held-out training S1).
Most important features (gain): {imp_top}.
**Threshold selection method:** grid search of macro-F0.5 on validation S1 against the full pool (test-mix
weighted by each labelled country's share of the test S1: {test_weights()}), after many-to-one resolution
and a cap of {CONFIG['MAX_PRED']} matches per S1. Selected threshold: **{THR}**.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro, validation, full pool):** {fmt(bm, ['F05', 'F05_testmix', 'F05_US', 'F05_India', 'mean_P', 'mean_R', 'singleton_acc'])}
- **Holdout (untouched):** {fmt(B.get('holdout') or {}, ['F05', 'F05_testmix', 'F05_US', 'F05_India']) or 'not run'}
- **Mini-world validation (optimistic, 10× fewer distractors):** {fmt(A.get('metrics', {}), ['F05', 'F05_testmix'])}
- **Effect of many-to-one resolution:** F0.5 {B.get('no_resolution', {}).get('F05', float('nan')):.4f} without →
  {bm.get('F05', float('nan')):.4f} with.
- **Leave-one-country-out (France proxy):** {json.dumps(A.get('loco', {}))}
- **Common false positives (wrong merges):** examples from validation:
{fp_lines}
- **Common false negatives (missed matches):**
{fn_lines}

---

## 6. Conclusion
Precision-oriented entity resolution: recall is secured by a two-channel, country-partitioned blocker, and
precision by a feature-rich gradient-boosted matcher plus many-to-one resolution and an F0.5-tuned threshold.
The largest remaining gap is cross-script (Devanagari ↔ Latin) names; a multilingual character model or
learned transliteration is the next lever, followed by per-country thresholds once France behaviour is known.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/src/er_pipeline.ipynb` — single notebook: data → preprocessing → blocking →
features → LightGBM → resolution → output files → validator. `README.md` gives the run steps,
`requirements.txt` pins versions. Runtime on a 2-CPU Colab: Phase A {A.get('minutes', '?')} min, Phase B
{B.get('minutes', '?')} min + test {T.get('minutes', '?')} min.

### B. Additional Results
Threshold sweep, per-country metrics, blocking recall by cut-off and feature importance are in `results.json`.
"""
with open(os.path.join(OUT, "Documentation.md"), "w", encoding="utf-8") as f:
    f.write(doc)
log("wrote requirements.txt, README.md, Documentation.md")

# %%
# 4.2 Package the submission outputs and download
zip_path = os.path.join(OUT, "submission_outputs.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in ("matching_results.tsv", "candidate_pairs.tsv"):
        z.write(os.path.join(OUT, fn), arcname=f"output/{fn}")
    for fn in ("README.md", "requirements.txt"):
        z.write(os.path.join(OUT, fn), arcname=f"code/business_entity_resolution/{fn}")
    z.write(os.path.join(OUT, "Documentation.md"), arcname="Documentation_template.md")
    z.write(os.path.join(OUT, "results.json"), arcname="results.json")
log(f"{zip_path}: {os.path.getsize(zip_path)/1e6:.0f} MB")
for fn in os.listdir(OUT):
    log(f"  {fn:28} {os.path.getsize(os.path.join(OUT, fn))/1e6:9.1f} MB")
if IN_COLAB:
    from google.colab import files
    files.download(os.path.join(OUT, "matching_results.tsv"))   # the leaderboard file (small)
    files.download(os.path.join(OUT, "results.json"))
    files.download(os.path.join(OUT, "Documentation.md"))
    print(f"Large file: download submission_outputs.zip from the Files panel (left sidebar) → {OUT}.")
    print("Add this notebook (File → Download .ipynb) as code/business_entity_resolution/src/er_pipeline.ipynb.")
elif ON_KAGGLE:
    print(f"All outputs are in {OUT}: after 'Save & Run All' open the notebook's Output tab to download them.")
    print("Add this notebook (File → Download notebook) as code/business_entity_resolution/src/er_pipeline.ipynb.")
