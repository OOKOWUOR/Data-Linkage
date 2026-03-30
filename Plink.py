import os, re
import numpy as np
import pandas as pd
import jellyfish

# =============== CONFIG ===============
CENSUS_PATH   = r"C:/Users/guest441/Downloads/D1_Census.xlsx"
REGISTRY_PATH = r"C:/Users/guest441/Downloads/D2_Registry.xlsx"

OUT_DIR = r"C:/Users/guest441/Downloads/linkage_outputs_probabilistic_FAST"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42

N_CENSUS   = 21766
N_REGISTRY = 75125

P_VALUES = [0.5, 0.4, 0.3]
D_VALUES = [0.10, 0.05, 0.01]
Z_95 = 1.96

# candidate controls (fast)
TOPK_PER_PASS_FULL    = 60
MAX_PER_CENSUS_FULL   = 80
TOPK_PER_PASS_SAMPLES = 80
MAX_PER_CENSUS_SAMPLES= 120

# calibration controls
CALIBRATION_SAMPLE = 5000   # deterministic pairs sampled to learn threshold (fast)
KEEP_TRUE_Q = 0.05          # threshold = 5th percentile of true-pair scores (keeps ~95% of true pairs)


# =============== RAPIDFUZZ (fast similarity) ===============
USE_RAPIDFUZZ = False
try:
    from rapidfuzz.distance import JaroWinkler
    USE_RAPIDFUZZ = True
except Exception:
    USE_RAPIDFUZZ = False

def jw(a, b):
    a = a or ""; b = b or ""
    if not a and not b:
        return 0.0
    if USE_RAPIDFUZZ:
        return float(JaroWinkler.similarity(a, b) / 100.0)
    return float(jellyfish.jaro_winkler_similarity(a, b))

def jw_list(a_list, b_list):
    out = np.empty(len(a_list), dtype=float)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        out[i] = jw(a, b)
    return out


# =============== BASIC CLEANING ===============
def clean_string(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_sex(s):
    s = clean_string(s)
    if s in ["m","male","man","1"]: return "m"
    if s in ["f","female","woman","2"]: return "f"
    return s

def split_first_last(fullname):
    parts = fullname.split()
    if len(parts) == 0: return "", ""
    if len(parts) == 1: return parts[0], ""  # common in your data
    return parts[0], parts[-1]

def safe_soundex(s):
    s = s or ""
    try:
        return jellyfish.soundex(s) if s else ""
    except Exception:
        return ""

def to_numeric_age(x):
    return pd.to_numeric(x, errors="coerce")

def age_sim_vec(a, b):
    diff = np.abs(a - b)
    out = np.zeros_like(diff, dtype=float)
    out[np.isnan(diff)] = 0.0
    out[(~np.isnan(diff)) & (diff <= 1)] = 1.0
    out[(~np.isnan(diff)) & (diff > 1) & (diff <= 2)] = 0.5
    return out


# =============== COLUMN DETECTION (simple + stable) ===============
def detect_column(df, patterns):
    for p in patterns:
        for c in df.columns:
            if p.lower() in c.lower():
                return c
    return None


# =============== PREPARE ===============
def prepare(df, name_col, sex_col, vill_col, age_col, prefix):
    df = df.copy()
    df["person_name"] = df[name_col].apply(clean_string)
    df["sex"] = df[sex_col].apply(normalize_sex) if sex_col else ""
    df["village"] = df[vill_col].apply(clean_string) if vill_col else ""
    df["age"] = df[age_col].apply(to_numeric_age) if age_col else np.nan

    df["first_name"], df["last_name"] = zip(*df["person_name"].apply(split_first_last))
    df["sx_first"] = df["first_name"].apply(safe_soundex)
    df["sx_last"]  = df["last_name"].apply(safe_soundex)
    df["vill3"] = df["village"].str[:3].fillna("")

    df["rid"] = [f"{prefix}{i}" for i in range(len(df))]
    return df


# =============== SCORING (fast + sensible) ===============
def score_pairs(m):
    # similarities
    full = jw_list(m["person_name_c"].tolist(), m["person_name_r"].tolist())
    fn   = jw_list(m["first_name_c"].tolist(),  m["first_name_r"].tolist())
    ln   = jw_list(m["last_name_c"].tolist(),   m["last_name_r"].tolist())
    vill = jw_list(m["village_c"].tolist(),     m["village_r"].tolist())
    sex  = (m["sex_c"].values == m["sex_r"].values).astype(float)
    ageS = age_sim_vec(m["age_c"].values, m["age_r"].values)

    # weights (full name drives)
    W = {"full":0.45, "fn":0.20, "ln":0.15, "vill":0.12, "sex":0.06, "age":0.02}

    # renormalize when missing
    full_ok = (m["person_name_c"].values!="") & (m["person_name_r"].values!="")
    fn_ok   = (m["first_name_c"].values!="") & (m["first_name_r"].values!="")
    ln_ok   = (m["last_name_c"].values!="") & (m["last_name_r"].values!="")
    vill_ok = (m["village_c"].values!="") & (m["village_r"].values!="")
    sex_ok  = (m["sex_c"].values!="") & (m["sex_r"].values!="")
    age_ok  = (~np.isnan(m["age_c"].values)) & (~np.isnan(m["age_r"].values))

    denom = (W["full"]*full_ok + W["fn"]*fn_ok + W["ln"]*ln_ok +
             W["vill"]*vill_ok + W["sex"]*sex_ok + W["age"]*age_ok).astype(float)
    denom = np.where(denom==0, 1.0, denom)

    score = (W["full"]*full_ok*full + W["fn"]*fn_ok*fn + W["ln"]*ln_ok*ln +
             W["vill"]*vill_ok*vill + W["sex"]*sex_ok*sex + W["age"]*age_ok*ageS) / denom

    out = m.copy()
    out["score"] = score
    return out


def topk_per_left(df, k):
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).groupby("rid_c", as_index=False).head(k)


def one_to_one(df):
    if df.empty:
        return df
    df = df.sort_values("score", ascending=False)
    used_c=set(); used_r=set(); keep=[]
    for _, row in df.iterrows():
        c=row["rid_c"]; r=row["rid_r"]
        if c in used_c or r in used_r:
            continue
        keep.append(row)
        used_c.add(c); used_r.add(r)
    return pd.DataFrame(keep)


# =============== FAST CANDIDATE GENERATION (NO MILLION+ MERGES) ===============
def build_candidates(census, registry, label, topk_per_pass, max_per_census):
    c = census.copy()
    r = registry.copy()

    # These keys are MUCH tighter than your previous PASS_B that created 2.1M candidates.
    c["k1"] = c["sx_first"].astype(str) + "|" + c["sex"].astype(str) + "|" + c["vill3"].astype(str)
    r["k1"] = r["sx_first"].astype(str) + "|" + r["sex"].astype(str) + "|" + r["vill3"].astype(str)

    c["k2"] = c["sx_first"].astype(str) + "|" + c["sex"].astype(str)
    r["k2"] = r["sx_first"].astype(str) + "|" + r["sex"].astype(str)

    c["k3"] = c["sx_last"].astype(str)  + "|" + c["sex"].astype(str) + "|" + c["vill3"].astype(str)
    r["k3"] = r["sx_last"].astype(str)  + "|" + r["sex"].astype(str) + "|" + r["vill3"].astype(str)

    passes = [("PASS1_SXFN_SEX_V3","k1"), ("PASS2_SXFN_SEX","k2"), ("PASS3_SXLN_SEX_V3","k3")]

    allc = []
    for pname, key in passes:
        print(f"\n[{label}] {pname} merge on {key} ...")
        m = c.merge(r, on=key, how="inner", suffixes=("_c","_r"))
        print(f"[{label}] {pname} raw candidates: {len(m):,}")
        if m.empty:
            continue

        need = ["rid_c","rid_r","person_name_c","person_name_r","first_name_c","first_name_r",
                "last_name_c","last_name_r","village_c","village_r","sex_c","sex_r","age_c","age_r"]
        m = m[[x for x in need if x in m.columns]].copy()
        m = score_pairs(m)
        m = topk_per_left(m, topk_per_pass)
        allc.append(m)

    if not allc:
        return pd.DataFrame(columns=["rid_c","rid_r","score"])

    cand = pd.concat(allc, ignore_index=True)
    cand = cand.sort_values("score", ascending=False).drop_duplicates(["rid_c","rid_r"], keep="first")
    cand = cand.sort_values("score", ascending=False).groupby("rid_c", as_index=False).head(max_per_census)
    print(f"\n[{label}] After union + cap {max_per_census} per census: {len(cand):,}")
    return cand


# =============== THRESHOLD CALIBRATION USING DETERMINISTIC "TRUE" PAIRS ===============
def deterministic_true_pairs(census, registry, max_pairs=5000, seed=42):
    # high-confidence deterministic links used only for calibration
    # exact first+last+sex (strong and consistent with your deterministic Phase 1 logic)
    left = census[["rid","first_name","last_name","sex","person_name","village","age"]].copy()
    right= registry[["rid","first_name","last_name","sex","person_name","village","age"]].copy()
    left.columns  = ["rid_c","first_name","last_name","sex","person_name","village","age"]
    right.columns = ["rid_r","first_name","last_name","sex","person_name","village","age"]

    m = left.merge(
        right,
        on=["first_name","last_name","sex"],
        how="inner",
        suffixes=("_c","_r")
    )

    if m.empty:
        return pd.DataFrame(columns=["rid_c","rid_r","person_name_c","person_name_r","first_name_c","first_name_r",
                                     "last_name_c","last_name_r","village_c","village_r","sex_c","sex_r","age_c","age_r"])

    # sample to keep calibration fast
    if len(m) > max_pairs:
        m = m.sample(n=max_pairs, random_state=seed)

    # rename to scoring format
    out = pd.DataFrame({
        "rid_c": m["rid_c"],
        "rid_r": m["rid_r"],
        "person_name_c": m["person_name_c"],
        "person_name_r": m["person_name_r"],
        "first_name_c": m["first_name_c"],
        "first_name_r": m["first_name_r"],
        "last_name_c": m["last_name_c"],
        "last_name_r": m["last_name_r"],
        "village_c": m["village_c"],
        "village_r": m["village_r"],
        "sex_c": m["sex_c"],
        "sex_r": m["sex_r"],
        "age_c": m["age_c"],
        "age_r": m["age_r"],
    })
    return out


def calibrate_threshold(census, registry):
    tp = deterministic_true_pairs(census, registry, max_pairs=CALIBRATION_SAMPLE, seed=RANDOM_SEED)
    if tp.empty:
        # fallback (still safe)
        print("\n[CALIBRATION] No deterministic calibration pairs found. Using fallback threshold 0.88")
        return 0.88

    tp_scored = score_pairs(tp)
    thr = float(tp_scored["score"].quantile(KEEP_TRUE_Q))

    # guardrails
    thr = max(0.75, min(0.95, thr))

    print("\n[CALIBRATION] True-pair score stats:")
    print(tp_scored["score"].describe(percentiles=[0.01,0.05,0.10,0.50,0.90,0.95,0.99]).to_string())
    print(f"[CALIBRATION] Threshold chosen (q={KEEP_TRUE_Q}): {thr:.3f}")
    return thr


def run_probabilistic(census, registry, label, threshold, topk_per_pass, max_per_census):
    print("\n" + "="*len(label))
    print(label)
    print("="*len(label))

    cand = build_candidates(census, registry, label, topk_per_pass, max_per_census)
    if cand.empty:
        return pd.DataFrame(columns=["rid_c","rid_r","score"]), cand

    above = cand[cand["score"] >= threshold].copy()
    links = one_to_one(above)
    return links, cand


# =============== Sampling ===============
def proportion_sample_size(N, p, d, z=1.96):
    n0 = (z**2) * p * (1 - p) / (d**2)
    n  = n0 / (1 + (n0 - 1) / N)
    return int(np.ceil(n))

def sample_df(df, n, seed):
    return df.sample(n=min(int(n), len(df)), random_state=seed).copy()


def main():
    print("rapidfuzz enabled:", USE_RAPIDFUZZ)

    census_raw = pd.read_excel(CENSUS_PATH)
    registry_raw = pd.read_excel(REGISTRY_PATH)
    print(f"Loaded rows: census={len(census_raw)} registry={len(registry_raw)}")

    c_name = detect_column(census_raw, ["hhmember","name","person"])
    c_sex  = detect_column(census_raw, ["sex","gender","hheadsex"])
    c_vill = detect_column(census_raw, ["vname","vill","village"])
    c_age  = detect_column(census_raw, ["age","dob"])

    r_name = detect_column(registry_raw, ["name","patient"])
    r_sex  = detect_column(registry_raw, ["gender","sex"])
    r_vill = detect_column(registry_raw, ["village","vill"])
    r_age  = detect_column(registry_raw, ["age","dob"])

    print("Selected columns:")
    print("Census:  ", c_name, c_sex, c_vill, c_age)
    print("Registry:", r_name, r_sex, r_vill, r_age)

    census = prepare(census_raw, c_name, c_sex, c_vill, c_age, "C")
    registry = prepare(registry_raw, r_name, r_sex, r_vill, r_age, "R")

    # -------- calibrate threshold on full data (fast) --------
    threshold = calibrate_threshold(census, registry)

    # -------- PHASE 1 --------
    links1, cand1 = run_probabilistic(
        census, registry,
        label="PHASE1_FULL_PROB_FAST",
        threshold=threshold,
        topk_per_pass=TOPK_PER_PASS_FULL,
        max_per_census=MAX_PER_CENSUS_FULL
    )

    print(f"\n[PHASE 1] Links (1-to-1): {len(links1)}")
    links1.to_csv(os.path.join(OUT_DIR, "phase1_links.csv"), index=False)

    # -------- Sampling grids --------
    grid_c = pd.DataFrame(index=D_VALUES, columns=P_VALUES, dtype=int)
    grid_r = pd.DataFrame(index=D_VALUES, columns=P_VALUES, dtype=int)
    for d in D_VALUES:
        for p in P_VALUES:
            grid_c.loc[d,p] = proportion_sample_size(N_CENSUS, p, d, Z_95)
            grid_r.loc[d,p] = proportion_sample_size(N_REGISTRY, p, d, Z_95)

    grid_c.to_csv(os.path.join(OUT_DIR, "phase2_sampling_grid_census.csv"))
    grid_r.to_csv(os.path.join(OUT_DIR, "phase2_sampling_grid_registry.csv"))

    # -------- PHASE 2 --------
    rows = []
    for d in D_VALUES:
        for p in P_VALUES:
            n_c = int(grid_c.loc[d,p]); n_r = int(grid_r.loc[d,p])
            c_s = sample_df(census, n_c, RANDOM_SEED)
            r_s = sample_df(registry, n_r, RANDOM_SEED)

            label = f"PHASE2_PROB_p{p}_d{d}"
            links, _ = run_probabilistic(
                c_s, r_s,
                label=label,
                threshold=threshold,  # use calibrated threshold consistently
                topk_per_pass=TOPK_PER_PASS_SAMPLES,
                max_per_census=MAX_PER_CENSUS_SAMPLES
            )

            outpath = os.path.join(OUT_DIR, f"phase2_links_p{p}_d{d}.csv")
            links.to_csv(outpath, index=False)
            rows.append({"p":p, "d":d, "Census_n":n_c, "Registry_n":n_r, "Pairs":len(links)})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "phase2_summary_all.csv"), index=False)
    print("\nPHASE 2 summary:")
    print(summary.to_string(index=False))

    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
