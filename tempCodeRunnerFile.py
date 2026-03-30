# ============================================================
# PROBABILISTIC LINKAGE - FULL WORKING VERSION + P/D GRID
# ------------------------------------------------------------
# Fixes:
# - Removes age from scoring
# - Uses consistent column names everywhere
# - Fixes household-context columns
# - Restores Phase 2 grid over p and d like original file
# - Saves phase1 full-run outputs and phase2 sample outputs
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import jellyfish

# ---------------- CONFIG ----------------
CENSUS_PATH   = r"C:/Users/guest441/Downloads/D1_Census.xlsx"
REGISTRY_PATH = r"C:/Users/guest441/Downloads/D2_Registry.xlsx"
OUT_DIR       = r"C:/Users/guest441/Downloads/linkage_outputs_probabilistic_reworked"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# original-style p and d grid
N_CENSUS = 21766
N_REGISTRY = 75125
P_VALUES = [0.5, 0.4, 0.3]
D_VALUES = [0.10, 0.05, 0.01]
Z_95 = 1.96

REGISTRY_CAP_PER_KEY = {
    "k1": 250,
    "k2": 250,
    "k3": 250,
    "k4": 200,
    "k5": 200,
    "k6": 150
}

TOPK_PER_PASS_FULL = 60
MAX_PER_CENSUS_FULL = 100

TOPK_PER_PASS_SAMPLES = 80
MAX_PER_CENSUS_SAMPLES = 120

THRESH_HIGH = 0.88
THRESH_POSSIBLE = 0.82
THRESH_REVIEW = 0.75

# ---------------- HELPERS ----------------
def clean_string(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_sex(x):
    x = clean_string(x)
    if x in {"m", "male", "man", "1"}:
        return "m"
    if x in {"f", "female", "woman", "2"}:
        return "f"
    return "u"

def normalize_relationship(x):
    x = clean_string(x)
    mapping = {
        "head": "head",
        "self": "head",
        "wife": "spouse",
        "husband": "spouse",
        "spouse": "spouse",
        "daughter son": "child",
        "son": "child",
        "daughter": "child",
        "child": "child",
        "grandchild": "grandchild",
        "father": "parent",
        "mother": "parent",
        "parent": "parent",
        "brother": "sibling",
        "sister": "sibling",
        "sibling": "sibling",
        "other": "other"
    }
    for k, v in mapping.items():
        if k in x:
            return v
    return x if x else ""

def split_first_last(full_name):
    parts = clean_string(full_name).split()
    if len(parts) == 0:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[-1]

def safe_soundex(x):
    x = clean_string(x)
    if not x:
        return ""
    try:
        return jellyfish.soundex(x)
    except Exception:
        return ""

def numeric_from_age_string(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else np.nan

def village_bucket(v):
    v = clean_string(v)
    return v[:3] if v else ""

# ---------------- SIMILARITY ----------------
def jw(a, b):
    a = clean_string(a)
    b = clean_string(b)
    if not a and not b:
        return 0.5
    return float(jellyfish.jaro_winkler_similarity(a, b))

def jw_list(a_list, b_list):
    out = np.empty(len(a_list), dtype=float)
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        out[i] = jw(a, b)
    return out

def exact_or_unknown(a, b):
    a = a if pd.notna(a) else "u"
    b = b if pd.notna(b) else "u"
    if a == "u" or b == "u":
        return 0.70
    return 1.0 if a == b else 0.0

def relation_sim(a, b):
    a = normalize_relationship(a)
    b = normalize_relationship(b)
    if not a and not b:
        return 0.5
    if not a or not b:
        return 0.4
    return 1.0 if a == b else 0.0

# ---------------- PREP ----------------
def prepare_census(df):
    x = df.copy()

    x["person_name"] = x["hhmember"].apply(clean_string)
    x["sex_std"] = x["sex"].apply(normalize_sex)
    x["village_std"] = x["vname"].apply(clean_string)
    x["relationship_std"] = x["rship"].apply(normalize_relationship)

    x["hhead_std"] = x["hhead"].apply(clean_string)
    x["hhno_std"] = x["hhno"].astype(str).str.strip()

    x["first_name"], x["last_name"] = zip(*x["person_name"].apply(split_first_last))
    x["head_first"], x["head_last"] = zip(*x["hhead_std"].apply(split_first_last))

    x["sx_first"] = x["first_name"].apply(safe_soundex)
    x["sx_last"] = x["last_name"].apply(safe_soundex)
    x["sx_head_last"] = x["head_last"].apply(safe_soundex)

    x["fi"] = x["first_name"].str[:1].fillna("")
    x["li"] = x["last_name"].str[:1].fillna("")
    x["vill3"] = x["village_std"].apply(village_bucket)

    x["rid"] = ["C" + str(i) for i in range(len(x))]
    return x

def prepare_registry(df):
    x = df.copy()

    x["person_name"] = x["name"].apply(clean_string)
    x["sex_std"] = x["gender"].apply(normalize_sex)
    x["village_std"] = x["village"].apply(clean_string)
    x["relationship_std"] = x["rship"].apply(normalize_relationship)

    x["kin_std"] = x["kin"].apply(clean_string)
    x["registry_age_num"] = x["age"].apply(numeric_from_age_string)

    x["first_name"], x["last_name"] = zip(*x["person_name"].apply(split_first_last))
    x["kin_first"], x["kin_last"] = zip(*x["kin_std"].apply(split_first_last))

    x["sx_first"] = x["first_name"].apply(safe_soundex)
    x["sx_last"] = x["last_name"].apply(safe_soundex)
    x["sx_kin_last"] = x["kin_last"].apply(safe_soundex)

    x["fi"] = x["first_name"].str[:1].fillna("")
    x["li"] = x["last_name"].str[:1].fillna("")
    x["vill3"] = x["village_std"].apply(village_bucket)

    x["rid"] = ["R" + str(i) for i in range(len(x))]
    return x

# ---------------- SCORING ----------------
def ensure_required_columns(m):
    needed_defaults = {
        "hhead_std_c": "",
        "kin_std_r": "",
        "relationship_std_c": "",
        "relationship_std_r": "",
        "sex_std_c": "u",
        "sex_std_r": "u",
        "village_std_c": "",
        "village_std_r": "",
        "person_name_c": "",
        "person_name_r": "",
        "first_name_c": "",
        "first_name_r": "",
        "last_name_c": "",
        "last_name_r": "",
    }
    for col, default in needed_defaults.items():
        if col not in m.columns:
            m[col] = default
    return m

def score_pairs(m):
    m = m.copy()
    m = ensure_required_columns(m)

    full_name_sim = jw_list(m["person_name_c"].tolist(), m["person_name_r"].tolist())
    first_sim = jw_list(m["first_name_c"].tolist(), m["first_name_r"].tolist())
    last_sim = jw_list(m["last_name_c"].tolist(), m["last_name_r"].tolist())
    village_sim = jw_list(m["village_std_c"].tolist(), m["village_std_r"].tolist())
    hh_context_sim = jw_list(m["hhead_std_c"].tolist(), m["kin_std_r"].tolist())

    sex_sim = np.array(
        [exact_or_unknown(a, b) for a, b in zip(m["sex_std_c"], m["sex_std_r"])],
        dtype=float
    )

    rel_sim = np.array(
        [relation_sim(a, b) for a, b in zip(m["relationship_std_c"], m["relationship_std_r"])],
        dtype=float
    )

    W = {
        "full": 0.36,
        "first": 0.18,
        "last": 0.16,
        "village": 0.14,
        "sex": 0.08,
        "hhctx": 0.06,
        "rel": 0.02
    }

    full_ok = (m["person_name_c"].values != "") & (m["person_name_r"].values != "")
    first_ok = (m["first_name_c"].values != "") & (m["first_name_r"].values != "")
    last_ok = (m["last_name_c"].values != "") & (m["last_name_r"].values != "")
    village_ok = (m["village_std_c"].values != "") & (m["village_std_r"].values != "")
    sex_ok = np.ones(len(m), dtype=bool)
    hh_ok = (m["hhead_std_c"].values != "") & (m["kin_std_r"].values != "")
    rel_ok = (m["relationship_std_c"].values != "") & (m["relationship_std_r"].values != "")

    denom = (
        W["full"] * full_ok.astype(float) +
        W["first"] * first_ok.astype(float) +
        W["last"] * last_ok.astype(float) +
        W["village"] * village_ok.astype(float) +
        W["sex"] * sex_ok.astype(float) +
        W["hhctx"] * hh_ok.astype(float) +
        W["rel"] * rel_ok.astype(float)
    )
    denom = np.where(denom == 0, 1.0, denom)

    score = (
        W["full"] * full_ok.astype(float) * full_name_sim +
        W["first"] * first_ok.astype(float) * first_sim +
        W["last"] * last_ok.astype(float) * last_sim +
        W["village"] * village_ok.astype(float) * village_sim +
        W["sex"] * sex_sim +
        W["hhctx"] * hh_ok.astype(float) * hh_context_sim +
        W["rel"] * rel_ok.astype(float) * rel_sim
    ) / denom

    out = m.copy()
    out["sim_full_name"] = full_name_sim
    out["sim_first_name"] = first_sim
    out["sim_last_name"] = last_sim
    out["sim_village"] = village_sim
    out["sim_household_context"] = hh_context_sim
    out["sim_sex"] = sex_sim
    out["sim_relationship"] = rel_sim
    out["score"] = score

    out["match_class"] = np.where(
        out["score"] >= THRESH_HIGH, "HIGH",
        np.where(
            out["score"] >= THRESH_POSSIBLE, "POSSIBLE",
            np.where(out["score"] >= THRESH_REVIEW, "REVIEW", "LOW")
        )
    )
    return out

# ---------------- MATCHING HELPERS ----------------
def cap_registry_per_key(reg, key, cap):
    if cap is None or cap <= 0:
        return reg
    reg = reg.sort_values("rid_r" if "rid_r" in reg.columns else "rid")
    return reg.groupby(key, as_index=False).head(cap)

def topk_per_left(df, k):
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).groupby("rid_c", as_index=False).head(k)

def one_to_one(df):
    if df.empty:
        return df.copy()

    df = df.sort_values(["score", "sim_full_name", "sim_village"], ascending=False)
    used_c = set()
    used_r = set()
    keep = []

    for _, row in df.iterrows():
        c = row["rid_c"]
        r = row["rid_r"]
        if c in used_c or r in used_r:
            continue
        keep.append(row)
        used_c.add(c)
        used_r.add(r)

    return pd.DataFrame(keep)

# ---------------- CANDIDATE GENERATION ----------------
def build_candidates(census, registry, label="FULL_RUN", topk_per_pass=60, max_per_census=100):
    c = census.copy()
    r = registry.copy()

    c["k1"] = c["sx_first"] + "|" + c["sx_last"] + "|" + c["vill3"]
    r["k1"] = r["sx_first"] + "|" + r["sx_last"] + "|" + r["vill3"]

    c["k2"] = c["sx_first"] + "|" + c["sex_std"] + "|" + c["vill3"]
    r["k2"] = r["sx_first"] + "|" + r["sex_std"] + "|" + r["vill3"]

    c["k3"] = c["sx_last"] + "|" + c["sex_std"] + "|" + c["vill3"]
    r["k3"] = r["sx_last"] + "|" + r["sex_std"] + "|" + r["vill3"]

    c["k4"] = c["fi"] + "|" + c["li"] + "|" + c["vill3"] + "|" + c["sex_std"]
    r["k4"] = r["fi"] + "|" + r["li"] + "|" + r["vill3"] + "|" + r["sex_std"]

    c["k5"] = c["sx_head_last"] + "|" + c["vill3"]
    r["k5"] = r["sx_kin_last"] + "|" + r["vill3"]

    c["k6"] = c["sx_first"] + "|" + c["vill3"]
    r["k6"] = r["sx_first"] + "|" + r["vill3"]

    passes = [
        ("PASS1_NAME_VILL", "k1"),
        ("PASS2_FNAME_SEX_VILL", "k2"),
        ("PASS3_LNAME_SEX_VILL", "k3"),
        ("PASS4_INITIALS_VILL_SEX", "k4"),
        ("PASS5_HHCTX_VILL", "k5"),
        ("PASS6_RELAXED_FNAME_VILL", "k6"),
    ]

    all_candidates = []

    for pass_name, key in passes:
        print(f"\n[{label}] {pass_name}: blocking on {key}")

        c_key = c[(c[key].notna()) & (c[key] != "")].copy()
        r_key = r[(r[key].notna()) & (r[key] != "")].copy()

        c_keep = c_key[[
            key, "rid", "person_name", "first_name", "last_name",
            "village_std", "sex_std", "relationship_std", "hhead_std", "hhno_std"
        ]].copy().rename(columns={
            "rid": "rid_c",
            "person_name": "person_name_c",
            "first_name": "first_name_c",
            "last_name": "last_name_c",
            "village_std": "village_std_c",
            "sex_std": "sex_std_c",
            "relationship_std": "relationship_std_c",
            "hhead_std": "hhead_std_c",
            "hhno_std": "hhno_std_c"
        })

        r_keep = r_key[[
            key, "rid", "person_name", "first_name", "last_name",
            "village_std", "sex_std", "relationship_std", "kin_std", "hospitalno"
        ]].copy().rename(columns={
            "rid": "rid_r",
            "person_name": "person_name_r",
            "first_name": "first_name_r",
            "last_name": "last_name_r",
            "village_std": "village_std_r",
            "sex_std": "sex_std_r",
            "relationship_std": "relationship_std_r",
            "kin_std": "kin_std_r"
        })

        r_keep = cap_registry_per_key(r_keep, key, REGISTRY_CAP_PER_KEY.get(key, 200))
        merged = c_keep.merge(r_keep, on=key, how="inner")

        print(f"[{label}] {pass_name}: raw candidates = {len(merged):,}")

        if merged.empty:
            continue

        merged = score_pairs(merged)
        merged = topk_per_left(merged, topk_per_pass)
        all_candidates.append(merged)

    if not all_candidates:
        return pd.DataFrame()

    cand = pd.concat(all_candidates, ignore_index=True)
    cand = cand.sort_values("score", ascending=False).drop_duplicates(["rid_c", "rid_r"], keep="first")
    cand = cand.sort_values("score", ascending=False).groupby("rid_c", as_index=False).head(max_per_census)

    return cand

# ---------------- CALIBRATION ----------------
def deterministic_seed_pairs(census, registry, max_pairs=5000):
    left = census[[
        "rid", "first_name", "last_name", "sex_std", "vill3",
        "person_name", "village_std", "relationship_std", "hhead_std"
    ]].copy().rename(columns={
        "rid": "rid_c",
        "person_name": "person_name_c",
        "village_std": "village_std_c",
        "relationship_std": "relationship_std_c",
        "hhead_std": "hhead_std_c"
    })

    right = registry[[
        "rid", "first_name", "last_name", "sex_std", "vill3",
        "person_name", "village_std", "relationship_std", "kin_std"
    ]].copy().rename(columns={
        "rid": "rid_r",
        "person_name": "person_name_r",
        "village_std": "village_std_r",
        "relationship_std": "relationship_std_r",
        "kin_std": "kin_std_r"
    })

    m = left.merge(
        right,
        on=["first_name", "last_name", "sex_std", "vill3"],
        how="inner"
    )

    if m.empty:
        return pd.DataFrame()

    if len(m) > max_pairs:
        m = m.sample(max_pairs, random_state=RANDOM_SEED)

    out = pd.DataFrame({
        "rid_c": m["rid_c"].values,
        "rid_r": m["rid_r"].values,
        "person_name_c": m["person_name_c"].values,
        "person_name_r": m["person_name_r"].values,
        "first_name_c": m["first_name"].values,
        "first_name_r": m["first_name"].values,
        "last_name_c": m["last_name"].values,
        "last_name_r": m["last_name"].values,
        "village_std_c": m["village_std_c"].values,
        "village_std_r": m["village_std_r"].values,
        "sex_std_c": m["sex_std"].values,
        "sex_std_r": m["sex_std"].values,
        "relationship_std_c": m["relationship_std_c"].values,
        "relationship_std_r": m["relationship_std_r"].values,
        "hhead_std_c": m["hhead_std_c"].values,
        "kin_std_r": m["kin_std_r"].values,
    })
    return out

def calibrate_threshold(census, registry):
    seed = deterministic_seed_pairs(census, registry, max_pairs=5000)

    if seed.empty:
        print("\n[CALIBRATION] No seed pairs found. Using default thresholds.")
        return THRESH_HIGH, THRESH_POSSIBLE, THRESH_REVIEW

    scored = score_pairs(seed)

    high = max(0.85, float(scored["score"].quantile(0.10)))
    possible = max(0.80, high - 0.05)
    review = max(0.72, possible - 0.07)

    print("\n[CALIBRATION] Seed-pair score summary:")
    print(scored["score"].describe(percentiles=[0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]).to_string())

    return round(high, 3), round(possible, 3), round(review, 3)

# ---------------- SAMPLING ----------------
def proportion_sample_size(N, p, d, z=1.96):
    n0 = (z ** 2) * p * (1 - p) / (d ** 2)
    n = n0 / (1 + (n0 - 1) / N)
    return int(np.ceil(n))

def sample_df(df, n, seed):
    return df.sample(n=min(int(n), len(df)), random_state=seed).copy()

# ---------------- RUNNER ----------------
def run_linkage(census_df, registry_df, label, topk_per_pass, max_per_census):
    candidates = build_candidates(
        census_df,
        registry_df,
        label=label,
        topk_per_pass=topk_per_pass,
        max_per_census=max_per_census
    )

    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()

    reviewable = candidates[candidates["score"] >= THRESH_REVIEW].copy()
    final_links = one_to_one(reviewable[reviewable["score"] >= THRESH_POSSIBLE].copy())

    reviewable["rank_within_census"] = (
        reviewable.sort_values("score", ascending=False)
        .groupby("rid_c")
        .cumcount() + 1
    )

    return final_links, reviewable

# ---------------- MAIN ----------------
def main():
    global THRESH_HIGH, THRESH_POSSIBLE, THRESH_REVIEW

    census_raw = pd.read_excel(CENSUS_PATH)
    registry_raw = pd.read_excel(REGISTRY_PATH)

    print(f"Loaded census rows   : {len(census_raw):,}")
    print(f"Loaded registry rows : {len(registry_raw):,}")

    census = prepare_census(census_raw)
    registry = prepare_registry(registry_raw)

    print("\nPrepared variables:")
    print("Census   -> hhmember, sex, vname, rship, hhead")
    print("Registry -> name, gender, village, rship, kin")

    THRESH_HIGH, THRESH_POSSIBLE, THRESH_REVIEW = calibrate_threshold(census, registry)

    print("\nThresholds:")
    print(f"  HIGH     >= {THRESH_HIGH}")
    print(f"  POSSIBLE >= {THRESH_POSSIBLE}")
    print(f"  REVIEW   >= {THRESH_REVIEW}")

    # -------- PHASE 1: FULL DATA --------
    phase1_links, phase1_reviewable = run_linkage(
        census, registry,
        label="PHASE1_FULL_REWORKED_RUN",
        topk_per_pass=TOPK_PER_PASS_FULL,
        max_per_census=MAX_PER_CENSUS_FULL
    )

    phase1_links.to_csv(os.path.join(OUT_DIR, "phase1_probabilistic_final_links.csv"), index=False)
    phase1_reviewable.to_csv(os.path.join(OUT_DIR, "phase1_probabilistic_candidates_reviewable.csv"), index=False)

    print(f"\n[PHASE 1] final one-to-one links: {len(phase1_links):,}")
    print(f"[PHASE 1] reviewable candidates: {len(phase1_reviewable):,}")

    # -------- PHASE 2: GRID OVER p AND d --------
    rows = []

    for d in D_VALUES:
        for p in P_VALUES:
            n_c = proportion_sample_size(N_CENSUS, p, d, Z_95)
            n_r = proportion_sample_size(N_REGISTRY, p, d, Z_95)

            c_s = sample_df(census, n_c, RANDOM_SEED)
            r_s = sample_df(registry, n_r, RANDOM_SEED)

            label = f"PHASE2_PROB_p{p}_d{d}"

            links, reviewable = run_linkage(
                c_s, r_s,
                label=label,
                topk_per_pass=TOPK_PER_PASS_SAMPLES,
                max_per_census=MAX_PER_CENSUS_SAMPLES
            )

            links.to_csv(os.path.join(OUT_DIR, f"phase2_prob_links_p{p}_d{d}.csv"), index=False)
            reviewable.to_csv(os.path.join(OUT_DIR, f"phase2_prob_reviewable_p{p}_d{d}.csv"), index=False)

            rows.append({
                "p": p,
                "d": d,
                "Census_n": int(n_c),
                "Registry_n": int(n_r),
                "Reviewable_candidates": int(len(reviewable)),
                "Final_pairs": int(len(links))
            })

            print(f"\n[PHASE 2] p={p}, d={d}, census_n={n_c}, registry_n={n_r}, final_pairs={len(links):,}")

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "phase2_probabilistic_summary_all.csv"), index=False)

    print("\nPHASE 2 summary:")
    print(summary.to_string(index=False))

    print(f"\nOutputs written to: {OUT_DIR}")

if __name__ == "__main__":
    main()