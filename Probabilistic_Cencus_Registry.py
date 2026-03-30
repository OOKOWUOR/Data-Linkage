# ============================================================
# PROBABILISTIC LINKAGE - REWORKED FOR D1_Census + D2_Registry
# ------------------------------------------------------------
# Main changes versus your current script:
# - REMOVES age from probabilistic scoring
# - Uses explicit columns from your two datasets
# - Adds household-context support using hhead (census) and kin (registry)
# - Uses stronger village + sex contribution
# - Produces reviewed match classes: HIGH / POSSIBLE / REVIEW
# - Writes candidate and final-link outputs to working folder
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

# candidate control
REGISTRY_CAP_PER_KEY = {
    "k1": 250,
    "k2": 250,
    "k3": 250,
    "k4": 200,
    "k5": 200,
    "k6": 150
}

TOPK_PER_PASS = 60
MAX_PER_CENSUS = 100

# thresholds
THRESH_HIGH = 0.88
THRESH_POSSIBLE = 0.82
THRESH_REVIEW = 0.75

# ---------------- STRING HELPERS ----------------
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
    # for sex
    a = a if pd.notna(a) else "u"
    b = b if pd.notna(b) else "u"
    if a == "u" or b == "u":
        return 0.70
    return 1.0 if a == b else 0.0

def relation_sim(a, b):
    # light support variable, not dominant
    a = normalize_relationship(a)
    b = normalize_relationship(b)
    if not a and not b:
        return 0.5
    if not a or not b:
        return 0.4
    return 1.0 if a == b else 0.0

def village_bucket(v):
    v = clean_string(v)
    return v[:3] if v else ""

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
    x["sx_last"]  = x["last_name"].apply(safe_soundex)
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
    x["registry_age_num"] = x["age"].apply(numeric_from_age_string)  # kept only for diagnostics, not scoring

    x["first_name"], x["last_name"] = zip(*x["person_name"].apply(split_first_last))
    x["kin_first"], x["kin_last"] = zip(*x["kin_std"].apply(split_first_last))

    x["sx_first"] = x["first_name"].apply(safe_soundex)
    x["sx_last"]  = x["last_name"].apply(safe_soundex)
    x["sx_kin_last"] = x["kin_last"].apply(safe_soundex)

    x["fi"] = x["first_name"].str[:1].fillna("")
    x["li"] = x["last_name"].str[:1].fillna("")
    x["vill3"] = x["village_std"].apply(village_bucket)

    x["rid"] = ["R" + str(i) for i in range(len(x))]
    return x

# ---------------- SCORING ----------------
def score_pairs(m):
    full_name_sim = jw_list(m["person_name_c"].tolist(), m["person_name_r"].tolist())
    first_sim     = jw_list(m["first_name_c"].tolist(), m["first_name_r"].tolist())
    last_sim      = jw_list(m["last_name_c"].tolist(), m["last_name_r"].tolist())
    village_sim   = jw_list(m["village_std_c"].tolist(), m["village_std_r"].tolist())

    # household context: census household head vs registry kin
    hh_context_sim = jw_list(m["hhead_std"].tolist(), m["kin_std"].tolist())

    sex_sim = np.array([
        exact_or_unknown(a, b)
        for a, b in zip(m["sex_std_c"], m["sex_std_r"])
    ], dtype=float)

    rel_sim = np.array([
        relation_sim(a, b)
        for a, b in zip(m["relationship_std_c"], m["relationship_std_r"])
    ], dtype=float)

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
    hh_ok = (m["hhead_std"].values != "") & (m["kin_std"].values != "")
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
        np.where(out["score"] >= THRESH_POSSIBLE, "POSSIBLE",
                 np.where(out["score"] >= THRESH_REVIEW, "REVIEW", "LOW"))
    )

    return out

# ---------------- CANDIDATE GENERATION ----------------
def cap_registry_per_key(reg, key, cap):
    if cap is None or cap <= 0:
        return reg
    reg = reg.sort_values("rid")
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

def build_candidates(census, registry, label="FULL_RUN"):
    c = census.copy()
    r = registry.copy()

    # blocking keys
    c["k1"] = c["sx_first"] + "|" + c["sx_last"] + "|" + c["vill3"]
    r["k1"] = r["sx_first"] + "|" + r["sx_last"] + "|" + r["vill3"]

    c["k2"] = c["sx_first"] + "|" + c["sex_std"] + "|" + c["vill3"]
    r["k2"] = r["sx_first"] + "|" + r["sex_std"] + "|" + r["vill3"]

    c["k3"] = c["sx_last"] + "|" + c["sex_std"] + "|" + c["vill3"]
    r["k3"] = r["sx_last"] + "|" + r["sex_std"] + "|" + r["vill3"]

    c["k4"] = c["fi"] + "|" + c["li"] + "|" + c["vill3"] + "|" + c["sex_std"]
    r["k4"] = r["fi"] + "|" + r["li"] + "|" + r["vill3"] + "|" + r["sex_std"]

    # household context block: census household head last soundex vs registry kin last soundex
    c["k5"] = c["sx_head_last"] + "|" + c["vill3"]
    r["k5"] = r["sx_kin_last"] + "|" + r["vill3"]

    # more relaxed block
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

        r_key = cap_registry_per_key(r_key, key, REGISTRY_CAP_PER_KEY.get(key, 200))
        merged = c_key.merge(r_key, on=key, how="inner", suffixes=("_c", "_r"))

        print(f"[{label}] {pass_name}: raw candidates = {len(merged):,}")

        if merged.empty:
            continue

        need = [
    "rid_c", "rid_r",
    "person_name_c", "person_name_r",
    "first_name_c", "first_name_r",
    "last_name_c", "last_name_r",
    "village_std_c", "village_std_r",
    "sex_std_c", "sex_std_r",
    "relationship_std_c", "relationship_std_r",
    "hhead_std", "kin_std",
    "hhno_std", "hospitalno"
]

        merged = merged[[col for col in need if col in merged.columns]].copy()
        merged = score_pairs(merged)
        merged = topk_per_left(merged, TOPK_PER_PASS)

        all_candidates.append(merged)

    if not all_candidates:
        return pd.DataFrame()

    cand = pd.concat(all_candidates, ignore_index=True)
    cand = cand.sort_values("score", ascending=False).drop_duplicates(["rid_c", "rid_r"], keep="first")
    cand = cand.sort_values("score", ascending=False).groupby("rid_c", as_index=False).head(MAX_PER_CENSUS)

    return cand

# ---------------- CALIBRATION ----------------
def deterministic_seed_pairs(census, registry, max_pairs=5000):
    """
    Seed pairs used only to estimate a reasonable threshold.
    Strict exact match on first + last + sex + village prefix.
    """
    left = census[["rid", "first_name", "last_name", "sex_std", "vill3",
                   "person_name", "village_std", "relationship_std", "hhead_std"]].copy()
    right = registry[["rid", "first_name", "last_name", "sex_std", "vill3",
                      "person_name", "village_std", "relationship_std", "kin_std"]].copy()

    left = left.rename(columns={
        "rid": "rid_c",
        "person_name": "person_name_c",
        "village_std": "village_std_c",
        "relationship_std": "relationship_std_c",
        "hhead_std": "hhead_std_c"
    })
    right = right.rename(columns={
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
        "rid_c": m["rid_c"],
        "rid_r": m["rid_r"],
        "person_name_c": m["person_name_c"],
        "person_name_r": m["person_name_r"],
        "first_name_c": m["first_name"],
        "first_name_r": m["first_name"],
        "last_name_c": m["last_name"],
        "last_name_r": m["last_name"],
        "village_std_c": m["village_std_c"],
        "village_std_r": m["village_std_r"],
        "sex_std_c": m["sex_std"],
        "sex_std_r": m["sex_std"],
        "relationship_std_c": m["relationship_std_c"],
        "relationship_std_r": m["relationship_std_r"],
        "hhead_std_c": m["hhead_std_c"],
        "kin_std_r": m["kin_std_r"],
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
    print(f"\nThresholds:")
    print(f"  HIGH     >= {THRESH_HIGH}")
    print(f"  POSSIBLE >= {THRESH_POSSIBLE}")
    print(f"  REVIEW   >= {THRESH_REVIEW}")

    candidates = build_candidates(census, registry, label="FULL_REWORKED_RUN")

    if candidates.empty:
        print("\nNo candidates found.")
        return

    print("\nCandidate score summary:")
    print(candidates["score"].describe(percentiles=[0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]).to_string())

    # keep reviewable set
    reviewable = candidates[candidates["score"] >= THRESH_REVIEW].copy()

    # final one-to-one links for POSSIBLE and HIGH only
    final_links = one_to_one(reviewable[reviewable["score"] >= THRESH_POSSIBLE].copy())

    # rank within each census record
    reviewable["rank_within_census"] = (
        reviewable.sort_values("score", ascending=False)
        .groupby("rid_c")
        .cumcount() + 1
    )

    # bring original identifiers back
    census_keep = census[[
        "rid", "hhmember", "sex", "vname", "rship", "hhead", "hhno", "enumno", "pno"
    ]].copy().rename(columns={"rid": "rid_c"})

    registry_keep = registry[[
        "rid", "name", "gender", "village", "rship", "kin", "hospitalno", "district", "subcounty"
    ]].copy().rename(columns={"rid": "rid_r"})

    reviewable = reviewable.merge(census_keep, on="rid_c", how="left")
    reviewable = reviewable.merge(registry_keep, on="rid_r", how="left", suffixes=("_census", "_registry"))

    final_links = final_links.merge(census_keep, on="rid_c", how="left")
    final_links = final_links.merge(registry_keep, on="rid_r", how="left", suffixes=("_census", "_registry"))

    # save outputs
    reviewable.to_csv(os.path.join(OUT_DIR, "probabilistic_candidates_reviewable.csv"), index=False)
    final_links.to_csv(os.path.join(OUT_DIR, "probabilistic_final_links.csv"), index=False)

    summary = (
        reviewable["match_class"]
        .value_counts(dropna=False)
        .rename_axis("match_class")
        .reset_index(name="n")
    )
    summary.to_csv(os.path.join(OUT_DIR, "probabilistic_match_summary.csv"), index=False)

    print("\nSaved files:")
    print(os.path.join(OUT_DIR, "probabilistic_candidates_reviewable.csv"))
    print(os.path.join(OUT_DIR, "probabilistic_final_links.csv"))
    print(os.path.join(OUT_DIR, "probabilistic_match_summary.csv"))

    print("\nMatch class counts:")
    print(summary.to_string(index=False))

    print(f"\nFinal one-to-one links retained: {len(final_links):,}")

if __name__ == "__main__":
    main()