# ==========================================================
# Robust Deterministic + Probabilistic Data Linkage (Full + Samples)
# - Strong column detection (header + data profiling)
# - Deterministic includes phonetic rules (prevents 0 matches)
# - Probabilistic uses multiple blocking passes
# - Phase II always runs and outputs clean tables
# ==========================================================

import os
import re
import numpy as np
import pandas as pd
import jellyfish

import recordlinkage

# ---------------------------
# 0. CONFIG
# ---------------------------
CENSUS_PATH   = r"C:/Users/guest441/Downloads/D1_Census.xlsx"
REGISTRY_PATH = r"C:/Users/guest441/Downloads/D2_Registry.xlsx"

OUT_DIR = r"C:/Users/guest441/Downloads/linkage_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42

# Population sizes (from your PDF)
N_CENSUS   = 21766
N_REGISTRY = 75125

# Phase 2 scenarios
P_VALUES = [0.5, 0.4, 0.3]
D_VALUES = [0.10, 0.05, 0.01]
Z_95 = 1.96

# ---------------------------
# 1) BASIC CLEANING HELPERS
# ---------------------------
def clean_string(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_sex(s):
    s = clean_string(s)
    if s in ["m", "male", "man", "1"]:
        return "m"
    if s in ["f", "female", "woman", "2"]:
        return "f"
    return s

def split_first_last(fullname):
    parts = fullname.split()
    if len(parts) == 0:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[-1]

def safe_soundex(s):
    s = s or ""
    try:
        return jellyfish.soundex(s) if s else ""
    except Exception:
        return ""

def safe_metaphone(s):
    s = s or ""
    try:
        return jellyfish.metaphone(s) if s else ""
    except Exception:
        return ""

def to_numeric_age(x):
    return pd.to_numeric(x, errors="coerce")

# ---------------------------
# 2) SMART COLUMN DETECTION
# ---------------------------
NAME_HINTS = ["name", "hhmember", "member", "person", "patient", "client", "full"]
VILL_HINTS = ["village", "vill", "vname", "location", "residence", "ward"]
SEX_HINTS  = ["sex", "gender"]
AGE_HINTS  = ["age", "dob", "years", "yr", "birth"]

def profile_col(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return dict(uniq=0, space_rate=0.0, avg_len=0.0, numeric_rate=0.0, mf_rate=0.0)

    s_str = s.astype(str)
    uniq = s_str.nunique()
    space_rate = s_str.str.contains(r"\s").mean()
    avg_len = s_str.str.len().mean()
    numeric_rate = pd.to_numeric(s, errors="coerce").notna().mean()

    # male/female rate
    ss = s_str.map(clean_string)
    mf_rate = ss.isin(["m","f","male","female","1","2","man","woman"]).mean()

    return dict(
        uniq=int(uniq),
        space_rate=float(space_rate),
        avg_len=float(avg_len),
        numeric_rate=float(numeric_rate),
        mf_rate=float(mf_rate),
    )

def header_score(colname: str, hints):
    c = colname.lower()
    score = 0
    for h in hints:
        if h in c:
            score += 3
    return score

def detect_columns(df: pd.DataFrame, label="DATASET"):
    cols = list(df.columns)

    prof = {c: profile_col(df[c]) for c in cols}

    # Build score tables for each type
    def score_name(c):
        p = prof[c]
        s = header_score(c, NAME_HINTS)
        # names: high uniq, spaces common, not numeric
        s += min(5, p["uniq"] / max(1, len(df)) * 50)   # scaled uniqueness
        s += 3 * p["space_rate"]
        s += 2 * (1 - p["numeric_rate"])
        return s

    def score_vill(c):
        p = prof[c]
        s = header_score(c, VILL_HINTS)
        # villages: low/moderate uniq, fewer spaces, not numeric
        s += 2 * (1 - p["numeric_rate"])
        # prefer moderate uniq (not 2, not 20000)
        uniq = p["uniq"]
        if 5 <= uniq <= max(50, int(len(df) * 0.05)):
            s += 3
        s += 1 * (1 - p["space_rate"])
        return s

    def score_sex(c):
        p = prof[c]
        s = header_score(c, SEX_HINTS)
        # sex: very low uniq and high mf_rate
        if p["uniq"] <= 10:
            s += 3
        s += 5 * p["mf_rate"]
        s += 1 * (1 - p["numeric_rate"])
        return s

    def score_age(c):
        p = prof[c]
        s = header_score(c, AGE_HINTS)
        # age: high numeric rate
        s += 8 * p["numeric_rate"]
        return s

    name_rank = sorted(cols, key=lambda c: score_name(c), reverse=True)
    vill_rank = sorted(cols, key=lambda c: score_vill(c), reverse=True)
    sex_rank  = sorted(cols, key=lambda c: score_sex(c), reverse=True)
    age_rank  = sorted(cols, key=lambda c: score_age(c), reverse=True)

    name_col = name_rank[0] if name_rank else None
    vill_col = vill_rank[0] if vill_rank else None
    sex_col  = sex_rank[0]  if sex_rank else None
    age_col  = age_rank[0]  if age_rank else None

    # Force name != village if possible
    if name_col == vill_col and len(name_rank) > 1:
        for c in name_rank[1:]:
            if c != vill_col:
                name_col = c
                break

    # Print a detection table (top 5 candidates)
    def top_table(rank_list, scorer, title):
        rows = []
        for c in rank_list[:5]:
            rows.append({
                "field": title,
                "candidate": c,
                "score": round(float(scorer(c)), 3),
                "uniq": prof[c]["uniq"],
                "space_rate": round(prof[c]["space_rate"], 3),
                "numeric_rate": round(prof[c]["numeric_rate"], 3),
                "mf_rate": round(prof[c]["mf_rate"], 3),
            })
        return rows

    rows = []
    rows += top_table(name_rank, score_name, "NAME")
    rows += top_table(vill_rank, score_vill, "VILLAGE")
    rows += top_table(sex_rank,  score_sex,  "SEX")
    rows += top_table(age_rank,  score_age,  "AGE")

    det_table = pd.DataFrame(rows)

    print(f"\n[{label}] Column detection candidates (top 5 each):")
    try:
        from tabulate import tabulate
        print(tabulate(det_table, headers="keys", tablefmt="github", showindex=False))
    except Exception:
        print(det_table.to_string(index=False))

    print(f"\n[{label}] Selected columns:")
    print(f"  name={name_col}, sex={sex_col}, village={vill_col}, age={age_col}")

    return name_col, sex_col, vill_col, age_col

# ---------------------------
# 3) PREPARE DATAFRAMES
# ---------------------------
def prepare_df(df, name_col, sex_col, vill_col, age_col, id_prefix):
    df = df.copy()

    df["person_name"] = df[name_col].apply(clean_string)

    df["sex"] = df[sex_col].apply(normalize_sex) if sex_col else ""
    df["village"] = df[vill_col].apply(clean_string) if vill_col else ""

    df["age"] = df[age_col].apply(to_numeric_age) if age_col else np.nan

    df["first_name"], df["last_name"] = zip(*df["person_name"].apply(split_first_last))

    df["sx_first"] = df["first_name"].apply(safe_soundex)
    df["sx_last"]  = df["last_name"].apply(safe_soundex)
    df["mp_first"] = df["first_name"].apply(safe_metaphone)
    df["mp_last"]  = df["last_name"].apply(safe_metaphone)

    df["rid"] = [f"{id_prefix}{i}" for i in range(len(df))]
    return df

# ---------------------------
# 4) MATCHING UTILITIES
# ---------------------------
def one_to_one_greedy(df, score_col=None):
    if df is None or df.empty:
        return df

    out = df.copy()
    if score_col and score_col in out.columns:
        out = out.sort_values(score_col, ascending=False)
    else:
        out = out.sort_values(["rid_c", "rid_r"])

    used_c = set()
    used_r = set()
    keep = []
    for _, row in out.iterrows():
        c = row["rid_c"]
        r = row["rid_r"]
        if c in used_c or r in used_r:
            continue
        keep.append(row)
        used_c.add(c)
        used_r.add(r)
    return pd.DataFrame(keep)

def summarize_matches(df, label):
    if df is None or df.empty:
        return pd.DataFrame([{"Run": label, "Pairs": 0, "Unique Census IDs": 0, "Unique Registry IDs": 0}])
    return pd.DataFrame([{
        "Run": label,
        "Pairs": int(len(df)),
        "Unique Census IDs": int(df["rid_c"].nunique()),
        "Unique Registry IDs": int(df["rid_r"].nunique()),
    }])

def pretty_print_df(df, title):
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))
    if df is None or df.empty:
        print("(empty)")
    else:
        try:
            from tabulate import tabulate
            print(tabulate(df, headers="keys", tablefmt="github", showindex=False))
        except Exception:
            print(df.to_string(index=False))

# ---------------------------
# 5) DETERMINISTIC (WITH PHONETIC RULES + BLOCKING)
# ---------------------------
def run_deterministic(census_df, registry_df, label):
    """
    Deterministic rules, but includes phonetic rules so you don't get 0 matches.
    Uses lightweight blocking to avoid huge merges.
    """
    rules = [
        # Exact strong
        (1, "EXACT_FN_LN_SEX_VILL", ["first_name", "last_name", "sex", "village"]),
        (2, "EXACT_FN_LN_SEX",      ["first_name", "last_name", "sex"]),
        (3, "EXACT_FN_LN",          ["first_name", "last_name"]),
        # Phonetic deterministic (still deterministic)
        (4, "SOUNDEX_LN_SEX_FN",    ["sx_last", "sex", "first_name"]),
        (5, "SOUNDEX_FN_LN",        ["sx_first", "sx_last"]),
        (6, "METAPHONE_FN_LN",      ["mp_first", "mp_last"]),
        # Weakest exact full string
        (7, "EXACT_FULLNAME",       ["person_name"]),
    ]

    all_pairs = []
    stats = []

    for rid, rname, cols in rules:
        left_cols = [c for c in cols if c in census_df.columns]
        right_cols = [c for c in cols if c in registry_df.columns]
        if not left_cols:
            continue

        merged = census_df.merge(
            registry_df,
            left_on=left_cols,
            right_on=right_cols,
            suffixes=("_c", "_r"),
            how="inner"
        )

        stats.append({"Rule": rid, "RuleName": rname, "Columns": " + ".join(cols), "Pairs": int(len(merged))})

        if merged.empty:
            continue

        out_cols = ["rid_c", "rid_r", "person_name_c", "person_name_r", "sex_c", "sex_r", "village_c", "village_r", "age_c", "age_r"]
        out_cols = [c for c in out_cols if c in merged.columns]
        out = merged[out_cols].copy()
        out["det_rule"] = rid
        out["det_rule_name"] = rname
        out["score"] = 1.0
        all_pairs.append(out)

    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(columns=["rid_c", "rid_r", "det_rule"])

    if not pairs.empty:
        pairs = pairs.sort_values("det_rule").drop_duplicates(subset=["rid_c", "rid_r"], keep="first")

    pairs_1to1 = one_to_one_greedy(pairs, score_col=None)

    return pairs_1to1, pd.DataFrame(stats).sort_values("Rule"), summarize_matches(pairs_1to1, label)

# ---------------------------
# 6) PROBABILISTIC (MULTI-BLOCK + SCORING)
# ---------------------------
def run_probabilistic(census_df, registry_df, label, threshold=0.85):
    """
    Probabilistic linkage:
    - multiple blocking passes
    - similarity features
    - score weighted mean
    """
    c = census_df.set_index("rid")
    r = registry_df.set_index("rid")

    indexer = recordlinkage.Index()

    # Multiple blocking passes (union):
    # pass 1: soundex last + sex
    if "sx_last" in c.columns and "sx_last" in r.columns:
        indexer.block(left_on="sx_last", right_on="sx_last")
    if "sex" in c.columns and "sex" in r.columns:
        indexer.block(left_on="sex", right_on="sex")

    pairs1 = indexer.index(c, r)

    # pass 2: metaphone last (helps when soundex fails)
    indexer2 = recordlinkage.Index()
    if "mp_last" in c.columns and "mp_last" in r.columns:
        indexer2.block(left_on="mp_last", right_on="mp_last")
    pairs2 = indexer2.index(c, r)

    # pass 3: village + sex (helps same area)
    indexer3 = recordlinkage.Index()
    if "village" in c.columns and "village" in r.columns:
        indexer3.block(left_on="village", right_on="village")
    if "sex" in c.columns and "sex" in r.columns:
        indexer3.block(left_on="sex", right_on="sex")
    pairs3 = indexer3.index(c, r)

    # union
    candidate_links = pairs1.union(pairs2).union(pairs3)
    print(f"\n[{label}] Candidate pairs after multi-block union: {len(candidate_links):,}")

    compare = recordlinkage.Compare()
    compare.string("first_name", "first_name", method="jarowinkler", label="fn_sim")
    compare.string("last_name",  "last_name",  method="jarowinkler", label="ln_sim")
    compare.string("village",    "village",    method="jarowinkler", label="vill_sim")
    compare.exact("sex", "sex", label="sex_exact")

    features = compare.compute(candidate_links, c, r)

    # age similarity
    left_age  = c.loc[features.index.get_level_values(0), "age"].values if "age" in c.columns else np.full(len(features), np.nan)
    right_age = r.loc[features.index.get_level_values(1), "age"].values if "age" in r.columns else np.full(len(features), np.nan)
    age_diff = np.abs(left_age - right_age)

    age_sim = np.where(np.isnan(age_diff), 0,
               np.where(age_diff <= 1, 1.0,
               np.where(age_diff <= 2, 0.5, 0.0)))
    features["age_sim"] = age_sim

    weights = {"fn_sim": 0.30, "ln_sim": 0.35, "sex_exact": 0.10, "vill_sim": 0.15, "age_sim": 0.10}
    for k in weights:
        if k not in features.columns:
            features[k] = 0.0

    features["score"] = sum(features[k] * w for k, w in weights.items()) / sum(weights.values())

    matched = features[features["score"] >= threshold].reset_index()
    matched = matched.rename(columns={"level_0": "rid_c", "level_1": "rid_r"})

    matched_1to1 = one_to_one_greedy(matched, score_col="score")

    thresh_tbl = pd.DataFrame([{
        "Threshold": threshold,
        "Candidates": int(len(features)),
        "Above threshold": int(len(matched)),
        "1-to-1 pairs": int(len(matched_1to1))
    }])

    return matched_1to1, thresh_tbl, summarize_matches(matched_1to1, label), features

# ---------------------------
# 7) HYBRID
# ---------------------------
def run_hybrid(census_df, registry_df, det_pairs, prob_threshold, label):
    if det_pairs is None:
        det_pairs = pd.DataFrame(columns=["rid_c", "rid_r"])

    used_c = set(det_pairs["rid_c"]) if not det_pairs.empty else set()
    used_r = set(det_pairs["rid_r"]) if not det_pairs.empty else set()

    c_rem = census_df[~census_df["rid"].isin(used_c)].copy()
    r_rem = registry_df[~registry_df["rid"].isin(used_r)].copy()

    prob_pairs, _, _, _ = run_probabilistic(c_rem, r_rem, f"{label}_PROB_REMAINING", threshold=prob_threshold)

    det_out = det_pairs.copy()
    det_out["method"] = "deterministic"
    if "score" not in det_out.columns:
        det_out["score"] = 1.0

    prob_pairs = prob_pairs.copy()
    prob_pairs["method"] = "probabilistic"

    combined = pd.concat([
        det_out[["rid_c", "rid_r", "score", "method"]],
        prob_pairs[["rid_c", "rid_r", "score", "method"]],
    ], ignore_index=True)

    return one_to_one_greedy(combined, score_col="score")

# ---------------------------
# 8) SAMPLING
# ---------------------------
def proportion_sample_size(N, p, d, z=1.96):
    n0 = (z**2) * p * (1 - p) / (d**2)
    n = n0 / (1 + (n0 - 1) / N)
    return int(np.ceil(n))

def sample_df(df, n, seed=42):
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).copy()

# ==========================================================
# MAIN EXECUTION
# ==========================================================
def main():
    census_raw   = pd.read_excel(CENSUS_PATH)
    registry_raw = pd.read_excel(REGISTRY_PATH)

    print(f"Loaded rows: census={len(census_raw)}  registry={len(registry_raw)}")

    # --- Detect columns robustly
    c_name, c_sex, c_vill, c_age = detect_columns(census_raw, label="CENSUS")
    r_name, r_sex, r_vill, r_age = detect_columns(registry_raw, label="REGISTRY")

    # Safety check
    if c_name is None or r_name is None:
        raise ValueError("Could not detect NAME columns. You can manually set them in code.")
    if c_name == c_vill:
        print("\nWARNING: Census name == village after detection. Please manually set correct columns.\n")

    # --- Prepare
    census   = prepare_df(census_raw,   c_name, c_sex, c_vill, c_age, id_prefix="C")
    registry = prepare_df(registry_raw, r_name, r_sex, r_vill, r_age, id_prefix="R")

    # =========================
    # PHASE 1
    # =========================
    det_pairs_full, det_rule_table, det_summary = run_deterministic(census, registry, "PHASE1_FULL_DETERMINISTIC")
    pretty_print_df(det_rule_table, "PHASE 1: Deterministic rule-level match counts")
    pretty_print_df(det_summary, "PHASE 1: Deterministic summary (1-to-1 enforced)")
    det_pairs_full.to_csv(os.path.join(OUT_DIR, "phase1_full_deterministic_pairs.csv"), index=False)

    prob_pairs_full, prob_thresh_table, prob_summary, _ = run_probabilistic(
        census, registry, label="PHASE1_FULL_PROBABILISTIC", threshold=0.85
    )
    pretty_print_df(prob_thresh_table, "PHASE 1: Probabilistic linkage threshold table")
    pretty_print_df(prob_summary, "PHASE 1: Probabilistic summary (1-to-1 enforced)")
    prob_pairs_full.to_csv(os.path.join(OUT_DIR, "phase1_full_probabilistic_pairs.csv"), index=False)

    # =========================
    # PHASE 2
    # =========================
    print("\n========================\nSTARTING PHASE II\n========================")

    # Sample size table
    sample_rows = []
    for p in P_VALUES:
        for d in D_VALUES:
            sample_rows.append({
                "p": p, "d": d, "z": Z_95,
                "n_census": proportion_sample_size(N_CENSUS, p, d, z=Z_95),
                "n_registry": proportion_sample_size(N_REGISTRY, p, d, z=Z_95),
            })
    sample_sizes_df = pd.DataFrame(sample_rows).sort_values(["p","d"], ascending=[False, False])
    pretty_print_df(sample_sizes_df, "PHASE 2: Sample sizes for scenarios (with finite population correction)")
    sample_sizes_df.to_csv(os.path.join(OUT_DIR, "phase2_sample_sizes.csv"), index=False)

    phase2_results = []

    for p in P_VALUES:
        for d in D_VALUES:
            n_c = proportion_sample_size(N_CENSUS, p, d, z=Z_95)
            n_r = proportion_sample_size(N_REGISTRY, p, d, z=Z_95)

            c_s = sample_df(census,   n_c, seed=RANDOM_SEED)
            r_s = sample_df(registry, n_r, seed=RANDOM_SEED)

            det_pairs, det_rules, det_sum = run_deterministic(c_s, r_s, label=f"PHASE2_DET_p{p}_d{d}")
            det_pairs["scenario_p"] = p
            det_pairs["scenario_d"] = d
            det_pairs["method"] = "deterministic"
            det_pairs.to_csv(os.path.join(OUT_DIR, f"phase2_pairs_det_p{p}_d{d}.csv"), index=False)
            det_rules.to_csv(os.path.join(OUT_DIR, f"phase2_det_rules_p{p}_d{d}.csv"), index=False)

            prob_pairs, prob_tbl, prob_sum, _ = run_probabilistic(c_s, r_s, label=f"PHASE2_PROB_p{p}_d{d}", threshold=0.85)
            prob_pairs["scenario_p"] = p
            prob_pairs["scenario_d"] = d
            prob_pairs["method"] = "probabilistic"
            prob_pairs.to_csv(os.path.join(OUT_DIR, f"phase2_pairs_prob_p{p}_d{d}.csv"), index=False)

            hybrid_pairs = run_hybrid(c_s, r_s, det_pairs, prob_threshold=0.85, label=f"PHASE2_HYBRID_p{p}_d{d}")
            hybrid_pairs["scenario_p"] = p
            hybrid_pairs["scenario_d"] = d
            hybrid_pairs["method"] = "hybrid"
            hybrid_pairs.to_csv(os.path.join(OUT_DIR, f"phase2_pairs_hybrid_p{p}_d{d}.csv"), index=False)

            phase2_results.append(summarize_matches(det_pairs,   f"PHASE2_DET p={p}, d={d}").iloc[0].to_dict())
            phase2_results.append(summarize_matches(prob_pairs,  f"PHASE2_PROB p={p}, d={d}").iloc[0].to_dict())
            phase2_results.append(summarize_matches(hybrid_pairs,f"PHASE2_HYBRID p={p}, d={d}").iloc[0].to_dict())

    phase2_summary_df = pd.DataFrame(phase2_results)
    pretty_print_df(phase2_summary_df, "PHASE 2: Summary across scenarios and linkage methods")
    phase2_summary_df.to_csv(os.path.join(OUT_DIR, "phase2_summary_all_methods.csv"), index=False)

    print(f"\nAll outputs written to: {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
