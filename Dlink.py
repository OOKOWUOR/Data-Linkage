# ==========================================================
# DETERMINISTIC DATA LINKAGE ONLY (RIGOROUS + NO PROBABILISTIC)
# Phase 1: Full datasets deterministic linkage
# After Phase 1: 3x3 sampling grids (rows=d, cols=p) for both datasets
# Phase 2: For each (d,p), draw samples and run deterministic linkage
# Outputs: Pretty tables + CSV outputs for report
# ==========================================================

import os, re
import numpy as np
import pandas as pd
import jellyfish

# ---------------------------
# 0) CONFIG
# ---------------------------
CENSUS_PATH   = r"C:/Users/guest441/Downloads/D1_Census.xlsx"
REGISTRY_PATH = r"C:/Users/guest441/Downloads/D2_Registry.xlsx"

OUT_DIR = r"C:/Users/guest441/Downloads/linkage_outputs_deterministic"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42

N_CENSUS   = 21766
N_REGISTRY = 75125

P_VALUES = [0.5, 0.4, 0.3]         # columns
D_VALUES = [0.10, 0.05, 0.01]      # rows
Z_95 = 1.96


# ---------------------------
# 1) PRETTY PRINT
# ---------------------------
def pretty_print_df(df, title, max_rows=30):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    if df is None or df.empty:
        print("(empty)")
        return
    view = df.head(max_rows)
    try:
        from tabulate import tabulate
        print(tabulate(view, headers="keys", tablefmt="github", showindex=False))
        if len(df) > max_rows:
            print(f"... ({len(df) - max_rows} more rows not shown)")
    except Exception:
        print(view.to_string(index=False))
        if len(df) > max_rows:
            print(f"... ({len(df) - max_rows} more rows not shown)")

def pretty_print_grid(grid: pd.DataFrame, title: str):
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    try:
        from tabulate import tabulate
        print(tabulate(grid.reset_index(), headers="keys", tablefmt="github", showindex=False))
    except Exception:
        print(grid.to_string())


# ---------------------------
# 2) CLEANING + FEATURES
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
# 3) SMART COLUMN DETECTION
# ---------------------------
NAME_HINTS = ["name", "hhmember", "member", "person", "patient", "client", "full", "hhead"]
VILL_HINTS = ["village", "vill", "vname", "location", "residence", "ward", "subcounty"]
SEX_HINTS  = ["sex", "gender"]
AGE_HINTS  = ["age", "dob", "years", "yr", "birth", "date"]

def profile_col(series: pd.Series):
    s = series.dropna()
    if len(s) == 0:
        return dict(uniq=0, space_rate=0.0, numeric_rate=0.0, mf_rate=0.0)
    s_str = s.astype(str)
    uniq = s_str.nunique()
    space_rate = s_str.str.contains(r"\s").mean()
    numeric_rate = pd.to_numeric(s, errors="coerce").notna().mean()
    mf_rate = s_str.map(clean_string).isin(["m","f","male","female","1","2","man","woman"]).mean()
    return dict(uniq=int(uniq), space_rate=float(space_rate), numeric_rate=float(numeric_rate), mf_rate=float(mf_rate))

def header_score(colname: str, hints):
    c = colname.lower()
    return sum(3 for h in hints if h in c)

def detect_columns(df: pd.DataFrame, label="DATASET"):
    cols = list(df.columns)
    prof = {c: profile_col(df[c]) for c in cols}

    def score_name(c):
        p = prof[c]
        s = header_score(c, NAME_HINTS)
        s += min(5, (p["uniq"] / max(1, len(df))) * 60)
        s += 3 * p["space_rate"]
        s += 2 * (1 - p["numeric_rate"])
        return s

    def score_vill(c):
        p = prof[c]
        s = header_score(c, VILL_HINTS)
        s += 2 * (1 - p["numeric_rate"])
        uniq = p["uniq"]
        if 5 <= uniq <= max(50, int(len(df) * 0.05)):
            s += 3
        s += (1 - p["space_rate"])
        return s

    def score_sex(c):
        p = prof[c]
        s = header_score(c, SEX_HINTS)
        if p["uniq"] <= 10:
            s += 3
        s += 5 * p["mf_rate"]
        s += (1 - p["numeric_rate"])
        return s

    def score_age(c):
        p = prof[c]
        s = header_score(c, AGE_HINTS)
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

    # prefer exact 'age'
    for c in cols:
        if c.lower() == "age":
            age_col = c

    # ensure name != village
    if name_col == vill_col and len(name_rank) > 1:
        for c in name_rank[1:]:
            if c != vill_col:
                name_col = c
                break

    # print detection table (top 5 each)
    rows = []
    def add(field, rank, scorer):
        for c in rank[:5]:
            rows.append({
                "field": field, "candidate": c,
                "score": round(float(scorer(c)), 3),
                "uniq": prof[c]["uniq"],
                "space_rate": round(prof[c]["space_rate"], 3),
                "numeric_rate": round(prof[c]["numeric_rate"], 3),
                "mf_rate": round(prof[c]["mf_rate"], 3),
            })
    add("NAME", name_rank, score_name)
    add("VILLAGE", vill_rank, score_vill)
    add("SEX", sex_rank, score_sex)
    add("AGE", age_rank, score_age)

    pretty_print_df(pd.DataFrame(rows), f"{label} detection table")
    print(f"\n[{label}] Selected columns: name={name_col}, sex={sex_col}, village={vill_col}, age={age_col}")
    return name_col, sex_col, vill_col, age_col


# ---------------------------
# 4) PREPARE
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
# 5) 1-to-1 + Summary
# ---------------------------
def one_to_one_greedy(df):
    if df is None or df.empty:
        return df
    out = df.sort_values(["det_rule", "rid_c", "rid_r"]).copy()
    used_c, used_r = set(), set()
    keep = []
    for _, row in out.iterrows():
        c = row["rid_c"]; r = row["rid_r"]
        if c in used_c or r in used_r:
            continue
        keep.append(row)
        used_c.add(c); used_r.add(r)
    return pd.DataFrame(keep)

def summarize_matches(df, run_label, census_n, registry_n):
    if df is None or df.empty:
        pairs = 0; uc = 0; ur = 0
    else:
        pairs = int(len(df))
        uc = int(df["rid_c"].nunique())
        ur = int(df["rid_r"].nunique())
    return pd.DataFrame([{
        "Run": run_label,
        "Pairs": pairs,
        "Matched Census": uc,
        "Matched Registry": ur,
        "Unmatched Census": int(census_n - uc),
        "Unmatched Registry": int(registry_n - ur),
    }])


# ---------------------------
# 6) DETERMINISTIC LINKAGE
# ---------------------------
def run_deterministic(census_df, registry_df, label):
    rules = [
        (1, "EXACT_FN_LN_SEX_VILL", ["first_name","last_name","sex","village"]),
        (2, "EXACT_FN_LN_SEX",      ["first_name","last_name","sex"]),
        (3, "EXACT_FN_LN",          ["first_name","last_name"]),
        (4, "SOUNDEX_LN_SEX_FN",    ["sx_last","sex","first_name"]),
        (5, "SOUNDEX_FN_LN",        ["sx_first","sx_last"]),
        (6, "METAPHONE_FN_LN",      ["mp_first","mp_last"]),
        (7, "EXACT_FULLNAME",       ["person_name"]),
    ]

    all_pairs = []
    rule_stats = []

    for rid, rname, cols in rules:
        merged = census_df.merge(
            registry_df,
            on=cols,               # IMPORTANT: join on shared column names
            how="inner",
            suffixes=("_c","_r")   # suffixes won't apply to join keys, only other overlapping cols
        )

        rule_stats.append({"Rule": rid, "RuleName": rname, "Columns": " + ".join(cols), "Pairs": int(len(merged))})
        if merged.empty:
            continue

        # Only person_name exists with suffixes (because it isn't always a join key).
        # Join keys (sex, village, etc.) appear ONCE in merged -> use without suffix.
        desired = [
            "rid_c", "rid_r",
            "person_name_c", "person_name_r",
            "first_name", "last_name",
            "sex", "village",
            "age_c", "age_r",
        ]
        keep_cols = [c for c in desired if c in merged.columns]
        out = merged[keep_cols].copy()

        out["det_rule"] = rid
        out["det_rule_name"] = rname
        all_pairs.append(out)

    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(columns=["rid_c","rid_r","det_rule"])
    if not pairs.empty:
        pairs = pairs.sort_values("det_rule").drop_duplicates(subset=["rid_c","rid_r"], keep="first")

    pairs_1to1 = one_to_one_greedy(pairs)

    rule_tbl = pd.DataFrame(rule_stats).sort_values("Rule")
    summary_tbl = summarize_matches(pairs_1to1, label, len(census_df), len(registry_df))

    return pairs_1to1, rule_tbl, summary_tbl


# ---------------------------
# 7) SAMPLING (3x3 grids)
# ---------------------------
def proportion_sample_size(N, p, d, z=1.96):
    n0 = (z**2) * p * (1 - p) / (d**2)
    n  = n0 / (1 + (n0 - 1) / N)
    return int(np.ceil(n))

def make_sample_grid(N, label):
    grid = pd.DataFrame(index=D_VALUES, columns=P_VALUES, dtype=int)
    for d in D_VALUES:
        for p in P_VALUES:
            grid.loc[d, p] = proportion_sample_size(N, p, d, z=Z_95)
    grid.index.name = "d"
    grid.columns.name = "p"
    pretty_print_grid(grid, f"SAMPLING GRID ({label}) — rows=d, cols=p")
    return grid

def sample_df(df, n, seed):
    n = min(int(n), len(df))
    return df.sample(n=n, random_state=seed).copy()


# ==========================================================
# MAIN
# ==========================================================
def main():
    census_raw = pd.read_excel(CENSUS_PATH)
    registry_raw = pd.read_excel(REGISTRY_PATH)
    print(f"Loaded rows: census={len(census_raw)}  registry={len(registry_raw)}")

    c_name, c_sex, c_vill, c_age = detect_columns(census_raw, "CENSUS")
    r_name, r_sex, r_vill, r_age = detect_columns(registry_raw, "REGISTRY")

    census = prepare_df(census_raw, c_name, c_sex, c_vill, c_age, "C")
    registry = prepare_df(registry_raw, r_name, r_sex, r_vill, r_age, "R")

    # -------------------------
    # PHASE 1: FULL DETERMINISTIC
    # -------------------------
    print("\n========================")
    print("PHASE 1 (FULL DATASETS) — DETERMINISTIC ONLY")
    print("========================")

    det_pairs_full, det_rules_full, det_sum_full = run_deterministic(census, registry, "PHASE1_FULL_DETERMINISTIC")
    pretty_print_df(det_rules_full, "PHASE 1: Deterministic rule-level match counts")
    pretty_print_df(det_sum_full, "PHASE 1: Deterministic summary (1-to-1 enforced)")

    det_pairs_full.to_csv(os.path.join(OUT_DIR, "phase1_full_deterministic_pairs.csv"), index=False)
    det_rules_full.to_csv(os.path.join(OUT_DIR, "phase1_full_deterministic_rules.csv"), index=False)
    det_sum_full.to_csv(os.path.join(OUT_DIR, "phase1_full_deterministic_summary.csv"), index=False)

    # -------------------------
    # SAMPLING TABLES
    # -------------------------
    print("\n==============================================")
    print("SAMPLING DESIGN TABLES (AFTER PHASE 1)")
    print("==============================================")

    census_grid = make_sample_grid(N_CENSUS, "CENSUS")
    registry_grid = make_sample_grid(N_REGISTRY, "REGISTRY")

    census_grid.to_csv(os.path.join(OUT_DIR, "phase2_sampling_grid_census.csv"))
    registry_grid.to_csv(os.path.join(OUT_DIR, "phase2_sampling_grid_registry.csv"))

    # Long-format sampling design (nice for report)
    sampling_rows = []
    for d in D_VALUES:
        for p in P_VALUES:
            sampling_rows.append({
                "d": d, "p": p,
                "Census_sample_n": int(census_grid.loc[d, p]),
                "Registry_sample_n": int(registry_grid.loc[d, p]),
                "Census_fraction": round(int(census_grid.loc[d, p]) / N_CENSUS, 4),
                "Registry_fraction": round(int(registry_grid.loc[d, p]) / N_REGISTRY, 4),
                "seed": RANDOM_SEED
            })
    sampling_long = pd.DataFrame(sampling_rows).sort_values(["d","p"], ascending=[False, False])
    pretty_print_df(sampling_long, "Sampling design (long format): n for each (d,p)")
    sampling_long.to_csv(os.path.join(OUT_DIR, "phase2_sampling_design_long.csv"), index=False)

    # -------------------------
    # PHASE 2: DETERMINISTIC ON SAMPLES
    # -------------------------
    print("\n========================")
    print("PHASE 2 (SAMPLES) — DETERMINISTIC ONLY")
    print("========================")

    phase2_summary_rows = []

    for d in D_VALUES:
        for p in P_VALUES:
            n_c = int(census_grid.loc[d, p])
            n_r = int(registry_grid.loc[d, p])

            print(f"\n[PHASE 2] Scenario d={d}, p={p} -> Census n={n_c}, Registry n={n_r}")

            census_s = sample_df(census, n_c, RANDOM_SEED)
            registry_s = sample_df(registry, n_r, RANDOM_SEED)

            det_pairs_s, det_rules_s, det_sum_s = run_deterministic(census_s, registry_s, f"PHASE2_DET_p{p}_d{d}")

            det_pairs_s.to_csv(os.path.join(OUT_DIR, f"phase2_pairs_det_p{p}_d{d}.csv"), index=False)
            det_rules_s.to_csv(os.path.join(OUT_DIR, f"phase2_det_rules_p{p}_d{d}.csv"), index=False)
            det_sum_s.to_csv(os.path.join(OUT_DIR, f"phase2_det_summary_p{p}_d{d}.csv"), index=False)

            row = det_sum_s.iloc[0].to_dict()
            row.update({"p": p, "d": d, "Census_n": n_c, "Registry_n": n_r})
            phase2_summary_rows.append(row)

    phase2_summary = pd.DataFrame(phase2_summary_rows)
    pretty_print_df(phase2_summary, "PHASE 2: Deterministic summary across all (d,p) scenarios")
    phase2_summary.to_csv(os.path.join(OUT_DIR, "phase2_deterministic_summary_all_scenarios.csv"), index=False)

    # Also show as a 3x3 grid of matched pairs
    grid_pairs = pd.DataFrame(index=D_VALUES, columns=P_VALUES, dtype=int)
    for r in phase2_summary_rows:
        grid_pairs.loc[r["d"], r["p"]] = r["Pairs"]
    grid_pairs.index.name = "d"
    grid_pairs.columns.name = "p"
    pretty_print_grid(grid_pairs, "PHASE 2: Deterministic matched PAIRS (3x3 grid)")
    grid_pairs.to_csv(os.path.join(OUT_DIR, "phase2_deterministic_pairs_grid.csv"))

    print(f"\nAll deterministic outputs written to: {OUT_DIR}")
    print("Deterministic pipeline complete.")


if __name__ == "__main__":

    main()
