# ==========================================================
# DETERMINISTIC DATA LINKAGE ONLY
# 4-variable deterministic linkage using:
#   1) sex
#   2) dob
#   3) facility_code
#   4) enrollment_date
#
# INPUT FILES
#   - merge_between_baseline_endline_deid.csv
#   - BL_SMSTrialSociodem_NRB_deid.csv
#
# OUTPUTS
#   - deterministic_pairs_4vars.csv
#   - deterministic_rule_counts_4vars.csv
#   - deterministic_summary_4vars.csv
#   - deterministic_linked_dataset_4vars.csv
#   - deterministic_left_unmatched_4vars.csv
#   - deterministic_right_unmatched_4vars.csv
# ==========================================================

import os
import re
import numpy as np
import pandas as pd

# ---------------------------
# 0) CONFIG
# ---------------------------
LEFT_PATH = r"C:\Users\guest441\Desktop\Linkage\data\merge_between_baseline_endline_deid.csv"
RIGHT_PATH = r"C:\Users\guest441\Desktop\Linkage\data\BL_SMSTrialSociodem_NRB_deid.csv"

OUT_DIR = r"C:\Users\guest441\Desktop\Linkage\data\linkage_outputs_deterministic_4vars"
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------
# 1) HELPERS
# ---------------------------
def clean_string(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x

def clean_code(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\.0$", "", x)
    x = re.sub(r"\s+", "", x)
    return x

def normalize_sex(x):
    x = clean_string(x)
    if x in ["m", "male", "1"]:
        return "m"
    if x in ["f", "female", "2"]:
        return "f"
    return "u" if x == "" else x

def parse_date(x):
    if pd.isna(x) or str(x).strip() == "":
        return pd.NaT
    return pd.to_datetime(str(x), errors="coerce", dayfirst=True)

def detect_column(df, patterns, required=True):
    for p in patterns:
        p = p.lower()

        # exact match first
        for c in df.columns:
            if c.lower() == p:
                return c

        # then contains
        for c in df.columns:
            if p in c.lower():
                return c

    if required:
        raise ValueError(f"Could not detect column for patterns: {patterns}")
    return None

def one_to_one_greedy(df):
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.sort_values(["det_rule", "rid_l", "rid_r"]).copy()
    used_l = set()
    used_r = set()
    keep = []

    for _, row in out.iterrows():
        l = row["rid_l"]
        r = row["rid_r"]
        if l in used_l or r in used_r:
            continue
        keep.append(row)
        used_l.add(l)
        used_r.add(r)

    return pd.DataFrame(keep)

def summarize_matches(df, run_label, left_n, right_n):
    if df is None or df.empty:
        pairs = 0
        ul = 0
        ur = 0
    else:
        pairs = int(len(df))
        ul = int(df["rid_l"].nunique())
        ur = int(df["rid_r"].nunique())

    return pd.DataFrame([{
        "Run": run_label,
        "Pairs": pairs,
        "Matched Left": ul,
        "Matched Right": ur,
        "Unmatched Left": int(left_n - ul),
        "Unmatched Right": int(right_n - ur),
        "Left Match Rate (%)": round((ul / left_n) * 100, 2) if left_n else 0,
        "Right Match Rate (%)": round((ur / right_n) * 100, 2) if right_n else 0
    }])


# ---------------------------
# 2) COLUMN DETECTION
# ---------------------------
def detect_left_columns(df):
    return {
        "facility_code": detect_column(df, ["facility_code_baseline"]),
        "enrollment_date": detect_column(df, ["enrollment_date_baseline"]),
        "sex": detect_column(df, ["sex_baseline"]),
        "dob": detect_column(df, ["dob_baseline"]),
    }

def detect_right_columns(df):
    return {
        "facility_code": detect_column(df, ["facility_code"]),
        "enrollment_date": detect_column(df, ["enrollment_date"]),
        "sex": detect_column(df, ["sex"]),
        "dob": detect_column(df, ["dob"]),
    }


# ---------------------------
# 3) PREPARE DATA
# ---------------------------
def prepare_df(df, cols, prefix):
    out = df.copy()

    out["facility_code_std"] = out[cols["facility_code"]].apply(clean_code)
    out["sex_std"] = out[cols["sex"]].apply(normalize_sex)

    out["dob_dt"] = out[cols["dob"]].apply(parse_date)
    out["enrollment_dt"] = out[cols["enrollment_date"]].apply(parse_date)

    out["dob_std"] = out["dob_dt"].dt.strftime("%Y-%m-%d").fillna("")
    out["enrollment_std"] = out["enrollment_dt"].dt.strftime("%Y-%m-%d").fillna("")

    out["dob_year"] = out["dob_dt"].dt.year.fillna(-999).astype(int).astype(str)
    out["enroll_year"] = out["enrollment_dt"].dt.year.fillna(-999).astype(int).astype(str)

    out["rid"] = [f"{prefix}{i}" for i in range(len(out))]
    return out


# ---------------------------
# 4) DETERMINISTIC LINKAGE
# ---------------------------
def run_deterministic(left_df, right_df, label):
    """
    Deterministic hierarchy:
      Rule 1: exact sex + dob + facility + enrollment
      Rule 2: exact sex + dob + facility
      Rule 3: exact sex + dob + enrollment
      Rule 4: exact sex + facility + enrollment
      Rule 5: exact dob + facility + enrollment
      Rule 6: exact sex + dob
      Rule 7: exact facility + enrollment + dob_year
    """
    rules = [
        (1, "EXACT_SEX_DOB_FACILITY_ENROLL", ["sex_std", "dob_std", "facility_code_std", "enrollment_std"]),
        (2, "EXACT_SEX_DOB_FACILITY",        ["sex_std", "dob_std", "facility_code_std"]),
        (3, "EXACT_SEX_DOB_ENROLL",          ["sex_std", "dob_std", "enrollment_std"]),
        (4, "EXACT_SEX_FACILITY_ENROLL",     ["sex_std", "facility_code_std", "enrollment_std"]),
        (5, "EXACT_DOB_FACILITY_ENROLL",     ["dob_std", "facility_code_std", "enrollment_std"]),
        (6, "EXACT_SEX_DOB",                 ["sex_std", "dob_std"]),
        (7, "EXACT_FACILITY_ENROLL_DOBYEAR", ["facility_code_std", "enroll_year", "dob_year"]),
    ]

    all_pairs = []
    rule_stats = []

    for rid, rname, cols in rules:
        merged = left_df.merge(
            right_df,
            on=cols,
            how="inner",
            suffixes=("_l", "_r")
        )

        rule_stats.append({
            "Rule": rid,
            "RuleName": rname,
            "Columns": " + ".join(cols),
            "Pairs": int(len(merged))
        })

        if merged.empty:
            continue

        desired = [
            "rid_l", "rid_r",
            "sex_std", "dob_std", "facility_code_std", "enrollment_std",
            "dob_year", "enroll_year"
        ]
        keep_cols = [c for c in desired if c in merged.columns]

        out = merged[keep_cols].copy()
        out["det_rule"] = rid
        out["det_rule_name"] = rname
        all_pairs.append(out)

    pairs = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(columns=["rid_l", "rid_r", "det_rule"])

    if not pairs.empty:
        pairs = pairs.sort_values("det_rule").drop_duplicates(subset=["rid_l", "rid_r"], keep="first")

    pairs_1to1 = one_to_one_greedy(pairs)

    rule_tbl = pd.DataFrame(rule_stats).sort_values("Rule")
    summary_tbl = summarize_matches(pairs_1to1, label, len(left_df), len(right_df))

    return pairs_1to1, rule_tbl, summary_tbl


# ---------------------------
# 5) MAIN
# ---------------------------
def main():
    left_raw = pd.read_csv(LEFT_PATH)
    right_raw = pd.read_csv(RIGHT_PATH)

    print(f"Loaded rows: left={len(left_raw)} right={len(right_raw)}")

    left_cols = detect_left_columns(left_raw)
    right_cols = detect_right_columns(right_raw)

    print("\nSelected linkage columns:")
    print("LEFT :", left_cols)
    print("RIGHT:", right_cols)

    left = prepare_df(left_raw, left_cols, "L")
    right = prepare_df(right_raw, right_cols, "R")

    print("\n[DIAGNOSTICS] Missing rates (left):")
    print("sex_u =", round((left["sex_std"] == "u").mean(), 4))
    print("dob_blank =", round((left["dob_std"] == "").mean(), 4))
    print("facility_blank =", round((left["facility_code_std"] == "").mean(), 4))
    print("enroll_blank =", round((left["enrollment_std"] == "").mean(), 4))

    print("\n[DIAGNOSTICS] Missing rates (right):")
    print("sex_u =", round((right["sex_std"] == "u").mean(), 4))
    print("dob_blank =", round((right["dob_std"] == "").mean(), 4))
    print("facility_blank =", round((right["facility_code_std"] == "").mean(), 4))
    print("enroll_blank =", round((right["enrollment_std"] == "").mean(), 4))

    print("\n========================")
    print("DETERMINISTIC LINKAGE")
    print("========================")

    det_pairs, det_rules, det_sum = run_deterministic(left, right, "DETERMINISTIC_4VARS")

    print("\nRule-level counts:")
    print(det_rules.to_string(index=False))

    print("\nDeterministic summary:")
    print(det_sum.to_string(index=False))

    # save pair-level results
    det_pairs.to_csv(os.path.join(OUT_DIR, "deterministic_pairs_4vars.csv"), index=False)
    det_rules.to_csv(os.path.join(OUT_DIR, "deterministic_rule_counts_4vars.csv"), index=False)
    det_sum.to_csv(os.path.join(OUT_DIR, "deterministic_summary_4vars.csv"), index=False)

    # save full linked dataset
    if not det_pairs.empty:
        left_with_rid = left_raw.copy()
        left_with_rid["rid_l"] = left["rid"]

        right_with_rid = right_raw.copy()
        right_with_rid["rid_r"] = right["rid"]

        linked_full = (
            det_pairs
            .merge(left_with_rid, on="rid_l", how="left")
            .merge(right_with_rid, on="rid_r", how="left", suffixes=("_leftfile", "_rightfile"))
        )

        linked_full.to_csv(
            os.path.join(OUT_DIR, "deterministic_linked_dataset_4vars.csv"),
            index=False
        )
    else:
        pd.DataFrame().to_csv(
            os.path.join(OUT_DIR, "deterministic_linked_dataset_4vars.csv"),
            index=False
        )

    # unmatched
    matched_left = set(det_pairs["rid_l"]) if not det_pairs.empty else set()
    matched_right = set(det_pairs["rid_r"]) if not det_pairs.empty else set()

    left_unmatched = left_raw.copy()
    left_unmatched["rid_l"] = left["rid"]
    left_unmatched = left_unmatched[~left_unmatched["rid_l"].isin(matched_left)]
    left_unmatched.to_csv(
        os.path.join(OUT_DIR, "deterministic_left_unmatched_4vars.csv"),
        index=False
    )

    right_unmatched = right_raw.copy()
    right_unmatched["rid_r"] = right["rid"]
    right_unmatched = right_unmatched[~right_unmatched["rid_r"].isin(matched_right)]
    right_unmatched.to_csv(
        os.path.join(OUT_DIR, "deterministic_right_unmatched_4vars.csv"),
        index=False
    )

    print(f"\nAll deterministic outputs written to: {OUT_DIR}")
    print("Deterministic pipeline complete.")


if __name__ == "__main__":
    main()