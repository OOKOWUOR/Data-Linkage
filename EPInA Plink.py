# ============================================================
# Probabilistic linkage using 4 strong variables only
# Variables used:
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
#   - probabilistic_links_4vars.csv
#   - probabilistic_linked_dataset_4vars.csv
#   - left_unmatched_4vars.csv
#   - right_unmatched_4vars.csv
#   - all_scored_candidates_4vars.csv
#   - linkage_summary_4vars.csv
# ============================================================

import os
import re
import numpy as np
import pandas as pd

# ===================== CONFIG =====================
LEFT_PATH  = r"C:\Users\guest441\Desktop\Linkage\data\merge_between_baseline_endline_deid.csv"
RIGHT_PATH = r"C:\Users\guest441\Desktop\Linkage\data\BL_SMSTrialSociodem_NRB_deid.csv"

OUT_DIR = r"C:\Users\guest441\Desktop\Linkage\data\linkage_outputs_probabilistic_4vars"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# candidate controls
TOPK_PER_PASS = 60
MAX_PER_LEFT = 100

# calibration
CALIBRATION_SAMPLE = 5000
KEEP_TRUE_Q = 0.10   # threshold = 10th percentile of deterministic-pair scores

# threshold bounds
THR_MIN = 0.70
THR_MAX = 0.95
THR_FALLBACK = 0.82

# relax threshold if 0 links
RELAX_STEPS = [0.00, -0.03, -0.06, -0.10, -0.15]


# ===================== HELPERS =====================
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
    x = re.sub(r"\.0$", "", x)   # remove trailing .0 from numeric-looking codes
    x = re.sub(r"\s+", "", x)    # codes usually shouldn't have spaces
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
    # flexible parsing, dayfirst because many survey exports are dd/mm/yyyy
    return pd.to_datetime(str(x), errors="coerce", dayfirst=True)


def detect_column(df, patterns, required=True):
    cols_lower = {c.lower(): c for c in df.columns}
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


def exact_sim_vec(a, b):
    a = np.array(a, dtype=object)
    b = np.array(b, dtype=object)
    out = np.zeros(len(a), dtype=float)
    both_missing = (a == "") & (b == "")
    one_missing = ((a == "") & (b != "")) | ((a != "") & (b == ""))
    both_present = (a != "") & (b != "")
    out[both_missing] = 0.5
    out[one_missing] = 0.5
    out[both_present] = (a[both_present] == b[both_present]).astype(float)
    return out


def date_sim_vec(a, b):
    out = np.zeros(len(a), dtype=float)
    a = pd.to_datetime(a, errors="coerce")
    b = pd.to_datetime(b, errors="coerce")

    both_missing = a.isna() & b.isna()
    one_missing = (a.isna() & ~b.isna()) | (~a.isna() & b.isna())
    both_present = ~a.isna() & ~b.isna()

    out[both_missing] = 0.5
    out[one_missing] = 0.5

    if both_present.any():
        diff = np.abs((a[both_present] - b[both_present]).days)

        vals = np.zeros(len(diff), dtype=float)
        vals[diff == 0] = 1.0
        vals[diff <= 1] = np.maximum(vals[diff <= 1], 0.9)
        vals[(diff >= 2) & (diff <= 7)] = 0.6
        vals[(diff >= 8) & (diff <= 30)] = 0.2

        out[np.where(both_present)[0]] = vals

    return out


def topk_per_left(df, k):
    if df.empty:
        return df
    return (
        df.sort_values("score", ascending=False)
          .groupby("rid_l", as_index=False)
          .head(k)
    )


def one_to_one(df):
    if df.empty:
        return df
    df = df.sort_values("score", ascending=False).copy()
    used_l = set()
    used_r = set()
    keep = []
    for _, row in df.iterrows():
        l = row["rid_l"]
        r = row["rid_r"]
        if l in used_l or r in used_r:
            continue
        keep.append(row)
        used_l.add(l)
        used_r.add(r)
    return pd.DataFrame(keep)


# ===================== COLUMN DETECTION =====================
def detect_left_columns(df):
    # left file is merge_between_baseline_endline_deid.csv
    # use baseline-side fields only
    return {
        "facility_code": detect_column(df, ["facility_code_baseline"]),
        "enrollment_date": detect_column(df, ["enrollment_date_baseline"]),
        "sex": detect_column(df, ["sex_baseline"]),
        "dob": detect_column(df, ["dob_baseline"]),
    }


def detect_right_columns(df):
    # right file is BL_SMSTrialSociodem_NRB_deid.csv
    return {
        "facility_code": detect_column(df, ["facility_code"]),
        "enrollment_date": detect_column(df, ["enrollment_date"]),
        "sex": detect_column(df, ["sex"]),
        "dob": detect_column(df, ["dob"]),
    }


# ===================== PREPARE =====================
def prepare(df, cols, prefix):
    out = df.copy()

    out["facility_code_std"] = out[cols["facility_code"]].apply(clean_code)
    out["enrollment_date_std"] = out[cols["enrollment_date"]].apply(parse_date)
    out["sex_std"] = out[cols["sex"]].apply(normalize_sex)
    out["dob_std"] = out[cols["dob"]].apply(parse_date)

    out["dob_year"] = out["dob_std"].dt.year
    out["enroll_year"] = out["enrollment_date_std"].dt.year

    out["rid"] = [f"{prefix}{i}" for i in range(len(out))]
    return out


# ===================== SCORING =====================
def score_pairs(m):
    sex = exact_sim_vec(m["sex_std_l"].values, m["sex_std_r"].values)
    dob = date_sim_vec(m["dob_std_l"].values, m["dob_std_r"].values)
    facility = exact_sim_vec(m["facility_code_std_l"].values, m["facility_code_std_r"].values)
    enroll = date_sim_vec(m["enrollment_date_std_l"].values, m["enrollment_date_std_r"].values)

    # weights for the chosen 4 variables
    W = {
        "sex": 0.15,
        "dob": 0.40,
        "facility": 0.20,
        "enroll": 0.25
    }

    # all four are allowed to contribute, with neutral 0.5 if missing
    score = (
        W["sex"] * sex +
        W["dob"] * dob +
        W["facility"] * facility +
        W["enroll"] * enroll
    )

    out = m.copy()
    out["sex_score"] = sex
    out["dob_score"] = dob
    out["facility_score"] = facility
    out["enrollment_score"] = enroll
    out["score"] = score
    return out


# ===================== CANDIDATE GENERATION =====================
def build_candidates(left, right, label):
    l = left.copy()
    r = right.copy()

    # blocking keys: similar spirit to your reference code, but for these 4 variables
    l["k1"] = l["sex_std"].astype(str) + "|" + l["facility_code_std"].astype(str)
    r["k1"] = r["sex_std"].astype(str) + "|" + r["facility_code_std"].astype(str)

    l["k2"] = l["sex_std"].astype(str) + "|" + l["dob_year"].fillna(-999).astype(int).astype(str)
    r["k2"] = r["sex_std"].astype(str) + "|" + r["dob_year"].fillna(-999).astype(int).astype(str)

    l["k3"] = l["facility_code_std"].astype(str) + "|" + l["enroll_year"].fillna(-999).astype(int).astype(str)
    r["k3"] = r["facility_code_std"].astype(str) + "|" + r["enroll_year"].fillna(-999).astype(int).astype(str)

    l["k4"] = l["sex_std"].astype(str) + "|" + l["facility_code_std"].astype(str) + "|" + l["dob_year"].fillna(-999).astype(int).astype(str)
    r["k4"] = r["sex_std"].astype(str) + "|" + r["facility_code_std"].astype(str) + "|" + r["dob_year"].fillna(-999).astype(int).astype(str)

    passes = [
        ("PASS1_SEX_FACILITY", "k1"),
        ("PASS2_SEX_DOBYEAR", "k2"),
        ("PASS3_FACILITY_ENROLLYEAR", "k3"),
        ("PASS4_SEX_FACILITY_DOBYEAR", "k4"),
    ]

    allc = []

    for pname, key in passes:
        print(f"\n[{label}] {pname} merge on {key} ...")

        ll = l[l[key].notna() & (l[key] != "")].copy()
        rr = r[r[key].notna() & (r[key] != "")].copy()

        m = ll.merge(rr, on=key, how="inner", suffixes=("_l", "_r"))
        print(f"[{label}] {pname} raw candidates: {len(m):,}")

        if m.empty:
            continue

        need = [
            "rid_l", "rid_r",
            "sex_std_l", "sex_std_r",
            "dob_std_l", "dob_std_r",
            "facility_code_std_l", "facility_code_std_r",
            "enrollment_date_std_l", "enrollment_date_std_r"
        ]

        m = m[need].copy()
        m = score_pairs(m)
        m = topk_per_left(m, TOPK_PER_PASS)
        m["blocking_pass"] = pname
        allc.append(m)

    if not allc:
        return pd.DataFrame(columns=["rid_l", "rid_r", "score"])

    cand = pd.concat(allc, ignore_index=True)
    cand = cand.sort_values("score", ascending=False).drop_duplicates(["rid_l", "rid_r"], keep="first")
    cand = cand.sort_values("score", ascending=False).groupby("rid_l", as_index=False).head(MAX_PER_LEFT)

    print(f"\n[{label}] After union + cap: {len(cand):,}")
    print(f"[{label}] Candidate score summary:")
    print(cand["score"].describe(percentiles=[0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]).to_string())

    return cand


# ===================== CALIBRATION =====================
def deterministic_true_pairs(left, right, max_pairs=5000, seed=42):
    """
    High-confidence pairs for calibration:
    exact match on the 4 chosen variables after cleaning.
    """
    l = left[["rid", "sex_std", "dob_std", "facility_code_std", "enrollment_date_std"]].copy()
    r = right[["rid", "sex_std", "dob_std", "facility_code_std", "enrollment_date_std"]].copy()

    l = l.rename(columns={"rid": "rid_l"})
    r = r.rename(columns={"rid": "rid_r"})

    m = l.merge(
        r,
        on=["sex_std", "dob_std", "facility_code_std", "enrollment_date_std"],
        how="inner"
    )

    if m.empty:
        return pd.DataFrame(columns=[
            "rid_l", "rid_r",
            "sex_std_l", "sex_std_r",
            "dob_std_l", "dob_std_r",
            "facility_code_std_l", "facility_code_std_r",
            "enrollment_date_std_l", "enrollment_date_std_r"
        ])

    if len(m) > max_pairs:
        m = m.sample(n=max_pairs, random_state=seed)

    out = pd.DataFrame({
        "rid_l": m["rid_l"].values,
        "rid_r": m["rid_r"].values,
        "sex_std_l": m["sex_std"].values,
        "sex_std_r": m["sex_std"].values,
        "dob_std_l": m["dob_std"].values,
        "dob_std_r": m["dob_std"].values,
        "facility_code_std_l": m["facility_code_std"].values,
        "facility_code_std_r": m["facility_code_std"].values,
        "enrollment_date_std_l": m["enrollment_date_std"].values,
        "enrollment_date_std_r": m["enrollment_date_std"].values,
    })

    return out


def calibrate_threshold(left, right):
    tp = deterministic_true_pairs(left, right, max_pairs=CALIBRATION_SAMPLE, seed=RANDOM_SEED)

    if not tp.empty:
        tp_scored = score_pairs(tp)
        thr = float(tp_scored["score"].quantile(KEEP_TRUE_Q))
        thr = max(THR_MIN, min(THR_MAX, thr))

        print("\n[CALIBRATION] Deterministic-pair score percentiles:")
        print(tp_scored["score"].describe(percentiles=[0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99]).to_string())
        print(f"[CALIBRATION] Threshold chosen (q={KEEP_TRUE_Q}): {thr:.3f}")
        return thr

    print("\n[CALIBRATION] No deterministic calibration pairs found. Using candidate-derived threshold.")
    cand = build_candidates(left, right, "CALIBRATION_FALLBACK")
    if cand.empty:
        print(f"[CALIBRATION] No candidates found. Using fallback threshold {THR_FALLBACK:.2f}")
        return THR_FALLBACK

    thr = float(cand["score"].quantile(0.92))
    thr = max(THR_MIN, min(THR_MAX, thr))
    print(f"[CALIBRATION] Candidate-derived threshold (q=0.92): {thr:.3f}")
    return thr


def run_probabilistic(left, right, label, threshold):
    print("\n" + "=" * len(label))
    print(label)
    print("=" * len(label))

    cand = build_candidates(left, right, label)
    if cand.empty:
        return pd.DataFrame(columns=["rid_l", "rid_r", "score"]), cand

    above = cand[cand["score"] >= threshold].copy()
    links = one_to_one(above)
    return links, cand


# ===================== MAIN =====================
def main():
    left_raw = pd.read_csv(LEFT_PATH)
    right_raw = pd.read_csv(RIGHT_PATH)

    print(f"Loaded rows: left={len(left_raw)} right={len(right_raw)}")

    left_cols = detect_left_columns(left_raw)
    right_cols = detect_right_columns(right_raw)

    print("\nSelected linkage columns:")
    print("LEFT :", left_cols)
    print("RIGHT:", right_cols)

    left = prepare(left_raw, left_cols, "L")
    right = prepare(right_raw, right_cols, "R")

    # diagnostics
    print("\n[DIAGNOSTICS] Missing rates (left):")
    print("sex_u =", round((left["sex_std"] == "u").mean(), 4))
    print("dob_na =", round(left["dob_std"].isna().mean(), 4))
    print("facility_blank =", round((left["facility_code_std"] == "").mean(), 4))
    print("enroll_na =", round(left["enrollment_date_std"].isna().mean(), 4))

    print("\n[DIAGNOSTICS] Missing rates (right):")
    print("sex_u =", round((right["sex_std"] == "u").mean(), 4))
    print("dob_na =", round(right["dob_std"].isna().mean(), 4))
    print("facility_blank =", round((right["facility_code_std"] == "").mean(), 4))
    print("enroll_na =", round(right["enrollment_date_std"].isna().mean(), 4))

    # calibration
    threshold0 = calibrate_threshold(left, right)

    used_thr = None
    links = pd.DataFrame()
    cand = None

    for delta in RELAX_STEPS:
        thr = max(THR_MIN, min(THR_MAX, threshold0 + delta))
        label = f"PROBABILISTIC_4VARS_thr{thr:.2f}"
        links, cand = run_probabilistic(left, right, label, thr)
        print(f"\n[LINKING] threshold={thr:.3f} links={len(links):,}")
        if len(links) > 0:
            used_thr = thr
            break

    if used_thr is None:
        used_thr = max(THR_MIN, min(THR_MAX, threshold0 + RELAX_STEPS[-1]))
        print(f"\n[LINKING] Still 0 links after relaxation. Keeping threshold={used_thr:.3f} for reporting.")

    # save scored candidates
    if cand is not None and not cand.empty:
        cand.to_csv(os.path.join(OUT_DIR, "all_scored_candidates_4vars.csv"), index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(OUT_DIR, "all_scored_candidates_4vars.csv"), index=False)

    # save links only
    if not links.empty:
        links.to_csv(os.path.join(OUT_DIR, "probabilistic_links_4vars.csv"), index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(OUT_DIR, "probabilistic_links_4vars.csv"), index=False)

    # full linked dataset
    if not links.empty:
        left_with_rid = left_raw.copy()
        left_with_rid["rid_l"] = left["rid"]

        right_with_rid = right_raw.copy()
        right_with_rid["rid_r"] = right["rid"]

        linked_full = (
            links[["rid_l", "rid_r", "score", "blocking_pass", "sex_score", "dob_score", "facility_score", "enrollment_score"]]
            .merge(left_with_rid, on="rid_l", how="left")
            .merge(right_with_rid, on="rid_r", how="left", suffixes=("_leftfile", "_rightfile"))
        )
        linked_full.to_csv(os.path.join(OUT_DIR, "probabilistic_linked_dataset_4vars.csv"), index=False)
    else:
        pd.DataFrame().to_csv(os.path.join(OUT_DIR, "probabilistic_linked_dataset_4vars.csv"), index=False)

    # unmatched
    matched_left = set(links["rid_l"]) if not links.empty else set()
    matched_right = set(links["rid_r"]) if not links.empty else set()

    left_unmatched = left_raw.copy()
    left_unmatched["rid_l"] = left["rid"]
    left_unmatched = left_unmatched[~left_unmatched["rid_l"].isin(matched_left)]
    left_unmatched.to_csv(os.path.join(OUT_DIR, "left_unmatched_4vars.csv"), index=False)

    right_unmatched = right_raw.copy()
    right_unmatched["rid_r"] = right["rid"]
    right_unmatched = right_unmatched[~right_unmatched["rid_r"].isin(matched_right)]
    right_unmatched.to_csv(os.path.join(OUT_DIR, "right_unmatched_4vars.csv"), index=False)

    # summary
    summary = pd.DataFrame({
        "metric": [
            "left_rows",
            "right_rows",
            "candidate_pairs",
            "final_links",
            "left_match_rate_pct",
            "right_match_rate_pct",
            "threshold_used"
        ],
        "value": [
            len(left_raw),
            len(right_raw),
            0 if cand is None else len(cand),
            len(links),
            round(100 * len(links) / len(left_raw), 2) if len(left_raw) else 0,
            round(100 * len(links) / len(right_raw), 2) if len(right_raw) else 0,
            round(used_thr, 4)
        ]
    })
    summary.to_csv(os.path.join(OUT_DIR, "linkage_summary_4vars.csv"), index=False)

    print("\nDone.")
    print(summary.to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()