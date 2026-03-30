import pandas as pd
import os

os.chdir(r"C:\Users\guest441\Desktop\Linkage\data")

left = pd.read_csv("BL_SMSTrialSociodem_NRB_deidentified.csv")
right = pd.read_csv("Merged_SMSTrial_WHOQOL_NRB_deidentified.csv")

if left["caseid"].duplicated().any():
    raise ValueError("Duplicate caseid found in LEFT dataset")

if right["caseid"].duplicated().any():
    raise ValueError("Duplicate caseid found in RIGHT dataset")

merged = left.merge(
    right,
    on="caseid",
    how="outer",
    indicator="match_status",
    validate="one_to_one"
)

matched = merged[merged["match_status"] == "both"].drop(columns=["match_status"])
left_unmatched = merged[merged["match_status"] == "left_only"].drop(columns=["match_status"])
right_unmatched = merged[merged["match_status"] == "right_only"].drop(columns=["match_status"])

matched.to_csv("matched_caseid.csv", index=False)
left_unmatched.to_csv("unmatched_left_caseid.csv", index=False)
right_unmatched.to_csv("unmatched_right_caseid.csv", index=False)

print("Matching complete")
print("Matched:", len(matched))
print("Left unmatched:", len(left_unmatched))
print("Right unmatched:", len(right_unmatched))

# We then match the baseline with the this latest dataset matched_caseid.csv


import pandas as pd
import os

# =========================================
# 1. SET WORKING DIRECTORY
# =========================================
os.chdir(r"C:\Users\guest441\Desktop\Linkage\data")

# =========================================
# 2. LOAD DATA
# =========================================
baseline = pd.read_csv("BL_SMSTrialSociodem_NRB_deidentified.csv")
latest = pd.read_csv("matched_caseid.csv")

# =========================================
# 3. CHECK THAT caseid EXISTS
# =========================================
if "caseid" not in baseline.columns:
    raise ValueError("caseid column not found in BL_SMSTrialSociodem_NRB_deidentified.csv")

if "caseid" not in latest.columns:
    raise ValueError("caseid column not found in matched_caseid.csv")

# =========================================
# 4. CHECK 1:1 MATCHING ASSUMPTION
# =========================================
if baseline["caseid"].duplicated().any():
    raise ValueError("Duplicate caseid found in baseline dataset")

if latest["caseid"].duplicated().any():
    raise ValueError("Duplicate caseid found in matched_caseid dataset")

# =========================================
# 5. MERGE AND IDENTIFY MATCH STATUS
# =========================================
merged_check = baseline.merge(
    latest,
    on="caseid",
    how="outer",
    indicator="match_status",
    validate="one_to_one",
    suffixes=("_baseline", "_latest")
)

# Keep only successful matches
matched = merged_check[merged_check["match_status"] == "both"].copy()

# Remove match indicator before saving
matched = matched.drop(columns=["match_status"])

# =========================================
# 6. CALCULATE MATCH PERCENTAGES
# =========================================
baseline_total = len(baseline)
latest_total = len(latest)
matched_total = len(matched)

baseline_match_pct = (matched_total / baseline_total) * 100 if baseline_total > 0 else 0
latest_match_pct = (matched_total / latest_total) * 100 if latest_total > 0 else 0

# =========================================
# 7. SAVE MATCHED DATA
# =========================================
matched.to_csv("merge_between_baseline_endline.csv", index=False)

# Optional: also save unmatched records separately
baseline_unmatched = merged_check[merged_check["match_status"] == "left_only"].copy()
latest_unmatched = merged_check[merged_check["match_status"] == "right_only"].copy()

baseline_unmatched.to_csv("baseline_unmatched_after_endline_merge.csv", index=False)
latest_unmatched.to_csv("latest_unmatched_after_baseline_merge.csv", index=False)

# =========================================
# 8. DISPLAY RESULTS
# =========================================
print("Matching complete")
print(f"Baseline records: {baseline_total}")
print(f"Latest records: {latest_total}")
print(f"Successful matches: {matched_total}")
print(f"Baseline match rate: {baseline_match_pct:.2f}%")
print(f"Latest dataset match rate: {latest_match_pct:.2f}%")
print("Saved matched file as: merge_between_baseline_endline.csv")