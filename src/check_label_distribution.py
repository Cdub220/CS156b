import csv
from pathlib import Path
from collections import Counter

csv_path = Path("/resnick/groups/CS156b/from_central/data/student_labels/train.csv")

if not csv_path.exists():
    print("Could not find train.csv")
    raise SystemExit

pathology_cols = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

counts = {col: Counter() for col in pathology_cols}
total_rows = 0

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        total_rows += 1

        for col in pathology_cols:
            val = row[col].strip() if row[col] is not None else ""

            if val == "1.0":
                counts[col]["positive"] += 1
            elif val == "0.0":
                counts[col]["uncertain"] += 1
            elif val == "-1.0":
                counts[col]["negative"] += 1
            elif val == "":
                counts[col]["missing"] += 1
            else:
                counts[col]["other"] += 1

print("Total rows:", total_rows)
print("\n=== LABEL DISTRIBUTION ===")

for col in pathology_cols:
    c = counts[col]

    pos = c["positive"]
    unc = c["uncertain"]
    neg = c["negative"]
    miss = c["missing"]
    other = c["other"]

    known = pos + unc + neg
    non_missing = total_rows - miss

    pos_rate_known = pos / known if known > 0 else 0.0
    unc_rate_known = unc / known if known > 0 else 0.0
    neg_rate_known = neg / known if known > 0 else 0.0

    pos_rate_non_missing = pos / non_missing if non_missing > 0 else 0.0

    print(f"\n{col}")
    print(f"  positive   : {pos}")
    print(f"  uncertain  : {unc}")
    print(f"  negative   : {neg}")
    print(f"  missing    : {miss}")
    if other > 0:
        print(f"  other      : {other}")
    print(f"  positive rate among known labels     : {pos_rate_known:.4f}")
    print(f"  uncertain rate among known labels    : {unc_rate_known:.4f}")
    print(f"  negative rate among known labels     : {neg_rate_known:.4f}")
    print(f"  positive rate among non-missing rows : {pos_rate_non_missing:.4f}")