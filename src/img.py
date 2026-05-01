import csv
import random
from pathlib import Path
from PIL import Image

csv_path = Path("/resnick/groups/CS156b/from_central/data/student_labels/train.csv")
data_root = Path("/resnick/groups/CS156b/from_central/data")

frontal_paths = []
lateral_paths = []

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rel_path = row["Path"]
        view = row["Frontal/Lateral"].strip() if row["Frontal/Lateral"] else ""

        if view == "Frontal":
            frontal_paths.append(rel_path)
        elif view == "Lateral":
            lateral_paths.append(rel_path)

random.seed(0)

frontal_sample = random.sample(frontal_paths, min(5, len(frontal_paths)))
lateral_sample = random.sample(lateral_paths, min(5, len(lateral_paths)))

print("=== FRONTAL SAMPLE ===")
for rel_path in frontal_sample:
    img_path = data_root / rel_path
    with Image.open(img_path) as img:
        print(rel_path, "->", img.size)

print("\n=== LATERAL SAMPLE ===")
for rel_path in lateral_sample:
    img_path = data_root / rel_path
    with Image.open(img_path) as img:
        print(rel_path, "->", img.size)