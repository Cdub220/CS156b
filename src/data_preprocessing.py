from pathlib import Path
import re
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_ROOT = Path("/resnick/groups/CS156b/from_central/data")
IMAGE_ROOT = DATA_ROOT / "train"
TRAIN_CSV = DATA_ROOT / "student_labels" / "train2023.csv"
OUT_DIR = Path("/resnick/groups/CS156b/from_central/2026/JSC/preprocessing_output")

SAMPLE_SIZE = 100
IMAGE_SIZE = 224
UNCERTAIN_POLICY = "ignore"  # ignore, zero, or one


def get_full_path(path_value):
    p = Path(str(path_value))

    if p.is_absolute():
        return p

    # Case 1: Path column already starts with train/...
    data_path = DATA_ROOT / p
    if data_path.exists():
        return data_path

    # Case 2: Path column starts with pid.../study...
    image_path = IMAGE_ROOT / p
    if image_path.exists():
        return image_path

    return image_path


def extract_patient_id(path_value):
    match = re.search(r"(pid\d+)", str(path_value))
    return match.group(1) if match else None


def infer_label_columns(df):
    metadata_cols = {"Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"}
    label_cols = []

    for col in df.columns:
        if col in metadata_cols:
            continue

        values = set(df[col].dropna().unique())
        allowed = {-1, 0, 1, -1.0, 0.0, 1.0}

        if len(values) > 0 and values.issubset(allowed):
            label_cols.append(col)

    return label_cols


def pad_to_square(img):
    width, height = img.size
    max_side = max(width, height)

    pad_left = (max_side - width) // 2
    pad_right = max_side - width - pad_left
    pad_top = (max_side - height) // 2
    pad_bottom = max_side - height - pad_top

    return ImageOps.expand(
        img,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=0,
    )


def preprocess_image(path):
    with Image.open(path) as img:
        original_size = img.size
        original_mode = img.mode

        img = img.convert("L")
        img = pad_to_square(img)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)

        raw = np.array(img).astype(np.float32) / 255.0

        # Grayscale to 3 channels for pretrained CNN compatibility.
        image = np.stack([raw, raw, raw], axis=0)

        # ImageNet normalization.
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        image = (image - mean) / std

        return image, raw, original_size, original_mode


def build_labels_and_mask(row, label_cols):
    labels = []
    mask = []

    for col in label_cols:
        value = row[col]

        if pd.isna(value):
            labels.append(0.0)
            mask.append(0.0)

        elif value == -1:
            if UNCERTAIN_POLICY == "ignore":
                labels.append(0.0)
                mask.append(0.0)
            elif UNCERTAIN_POLICY == "zero":
                labels.append(0.0)
                mask.append(1.0)
            elif UNCERTAIN_POLICY == "one":
                labels.append(1.0)
                mask.append(1.0)
            else:
                raise ValueError("Bad UNCERTAIN_POLICY")

        elif value == 0:
            labels.append(0.0)
            mask.append(1.0)

        elif value == 1:
            labels.append(1.0)
            mask.append(1.0)

        else:
            labels.append(0.0)
            mask.append(0.0)

    return np.array(labels, dtype=np.float32), np.array(mask, dtype=np.float32)


def save_preview_grid(raw_images, manifest):
    n = min(25, len(raw_images))
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i in range(n):
        axes[i].imshow(raw_images[i], cmap="gray")
        axes[i].set_title(
            f"{i}: {manifest[i]['view']} | known={manifest[i]['num_known_labels']}",
            fontsize=8,
        )

    plt.tight_layout()
    out_path = OUT_DIR / "preview_grid.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== DATA PREPROCESSING TEST ===")
    print(f"CSV: {TRAIN_CSV}")
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Uncertain policy: {UNCERTAIN_POLICY}")

    df = pd.read_csv(TRAIN_CSV)
    label_cols = infer_label_columns(df)

    print("\nDetected label columns:")
    for col in label_cols:
        print(f"  - {col}")

    # Shuffle first so we do not scan/check the whole dataset.
    sample_candidates = df.sample(
        n=min(1000, len(df)),
        random_state=42,
    ).copy()

    images = []
    raw_images = []
    labels_all = []
    masks_all = []
    manifest = []

    checked = 0
    skipped_missing = 0

    for csv_index, row in sample_candidates.iterrows():
        if len(images) >= SAMPLE_SIZE:
            break

        checked += 1
        image_path = get_full_path(row["Path"])

        if not image_path.exists():
            skipped_missing += 1
            continue

        try:
            image, raw, original_size, original_mode = preprocess_image(image_path)
            labels, mask = build_labels_and_mask(row, label_cols)
        except Exception as e:
            print(f"Skipping bad image: {image_path} | {e}")
            continue

        images.append(image)
        raw_images.append(raw)
        labels_all.append(labels)
        masks_all.append(mask)

        manifest.append({
            "csv_index": int(csv_index),
            "path": row["Path"],
            "full_path": str(image_path),
            "patient_id": extract_patient_id(row["Path"]),
            "sex": row["Sex"] if "Sex" in row else None,
            "age": row["Age"] if "Age" in row else None,
            "view": row["Frontal/Lateral"] if "Frontal/Lateral" in row else None,
            "ap_pa": row["AP/PA"] if "AP/PA" in row else None,
            "original_width": original_size[0],
            "original_height": original_size[1],
            "original_mode": original_mode,
            "processed_shape": str(image.shape),
            "num_positive_labels": int((labels == 1).sum()),
            "num_known_labels": int(mask.sum()),
            "num_ignored_labels": int((mask == 0).sum()),
            "raw_min": float(raw.min()),
            "raw_max": float(raw.max()),
            "raw_mean": float(raw.mean()),
            "raw_std": float(raw.std()),
            "norm_min": float(image.min()),
            "norm_max": float(image.max()),
            "norm_mean": float(image.mean()),
            "norm_std": float(image.std()),
        })

        if len(images) % 10 == 0:
            print(f"Processed {len(images)}/{SAMPLE_SIZE} images")

    if len(images) == 0:
        print("\nNo images were processed. First 10 attempted paths:")
        for _, row in sample_candidates.head(10).iterrows():
            print(get_full_path(row["Path"]))
        raise SystemExit

    images = np.stack(images, axis=0)
    labels_all = np.stack(labels_all, axis=0)
    masks_all = np.stack(masks_all, axis=0)

    manifest_df = pd.DataFrame(manifest)

    np.savez_compressed(
        OUT_DIR / "preprocessed_sample_100.npz",
        images=images,
        labels=labels_all,
        masks=masks_all,
        label_cols=np.array(label_cols),
        paths=np.array([m["path"] for m in manifest]),
    )

    manifest_df.to_csv(OUT_DIR / "sample_manifest.csv", index=False)

    save_preview_grid(raw_images, manifest)

    summary = {
        "num_samples_processed": int(images.shape[0]),
        "num_csv_rows_checked": int(checked),
        "num_missing_paths_skipped": int(skipped_missing),
        "image_tensor_shape": list(images.shape),
        "labels_shape": list(labels_all.shape),
        "masks_shape": list(masks_all.shape),
        "label_columns": label_cols,
        "uncertain_policy": UNCERTAIN_POLICY,
        "image_size": IMAGE_SIZE,
        "mean_known_labels_per_image": float(masks_all.sum(axis=1).mean()),
        "mean_positive_labels_per_image": float(labels_all.sum(axis=1).mean()),
        "total_known_label_entries": int(masks_all.sum()),
        "total_positive_label_entries": int(labels_all.sum()),
        "view_counts": manifest_df["view"].value_counts(dropna=False).to_dict(),
        "sex_counts": manifest_df["sex"].value_counts(dropna=False).to_dict(),
        "ap_pa_counts": manifest_df["ap_pa"].value_counts(dropna=False).to_dict(),
    }

    with open(OUT_DIR / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(OUT_DIR / "preprocessing_summary.txt", "w") as f:
        f.write("DATA PREPROCESSING TEST SUMMARY\n")
        f.write("=" * 40 + "\n")
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print("\n=== DONE ===")
    print(f"CSV rows checked: {checked}")
    print(f"Missing paths skipped: {skipped_missing}")
    print(f"Processed image tensor shape: {images.shape}")
    print(f"Labels shape: {labels_all.shape}")
    print(f"Masks shape: {masks_all.shape}")
    print(f"Mean known labels per image: {summary['mean_known_labels_per_image']:.2f}")
    print(f"Mean positive labels per image: {summary['mean_positive_labels_per_image']:.2f}")
    print(f"Saved output to: {OUT_DIR}")


if __name__ == "__main__":
    main()