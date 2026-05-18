from pathlib import Path
import json
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    LABEL_COLS,
    ChestXrayDataset,
    compute_label_means,
    make_eval_transform,
    make_train_transform,
)
from model import DenseNet121


# Bump RUN_NAME for every new training run so outputs don't overwrite.
# Each run gets its own directory under outputs/.
RUN_NAME = "v2_18models_imputed"

DATA_ROOT = Path("/resnick/groups/CS156b/from_central/data")
TRAIN_CSV = DATA_ROOT / "student_labels" / "train2023.csv"
CACHE_DIR = Path("/resnick/groups/CS156b/from_central/2026/JSC/cache_320")
OUT_DIR = Path("/resnick/groups/CS156b/from_central/2026/JSC/outputs") / RUN_NAME

# We train 18 models: one per. Each model is single-output
# and trained only on images of its view.
VIEWS = ["Frontal", "Lateral"]

IMAGE_SIZE = 320
BATCH_SIZE = 64
EPOCHS = 5
NUM_WORKERS = 8

LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-5

WARMUP_EPOCHS = 1

VAL_PATIENT_FRAC = 0.1
IGNORE_UNCERTAIN = False
USE_HFLIP = True
USE_AMP = True
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_patient_id(path_value):
    match = re.search(r"(pid\d+)", str(path_value))
    return match.group(1) if match else None


def make_patient_split(df):
    df = df.copy()
    df["patient_id"] = df["Path"].apply(extract_patient_id)

    patients = sorted(df["patient_id"].dropna().unique())
    rng = random.Random(SEED)
    rng.shuffle(patients)

    num_val = max(1, int(len(patients) * VAL_PATIENT_FRAC))
    val_patients = set(patients[:num_val])

    val_df = df[df["patient_id"].isin(val_patients)].copy()
    train_df = df[~df["patient_id"].isin(val_patients)].copy()

    return train_df, val_df


def slugify(name):
    return name.lower().replace(" ", "_").replace("/", "_")


def masked_mse_loss(preds, labels, mask):
    sq_err = (preds - labels) ** 2
    masked = sq_err * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()

    total_loss = 0.0
    total_batches = 0
    start = time.time()

    for batch_idx, (images, labels, mask) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            preds = model(images)
            loss = masked_mse_loss(preds, labels, mask)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start
            it_per_sec = total_batches / elapsed
            print(
                f"    batch {batch_idx + 1}/{len(loader)}  "
                f"loss={loss.item():.4f}  "
                f"avg={total_loss / total_batches:.4f}  "
                f"({it_per_sec:.1f} it/s)",
                flush=True,
            )

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []
    all_mask = []

    for images, labels, mask in loader:
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            preds = model(images).float()

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_mask.append(mask.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    masks = np.concatenate(all_mask)

    sq_err = (preds - labels) ** 2
    overall_mse = float((sq_err * masks).sum() / max(masks.sum(), 1))
    abs_err = np.abs(preds - labels)
    overall_mae = float((abs_err * masks).sum() / max(masks.sum(), 1))

    return {"mse": overall_mse, "mae": overall_mae}


def save_checkpoint(model, path, epoch, val_mse, view, pathology):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "val_mse": val_mse,
            "view": view,
            "pathology": pathology,
            "image_size": IMAGE_SIZE,
        },
        path,
    )


def train_one_model(view, pathology, train_df, val_df, label_means, device):
    model_dir = OUT_DIR / view.lower() / slugify(pathology)
    best_path = model_dir / "best.pt"

    # Resumability: skip if this model is already trained.
    if best_path.exists():
        print(f"Skipping {view} / {pathology}: best.pt already exists")
        return

    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Training {view} / {pathology} ===")
    print(f"  Train rows: {len(train_df):,}  Val rows: {len(val_df):,}")
    print(f"  Output: {model_dir}")

    train_dataset = ChestXrayDataset(
        train_df,
        CACHE_DIR,
        transform=make_train_transform(IMAGE_SIZE, hflip=USE_HFLIP),
        ignore_uncertain=IGNORE_UNCERTAIN,
        label_means=label_means,
        target_col=pathology,
    )

    val_dataset = ChestXrayDataset(
        val_df,
        CACHE_DIR,
        transform=make_eval_transform(IMAGE_SIZE),
        ignore_uncertain=IGNORE_UNCERTAIN,
        label_means=label_means,
        target_col=pathology,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    model = DenseNet121(num_outputs=1, pretrained=True).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_params(), "lr": LR_BACKBONE},
            {"params": model.head_params(), "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS * len(train_loader),
    )

    use_amp = USE_AMP and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    history = []
    best_mse = float("inf")

    for epoch in range(1, EPOCHS + 1):
        head_only = epoch <= WARMUP_EPOCHS
        model.freeze_backbone(freeze=head_only)

        phase = "head-only warmup" if head_only else "full fine-tune"
        print(f"\n  Epoch {epoch}/{EPOCHS}  ({phase})")

        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        scheduler.step()

        val = evaluate(model, val_loader, device)
        epoch_minutes = (time.time() - t0) / 60

        history.append({
            "epoch": epoch,
            "phase": phase,
            "train_loss": train_loss,
            "val_mse": val["mse"],
            "val_mae": val["mae"],
            "minutes": epoch_minutes,
        })

        print(
            f"    train_loss={train_loss:.4f}  "
            f"val_mse={val['mse']:.4f}  "
            f"val_mae={val['mae']:.4f}  "
            f"({epoch_minutes:.1f} min)"
        )

        if val["mse"] < best_mse:
            best_mse = val["mse"]
            save_checkpoint(model, best_path, epoch, best_mse, view, pathology)
            print(f"    ** new best val MSE {best_mse:.4f}")

        with open(model_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    save_checkpoint(model, model_dir / "last.pt", EPOCHS, history[-1]["val_mse"],
                    view, pathology)


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== MULTI-MODEL TRAIN ({RUN_NAME}) ===")
    print(f"CSV: {TRAIN_CSV}")
    print(f"Cache: {CACHE_DIR}")
    print(f"Output: {OUT_DIR}")
    print(f"Views: {VIEWS}")
    print(f"Pathologies: {len(LABEL_COLS)}")
    print(f"Total models to train: {len(VIEWS) * len(LABEL_COLS)}")

    df = pd.read_csv(TRAIN_CSV)
    print(f"\nLoaded {len(df):,} rows from CSV")

    # Drop rows whose preprocessed PNG isn't in the cache.
    print("Scanning cache directory...")
    cache_set = {
        str(p.relative_to(CACHE_DIR).with_suffix(""))
        for p in CACHE_DIR.rglob("*.png")
    }
    print(f"Found {len(cache_set):,} cached images")

    mask = df["Path"].apply(lambda p: str(Path(p).with_suffix("")) in cache_set)
    n_missing = int((~mask).sum())
    df = df[mask].reset_index(drop=True)
    print(f"Dropped {n_missing:,} rows with no cached image; {len(df):,} remain")

    train_df, val_df = make_patient_split(df)

    print(f"\nTrain rows: {len(train_df):,}  patients: {train_df['patient_id'].nunique():,}")
    print(f"Val rows:   {len(val_df):,}  patients: {val_df['patient_id'].nunique():,}")

    # Save splits so predict.py uses the same held-out patients.
    train_df.to_csv(OUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUT_DIR / "val_split.csv", index=False)

    # Compute per-pathology label means on training data only (no val leakage).
    label_means = compute_label_means(train_df)
    with open(OUT_DIR / "label_means.json", "w") as f:
        json.dump(label_means, f, indent=2)
    print("\nLabel means (for NaN imputation):")
    for col, m in label_means.items():
        print(f"  {col:30s}  mean={m:+.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")


    all_pairs = [(view, p) for view in VIEWS for p in LABEL_COLS]

    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")

    if array_task is not None:
        # Running as one task of a SLURM array job -- train just this pair.
        idx = int(array_task)
        if idx < 0 or idx >= len(all_pairs):
            raise ValueError(f"SLURM_ARRAY_TASK_ID={idx} out of range "
                             f"[0, {len(all_pairs) - 1}]")
        pairs_to_train = [all_pairs[idx]]
        print(f"\n[SLURM array task {idx}] training 1 of {len(all_pairs)} models")
    else:
        pairs_to_train = all_pairs
        print(f"\n[no SLURM array] training all {len(all_pairs)} models sequentially")

    overall_start = time.time()

    for i, (view, pathology) in enumerate(pairs_to_train, start=1):
        view_train = train_df[train_df["Frontal/Lateral"] == view]
        view_val = val_df[val_df["Frontal/Lateral"] == view]

        if len(view_train) == 0:
            print(f"\nNo training images for view '{view}', skipping")
            continue

        elapsed_min = (time.time() - overall_start) / 60
        print(f"\n--- {i}/{len(pairs_to_train)}  "
              f"(elapsed {elapsed_min:.1f} min) ---")
        train_one_model(view, pathology, view_train, view_val,
                        label_means, device)

    total_min = (time.time() - overall_start) / 60
    print(f"\n=== DONE === total time {total_min:.1f} min")


if __name__ == "__main__":
    main()
