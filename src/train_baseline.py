from pathlib import Path
import re
import random
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models


DATA_ROOT = Path("/resnick/groups/CS156b/from_central/data")
IMAGE_ROOT = DATA_ROOT / "train"
TRAIN_CSV = DATA_ROOT / "student_labels" / "train2023.csv"

OUT_DIR = Path("/resnick/groups/CS156b/from_central/2026/JSC/outputs/densenet_sanity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

IMAGE_SIZE = 224
BATCH_SIZE = 8
MAX_TRAIN_IMAGES = 500
MAX_VAL_IMAGES = 100
VAL_PATIENT_FRAC = 0.1
EPOCHS = 1
LR = 1e-3
UNCERTAIN_POLICY = "ignore"  # 0 means uncertain; baseline ignores it
FREEZE_BACKBONE = True
USE_PRETRAINED = True
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_patient_id(path_value):
    match = re.search(r"(pid\d+)", str(path_value))
    return match.group(1) if match else None


def get_full_path(path_value):
    p = Path(str(path_value))

    if p.is_absolute():
        return p

    data_path = DATA_ROOT / p
    if data_path.exists():
        return data_path

    image_path = IMAGE_ROOT / p
    return image_path


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
        img = img.convert("L")
        img = pad_to_square(img)
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)

        arr = np.array(img).astype(np.float32) / 255.0

        # grayscale -> 3 channels for ImageNet-pretrained DenseNet
        arr = np.stack([arr, arr, arr], axis=0)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / std

        return torch.tensor(arr, dtype=torch.float32)


def build_labels_and_mask(row):
    labels = []
    mask = []

    for col in LABEL_COLS:
        value = row[col]

        if pd.isna(value):
            # missing label: ignore
            labels.append(0.0)
            mask.append(0.0)

        elif value == 1:
            # positive label
            labels.append(1.0)
            mask.append(1.0)

        elif value == -1:
            # negative label
            labels.append(0.0)
            mask.append(1.0)

        elif value == 0:
            # uncertain label
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

        else:
            labels.append(0.0)
            mask.append(0.0)

    return (
        torch.tensor(labels, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32),
    )


class ChestXrayDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = get_full_path(row["Path"])

        image = preprocess_image(image_path)
        labels, mask = build_labels_and_mask(row)

        return image, labels, mask


class DenseNetBaseline(nn.Module):
    def __init__(self, num_outputs=9, use_pretrained=True, freeze_backbone=True):
        super().__init__()

        if use_pretrained:
            try:
                weights = models.DenseNet121_Weights.DEFAULT
                self.model = models.densenet121(weights=weights)
                print("Loaded DenseNet121 with ImageNet pretrained weights.")
            except Exception as e:
                print(f"Could not load pretrained weights: {e}")
                print("Falling back to randomly initialized DenseNet121.")
                self.model = models.densenet121(weights=None)
        else:
            self.model = models.densenet121(weights=None)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_outputs)

        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False

            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.model(x)


def masked_bce_loss(logits, labels, mask):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    raw_loss = loss_fn(logits, labels)

    masked_loss = raw_loss * mask
    denom = mask.sum().clamp_min(1.0)

    return masked_loss.sum() / denom


def make_patient_split(df):
    df = df.copy()
    df["patient_id"] = df["Path"].apply(extract_patient_id)

    patients = sorted(df["patient_id"].dropna().unique())
    random.shuffle(patients)

    num_val = max(1, int(len(patients) * VAL_PATIENT_FRAC))
    val_patients = set(patients[:num_val])
    train_patients = set(patients[num_val:])

    train_df = df[df["patient_id"].isin(train_patients)].copy()
    val_df = df[df["patient_id"].isin(val_patients)].copy()

    train_df = train_df.sample(
        n=min(MAX_TRAIN_IMAGES, len(train_df)),
        random_state=SEED,
    ).copy()

    val_df = val_df.sample(
        n=min(MAX_VAL_IMAGES, len(val_df)),
        random_state=SEED,
    ).copy()

    return train_df, val_df


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_batches = 0

    for batch_idx, (images, labels, mask) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = masked_bce_loss(logits, labels, mask)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if batch_idx == 0:
            print("\nFirst training batch:")
            print(f"  images: {tuple(images.shape)}")
            print(f"  labels: {tuple(labels.shape)}")
            print(f"  mask:   {tuple(mask.shape)}")
            print(f"  logits: {tuple(logits.shape)}")
            print(f"  known label entries in batch: {int(mask.sum().item())}")

        if (batch_idx + 1) % 5 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)} | loss {loss.item():.4f}")

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_loss = 0.0
    total_batches = 0
    total_known = 0
    total_correct = 0

    for images, labels, mask in loader:
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        logits = model(images)
        loss = masked_bce_loss(logits, labels, mask)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        known = mask == 1
        correct = ((preds == labels) & known).sum().item()

        total_correct += correct
        total_known += known.sum().item()

        total_loss += loss.item()
        total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    known_acc = total_correct / total_known if total_known > 0 else 0.0

    return avg_loss, known_acc, total_known


def main():
    set_seed(SEED)

    print("=== DENSENET121 BASELINE SANITY TRAINING ===")
    print(f"CSV: {TRAIN_CSV}")
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Labels: {LABEL_COLS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max train images: {MAX_TRAIN_IMAGES}")
    print(f"Max val images: {MAX_VAL_IMAGES}")
    print(f"Epochs: {EPOCHS}")
    print(f"LR: {LR}")
    print(f"Uncertain policy: {UNCERTAIN_POLICY}")
    print(f"Freeze backbone: {FREEZE_BACKBONE}")
    print(f"Use pretrained: {USE_PRETRAINED}")

    df = pd.read_csv(TRAIN_CSV)

    missing_cols = [col for col in LABEL_COLS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing label columns: {missing_cols}")

    train_df, val_df = make_patient_split(df)

    print(f"\nTrain rows used: {len(train_df)}")
    print(f"Val rows used: {len(val_df)}")
    print(f"Train patients: {train_df['patient_id'].nunique()}")
    print(f"Val patients: {val_df['patient_id'].nunique()}")

    train_dataset = ChestXrayDataset(train_df)
    val_dataset = ChestXrayDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = DenseNetBaseline(
        num_outputs=len(LABEL_COLS),
        use_pretrained=USE_PRETRAINED,
        freeze_backbone=FREEZE_BACKBONE,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,}")
    print(f"Total params: {total_params:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    history = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_known_acc, val_known_count = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_known_accuracy": val_known_acc,
            "val_known_label_entries": int(val_known_count),
        }
        history.append(row)

        print(f"\nEpoch {epoch + 1} summary:")
        print(f"  train loss: {train_loss:.4f}")
        print(f"  val loss:   {val_loss:.4f}")
        print(f"  val known-label accuracy: {val_known_acc:.4f}")
        print(f"  val known label entries: {int(val_known_count)}")

    model_path = OUT_DIR / "densenet121_sanity.pt"
    torch.save(model.state_dict(), model_path)

    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    train_df.to_csv(OUT_DIR / "train_subset.csv", index=False)
    val_df.to_csv(OUT_DIR / "val_subset.csv", index=False)

    print("\n=== DONE ===")
    print(f"Saved model to: {model_path}")
    print(f"Saved history to: {OUT_DIR / 'training_history.json'}")


if __name__ == "__main__":
    main()