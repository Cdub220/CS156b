from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


_CLAHE_OP = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def apply_clahe(pil_img_grayscale):
    arr = np.array(pil_img_grayscale, dtype=np.uint8)
    arr = _CLAHE_OP.apply(arr)
    return Image.fromarray(arr)


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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def compute_label_means(df):
    # Mean of non-NaN labels per pathology, computed once on train data.
    return {col: float(df[col].dropna().mean()) for col in LABEL_COLS}


def build_labels_and_mask(row, ignore_uncertain=False, label_means=None):
    labels = np.zeros(len(LABEL_COLS), dtype=np.float32)
    mask = np.zeros(len(LABEL_COLS), dtype=np.float32)

    for i, col in enumerate(LABEL_COLS):
        value = row[col]

        if pd.isna(value):
            if label_means is not None:
                # Mean imputation: fill with the pathology's average label.
                labels[i] = label_means[col]
                mask[i] = 1.0
            # else: leave as zero with mask=0 (label is ignored in loss)
            continue

        if value == 0 and ignore_uncertain:
            continue

        labels[i] = float(value)
        mask[i] = 1.0

    return labels, mask


def make_train_transform(image_size, hflip=True):
    ops = []

    if hflip:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))

    ops += [
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=0,
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(ops)


def make_eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class ChestXrayDataset(Dataset):
    def __init__(self, df, cache_root, transform, ignore_uncertain=False,
                 label_means=None, target_col=None, use_clahe=False):
        self.df = df.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.transform = transform
        self.ignore_uncertain = ignore_uncertain
        self.label_means = label_means
        self.use_clahe = use_clahe

        # If target_col is set, only return that one pathology's label
        # (so single-output models can use the same dataset class).
        self.target_col = target_col
        if target_col is not None:
            self.target_idx = LABEL_COLS.index(target_col)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # The cache mirrors the source layout but with .png extensions.
        path = self.cache_root / Path(row["Path"]).with_suffix(".png")

        img = Image.open(path).convert("L")
        if self.use_clahe:
            img = apply_clahe(img)
        img = self.transform(img)

        labels_np, mask_np = build_labels_and_mask(
            row, self.ignore_uncertain, self.label_means,
        )

        if self.target_col is not None:
            labels = torch.tensor([labels_np[self.target_idx]], dtype=torch.float32)
            mask = torch.tensor([mask_np[self.target_idx]], dtype=torch.float32)
        else:
            labels = torch.from_numpy(labels_np)
            mask = torch.from_numpy(mask_np)

        return img, labels, mask
