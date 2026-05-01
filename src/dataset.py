from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


#label convention:
#   1  -> positive
#   0  -> uncertain
#  -1  -> negative
#  NaN -> missing / unmentioned
#

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


def build_labels_and_mask(row, ignore_uncertain=False):
    labels = np.zeros(len(LABEL_COLS), dtype=np.float32)
    mask = np.zeros(len(LABEL_COLS), dtype=np.float32)

    for i, col in enumerate(LABEL_COLS):
        value = row[col]

        if pd.isna(value):
            # Missing not graded, don't compute loss for it.
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
    def __init__(self, df, cache_root, transform, ignore_uncertain=False):
        self.df = df.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.transform = transform
        self.ignore_uncertain = ignore_uncertain

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # The cache mirrors the source layout but with .png extensions.
        path = self.cache_root / Path(row["Path"]).with_suffix(".png")

        img = Image.open(path).convert("L")
        img = self.transform(img)

        labels_np, mask_np = build_labels_and_mask(row, self.ignore_uncertain)
        labels = torch.from_numpy(labels_np)
        mask = torch.from_numpy(mask_np)

        return img, labels, mask
