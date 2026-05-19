from pathlib import Path
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import LABEL_COLS, apply_clahe, make_eval_transform
from model import make_backbone


# RUN_NAME picks which run's checkpoint we load. The backbone name is
# read from the checkpoint itself (saved during training), so we don't
# need to know it here.
RUN_NAME = os.environ.get("CS156B_RUN_NAME", "v4_multitask_no_imputation")

DATA_ROOT = Path("/resnick/groups/CS156b/from_central/data")
TEST_CSV = DATA_ROOT / "student_labels" / "test_ids.csv"
CACHE_DIR = Path("/resnick/groups/CS156b/from_central/2026/JSC/cache_320")

MODEL_DIR = Path("/resnick/groups/CS156b/from_central/2026/JSC/outputs") / RUN_NAME
CKPT_PATH = MODEL_DIR / "best.pt"
OUT_PATH = MODEL_DIR / "predictions.csv"

IMAGE_SIZE = 384  # must match training
BATCH_SIZE = 64
NUM_WORKERS = 8

# Average predictions on the original image and its horizontal flip.
USE_TTA = True

CLIP_MIN = -1.0
CLIP_MAX = 1.0


class InferenceDataset(Dataset):
    def __init__(self, df, cache_root, transform, use_clahe=False):
        self.df = df.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.transform = transform
        self.use_clahe = use_clahe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.cache_root / Path(row["Path"]).with_suffix(".png")

        img = Image.open(path).convert("L")
        if self.use_clahe:
            img = apply_clahe(img)
        img = self.transform(img)

        return img, idx


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()

    all_idx = []
    all_preds = []

    use_amp = torch.cuda.is_available()

    for images, idx in loader:
        images = images.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            preds = model(images).float()

            if USE_TTA:
                preds_flip = model(torch.flip(images, dims=[3])).float()
                preds = (preds + preds_flip) / 2.0

        all_idx.append(idx.numpy())
        all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_idx), np.concatenate(all_preds)


def main():
    print(f"=== PREDICT ({RUN_NAME}) ===")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Cache: {CACHE_DIR}")
    print(f"Output: {OUT_PATH}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"TTA: {USE_TTA}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    backbone_name = ckpt.get("backbone", "densenet121")
    use_clahe = ckpt.get("use_clahe", False)
    print(f"Backbone (from checkpoint): {backbone_name}")
    print(f"CLAHE (from checkpoint):    {use_clahe}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_backbone(backbone_name, num_outputs=len(LABEL_COLS),
                          pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    df = pd.read_csv(TEST_CSV)
    print(f"\nPredicting {len(df):,} rows")

    dataset = InferenceDataset(df, CACHE_DIR, make_eval_transform(IMAGE_SIZE),
                                use_clahe=use_clahe)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    t0 = time.time()
    idx, preds = run_inference(model, loader, device)
    print(f"  inference took {(time.time() - t0) / 60:.1f} min")

    # The DataLoader doesn't preserve order across workers, so reorder
    # predictions to match the original CSV row order.
    order = np.argsort(idx)
    preds = preds[order]

    preds = np.clip(preds, CLIP_MIN, CLIP_MAX)

    out_df = df.copy()
    for i, col in enumerate(LABEL_COLS):
        out_df[col] = preds[:, i]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
