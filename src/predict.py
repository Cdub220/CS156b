from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import LABEL_COLS, build_labels_and_mask, make_eval_transform
from model import DenseNet121


DATA_ROOT = Path("/resnick/groups/CS156b/from_central/data")
TEST_CSV = DATA_ROOT / "student_labels" / "test_ids.csv"
CACHE_DIR = Path("/resnick/groups/CS156b/from_central/2026/JSC/cache_320")

CKPT_PATH = Path("/resnick/groups/CS156b/from_central/2026/JSC/outputs/densenet121/best.pt")
OUT_PATH = Path("/resnick/groups/CS156b/from_central/2026/JSC/outputs/densenet121/predictions.csv")

IMAGE_SIZE = 320
BATCH_SIZE = 128
NUM_WORKERS = 8

USE_TTA = True

CLIP_MIN = -1.0
CLIP_MAX = 1.0


class InferenceDataset(Dataset):
    def __init__(self, df, cache_root, transform):
        self.df = df.reset_index(drop=True)
        self.cache_root = Path(cache_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.cache_root / Path(row["Path"]).with_suffix(".png")

        img = Image.open(path).convert("L")
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
    print("=== DENSENET121 PREDICT ===")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Cache: {CACHE_DIR}")
    print(f"Output: {OUT_PATH}")
    print(f"TTA: {USE_TTA}")

    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(num_outputs=len(LABEL_COLS), pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    df = pd.read_csv(TEST_CSV)
    print(f"\nPredicting {len(df):,} rows")

    dataset = InferenceDataset(df, CACHE_DIR, make_eval_transform(IMAGE_SIZE))
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

    # If the input CSV has ground-truth labels (e.g. the val split), compute
    # MSE so we can directly see how the model generalizes on held-out data.
    if all(col in df.columns for col in LABEL_COLS):
        truths = np.zeros((len(df), len(LABEL_COLS)), dtype=np.float32)
        masks = np.zeros((len(df), len(LABEL_COLS)), dtype=np.float32)
        for i in range(len(df)):
            t, m = build_labels_and_mask(df.iloc[i])
            truths[i] = t
            masks[i] = m

        sq_err = (preds - truths) ** 2
        overall_mse = float((sq_err * masks).sum() / max(masks.sum(), 1))
        print(f"\nHeld-out MSE: {overall_mse:.4f}")

        for i, name in enumerate(LABEL_COLS):
            known = masks[:, i] == 1
            if known.sum() == 0:
                print(f"  {name:30s}  MSE=n/a")
            else:
                col_mse = float(((preds[known, i] - truths[known, i]) ** 2).mean())
                print(f"  {name:30s}  MSE={col_mse:.4f}  (n={int(known.sum())})")

    out_df = df.copy()
    for i, col in enumerate(LABEL_COLS):
        out_df[col] = preds[:, i]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
