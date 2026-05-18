from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import LABEL_COLS, make_eval_transform
from model import DenseNet121


# Must match the RUN_NAME used in train.py.
RUN_NAME = "v2_18models_imputed"

DATA_ROOT = Path("/resnick/groups/CS156b/from_central/data")
TEST_CSV = DATA_ROOT / "student_labels" / "test_ids.csv"
CACHE_DIR = Path("/resnick/groups/CS156b/from_central/2026/JSC/cache_320")

MODELS_DIR = Path("/resnick/groups/CS156b/from_central/2026/JSC/outputs") / RUN_NAME
OUT_PATH = MODELS_DIR / "predictions.csv"

VIEWS = ["Frontal", "Lateral"]

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


def slugify(name):
    return name.lower().replace(" ", "_").replace("/", "_")


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


def predict_for_view(view, view_df, device):
    # Runs all 9 pathology models for this view and returns a array.
    n = len(view_df)
    view_preds = np.zeros((n, len(LABEL_COLS)), dtype=np.float32)

    dataset = InferenceDataset(view_df, CACHE_DIR, make_eval_transform(IMAGE_SIZE))
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    for pathology_idx, pathology in enumerate(LABEL_COLS):
        ckpt_path = MODELS_DIR / view.lower() / slugify(pathology) / "best.pt"

        if not ckpt_path.exists():
            print(f"  WARNING: missing checkpoint {ckpt_path}, filling zeros")
            continue

        print(f"  [{pathology_idx + 1}/{len(LABEL_COLS)}] {pathology}", flush=True)

        model = DenseNet121(num_outputs=1, pretrained=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

        idx, preds = run_inference(model, loader, device)
        order = np.argsort(idx)
        preds = preds[order].squeeze(-1)  # (n,)

        view_preds[:, pathology_idx] = preds

        del model
        torch.cuda.empty_cache()

    return view_preds


def main():
    print(f"=== MULTI-MODEL PREDICT ({RUN_NAME}) ===")
    print(f"Models: {MODELS_DIR}")
    print(f"Test CSV: {TEST_CSV}")
    print(f"Cache: {CACHE_DIR}")
    print(f"Output: {OUT_PATH}")
    print(f"TTA: {USE_TTA}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(TEST_CSV)
    print(f"\nPredicting {len(df):,} rows")

    # Pre-allocate output array; fill it view-by-view.
    all_preds = np.zeros((len(df), len(LABEL_COLS)), dtype=np.float32)

    for view in VIEWS:
        view_mask = (df["Frontal/Lateral"] == view).values
        view_indices = np.where(view_mask)[0]

        if len(view_indices) == 0:
            print(f"\n[{view}] no rows to predict, skipping")
            continue

        print(f"\n[{view}] {len(view_indices):,} rows")
        view_df = df.iloc[view_indices].reset_index(drop=True)

        t0 = time.time()
        view_preds = predict_for_view(view, view_df, device)
        print(f"  done in {(time.time() - t0) / 60:.1f} min")

        all_preds[view_indices] = view_preds

    # Any rows whose Frontal/Lateral was missing/unknown stay at zeros.
    unknown = (~df["Frontal/Lateral"].isin(VIEWS)).sum()
    if unknown > 0:
        print(f"\nWarning: {unknown} rows have unknown view, predictions left as 0")

    all_preds = np.clip(all_preds, CLIP_MIN, CLIP_MAX)

    out_df = df.copy()
    for i, col in enumerate(LABEL_COLS):
        out_df[col] = all_preds[:, i]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
