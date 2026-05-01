from pathlib import Path
from multiprocessing import Pool
import time

import pandas as pd
from PIL import Image, ImageOps


DATA_ROOT = Path("/resnick/groups/CS156b/from_central/data")
TRAIN_CSV = DATA_ROOT / "student_labels" / "train2023.csv"
OUT_ROOT = Path("/resnick/groups/CS156b/from_central/2026/JSC/cache_320")

IMAGE_SIZE = 320
NUM_WORKERS = 8



def get_full_path(path_value):
    p = Path(str(path_value))
    if p.is_absolute():
        return p

    data_path = DATA_ROOT / p
    if data_path.exists():
        return data_path

    return DATA_ROOT / "train" / p


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


def process_one(rel_path):
    src = get_full_path(rel_path)
    dst = OUT_ROOT / Path(rel_path).with_suffix(".png")

    if dst.exists():
        return "skip", rel_path, None

    try:
        with Image.open(src) as img:
            img = img.convert("L")
            img = pad_to_square(img)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.BILINEAR)

            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, format="PNG")

        return "ok", rel_path, None

    except Exception as e:
        return "err", rel_path, repr(e)


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TRAIN_CSV)
    paths = df["Path"].tolist()

    print(f"Preprocessing {len(paths):,} images")
    print(f"Output: {OUT_ROOT}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Workers: {NUM_WORKERS}")

    start = time.time()
    n_ok = 0
    n_skip = 0
    n_err = 0

    err_log = open(OUT_ROOT / "_errors.txt", "w")

    with Pool(NUM_WORKERS) as pool:
        for i, (status, rel_path, err) in enumerate(
            pool.imap_unordered(process_one, paths, chunksize=32),
            start=1,
        ):
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                err_log.write(f"{rel_path}\t{err}\n")

            if i % 1000 == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                eta_min = (len(paths) - i) / rate / 60
                print(
                    f"  {i:,}/{len(paths):,}  "
                    f"ok={n_ok} skip={n_skip} err={n_err}  "
                    f"{rate:.1f} img/s  eta {eta_min:.1f} min",
                    flush=True,
                )

    err_log.close()

    print(f"\nDone in {(time.time() - start) / 60:.1f} min")
    print(f"  ok:      {n_ok:,}")
    print(f"  skipped: {n_skip:,}  (already on disk)")
    print(f"  errors:  {n_err:,}")


if __name__ == "__main__":
    main()
