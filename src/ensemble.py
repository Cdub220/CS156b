from pathlib import Path

import pandas as pd

from dataset import LABEL_COLS


# Each entry: (path-to-predictions.csv, weight). Weights are renormalized
# to sum to 1 automatically. Equal weights = simple average.
INPUTS = [
    (Path.home() / "Downloads/20260501_120326_submission1.csv", 1.0),  # v1
    (Path.home() / "Downloads/predictions_(4).csv",             1.0),  # v4
    (Path.home() / "Downloads/predictions(4)res.csv",           1.0),  # v5
    (Path.home() / "Downloads/predictions(5)dense.csv",         1.0),  # v6
    (Path.home() / "Downloads/predictions_v7_clahe.csv",        1.0),  # v7 CLAHE
]


OUT_PATH = Path.home() / "Downloads/predictions_ensemble8.csv"


def main():
    print(f"Ensembling {len(INPUTS)} models:")
    total_w = sum(w for _, w in INPUTS)
    weights = [w / total_w for _, w in INPUTS]
    for (path, _), w in zip(INPUTS, weights):
        print(f"  weight {w:.3f}  {path}")

    dfs = [pd.read_csv(path) for path, _ in INPUTS]

    n = len(dfs[0])
    for i, df in enumerate(dfs[1:], start=1):
        if len(df) != n:
            raise ValueError(f"Row count mismatch: file 0 has {n}, file {i} has {len(df)}")

    # If all files have an Id column, double-check they line up.
    if all("Id" in df.columns for df in dfs):
        first_ids = dfs[0]["Id"].values
        for i, df in enumerate(dfs[1:], start=1):
            if not (df["Id"].values == first_ids).all():
                raise ValueError(f"Id columns don't match between file 0 and file {i}")
        print(f"Verified Id alignment for {n:,} rows")

    out_df = dfs[0].copy()
    for col in LABEL_COLS:
        for df in dfs:
            if col not in df.columns:
                raise ValueError(f"Missing pathology column {col!r}")

        # Weighted sum across all input files for this pathology.
        combined = sum(w * df[col] for df, w in zip(dfs, weights))
        out_df[col] = combined

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")

    # Sanity: mean prediction per pathology from each model and the ensemble.
    print("\nPer-pathology mean prediction (one column per input, then ensemble):")
    header = "  " + "Pathology".ljust(30) + "  " + "  ".join(
        f"M{i}".rjust(8) for i in range(len(dfs))
    ) + "       ENS"
    print(header)
    for col in LABEL_COLS:
        means = [df[col].mean() for df in dfs]
        ens = out_df[col].mean()
        means_str = "  ".join(f"{m:+.4f}" for m in means)
        print(f"  {col:30s}  {means_str}    {ens:+.4f}")


if __name__ == "__main__":
    main()
