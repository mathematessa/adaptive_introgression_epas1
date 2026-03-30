from __future__ import annotations

from pathlib import Path
import pandas as pd

DATASET_DIR = Path("dataset5")
LABELS_FILE = DATASET_DIR / "labels.csv"
OUTPUT_FULL = DATASET_DIR / "ml_dataset_full.csv"
OUTPUT_X = DATASET_DIR / "ml_dataset_X.csv"
OUTPUT_Y = DATASET_DIR / "ml_dataset_y.csv"
FEATURE_COLUMNS = [
    "Chromosome",
    "Position",
    "Frequency in Tibetians",
    "Mean non-introgressed (0) tract length",
    "Mean introgressed (1) tract length",
    "Variance of non-introgressed (0) tract length",
    "Variance of introgressed tract length (1)",
]

TARGET_COLUMN = "neutrality_time"

def main() -> None:
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")
    labels = pd.read_csv(LABELS_FILE)
    if labels.empty:
        raise ValueError("labels.csv is empty.")
    required_label_cols = {"sim_id", "file", TARGET_COLUMN}
    missing_label_cols = required_label_cols - set(labels.columns)
    if missing_label_cols:
        raise ValueError(f"labels.csv is missing required columns: {sorted(missing_label_cols)}")
    frames: list[pd.DataFrame] = []
    for _, row in labels.iterrows():
        sim_id = row["sim_id"]
        csv_name = row["file"]
        feature_file = DATASET_DIR / csv_name
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")
        df = pd.read_csv(feature_file)
        missing_feature_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing_feature_cols:
            raise ValueError(f"{feature_file} is missing required feature columns: {missing_feature_cols}")
        df = df[FEATURE_COLUMNS].copy()
        df["sim_id"] = sim_id
        df["trees_file"] = row["trees_file"] if "trees_file" in row.index else None
        df["sel_coeff"] = row["sel_coeff"] if "sel_coeff" in row.index else None
        df["admix_prop"] = row["admix_prop"] if "admix_prop" in row.index else None
        df[TARGET_COLUMN] = row[TARGET_COLUMN]
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)
    ordered_cols = [
        "sim_id",
        "trees_file",
        "sel_coeff",
        "admix_prop",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]
    dataset = dataset[ordered_cols]
    X = dataset[FEATURE_COLUMNS].copy()
    y = dataset[[TARGET_COLUMN]].copy()
    dataset.to_csv(OUTPUT_FULL, index=False)
    X.to_csv(OUTPUT_X, index=False)
    y.to_csv(OUTPUT_Y, index=False)
    print(f"Saved full dataset to: {OUTPUT_FULL}")
    print(f"Saved X to: {OUTPUT_X}")
    print(f"Saved y to: {OUTPUT_Y}")
    print(f"Rows: {len(dataset)}")
    print(f"Number of simulations: {dataset['sim_id'].nunique()}")


if __name__ == "__main__":
    main()
