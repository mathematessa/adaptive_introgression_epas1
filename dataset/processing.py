from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

DATASET_DIR = Path("dataset5")
INPUT_FILE = DATASET_DIR / "ml_dataset_full.csv"
OUTPUT_DIR = DATASET_DIR / "learning_ready"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCALER_FILE = OUTPUT_DIR / "standard_scaler.joblib"
METADATA_FILE = OUTPUT_DIR / "metadata.json"
TRAIN_CSV = OUTPUT_DIR / "train.csv"
VAL_CSV = OUTPUT_DIR / "val.csv"
TEST_CSV = OUTPUT_DIR / "test.csv"
X_TRAIN_NPY = OUTPUT_DIR / "X_train.npy"
X_VAL_NPY = OUTPUT_DIR / "X_val.npy"
X_TEST_NPY = OUTPUT_DIR / "X_test.npy"
Y_TRAIN_NPY = OUTPUT_DIR / "y_train.npy"
Y_VAL_NPY = OUTPUT_DIR / "y_val.npy"
Y_TEST_NPY = OUTPUT_DIR / "y_test.npy"

FEATURE_COLUMNS = [
    "Chromosome",
    "Position",
    "Frequency in Tibetians",
    "Mean non-introgressed (0) tract length",
    "Mean introgressed (1) tract length",
    "Variance of non-introgressed (0) tract length",
    "Variance of introgressed tract length (1)",
]

ENGINEERED_COLUMNS = [
    "Chromosome",
    "Position_norm",
    "Frequency_logit",
    "Mean_nonintr_norm",
    "Mean_intr_norm",
    "Var_nonintr_norm_sqrt",
    "Var_intr_norm_sqrt",
]

TARGET_COLUMN = "neutrality_time"
GROUP_COLUMN = "sim_id"

GENOME_LENGTH = 3e8
MAX_TRACT_LENGTH = 300_000.0
MAX_TRACT_VAR = MAX_TRACT_LENGTH ** 2
LOGIT_EPS = 1e-6

TIME_TO_CLASS = {
    0: 0, 50: 1, 100: 2, 200: 3, 500: 4,
    750: 5, 1000: 6, 1300: 7, 1600: 8, 1900: 9,
}

TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TRAIN = 0.20
RANDOM_STATE = 42

def check_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["Chromosome"] = df["Chromosome"].astype(np.float64)
    out["Position_norm"] = (df["Position"].astype(np.float64) / GENOME_LENGTH)
    freq = df["Frequency in Tibetians"].astype(np.float64).clip(LOGIT_EPS, 1 - LOGIT_EPS)
    out["Frequency_logit"] = np.log(freq / (1.0 - freq))
    out["Mean_nonintr_norm"] = (df["Mean non-introgressed (0) tract length"].astype(np.float64) / MAX_TRACT_LENGTH)
    out["Mean_intr_norm"] = (df["Mean introgressed (1) tract length"].astype(np.float64) / MAX_TRACT_LENGTH)
    out["Var_nonintr_norm_sqrt"] = np.sqrt(
        df["Variance of non-introgressed (0) tract length"].astype(np.float64).clip(0)
        / MAX_TRACT_VAR
    )
    out["Var_intr_norm_sqrt"] = np.sqrt(
        df["Variance of introgressed tract length (1)"].astype(np.float64).clip(0)
        / MAX_TRACT_VAR
    )
    return out


def make_group_split(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
    random_state: int,
):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return train_idx, test_idx


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    check_required_columns(df, [GROUP_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS)
    before_drop = len(df)
    df = df.dropna(subset=[GROUP_COLUMN, TARGET_COLUMN] + FEATURE_COLUMNS).copy()
    after_drop = len(df)
    print(f"Loaded rows:                    {before_drop}")
    print(f"Rows after dropping NaNs:       {after_drop}")
    if df.empty:
        raise ValueError("Dataset is empty after dropping missing values.")
    unknown_times = sorted(set(df[TARGET_COLUMN].unique()) - set(TIME_TO_CLASS.keys()))
    if unknown_times:
        raise ValueError(
            f"Found target values not in TIME_TO_CLASS mapping: {unknown_times}"
        )
    df["target_class"] = df[TARGET_COLUMN].map(TIME_TO_CLASS).astype(np.int64)
    X_eng = engineer_features(df)
    print("\nEngineered feature stats (pre-scaler):")
    print(X_eng.describe().T[["mean", "std", "min", "max"]].to_string())
    y      = df["target_class"].copy()
    groups = df[GROUP_COLUMN].copy()
    train_val_idx, test_idx = make_group_split(X_eng, y, groups, TEST_SIZE, RANDOM_STATE)
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test      = df.iloc[test_idx].reset_index(drop=True)
    X_eng_tv = engineer_features(df_train_val)
    y_tv     = df_train_val["target_class"]
    g_tv     = df_train_val[GROUP_COLUMN]
    train_idx_rel, val_idx_rel = make_group_split(X_eng_tv, y_tv, g_tv, VAL_SIZE_WITHIN_TRAIN, RANDOM_STATE)
    df_train = df_train_val.iloc[train_idx_rel].reset_index(drop=True)
    df_val   = df_train_val.iloc[val_idx_rel].reset_index(drop=True)
    X_train_eng = engineer_features(df_train).to_numpy(dtype=np.float64)
    X_val_eng   = engineer_features(df_val).to_numpy(dtype=np.float64)
    X_test_eng  = engineer_features(df_test).to_numpy(dtype=np.float64)
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train_eng)
    X_val   = scaler.transform(X_val_eng)
    X_test  = scaler.transform(X_test_eng)
    y_train = df_train["target_class"].to_numpy(dtype=np.int64)
    y_val   = df_val["target_class"].to_numpy(dtype=np.int64)
    y_test  = df_test["target_class"].to_numpy(dtype=np.int64)
    for df_split, X_scaled, out_path in [
        (df_train, X_train, TRAIN_CSV),
        (df_val,   X_val,   VAL_CSV),
        (df_test,  X_test,  TEST_CSV),
    ]:
        df_out = df_split[[GROUP_COLUMN, TARGET_COLUMN, "target_class"]].copy()
        for i, col in enumerate(ENGINEERED_COLUMNS):
            df_out[col] = X_scaled[:, i]
        df_out.to_csv(out_path, index=False)

    np.save(X_TRAIN_NPY, X_train)
    np.save(X_VAL_NPY,   X_val)
    np.save(X_TEST_NPY,  X_test)
    np.save(Y_TRAIN_NPY, y_train)
    np.save(Y_VAL_NPY,   y_val)
    np.save(Y_TEST_NPY,  y_test)
    joblib.dump(scaler, SCALER_FILE)
    metadata = {
        "input_file": str(INPUT_FILE),
        "raw_feature_columns": FEATURE_COLUMNS,
        "engineered_feature_columns": ENGINEERED_COLUMNS,
        "feature_transforms": {
            "Chromosome":            "none",
            "Position_norm":         f"position / {GENOME_LENGTH:.0e}",
            "Frequency_logit":       "log(p / (1-p)), clipped at 1e-6",
            "Mean_nonintr_norm":     f"mean / {MAX_TRACT_LENGTH:.0f}",
            "Mean_intr_norm":        f"mean / {MAX_TRACT_LENGTH:.0f}",
            "Var_nonintr_norm_sqrt": f"sqrt(var / {MAX_TRACT_VAR:.2e})",
            "Var_intr_norm_sqrt":    f"sqrt(var / {MAX_TRACT_VAR:.2e})",
        },
        "target_column_raw": TARGET_COLUMN,
        "target_column_processed": "target_class",
        "group_column": GROUP_COLUMN,
        "time_to_class": TIME_TO_CLASS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "val_size_within_train": VAL_SIZE_WITHIN_TRAIN,
        "n_rows_total":  int(len(df)),
        "n_rows_train":  int(len(df_train)),
        "n_rows_val":    int(len(df_val)),
        "n_rows_test":   int(len(df_test)),
        "n_simulations_total": int(df[GROUP_COLUMN].nunique()),
        "n_simulations_train": int(df_train[GROUP_COLUMN].nunique()),
        "n_simulations_val":   int(df_val[GROUP_COLUMN].nunique()),
        "n_simulations_test":  int(df_test[GROUP_COLUMN].nunique()),
        "class_distribution_train": df_train["target_class"].value_counts().sort_index().to_dict(),
        "class_distribution_val":   df_val["target_class"].value_counts().sort_index().to_dict(),
        "class_distribution_test":  df_test["target_class"].value_counts().sort_index().to_dict(),
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved train table:  {TRAIN_CSV}")
    print(f"Saved val table:    {VAL_CSV}")
    print(f"Saved test table:   {TEST_CSV}")
    print(f"Saved scaler:       {SCALER_FILE}")
    print(f"Saved metadata:     {METADATA_FILE}")
    print(f"Saved arrays:       {OUTPUT_DIR}")
    print("\nShapes:")
    print(f"  X_train: {X_train.shape},  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")
    print("\nPost-scaler feature stats (train):")
    df_scaled = pd.DataFrame(X_train, columns=ENGINEERED_COLUMNS)
    print(df_scaled.describe().T[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
