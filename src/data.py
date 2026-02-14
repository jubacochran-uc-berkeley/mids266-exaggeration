"""
Data loading, cross validation for the copenlu/scientific-exaggeration-detection dataset.

useage:
    one time generate and save splits
    python src/data.py

"""
import json
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold
from pyprojroot import here
import pandas as pd
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 120)
pd.set_option("display.expand_frame_repr", False)


DATASET_NAME = "copenlu/scientific-exaggeration-detection"
SPLITS_DIR = here("data/splits")

ENCODED_LABELS = {"downplays": 0, "same": 1, "exaggerates": 2}
DECODED_LABELS = {v: k for k, v in ENCODED_LABELS.items()}


# =====================================================================
# Loading
# =====================================================================
print(f"Loading dataset, {DATASET_NAME}")

def load_hf_dataset(ds_name: str) -> DatasetDict:
    """
    Load hugging face dataset by name

    :param ds_name: hugging face dataset identifier
    """
    return load_dataset(ds_name)

full_data = load_hf_dataset(DATASET_NAME)
print(full_data['train'])


# =====================================================================
# Processing
# =====================================================================

def process_full_data(train_data: Dataset, test_data: Dataset):
    """
    Convert HF Dataset train/test splits pandas DataFrames and encodes. The splits are preserved as is.

    :param train_data: HF train split
    :param test_data: HF test split
    """
    train_df = train_data.to_pandas()
    train_df["exaggeration_label"] = (
        train_df["exaggeration_label"].astype(str).str.strip().replace(ENCODED_LABELS).astype(int)
    )
    bad_train = train_df["exaggeration_label"].apply(lambda x: not isinstance(x, (int, np.integer)))
    assert not bad_train.any(), f"Unmapped labels in train: {train_df.loc[bad_train, 'exaggeration_label'].unique()}"

    test_df = test_data.to_pandas()
    test_df["exaggeration_label"] = (
        test_df["exaggeration_label"].astype(str).str.strip().replace(ENCODED_LABELS).astype(int)
    )
    bad_test = test_df["exaggeration_label"].apply(lambda x: not isinstance(x, (int, np.integer)))
    assert not bad_test.any(), f"Unmapped labels in test: {test_df.loc[bad_test, 'exaggeration_label'].unique()}"

    return train_df, test_df

train_df, test_df = process_full_data(full_data['train'], full_data['test'])


# =====================================================================
# Stratified KFold Splits
# =====================================================================

def strat_kfold_splits_all(df_train: pd.DataFrame, df_test: pd.DataFrame, splits: int = 5):
    """
    Docstring for strat_kfold_splits_all

    :param df_train: encoded train dataframe
    :type df_train: pd.DataFrame
    :param df_test: encoded test dataframe
    :type df_test: pd.DataFrame
    :param splits: number of kfold splits
    :type splits: int
    """
    # combining so we train on all labels using crossfold validation
    # we're using stratified kfold to preserve the population variance of labels across folds
    full_df = pd.concat([df_train, df_test], ignore_index=True)

    y = full_df["exaggeration_label"].astype(int)
    X = np.zeros(len(full_df))

    print(full_df["exaggeration_label"].dtype)
    print(full_df["exaggeration_label"].unique())

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=7)
    print(f"applying split of: {splits}")

    folds = []
    for i, (train_index, val_index) in enumerate(skf.split(X, y), start=1):
        print(f"Fold {i}: train={len(train_index)} val={len(val_index)}")
        folds.append((train_index, val_index))

    return full_df, folds

full_dataset, kfold_val = strat_kfold_splits_all(train_df, test_df)


# =====================================================================
# Save Splits to Disk
# =====================================================================

def save_splits(full_df: pd.DataFrame, folds: list, output_dir=SPLITS_DIR, k: int = 5, seed: int = 7):
    """
    Save fold indices and metadata to a JSON file. Indices only — small, diffable, version-controllable.

    :param full_df: pooled dataframe with encoded labels
    :type full_df: pd.DataFrame
    :param folds: list of (train_index, val_index) tuples from StratifiedKFold
    :type folds: list
    :param output_dir: where to write the JSON
    :param k: number of folds
    :type k: int
    :param seed: random_state used in StratifiedKFold
    :type seed: int
    """
    label_counts = full_df["exaggeration_label"].value_counts().to_dict()

    splits = {
        "metadata": {
            "dataset": DATASET_NAME,
            "n_examples": len(full_df),
            "k": k,
            "seed": seed,
            "label_map": DECODED_LABELS,
            "label_distribution": {
                DECODED_LABELS[lbl]: int(cnt) for lbl, cnt in sorted(label_counts.items())
            },
            "pooling_rationale": (
                "HF train/test split (100/563) was designed for MT-PET "
                "semi-supervised learning. Pooled and re-split for standard "
                "supervised fine-tuning with stratified k-fold CV."
            ),
        },
        "folds": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        y_train = full_df.iloc[train_idx]["exaggeration_label"]
        y_val = full_df.iloc[val_idx]["exaggeration_label"]

        fold_info = {
            "fold": fold_idx,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "train_label_dist": {
                DECODED_LABELS[lbl]: int(cnt) for lbl, cnt in sorted(y_train.value_counts().items())
            },
            "val_label_dist": {
                DECODED_LABELS[lbl]: int(cnt) for lbl, cnt in sorted(y_val.value_counts().items())
            },
        }
        splits["folds"].append(fold_info)

    output_dir.mkdir(parents=True, exist_ok=True)
    splits_path = output_dir / f"splits_k{k}_seed{seed}.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Splits saved to {splits_path}")
    return splits_path


# =====================================================================
# Load Splits from Disk
# =====================================================================

def load_splits(k: int = 5, seed: int = 7, splits_dir=SPLITS_DIR) -> dict:
    """
    Load previously saved splits JSON.

    :param k: number of folds
    :type k: int
    :param seed: random_state used when splits were generated
    :type seed: int
    :param splits_dir: directory containing the splits JSON
    """
    splits_path = splits_dir / f"splits_k{k}_seed{seed}.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"No splits at {splits_path}. Run save_splits() first."
        )
    with open(splits_path) as f:
        return json.load(f)


def get_fold_from_disk(full_df: pd.DataFrame, fold: int, k: int = 5, seed: int = 7):
    """
    Load a specific fold's train/val DataFrames from saved indices.

    full_df must be the same pooled DataFrame (same row order) that
    was passed to save_splits(). Guaranteed by: concat([train, test], ignore_index=True).

    :param full_df: pooled dataframe (same as passed to save_splits)
    :type full_df: pd.DataFrame
    :param fold: fold index (0 to k-1)
    :type fold: int
    :param k: number of folds
    :type k: int
    :param seed: random_state used when splits were generated
    :type seed: int
    """
    splits = load_splits(k=k, seed=seed)
    fold_info = splits["folds"][fold]

    train_idx = fold_info["train_indices"]
    val_idx = fold_info["val_indices"]

    return full_df.iloc[train_idx].copy(), full_df.iloc[val_idx].copy()


# =====================================================================
# Sanity Check
# =====================================================================

def sanity_check(full_dataset: pd.DataFrame, kfold_val, fold_num: int = 1, n_examples: int = 3):
    """
    Validates kfold splits: leakage, distribution, sample rows, duplicates.

    :param full_dataset: pooled dataframe
    :type full_dataset: pd.DataFrame
    :param kfold_val: list of (train_index, val_index) tuples
    :param fold_num: which fold to show sample rows from
    :type fold_num: int
    :param n_examples: how many sample rows to print
    :type n_examples: int
    """
    label_map = DECODED_LABELS

    print("\n==================== SANITY CHECK: KFOLD SPLITS ====================")
    print(f"Total samples: {len(full_dataset)} | Num folds: {len(kfold_val)}")
    print(f"label dtype: {full_dataset['exaggeration_label'].dtype}")
    print("unique labels:", sorted(full_dataset["exaggeration_label"].dropna().unique().tolist()))

    print("\nOverall label counts:")
    print(full_dataset["exaggeration_label"].value_counts(dropna=False).rename(index=label_map))
    print("\nOverall label %:")
    print((full_dataset["exaggeration_label"].value_counts(normalize=True) * 100).rename(index=label_map).round(2))

    for i, (train_idx, val_idx) in enumerate(kfold_val, start=1):
        y_train = full_dataset.loc[train_idx, "exaggeration_label"]
        y_val = full_dataset.loc[val_idx, "exaggeration_label"]
        print(f"\nFold {i}")
        print(f"  Train size={len(train_idx)} | Val size={len(val_idx)}")
        print("  Train label %:")
        print((y_train.value_counts(normalize=True) * 100).rename(index=label_map).sort_index().round(2))
        print("  Val label %:")
        print((y_val.value_counts(normalize=True) * 100).rename(index=label_map).sort_index().round(2))

    train_idx, val_idx = kfold_val[fold_num - 1]
    cols_to_view = [c for c in ["press_release_conclusion", "abstract_conclusion", "exaggeration_label"] if c in full_dataset.columns]

    print(f"\nSample rows from Fold {fold_num} (val), n={n_examples}:")
    for _, row in full_dataset.loc[val_idx, cols_to_view].head(n_examples).iterrows():
        for col in cols_to_view:
            val = label_map.get(row[col], row[col]) if col == "exaggeration_label" else row[col]
            print(f"  {col}: {str(val)[:200]}")
        print("  ---")

    for i, (train_idx, val_idx) in enumerate(kfold_val, start=1):
        overlap = set(train_idx).intersection(set(val_idx))
        assert len(overlap) == 0, f"Leakage in fold {i}: {len(overlap)} overlapping indices"
    print("\nNo leakage: train/val indices are disjoint in every fold.")

    if all(c in full_dataset.columns for c in ["press_release_conclusion", "abstract_conclusion"]):
        dup_pairs = full_dataset.duplicated(subset=["press_release_conclusion", "abstract_conclusion"], keep=False).sum()
        print("Duplicate sentence-pairs:", int(dup_pairs))
    else:
        print("Duplicate sentence-pairs: skipped (missing text columns)")

    print("====================================================================\n")

sanity_check(full_dataset, kfold_val)


# =====================================================================
# Save & Verify
# =====================================================================

save_splits(full_dataset, kfold_val, k=5, seed=7)

# round-trip verify: load back and confirm indices match
print("Verifying round-trip from disk...")
loaded = load_splits(k=5, seed=7)
for fold_idx, (train_idx, val_idx) in enumerate(kfold_val):
    assert loaded["folds"][fold_idx]["train_indices"] == train_idx.tolist(), f"Fold {fold_idx} train mismatch"
    assert loaded["folds"][fold_idx]["val_indices"] == val_idx.tolist(), f"Fold {fold_idx} val mismatch"
print(f"Round-trip OK — {len(kfold_val)} folds verified against {SPLITS_DIR / 'splits_k5_seed7.json'}")