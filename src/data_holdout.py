"""
Data loading, cross validation for the copenlu/scientific-exaggeration-detection dataset.

Usage:
    One-time generate and save splits:
        python src/data_holdout.py

    Import in other modules:
        from src.data_holdout import get_pooled_df, get_fold_from_disk, get_test_from_disk
"""
import json
import numpy as np
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold, train_test_split
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

def load_hf_dataset(ds_name: str) -> DatasetDict:
    """
    Load hugging face dataset by name.

    :param ds_name: hugging face dataset identifier
    :type ds_name: str
    """
    return load_dataset(ds_name)


# =====================================================================
# Processing
# =====================================================================

def process_and_pool_data(train_data, test_data):
    """
    Convert HF splits to pandas, pool them, and encode labels.

    :param train_data: HF train split
    :param test_data: HF test split
    :return: pooled dataframe with integer-encoded labels
    :rtype: pd.DataFrame
    """
    train_df = train_data.to_pandas()
    test_df = test_data.to_pandas()

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    full_df["exaggeration_label"] = (
        full_df["exaggeration_label"]
        .astype(str)
        .str.strip()
        .replace(ENCODED_LABELS)
        .astype(int)
    )

    bad = full_df["exaggeration_label"].apply(lambda x: not isinstance(x, (int, np.integer)))
    assert not bad.any(), f"Unmapped labels: {full_df.loc[bad, 'exaggeration_label'].unique()}"

    return full_df


def get_pooled_df() -> pd.DataFrame:
    """
    Load dataset from HuggingFace, pool train+test, encode labels.
    Convenience function for use in notebooks and training scripts.

    :return: pooled dataframe with integer-encoded labels
    :rtype: pd.DataFrame
    """
    full_data = load_hf_dataset(DATASET_NAME)
    return process_and_pool_data(full_data["train"], full_data["test"])


# =====================================================================
# Holdout test data and make stratified KFold splits on training data
# =====================================================================

def make_train_dev_test_split(full_df, test_size=0.2, seed=7):
    """
    Create a held-out test set from the pooled dataframe.

    :param full_df: pooled dataframe
    :type full_df: pd.DataFrame
    :param test_size: fraction of data to hold out for test
    :type test_size: float
    :param seed: random state for reproducibility
    :type seed: int
    :return: (train_dev_indices, test_indices)
    """
    y = full_df["exaggeration_label"].astype(int)

    train_dev_idx, test_idx = train_test_split(
        np.arange(len(full_df)),
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    print(f"Created held-out split: train_dev={len(train_dev_idx)} test={len(test_idx)}")
    return train_dev_idx, test_idx


def strat_kfold_splits_train_dev(full_df, train_dev_idx, splits=5, seed=7):
    """
    Run stratified k-fold only on the train_dev subset.
    Returns folds as global indices into full_df.

    :param full_df: pooled dataframe
    :type full_df: pd.DataFrame
    :param train_dev_idx: indices of the train_dev subset
    :param splits: number of folds
    :type splits: int
    :param seed: random state for reproducibility
    :type seed: int
    :return: list of (train_global_indices, val_global_indices) tuples
    """
    train_dev_df = full_df.iloc[train_dev_idx].copy()

    y = train_dev_df["exaggeration_label"].astype(int)
    X = np.zeros(len(train_dev_df))

    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    print(f"Applying stratified {splits}-fold split")

    folds = []
    for i, (train_local, val_local) in enumerate(skf.split(X, y), start=1):
        train_global = train_dev_df.iloc[train_local].index.to_numpy()
        val_global = train_dev_df.iloc[val_local].index.to_numpy()

        print(f"Fold {i}: train={len(train_global)} val={len(val_global)}")
        folds.append((train_global, val_global))

    return folds


# =====================================================================
# Save Splits to Disk
# =====================================================================

def save_splits(full_df, train_dev_indices, test_indices, folds, output_dir=SPLITS_DIR, k=5, seed=7, test_size=0.2):
    """
    Save held-out test indices plus cross-fold validation indices to JSON.
    """
    label_counts = full_df["exaggeration_label"].value_counts().to_dict()

    train_dev_labels = full_df.iloc[train_dev_indices]["exaggeration_label"]
    test_labels = full_df.iloc[test_indices]["exaggeration_label"]

    splits = {
        "metadata": {
            "dataset": DATASET_NAME,
            "n_examples": len(full_df),
            "n_train_dev": len(train_dev_indices),
            "n_test": len(test_indices),
            "k": k,
            "seed": seed,
            "test_size": test_size,
            "label_map": DECODED_LABELS,
            "label_distribution": {
                DECODED_LABELS[lbl]: int(cnt) for lbl, cnt in sorted(label_counts.items())
            },
            "split_rationale": (
                "Pooled all labeled examples, created a stratified held-out test set, "
                "then ran stratified k-fold CV on the remaining train_dev data."
            ),
        },
        "train_dev_indices": train_dev_indices.tolist(),
        "test_indices": test_indices.tolist(),
        "train_dev_label_dist": {
            DECODED_LABELS[lbl]: int(cnt) for lbl, cnt in sorted(train_dev_labels.value_counts().items())
        },
        "test_label_dist": {
            DECODED_LABELS[lbl]: int(cnt) for lbl, cnt in sorted(test_labels.value_counts().items())
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
    splits_path = output_dir / f"splits_holdout20_k{k}_seed{seed}.json"
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
    splits_path = splits_dir / f"splits_holdout20_k{k}_seed{seed}.json"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"No splits at {splits_path}. Run save_splits() first."
        )
    with open(splits_path) as f:
        return json.load(f)


def get_fold_from_disk(full_df: pd.DataFrame, fold: int = None, k: int = 5, seed: int = 7):
    """
    Load fold train/val DataFrames from saved indices.

    full_df must be the same pooled DataFrame (same row order) that
    was passed to save_splits(). Guaranteed by: concat([train, test], ignore_index=True).

    :param full_df: pooled dataframe (same as passed to save_splits)
    :type full_df: pd.DataFrame
    :param fold: fold index (0 to k-1), or None for all folds
    :type fold: int or None
    :param k: number of folds
    :type k: int
    :param seed: random_state used when splits were generated
    :type seed: int
    """
    splits = load_splits(k=k, seed=seed)

    if fold is not None:
        fold_info = splits["folds"][fold]
        train_idx = fold_info["train_indices"]
        val_idx = fold_info["val_indices"]
        return full_df.iloc[train_idx].copy(), full_df.iloc[val_idx].copy()

    train_dfs, val_dfs = [], []
    for f in range(k):
        fold_info = splits["folds"][f]
        train_dfs.append(full_df.iloc[fold_info["train_indices"]].copy())
        val_dfs.append(full_df.iloc[fold_info["val_indices"]].copy())
    return train_dfs, val_dfs


def get_test_from_disk(full_df, k=5, seed=7):
    """
    Load held-out test dataframe from saved indices.

    :param full_df: pooled dataframe
    :type full_df: pd.DataFrame
    :param k: number of folds
    :type k: int
    :param seed: random_state used when splits were generated
    :type seed: int
    """
    splits = load_splits(k=k, seed=seed)
    test_idx = splits["test_indices"]
    return full_df.iloc[test_idx].copy()


def get_train_dev_from_disk(full_df, k=5, seed=7):
    """
    Load train_dev dataframe from saved indices.

    :param full_df: pooled dataframe
    :type full_df: pd.DataFrame
    :param k: number of folds
    :type k: int
    :param seed: random_state used when splits were generated
    :type seed: int
    """
    splits = load_splits(k=k, seed=seed)
    train_dev_idx = splits["train_dev_indices"]
    return full_df.iloc[train_dev_idx].copy()


# =====================================================================
# Sanity Check
# =====================================================================

def sanity_check(full_dataset, kfold_val, train_dev_indices, test_indices, fold_num=1, n_examples=3):
    """
    Validates kfold splits: leakage, distribution, sample rows, duplicates.

    :param full_dataset: pooled dataframe
    :type full_dataset: pd.DataFrame
    :param kfold_val: list of (train_index, val_index) tuples
    :param train_dev_indices: indices of train_dev subset
    :param test_indices: indices of held-out test set
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

    train_dev_labels = full_dataset.iloc[train_dev_indices]["exaggeration_label"]
    test_labels = full_dataset.iloc[test_indices]["exaggeration_label"]

    print("\nTrain/dev label counts:")
    print(train_dev_labels.value_counts(dropna=False).rename(index=label_map))
    print("\nTrain/dev label %:")
    print((train_dev_labels.value_counts(normalize=True) * 100).rename(index=label_map).round(2))

    print("\nHeld-out test label counts:")
    print(test_labels.value_counts(dropna=False).rename(index=label_map))
    print("\nHeld-out test label %:")
    print((test_labels.value_counts(normalize=True) * 100).rename(index=label_map).round(2))

    test_overlap = set(train_dev_indices).intersection(set(test_indices))
    assert len(test_overlap) == 0, f"Leakage between train_dev and test: {len(test_overlap)} overlapping indices"
    print("\nNo leakage: train_dev and test indices are disjoint.")

    print("====================================================================\n")


# =====================================================================
# CLI: Generate, verify, and save splits
# =====================================================================

if __name__ == "__main__":
    print(f"Loading dataset: {DATASET_NAME}")
    full_data = load_hf_dataset(DATASET_NAME)
    print(full_data["train"])

    full_df = process_and_pool_data(full_data["train"], full_data["test"])

    train_dev_indices, test_indices = make_train_dev_test_split(full_df, test_size=0.2, seed=7)
    kfold_val = strat_kfold_splits_train_dev(full_df, train_dev_indices, splits=5, seed=7)

    sanity_check(full_df, kfold_val, train_dev_indices, test_indices)

    save_splits(full_df, train_dev_indices, test_indices, kfold_val, k=5, seed=7, test_size=0.2)

    print("Verifying round-trip from disk...")
    loaded = load_splits(k=5, seed=7)

    for fold_idx, (train_idx, val_idx) in enumerate(kfold_val):
        assert loaded["folds"][fold_idx]["train_indices"] == train_idx.tolist(), f"Fold {fold_idx} train mismatch"
        assert loaded["folds"][fold_idx]["val_indices"] == val_idx.tolist(), f"Fold {fold_idx} val mismatch"

    assert loaded["train_dev_indices"] == train_dev_indices.tolist(), "train_dev indices mismatch"
    assert loaded["test_indices"] == test_indices.tolist(), "test indices mismatch"

    print(f"Round-trip OK — {len(kfold_val)} folds + held-out test verified against {SPLITS_DIR / 'splits_holdout20_k5_seed7.json'}")