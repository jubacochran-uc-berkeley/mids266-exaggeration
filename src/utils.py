"""utils module."""
# src/util.py

import pandas as pd
from src.data import load_hf_dataset, process_full_data, DATASET_NAME


def get_pooled_df() -> pd.DataFrame:
    """Reconstruct the pooled DataFrame used to generate splits."""
    full_data = load_hf_dataset(DATASET_NAME)
    train_df, test_df = process_full_data(full_data['train'], full_data['test'])
    return pd.concat([train_df, test_df], ignore_index=True)