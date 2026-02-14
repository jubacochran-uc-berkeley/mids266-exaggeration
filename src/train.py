"""train module."""
from src.util import get_pooled_df
from src.data import get_fold_from_disk

full_df = get_pooled_df()
train_fold, val_fold = get_fold_from_disk(full_df, fold=0, k=5, seed=7)