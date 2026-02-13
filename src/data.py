"""
Data loading, cross vallidation for the copenlu/scientific-exaggeration-detection dataset.

useage:
    one time generate and save splits

"""
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import StratifiedKFold


DATASET_NAME = "copenlu/scientific-exaggeration-detection"
#Loading
print(f"Loading dataset, {DATASET_NAME}")

def load_hf_dataset (ds_name:str) -> DatasetDict:
    """
    Load hugging face dataset by name
    
    :param ds_name: Description
    """
    return load_dataset(ds_name)

full_data = load_hf_dataset(DATASET_NAME)
print(full_data['train'])

#Processing

def process_full_data(train_data: Dataset, test_data: Dataset):
    """
    Convert HF Dataset splits to pandas DataFrames.
    """
    train_df = train_data.to_pandas()
    test_df = test_data.to_pandas()
    return train_df, test_df

train_df, test_df = process_full_data(full_data['train'],full_data['test'])
print(train_df.head())



