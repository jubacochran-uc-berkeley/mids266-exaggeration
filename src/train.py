"""
Training pipeline for exaggeration detection experiments.

Usage:
    python src/train.py --config configs/roberta_full.yaml
"""
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from pyprojroot import here

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report

from src.config import load_config
from src.data_holdout import (
    process_and_pool_data,
    load_hf_dataset,
    get_fold_from_disk,
    get_test_from_disk,
    DATASET_NAME,
)

# =====================================================================
# CLI
# =====================================================================

#args parse to build CLI interfaces with python
#This captures the config from the cli
def parse_args():
    """
    Parse command line arguments
    https://realpython.com/command-line-interfaces-python-argparse/
    """
    parser = argparse.ArgumentParser(description="Train exaggeration detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    return parser.parse_args()

# =====================================================================
# Tokenization
# =====================================================================

def tokenize_fold(df, tokenizer, max_length):

    ds = Dataset.from_pandas(df, preserve_index=False)

    def _tokenize(examples):
        return tokenizer(examples["abstract_conclusion"],
        examples["press_release_conclusion"], 
        truncation=True, padding="max_length", 
        max_length=maxlength
        )
    
    cols_to_remove = [c for c in ds.column_names if c != "exaggeration_label"]
    ds = ds.map(_tokenizer, batched=True, remove_columns=cols_to_remove)
    ds = ds.rename_column("exaggeration_label", "labels")
    ds.set_format("torch")
    return ds

# =====================================================================
# Metrics
# =====================================================================

def compute_metrics(eval_pred):
    """ Compute macro f1 from trainers eval predictions"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"macro_f1": f1_score(labels, preds, average="macro")}

