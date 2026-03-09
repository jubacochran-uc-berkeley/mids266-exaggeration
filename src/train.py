
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

def tokenize_df(df, tokenizer, max_length):

    ds = Dataset.from_pandas(df, preserve_index=False)

    def _tokenize(examples):
        return tokenizer(examples["abstract_conclusion"],
        examples["press_release_conclusion"], 
        truncation=True, padding="max_length", 
        max_length=max_length
        )
    
    cols_to_remove = [c for c in ds.column_names if c != "exaggeration_label"]
    ds = ds.map(_tokenize, batched=True, remove_columns=cols_to_remove)
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

# =====================================================================
# Fold Training
# =====================================================================

def train_single_fold(fold_idx, train_ds, val_ds, config):
    """
    Train and evaluate a single fold.

    :param fold_idx: fold number (0 to k-1)
    :type fold_idx: int
    :param train_ds: tokenized training dataset for this fold
    :type train_ds: Dataset
    :param val_ds: tokenized validation dataset for this fold
    :type val_ds: Dataset
    :param config: merged experiment config
    :type config: dict
    :return: tuple of (fold_result dict, trained Trainer instance)
    :rtype: tuple
    """

    print(f"\n{'='*60}")
    print(f"  Fold {fold_idx} | {config['model_name']} | {config['finetune_method']}")
    print(f"{'='*60}")

    #New model per fold
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"]
        num_labels=3,
    )

    fold_output_dir = srt(here(config["output_dir"]) / f"fold-{fold_idx}")

    #Training Args
    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        eval_strategy=config["eval_strategy"],
        save_strategy=config["save_strategy"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        num_train_epochs=config["num_train_epochs"],
        load_best_model_at_end=config["load_best_model_at_end"],
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=config["greater_is_better"],
        fp16=config["fp16"],
        seed=config["seed"],
        logging_steps=10,
        report_to=config["report_to"],
        save_total_limit=1,
    )

    #Trainer
    trainer = Trainer(
        model = model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=config["early_stopping_patience"]
        )],
    )

    #Now training
    trainer.train()
    eval_results = trainer.evaluate()

    
    





# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args.config)
