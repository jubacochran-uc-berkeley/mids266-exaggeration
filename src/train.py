
"""
Training pipeline for exaggeration detection experiments.

Usage:
    python src/train.py --config configs/roberta_full.yaml
"""
import torch
import argparse
import json
import numpy as np
from datetime import datetime
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
    get_pooled_df,
    get_fold_from_disk,
    get_test_from_disk,
    DECODED_LABELS,
)
from peft import LoraConfig, get_peft_model, TaskType

# =====================================================================
# CLI
# =====================================================================

#args parse to build CLI interfaces with python
#This captures the config from the cli
def parse_args():
    parser = argparse.ArgumentParser(description="Train exaggeration detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--multi-seed", action="store_true", help="Run multi-seed statistical evaluation")
    parser.add_argument("--n-runs", type=int, default=100, help="Number of seed runs (default: 100)")
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
        config["model_name"],
        num_labels=3
    )

    if config["finetune_method"] == "lora":
        model = apply_lora(model, config)
    elif config["finetune_method"] == "frozen":
        model = freeze_base(model)

    #Parameter count
    #https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    #flushing memory stats
    torch.cuda.reset_peak_memory_stats()

    fold_output_dir = str(here(config["output_dir"]) / f"fold-{fold_idx}")

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

    # Peak memory after training
    peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    print(f"Peak GPU memory: {peak_memory_mb:.0f} MB")

    # Detailed predictions for classification report
    predictions = trainer.predict(val_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    report = classification_report(
        labels, preds,
        target_names=list(DECODED_LABELS.values()),
        output_dict=True,
    )
    #Gathering information for Lora comparison and model comparision
    fold_result = {
        "fold": fold_idx,
        "macro_f1": eval_results["eval_macro_f1"],
        "eval_loss": eval_results["eval_loss"],
        "classification_report": report,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "peak_gpu_memory_mb": round(peak_memory_mb,1),
        "total_training_steps": trainer.state.global_step, #total gradient changes, delta gradient
        "epochs_completed": trainer.state.epoch, # number of steps before early stopping
        "total_parameter_updates": trainer.state.global_step * trainable_params, #full finetuning gradient update
    }

    print(f"Fold {fold_idx} macro_f1: {eval_results['eval_macro_f1']:.4f}")
    print(classification_report(labels, preds, target_names=list(DECODED_LABELS.values())))

    return fold_result, trainer
    
# =====================================================================
# Test evaluation
# =====================================================================

def evaluate_on_test(trainer, test_ds):
    """
    Evaluate a trained model on the held-out test set.

    :param trainer: trained Trainer instance (best fold model)
    :param test_ds: tokenized held-out test dataset
    :type test_ds: Dataset
    :return: dictionary with test set results
    :rtype: dict
    """

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    macro_f1 = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, target_names=list(DECODED_LABELS.values()),
                                   output_dict=True,
                                   )
    
    print(f"\nHeld-out test macro_f1: {macro_f1:.4f}")
    print(classification_report(labels, preds, target_names=list(DECODED_LABELS.values())))

    return {
        "macro_f1": macro_f1,
        "classification_report": report,
    }
    
# =====================================================================
# Experiment Orchestrator
# =====================================================================

def run_experiment(config_path):
    """
    Run a full experiment: train across all folds, evaluate on hold out test.

    :param config_path: path to experiment YAML config
    :type config_path: str
    """
    # --- Step 1: Load config ---
    config = load_config(config_path)
    print(f"\nExperiment: {config['model_name']} | {config['finetune_method']}")
    print(f"Config: {config_path}")

    # --- Step 2: Load data ---
    print("\nLoading data...")
    full_df = get_pooled_df()
    train_folds, val_folds = get_fold_from_disk(full_df, k=config["cv_k"], seed=config["cv_seed"])
    test_df = get_test_from_disk(full_df, k=config["cv_k"], seed=config["cv_seed"])

    n_folds = len(train_folds)
    print(f"Loaded {n_folds} folds + held-out test set ({len(test_df)} examples)")

    # --- Step 3: Tokenize ---
    print(f"\nTokenizing with {config['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    train_datasets = [tokenize_df(df, tokenizer, config["max_length"]) for df in train_folds]
    val_datasets = [tokenize_df(df, tokenizer, config["max_length"]) for df in val_folds]
    test_ds = tokenize_df(test_df, tokenizer, config["max_length"])

    for i in range(n_folds):
        print(f"  Fold {i}: train={len(train_datasets[i])} val={len(val_datasets[i])}")
    print(f"  Test: {len(test_ds)}")

    # --- Step 4: Train all folds ---
    fold_results = []
    best_f1 = -1
    best_trainer = None

    for fold_idx in range(n_folds):
        result, trainer = train_single_fold(
            fold_idx=fold_idx,
            train_ds=train_datasets[fold_idx],
            val_ds=val_datasets[fold_idx],
            config=config,
        )
        fold_results.append(result)

        # Track best fold for test evaluation
        if result["macro_f1"] > best_f1:
            best_f1 = result["macro_f1"]
            best_trainer = trainer

    # --- Step 5: Cross-validation summary ---
    f1_scores = [r["macro_f1"] for r in fold_results]
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print(f"\n{'='*60}")
    print(f"  CV Results: {config['model_name']} | {config['finetune_method']}")
    print(f"  macro_f1 = {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Per-fold: {[f'{s:.4f}' for s in f1_scores]}")
    print(f"{'='*60}")

    # --- Step 6: Evaluate best fold model on held-out test ---
    ## COMMENTED OUT DURING HYPERPARAMETER TUNING
    best_fold_idx = f1_scores.index(best_f1)
    print(f"\nEvaluating best fold (fold {best_fold_idx}) on held-out test set...")
    test_results = evaluate_on_test(best_trainer, test_ds)

    # --- Step 7: Save results ---
    output_dir = here(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_record = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "cv_summary": {
            "mean_macro_f1": mean_f1,
            "std_macro_f1": std_f1,
            "per_fold_f1": f1_scores,
        },
        "fold_results": fold_results,
        #"test_results": test_results,
    }

    results_path = output_dir / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(experiment_record, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")


# =====================================================================
# Lora finetuning
# =====================================================================

def apply_lora(model, config):
    """Wrap a pretrained model with Lora adapters.
       This will only train A and B matricies + classification head while other are frozen.
    """
    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],
        target_modules=config["target_modules"],
        task_type=TaskType[config["task_type"]],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

    
# =====================================================================
# multi seed runs
# =====================================================================

def run_multi_seed(config_path, n_runs=100, seed_start=1, seed_step=2):
    """
    Run experiment across multiple seeds for statistical testing.
    
    Generates seeds: [1, 3, 5,...]
    Results saved to results/multi_seed/{model-method}/
    
    :param config_path: path to experiment YAML config
    :param n_runs: number of independent training runs
    :param seed_start: first seed value
    :param seed_step: step between seeds
    """
    seeds = list(range(seed_start, seed_start + n_runs * seed_step, seed_step))
    
    config = load_config(config_path)
    model_label = f"{config['model_name'].split('/')[-1]}-{config['finetune_method']}"
    
    multi_seed_dir = here("results") / "multi_seed" / model_label
    multi_seed_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  Run {i+1}/{n_runs} | Seed {seed} | {model_label}")
        print(f"{'='*60}")
        
        config = load_config(config_path)
        config["seed"] = seed
        config["output_dir"] = str(multi_seed_dir / f"seed-{seed}")
        
        result = _run_single_seed(config)
        all_results.append(result)
        
        # Save incrementally if failure happens
        _save_multi_seed_summary(all_results, seeds[:i+1], config_path, model_label, multi_seed_dir)
    
    summary = _save_multi_seed_summary(all_results, seeds, config_path, model_label, multi_seed_dir)
    
    print(f"\n{'='*60}")
    print(f"  Multi-seed complete: {model_label} ({n_runs} runs)")
    print(f"  Test F1:  {summary['test_f1_mean']:.4f} ± {summary['test_f1_std']:.4f}")
    print(f"  CV F1:    {summary['cv_f1_mean']:.4f} ± {summary['cv_f1_std']:.4f}")
    print(f"{'='*60}")
    
    return summary


def _run_single_seed(config):
    """
    Run one full experiment with a given config.
    Same logic as run_experiment but returns a summary dict.
    """
    full_df = get_pooled_df()
    train_folds, val_folds = get_fold_from_disk(full_df, k=config["cv_k"], seed=config["cv_seed"])
    test_df = get_test_from_disk(full_df, k=config["cv_k"], seed=config["cv_seed"])
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_datasets = [tokenize_df(df, tokenizer, config["max_length"]) for df in train_folds]
    val_datasets = [tokenize_df(df, tokenizer, config["max_length"]) for df in val_folds]
    test_ds = tokenize_df(test_df, tokenizer, config["max_length"])
    
    fold_results = []
    best_f1 = -1
    best_trainer = None
    
    for fold_idx in range(len(train_folds)):
        result, trainer = train_single_fold(
            fold_idx=fold_idx,
            train_ds=train_datasets[fold_idx],
            val_ds=val_datasets[fold_idx],
            config=config,
        )
        fold_results.append(result)
        if result["macro_f1"] > best_f1:
            best_f1 = result["macro_f1"]
            best_trainer = trainer
    
    f1_scores = [r["macro_f1"] for r in fold_results]
    #test_results = evaluate_on_test(best_trainer, test_ds)
    
    return {
        "seed": config["seed"],
        "cv_mean_f1": round(float(np.mean(f1_scores)), 4),
        "cv_std_f1": round(float(np.std(f1_scores)), 4),
        "cv_per_fold": f1_scores,
        #"test_macro_f1": test_results["macro_f1"],
    }


def _save_multi_seed_summary(all_results, seeds_completed, config_path, model_label, multi_seed_dir):
    """
    Save summary JSON. Called incrementally so progress survives crashes.
    """
    test_f1s = [r["test_macro_f1"] for r in all_results]
    cv_f1s = [r["cv_mean_f1"] for r in all_results]
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config_path": config_path,
        "model_label": model_label,
        "n_runs_completed": len(all_results),
        "seeds_completed": seeds_completed,
        "per_seed_results": all_results,
        "test_f1_scores": test_f1s,
        "test_f1_mean": round(float(np.mean(test_f1s)), 4),
        "test_f1_std": round(float(np.std(test_f1s)), 4),
        "cv_f1_scores": cv_f1s,
        "cv_f1_mean": round(float(np.mean(cv_f1s)), 4),
        "cv_f1_std": round(float(np.std(cv_f1s)), 4),
    }
    
    summary_path = multi_seed_dir / "multi_seed_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary   

# =====================================================================
# Freeze all layers but classifier
# =====================================================================

def freeze_base(model):
    """
    Freeze all base model weights. Only classification head trains.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    return model

# =====================================================================
# Entry Point
# =====================================================================

if __name__ == "__main__":
    args = parse_args()
    if args.multi_seed:
        run_multi_seed(args.config, n_runs=args.n_runs)
    else:
        run_experiment(args.config)

