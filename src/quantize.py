"""
Post-training quantization evaluation pipeline.
 
Evaluates trained models at FP32, FP16, and INT8 precision
to measure accuracy persistence under quantization (RQ3).
 
Usage:
    python src/quantize.py --config configs/roberta_full.yaml
    python src/quantize.py --config configs/pubmedbert_full.yaml
"""
#All concepts taken from this site: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
#https://arxiv.org/pdf/2106.08295 white paper on neural network quantization
 
import argparse
import json
import time
import os
import copy
import tempfile
import numpy as np
import torch
import torch.quantization as quant
from datetime import datetime
from pyprojroot import here
 
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
from sklearn.metrics import f1_score, classification_report


from src.config import load_config
from src.data_holdout import (
    get_pooled_df,
    get_test_from_disk,
    DECODED_LABELS,
)
from src.train import compute_metrics

# =====================================================================
# CLI
# =====================================================================
 
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Post-training quantization evaluation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config (e.g., configs/roberta_full.yaml)",
    )
    return parser.parse_args()

# =====================================================================
# Tokenization (same as train.py)
# =====================================================================
 
def tokenize_df(df, tokenizer, max_length):
    """Tokenize a DataFrame of sentence pairs into a HuggingFace Dataset."""
    ds = Dataset.from_pandas(df, preserve_index=False)
 
    def _tokenize(examples):
        return tokenizer(
            examples["abstract_conclusion"],
            examples["press_release_conclusion"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
 
    cols_to_remove = [c for c in ds.column_names if c != "exaggeration_label"]
    ds = ds.map(_tokenize, batched=True, remove_columns=cols_to_remove)
    ds = ds.rename_column("exaggeration_label", "labels")
    ds.set_format("torch")
    return ds

# =====================================================================
# Get MOdel
# =====================================================================

def get_model_size_mb(model):
    """
    Measure model size on disk in MB.
 
    Saves to a temp file, measures, then cleans up.
    https://pytorch.org/docs/stable/generated/torch.save.html
    """
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_mb = os.path.getsize(f.name) / (1024 * 1024)
        os.unlink(f.name)
    return round(size_mb, 2)

# =====================================================================
# Load Trained Checkpoint
# =====================================================================

def load_trained_model(config):
    """
    Find and load model checkpoint from trained run this will be the fp(32) version.
    Reads trainer_state.json to find the best model checkpoint.
    """
    output_dir = here(config["output_dir"])
    
    best_checkpoint = None
    best_f1 = -1

    # glob allows iteration with wildcards to desired path. We can search through many folds and checkpoints efficiently
    for state_path in sorted(output_dir.glob("fold-*/checkpoint-*/trainer_state.json")):
        with state_path.open(mode='r') as f: 
            trainer_state = json.load(f) #load in as json since trainer.state is json
        
        checkpoint_path = trainer_state.get("best_model_checkpoint") #keying off of best_model_checkpoint
        best_metric = trainer_state.get("best_metric", 0) #keying off best metric
        
        if checkpoint_path and best_metric > best_f1:
            best_f1 = best_metric
            best_checkpoint = checkpoint_path
    
    if best_checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoints found in {output_dir}. Run train.py first." #I may share checkponts in repo later if we have time
        )
    
    print(f"Best checkpoint: {best_checkpoint} (macro_f1: {best_f1:.4f})")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        best_checkpoint,
        num_labels=3,
    )
    model.eval()
    return model

# =====================================================================
# quantization error measurement
# =====================================================================
#https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization

def compute_quantization_error(w_original, w_dequantized, layer_name):
    """
    Compare original FP32 weights against dequantized weights.
    delta W = W_original - W_dequantized
    """
    error = w_original - w_dequantized #critical understanding of loss of information
    
    signal_power = np.mean(w_original ** 2)
    noise_power = np.mean(error ** 2)
    sqnr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    mean_abs_weight = np.mean(np.abs(w_original))
    relative_error = (np.mean(np.abs(error)) / mean_abs_weight * 100) if mean_abs_weight > 0 else 0.0
    
    return {
        "layer_name": layer_name,
        "sqnr_db": round(float(sqnr_db), 2),
        "relative_error_pct": round(float(relative_error), 4),
    }

# =====================================================================
# fp(16) half precision
# =====================================================================

def convert_to_fp16(model):
    """
    Convert model from FP32 to FP16.
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.half
    """
    model_fp16 = copy.deepcopy(model)
    model_fp16.half()
    return model_fp16


# =====================================================================
# INT8 dynamic quantization
# =====================================================================

def convert_to_int8(model):
    """
    Convert model to INT8 using dynamic quantization on Linear layers only.
    
    https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization
    """
    fp32_weights = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            fp32_weights[name] = param.detach().cpu().float().numpy()
    #setting dynamic here
    #This is the simplest approach no calibration
    model_int8 = torch.quantization.quantize_dynamic(
        copy.deepcopy(model).cpu(),#torch forces us to use cpu..to use GPU we'd have to shift to onnx
        {torch.nn.Linear},
        dtype=torch.qint8, #transform to int8
    )
    
    return model_int8, fp32_weights

# =====================================================================
# Evaluation
# =====================================================================

def evaluate_precision(model, test_ds, precision_label, use_cpu=False):
    """Run inference on test set and measure macro F1."""
    args = TrainingArguments(
        output_dir="/tmp/eval", #This is required but this isn't needed for our eval...ignore
        use_cpu=use_cpu,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args, compute_metrics=compute_metrics)
    results = trainer.predict(test_ds)
    
    macro_f1 = results.metrics["test_macro_f1"]
    model_size = get_model_size_mb(model)
    
    preds = np.argmax(results.predictions, axis=-1)
    print(f"\n  {precision_label}: macro_f1={macro_f1:.4f} | size={model_size:.1f} MB")
    print(classification_report(results.label_ids, preds, target_names=list(DECODED_LABELS.values())))
    
    return {"precision": precision_label, "macro_f1": macro_f1, "model_size_mb": model_size}

# =====================================================================
# Orchestrator
# =====================================================================

def run_quantization(config_path):
    """
    Run PTQ evaluation at FP32, FP16, and INT8.
    Saves results to output_dir/quantization_results.json
    """
    config = load_config(config_path)
    print(f"\nQuantization Evaluation: {config['model_name']} | {config['finetune_method']}")

    # Load model and test data
    model = load_trained_model(config)
    full_df = get_pooled_df()
    test_df = get_test_from_disk(full_df, k=config["cv_k"], seed=config["cv_seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    test_ds = tokenize_df(test_df, tokenizer, config["max_length"])
    print(f"Test set: {len(test_ds)} examples")

    # Evaluate FP32
    fp32_result = evaluate_precision(model, test_ds, "FP32")

    # Evaluate FP16
    model_fp16 = convert_to_fp16(model)
    fp16_result = evaluate_precision(model_fp16, test_ds, "FP16")
    del model_fp16

    # Evaluate INT8
    model_int8, fp32_weights = convert_to_int8(model)
    int8_result = evaluate_precision(model_int8, test_ds, "INT8", use_cpu=True)
    del model_int8

    # Save results
    output_dir = here(config["output_dir"])
    record = {
        "timestamp": datetime.now().isoformat(),
        "config_path": config_path,
        "model_name": config["model_name"],
        "finetune_method": config["finetune_method"],
        "test_size": len(test_ds),
        "results": [fp32_result, fp16_result, int8_result],
    }

    results_path = output_dir / "quantization_results.json"
    with open(results_path, "w") as f:
        json.dump(record, f, indent=2)

    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    args = parse_args()
    run_quantization(args.config)