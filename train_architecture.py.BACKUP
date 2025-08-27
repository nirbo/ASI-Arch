#!/usr/bin/env python3
"""
Custom training script for transformer architecture research.
Focus: Discovering novel optimizations to address quadratic memory scaling.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

# Add pipeline directory to path for architecture import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipeline"))

# Import the current architecture
from current_architecture import BasicTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Simple text dataset for language modeling."""

    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()

        # For language modeling, labels are input_ids shifted
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics."""
    if torch.cuda.is_available():
        return {
            "cuda_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "cuda_reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "cuda_max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    return {"cpu_only": True}


def measure_complexity_scaling(
    model, tokenizer, device, max_seq_lengths=[128, 256, 512, 1024]
) -> Dict[str, Any]:
    """Measure memory and time complexity scaling with sequence length."""
    results = {}
    model.eval()

    for seq_len in max_seq_lengths:
        if seq_len > 2048:  # Skip if beyond positional embedding limit
            continue

        try:
            # Create dummy input
            batch_size = max(1, 8192 // seq_len)  # Adjust batch size for memory
            input_ids = torch.randint(
                0, tokenizer.vocab_size, (batch_size, seq_len)
            ).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Warm-up
            with torch.no_grad():
                _ = model(input_ids, attention_mask)

            # Measure forward pass
            start_time = time.time()
            start_memory = (
                torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            )

            with torch.no_grad():
                outputs = model(input_ids, attention_mask)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end_time = time.time()
            peak_memory = (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )

            results[seq_len] = {
                "forward_time_ms": (end_time - start_time) * 1000,
                "peak_memory_mb": peak_memory / 1e6,
                "memory_per_token": (
                    peak_memory / (batch_size * seq_len) if peak_memory > 0 else 0
                ),
                "time_per_token_us": ((end_time - start_time) * 1e6)
                / (batch_size * seq_len),
                "batch_size": batch_size,
            }

            logger.info(
                f"Seq {seq_len}: {results[seq_len]['forward_time_ms']:.2f}ms, "
                f"{results[seq_len]['peak_memory_mb']:.1f}MB, "
                f"{results[seq_len]['time_per_token_us']:.2f}μs/token"
            )

        except RuntimeError as e:
            logger.warning(f"Failed at sequence length {seq_len}: {e}")
            results[seq_len] = {"error": str(e)}

        torch.cuda.empty_cache()

    return results


def train_epoch(
    model, dataloader, optimizer, criterion, device, epoch_num
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids, attention_mask)

        # Compute loss (ignore padding tokens)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_tokens += attention_mask.sum().item()
        num_batches += 1

        if batch_idx % 100 == 0:
            logger.info(
                f"Epoch {epoch_num}, Batch {batch_idx}, Loss: {loss.item():.4f}"
            )

    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    return {
        "loss": avg_loss,
        "perplexity": np.exp(avg_loss),
        "tokens_per_second": total_tokens / epoch_time,
        "epoch_time": epoch_time,
    }


def evaluate_model(model, dataloader, criterion, device) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = criterion(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            total_loss += loss.item()
            total_tokens += attention_mask.sum().item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")

    return {"eval_loss": avg_loss, "eval_perplexity": np.exp(avg_loss)}


def save_results(results: Dict[str, Any], experiment_name: str):
    """Save results to CSV files."""
    os.makedirs("files/analysis", exist_ok=True)

    # Save training metrics
    loss_file = "files/analysis/loss.csv"
    benchmark_file = "files/analysis/benchmark.csv"

    # Training results
    loss_data = {
        "experiment": experiment_name,
        "final_loss": results.get("final_loss", float("inf")),
        "final_perplexity": results.get("final_perplexity", float("inf")),
        "best_loss": results.get("best_loss", float("inf")),
        "training_time": results.get("total_training_time", 0),
        "tokens_per_second": results.get("tokens_per_second", 0),
    }

    # Memory efficiency metrics
    complexity_results = results.get("complexity_analysis", {})
    if complexity_results:
        # Calculate quadratic scaling coefficient
        seq_lens = sorted([k for k in complexity_results.keys() if isinstance(k, int)])
        if len(seq_lens) >= 2:
            memory_ratios = []
            for i in range(1, len(seq_lens)):
                if (
                    seq_lens[i - 1] in complexity_results
                    and seq_lens[i] in complexity_results
                ):
                    prev_mem = complexity_results[seq_lens[i - 1]].get(
                        "memory_per_token", 0
                    )
                    curr_mem = complexity_results[seq_lens[i]].get(
                        "memory_per_token", 0
                    )
                    if prev_mem > 0:
                        ratio = curr_mem / prev_mem
                        length_ratio = seq_lens[i] / seq_lens[i - 1]
                        memory_ratios.append(
                            ratio / length_ratio
                        )  # Should be ~1 for linear, >1 for quadratic

            if memory_ratios:
                loss_data["memory_scaling_factor"] = np.mean(memory_ratios)

    # Benchmark results
    benchmark_data = {
        "experiment": experiment_name,
        "architecture": "BasicTransformer",
        "memory_efficiency": 1.0
        / loss_data.get("memory_scaling_factor", 1.0),  # Higher is better
        "speed_tokens_per_sec": loss_data["tokens_per_second"],
        "model_quality_inv_perplexity": (
            1.0 / loss_data["final_perplexity"]
            if loss_data["final_perplexity"] < float("inf")
            else 0
        ),
        "complexity_score": results.get(
            "complexity_score", 0
        ),  # Custom metric for sub-quadratic performance
    }

    # Write results
    import csv

    # Write loss file
    write_header = not os.path.exists(loss_file)
    with open(loss_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=loss_data.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(loss_data)

    # Write benchmark file
    write_header = not os.path.exists(benchmark_file)
    with open(benchmark_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=benchmark_data.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(benchmark_data)

    logger.info(f"Results saved for experiment: {experiment_name}")


def main():
    parser = argparse.ArgumentParser(description="Train transformer architecture")
    parser.add_argument("experiment_name", help="Name of the experiment")
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=5000,
        help="Number of samples to use (default: 5000 for research quality)",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")

    args = parser.parse_args()

    try:
        logger.info(f"Starting experiment: {args.experiment_name}")

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Initialize wandb if requested
        if args.wandb:
            try:
                wandb.init(
                    project="asi-arch-transformer-research",
                    name=args.experiment_name,
                    config=vars(args),
                )
            except:
                logger.warning("Failed to initialize wandb, continuing without it")

        # Load tokenizer and dataset
        logger.info("Loading dataset and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Load WikiText-103 for more substantial pretraining data
        # First load full dataset, then filter for non-empty articles
        logger.info("Loading WikiText-103 dataset...")
        full_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

        # Filter out empty lines and very short articles (< 100 chars)
        def is_substantial_text(example):
            text = example["text"].strip()
            return len(text) > 100 and not text.startswith("=") and text != ""

        logger.info("Filtering for substantial articles...")
        dataset = full_dataset.filter(is_substantial_text)
        logger.info(f"Filtered dataset size: {len(dataset)} substantial articles")

        # Intelligent dataset sizing based on experiment type
        total_available = len(dataset)

        # Auto-adjust dataset size based on experiment name patterns
        if args.experiment_name.startswith(("test_", "debug_", "quick_")):
            suggested_size = min(500, args.dataset_size)  # Quick tests
            logger.info(
                f"Detected test experiment - using smaller dataset size: {suggested_size}"
            )
        elif args.experiment_name.startswith(("research_", "evolution_", "pipeline_")):
            suggested_size = max(5000, args.dataset_size)  # Research quality
            logger.info(
                f"Detected research experiment - ensuring adequate dataset size: {suggested_size}"
            )
        else:
            suggested_size = args.dataset_size

        # Select the final number of samples
        final_size = min(suggested_size, total_available)
        if final_size < total_available:
            # Take a diverse sample by selecting every N-th article for better coverage
            step = max(1, total_available // final_size)
            indices = list(range(0, total_available, step))[:final_size]
            dataset = dataset.select(indices)
            logger.info(
                f"Selected {len(dataset)} diverse samples (every {step}th article)"
            )
        else:
            dataset = dataset.select(range(final_size))

        # Create datasets
        train_dataset = TextDataset(dataset["text"], tokenizer, args.max_length)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )

        # Initialize model
        logger.info("Initializing model...")
        model = BasicTransformer(
            d_model=512,
            num_heads=8,
            num_layers=4,
            d_ff=2048,
            dropout=0.1,
        ).to(device)

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")

        # Setup training
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Measure complexity scaling before training
        logger.info("Measuring complexity scaling...")
        complexity_results = measure_complexity_scaling(model, tokenizer, device)

        # Training loop
        best_loss = float("inf")
        training_start_time = time.time()

        for epoch in range(args.epochs):
            logger.info(f"Starting epoch {epoch + 1}/{args.epochs}")

            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch + 1
            )
            scheduler.step()

            # Evaluate
            eval_metrics = evaluate_model(
                model, train_loader, criterion, device
            )  # Using train for simplicity

            # Log metrics
            current_loss = eval_metrics["eval_loss"]
            if current_loss < best_loss:
                best_loss = current_loss

            memory_stats = get_memory_usage()

            epoch_metrics = {
                **train_metrics,
                **eval_metrics,
                **memory_stats,
                "epoch": epoch + 1,
                "learning_rate": scheduler.get_last_lr()[0],
            }

            logger.info(
                f"Epoch {epoch + 1} - Loss: {current_loss:.4f}, "
                f"Perplexity: {eval_metrics['eval_perplexity']:.2f}"
            )

            if args.wandb:
                try:
                    wandb.log(epoch_metrics)
                except:
                    pass

        total_training_time = time.time() - training_start_time

        # Calculate complexity score (higher is better for sub-quadratic)
        complexity_score = 0
        if complexity_results:
            seq_lens = sorted(
                [k for k in complexity_results.keys() if isinstance(k, int)]
            )
            if len(seq_lens) >= 2:
                # Check if scaling is better than quadratic
                time_scaling_factors = []
                memory_scaling_factors = []

                for i in range(1, len(seq_lens)):
                    if (
                        seq_lens[i - 1] in complexity_results
                        and seq_lens[i] in complexity_results
                        and "error" not in complexity_results[seq_lens[i - 1]]
                        and "error" not in complexity_results[seq_lens[i]]
                    ):

                        prev = complexity_results[seq_lens[i - 1]]
                        curr = complexity_results[seq_lens[i]]
                        length_ratio = seq_lens[i] / seq_lens[i - 1]

                        if prev.get("time_per_token_us", 0) > 0:
                            time_ratio = curr.get("time_per_token_us", 0) / prev.get(
                                "time_per_token_us", 1
                            )
                            time_scaling_factors.append(time_ratio)

                        if prev.get("memory_per_token", 0) > 0:
                            memory_ratio = curr.get("memory_per_token", 0) / prev.get(
                                "memory_per_token", 1
                            )
                            memory_scaling_factors.append(memory_ratio)

                # Score: closer to 1.0 (linear) is better, >2.0 (quadratic) is worse
                if time_scaling_factors or memory_scaling_factors:
                    avg_scaling = np.mean(time_scaling_factors + memory_scaling_factors)
                    complexity_score = max(
                        0, 2.0 - avg_scaling
                    )  # Higher score for better scaling

        # Prepare results
        results = {
            "final_loss": current_loss,
            "final_perplexity": eval_metrics["eval_perplexity"],
            "best_loss": best_loss,
            "total_training_time": total_training_time,
            "tokens_per_second": train_metrics.get("tokens_per_second", 0),
            "complexity_analysis": complexity_results,
            "complexity_score": complexity_score,
            "num_parameters": num_params,
        }

        # Save results
        save_results(results, args.experiment_name)

        logger.info(f"Training completed successfully!")
        logger.info(f"Final loss: {current_loss:.4f}")
        logger.info(f"Final perplexity: {eval_metrics['eval_perplexity']:.2f}")
        logger.info(f"Complexity score: {complexity_score:.3f}")

        if args.wandb:
            try:
                wandb.finish()
            except:
                pass

        return 0

    except Exception as e:
        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)

        # Save error to debug file
        os.makedirs("files/debug", exist_ok=True)
        with open("files/debug/training_error.txt", "w") as f:
            f.write(f"Experiment: {args.experiment_name}\n")
            f.write(f"Error: {error_msg}\n")
            f.write(f"Arguments: {vars(args)}\n")

        return 1


if __name__ == "__main__":
    sys.exit(main())
