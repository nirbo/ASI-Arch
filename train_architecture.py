#!/usr/bin/env python3
"""
Training Script for ASI-Arch Architecture Discovery
Trains and evaluates generated neural network architectures.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import time
import importlib.util
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import traceback

# Configuration
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2  # Short for rapid iteration
VOCAB_SIZE = 32000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_message(message):
    """Log message with timestamp"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def load_architecture(architecture_name, source_file):
    """Load the architecture from the source file"""
    try:
        log_message(f"Loading architecture: {architecture_name}")
        log_message(f"Source file: {source_file}")
        
        # Import the module dynamically
        spec = importlib.util.spec_from_file_location("architecture_module", source_file)
        architecture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(architecture_module)
        
        # Try to find the model class
        # Common patterns: Model, Architecture, LLM, etc.
        model_class = None
        for attr_name in dir(architecture_module):
            attr = getattr(architecture_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, nn.Module) and 
                attr != nn.Module):
                model_class = attr
                break
        
        if model_class is None:
            raise ValueError("No PyTorch model class found in the architecture file")
        
        log_message(f"Found model class: {model_class.__name__}")
        
        # Initialize model
        model = model_class(vocab_size=VOCAB_SIZE)
        model = model.to(DEVICE)
        
        log_message(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except Exception as e:
        log_message(f"Error loading architecture: {str(e)}")
        log_message(f"Traceback: {traceback.format_exc()}")
        raise

def prepare_dataset():
    """Prepare WikiText-2 dataset for training"""
    try:
        log_message("Loading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-v1")
        
        # Use GPT-2 tokenizer as a reasonable default
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt"
            )
        
        train_dataset = dataset["train"].map(tokenize_function, batched=True)
        valid_dataset = dataset["validation"].map(tokenize_function, batched=True)
        
        # Remove text column and set format
        train_dataset = train_dataset.remove_columns(["text"])
        valid_dataset = valid_dataset.remove_columns(["text"])
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        valid_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        log_message(f"Training samples: {len(train_dataset)}")
        log_message(f"Validation samples: {len(valid_dataset)}")
        
        return train_dataloader, valid_dataloader
        
    except Exception as e:
        log_message(f"Error preparing dataset: {str(e)}")
        raise

def train_model(model, train_dataloader, valid_dataloader):
    """Train the model and return metrics"""
    try:
        log_message("Starting training...")
        
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        train_losses = []
        valid_losses = []
        
        for epoch in range(NUM_EPOCHS):
            log_message(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
            
            # Training
            model.train()
            epoch_train_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                if batch_idx >= 100:  # Limit batches for quick iteration
                    break
                    
                input_ids = batch["input_ids"].to(DEVICE)
                
                # Create targets (next token prediction)
                targets = input_ids[:, 1:].contiguous()
                inputs = input_ids[:, :-1].contiguous()
                
                optimizer.zero_grad()
                
                try:
                    outputs = model(inputs)
                    
                    # Handle different output formats
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss.backward()
                    optimizer.step()
                    
                    epoch_train_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 20 == 0:
                        log_message(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    log_message(f"Error in training batch {batch_idx}: {str(e)}")
                    continue
            
            avg_train_loss = epoch_train_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            epoch_valid_loss = 0.0
            num_valid_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_dataloader):
                    if batch_idx >= 50:  # Limit validation batches
                        break
                        
                    input_ids = batch["input_ids"].to(DEVICE)
                    targets = input_ids[:, 1:].contiguous()
                    inputs = input_ids[:, :-1].contiguous()
                    
                    try:
                        outputs = model(inputs)
                        
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        elif isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                        
                        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                        epoch_valid_loss += loss.item()
                        num_valid_batches += 1
                        
                    except Exception as e:
                        log_message(f"Error in validation batch {batch_idx}: {str(e)}")
                        continue
            
            avg_valid_loss = epoch_valid_loss / max(num_valid_batches, 1)
            valid_losses.append(avg_valid_loss)
            
            log_message(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
        
        return train_losses, valid_losses
        
    except Exception as e:
        log_message(f"Error during training: {str(e)}")
        log_message(f"Traceback: {traceback.format_exc()}")
        raise

def save_results(architecture_name, train_losses, valid_losses):
    """Save training results to CSV files"""
    try:
        log_message("Saving results...")
        
        # Save training losses
        loss_df = pd.DataFrame({
            'epoch': range(1, len(train_losses) + 1),
            'train_loss': train_losses,
            'architecture': architecture_name
        })
        loss_df.to_csv("./files/analysis/loss.csv", index=False)
        
        # Save benchmark results (using final validation loss as benchmark)
        final_valid_loss = valid_losses[-1] if valid_losses else float('inf')
        perplexity = torch.exp(torch.tensor(final_valid_loss)).item()
        
        benchmark_df = pd.DataFrame({
            'architecture': [architecture_name],
            'final_valid_loss': [final_valid_loss],
            'perplexity': [perplexity],
            'num_epochs': [len(train_losses)],
            'success': [True]
        })
        benchmark_df.to_csv("./files/analysis/benchmark.csv", index=False)
        
        log_message(f"Final validation loss: {final_valid_loss:.4f}")
        log_message(f"Perplexity: {perplexity:.2f}")
        log_message("Results saved successfully")
        
        return final_valid_loss, perplexity
        
    except Exception as e:
        log_message(f"Error saving results: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Train architecture")
    parser.add_argument("architecture_name", help="Name of the architecture to train")
    args = parser.parse_args()
    
    try:
        log_message(f"Starting training for architecture: {args.architecture_name}")
        log_message(f"Device: {DEVICE}")
        log_message(f"PyTorch version: {torch.__version__}")
        
        # Load architecture
        model = load_architecture(args.architecture_name, "./current_architecture.py")
        
        # Prepare dataset
        train_dataloader, valid_dataloader = prepare_dataset()
        
        # Train model
        train_losses, valid_losses = train_model(model, train_dataloader, valid_dataloader)
        
        # Save results
        final_loss, perplexity = save_results(args.architecture_name, train_losses, valid_losses)
        
        log_message("Training completed successfully!")
        return 0
        
    except Exception as e:
        log_message(f"Training failed: {str(e)}")
        
        # Save failure results
        try:
            error_df = pd.DataFrame({
                'architecture': [args.architecture_name],
                'final_valid_loss': [float('inf')],
                'perplexity': [float('inf')],
                'num_epochs': [0],
                'success': [False],
                'error': [str(e)]
            })
            error_df.to_csv("./files/analysis/benchmark.csv", index=False)
            
            # Write error to debug file
            with open("./files/debug/training_error.txt", "w") as f:
                f.write(f"Architecture: {args.architecture_name}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
                
        except Exception as save_error:
            log_message(f"Error saving failure results: {str(save_error)}")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())