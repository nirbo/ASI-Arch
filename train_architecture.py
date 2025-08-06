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
import re
import math

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

def prepare_reasoning_dataset():
    """Prepare simple mathematical reasoning dataset for evaluation"""
    try:
        log_message("Creating simple mathematical reasoning dataset...")
        
        # Generate basic arithmetic problems
        problems = []
        for i in range(200):  # Small dataset for quick evaluation
            a, b = torch.randint(1, 100, (2,)).tolist()
            operation = torch.randint(0, 4, (1,)).item()
            
            if operation == 0:  # Addition
                problem = f"What is {a} + {b}?"
                answer = str(a + b)
            elif operation == 1:  # Subtraction  
                problem = f"What is {max(a,b)} - {min(a,b)}?"
                answer = str(max(a,b) - min(a,b))
            elif operation == 2:  # Multiplication
                a, b = min(a, 20), min(b, 20)  # Keep numbers smaller
                problem = f"What is {a} * {b}?"
                answer = str(a * b)
            else:  # Division
                b = max(b, 1)
                result = (a * b) // b  # Ensure integer division
                problem = f"What is {a * b} / {b}?"
                answer = str(result)
            
            problems.append({"problem": problem, "answer": answer})
        
        return problems
        
    except Exception as e:
        log_message(f"Error preparing reasoning dataset: {str(e)}")
        return []

def evaluate_reasoning(model, tokenizer, problems, max_problems=50):
    """Evaluate model on reasoning tasks"""
    try:
        log_message("Evaluating reasoning capabilities...")
        
        model.eval()
        correct = 0
        total = min(len(problems), max_problems)
        
        with torch.no_grad():
            for i, problem_data in enumerate(problems[:total]):
                problem = problem_data["problem"]
                correct_answer = problem_data["answer"]
                
                # Tokenize problem
                inputs = tokenizer(
                    problem, 
                    return_tensors="pt", 
                    max_length=64,
                    truncation=True,
                    padding=True
                ).to(DEVICE)
                
                try:
                    # Generate response (simple greedy decoding)
                    input_ids = inputs["input_ids"]
                    
                    # Generate a few tokens for the answer
                    for _ in range(10):  # Max 10 tokens for answer
                        outputs = model(input_ids)
                        
                        if hasattr(outputs, 'logits'):
                            logits = outputs.logits
                        elif isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs
                        
                        # Get next token
                        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        
                        # Stop if we hit end token or max length
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                    
                    # Decode generated response
                    generated_text = tokenizer.decode(
                        input_ids[0][inputs["input_ids"].shape[1]:], 
                        skip_special_tokens=True
                    ).strip()
                    
                    # Extract numbers from generated text
                    numbers = re.findall(r'\d+', generated_text)
                    
                    if numbers and numbers[0] == correct_answer:
                        correct += 1
                        
                    if i < 5:  # Log first few examples
                        log_message(f"Problem: {problem}")
                        log_message(f"Generated: {generated_text}")
                        log_message(f"Expected: {correct_answer}")
                        log_message(f"Correct: {numbers[0] == correct_answer if numbers else False}")
                        log_message("---")
                        
                except Exception as e:
                    log_message(f"Error in reasoning evaluation {i}: {str(e)}")
                    continue
        
        accuracy = correct / total if total > 0 else 0.0
        log_message(f"Reasoning accuracy: {correct}/{total} = {accuracy:.2%}")
        
        return accuracy
        
    except Exception as e:
        log_message(f"Error in reasoning evaluation: {str(e)}")
        return 0.0

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

def save_results(architecture_name, train_losses, valid_losses, reasoning_accuracy=0.0, reasoning_speed=0.0):
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
        
        # Compute composite score (lower is better)
        # Combine language modeling (perplexity) with reasoning accuracy
        language_score = perplexity
        reasoning_score = 1.0 / (reasoning_accuracy + 0.01)  # Invert accuracy (lower is better)
        composite_score = language_score + reasoning_score
        
        benchmark_df = pd.DataFrame({
            'architecture': [architecture_name],
            'final_valid_loss': [final_valid_loss],
            'perplexity': [perplexity],
            'reasoning_accuracy': [reasoning_accuracy],
            'reasoning_speed': [reasoning_speed],
            'composite_score': [composite_score],
            'num_epochs': [len(train_losses)],
            'success': [True]
        })
        benchmark_df.to_csv("./files/analysis/benchmark.csv", index=False)
        
        log_message(f"Final validation loss: {final_valid_loss:.4f}")
        log_message(f"Perplexity: {perplexity:.2f}")
        log_message(f"Reasoning accuracy: {reasoning_accuracy:.2%}")
        log_message(f"Composite score: {composite_score:.4f}")
        log_message("Results saved successfully")
        
        return final_valid_loss, perplexity, reasoning_accuracy
        
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
        
        # Prepare datasets
        train_dataloader, valid_dataloader = prepare_dataset()
        reasoning_problems = prepare_reasoning_dataset()
        
        # Get tokenizer for reasoning evaluation
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Train model
        train_losses, valid_losses = train_model(model, train_dataloader, valid_dataloader)
        
        # Evaluate reasoning capabilities
        start_time = time.time()
        reasoning_accuracy = evaluate_reasoning(model, tokenizer, reasoning_problems)
        reasoning_time = time.time() - start_time
        reasoning_speed = len(reasoning_problems[:50]) / reasoning_time if reasoning_time > 0 else 0
        
        # Save results with reasoning metrics
        final_loss, perplexity, reasoning_acc = save_results(
            args.architecture_name, train_losses, valid_losses, 
            reasoning_accuracy, reasoning_speed
        )
        
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