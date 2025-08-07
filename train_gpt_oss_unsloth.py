#!/usr/bin/env python3
"""
Unsloth QLoRA Fine-tuning Script for GPT-OSS-20B
Features: Checkpoint saving, auto-resumption, efficient training
"""

import unsloth
import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from datetime import datetime
import logging
import shutil
import math

# Sophia optimizers
try:
    from sophia import SophiaG, SophiaH
    SOPHIA_AVAILABLE = True
except ImportError:
    SOPHIA_AVAILABLE = False
    SophiaG = SophiaH = None


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce HuggingFace transformers logging verbosity
import transformers
transformers.logging.set_verbosity_error()

def dequantize_mxfp4_model(input_path: str, output_path: str, max_memory_gb: float = 28.0) -> bool:
    """
    Dequantize MXFP4 model to BF16 format and save to disk.
    
    Args:
        input_path: Path to MXFP4 model
        output_path: Path to save dequantized BF16 model
        max_memory_gb: Maximum GPU memory to use
        
    Returns:
        True if successful, False otherwise
    """
    import shutil
    from transformers import Mxfp4Config
    
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    logger.info(f"🔧 Dequantizing MXFP4 model to BF16")
    logger.info(f"   Input:  {input_path}")
    logger.info(f"   Output: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # Load tokenizer first (lightweight)  
        from transformers import AutoTokenizer
        logger.info("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            input_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Save tokenizer to output
        tokenizer.save_pretrained(output_path)
        
        # Configure MXFP4 dequantization
        logger.info("🏗️  Loading MXFP4 model with dequantization...")
        from transformers import AutoModelForCausalLM
        quantization_config = Mxfp4Config(dequantize=True)
        
        # Load model with dequantization enabled - use more conservative memory settings
        try:
            model = AutoModelForCausalLM.from_pretrained(
                input_path,
                trust_remote_code=True,
                device_map="auto",
                max_memory={0: f"{max_memory_gb}GB", "cpu": "8GB"},  # Limit CPU memory usage
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True,
                quantization_config=quantization_config
            )
        except Exception as e:
            if "offloaded" in str(e) or "cpu memory" in str(e).lower():
                logger.warning(f"Memory issue during dequantization: {e}")
                logger.info("Trying with sequential loading...")
                
                # Try sequential loading with smaller shards
                model = AutoModelForCausalLM.from_pretrained(
                    input_path,
                    trust_remote_code=True,
                    device_map="sequential",  # Load sequentially instead of auto
                    max_memory={0: f"{max_memory_gb-2}GB"},  # Leave more buffer
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    quantization_config=quantization_config
                )
            else:
                raise
        
        logger.info("✅ Model loaded and dequantized in memory")
        
        # Save the dequantized model with very small shards for reliability
        logger.info("💾 Saving dequantized BF16 model...")
        try:
            model.save_pretrained(
                output_path,
                safe_serialization=True,
                max_shard_size="500MB"  # Use very small shards for maximum compatibility
            )
        except Exception as save_error:
            logger.warning(f"Save failed with 500MB shards: {save_error}")
            logger.info("Retrying with 200MB shards...")
            
            # Try with even smaller shards
            model.save_pretrained(
                output_path,
                safe_serialization=True,
                max_shard_size="200MB"
            )
        
        # Copy config files
        config_files = ['config.json', 'generation_config.json', 'tokenizer_config.json']
        for config_file in config_files:
            src_file = os.path.join(input_path, config_file)
            if os.path.exists(src_file):
                dst_file = os.path.join(output_path, config_file)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
        
        # Create marker file
        marker_info = {
            "source_model": input_path,
            "target_dtype": "bfloat16",
            "dequantization_method": "Mxfp4Config",
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(output_path, "dequantization_info.json"), 'w') as f:
            json.dump(marker_info, f, indent=2)
        
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("✅ Dequantization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dequantization failed: {e}")
        if os.path.exists(output_path):
            try:
                shutil.rmtree(output_path)
            except:
                pass
        return False

from transformers import TrainerCallback


class BestModelCallback(TrainerCallback):
    """Custom callback to save the best model based on evaluation loss."""
    
    def __init__(self, output_dir: str, use_mxfp4: bool = False):
        self.output_dir = output_dir
        self.use_mxfp4 = use_mxfp4
        self.best_eval_loss = float('inf')
        self.best_model_dir = os.path.join(output_dir, "best-model")
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Called after evaluation."""
        if not state.log_history:
            return
            
        # Find the most recent eval_loss
        current_eval_loss = None
        for log_entry in reversed(state.log_history):
            if 'eval_loss' in log_entry:
                current_eval_loss = log_entry['eval_loss']
                break
        
        if current_eval_loss is not None and current_eval_loss < self.best_eval_loss:
            self.best_eval_loss = current_eval_loss
            logger.info(f"New best model! Eval loss: {current_eval_loss:.4f}")
            
            # Save best model
            logger.info(f"Saving best model to {self.best_model_dir}")
            
            # Remove existing best model
            if os.path.exists(self.best_model_dir):
                shutil.rmtree(self.best_model_dir)
            
            os.makedirs(self.best_model_dir, exist_ok=True)
            
            # Save model and tokenizer from kwargs
            if 'model' in kwargs and 'tokenizer' in kwargs:
                kwargs['model'].save_pretrained(self.best_model_dir)
                kwargs['tokenizer'].save_pretrained(self.best_model_dir)
            else:
                logger.warning("Model or tokenizer not available in callback kwargs")
            
            # Save best model metadata
            best_model_info = {
                "eval_loss": current_eval_loss,
                "step": state.global_step,
                "epoch": state.epoch,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(os.path.join(self.best_model_dir, "best_model_info.json"), 'w') as f:
                json.dump(best_model_info, f, indent=2)
            
            logger.info(f"Best model saved (step {state.global_step}, loss {current_eval_loss:.15f})")

def setup_unsloth():
    """Setup Unsloth with optimizations for QLoRA training."""
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        import torch
        
        # Check GPU availability
        if not torch.cuda.is_available():
            logger.error("CUDA not available! This script requires GPU.")
            return None, None, None
            
        gpu_stats = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {gpu_stats.name}, VRAM: {gpu_stats.total_memory / 1024**3:.1f} GB")
        
        return FastLanguageModel, get_chat_template, torch
        
    except ImportError as e:
        logger.error(f"Failed to import Unsloth: {e}")
        logger.error("Install with: pip install unsloth[cu128-torch270]")  # Adjust CUDA version as needed
        return None, None, None

def load_model_native_mxfp4(model_name: str, max_seq_length: int = 2048):
    """Load GPT-OSS model with native MXFP4 quantization (no additional quantization)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    import torch
    import os
    import gc
    
    def clear_gpu_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        gc.collect()
    
    model_path = os.path.abspath(model_name)
    logger.info(f"🔧 Loading MXFP4 model from {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    clear_gpu_cache()
    
    # Load tokenizer
    logger.info("📝 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model first, then convert from MXFP4 to trainable format
    logger.info("🏗️  Loading MXFP4 model and converting to trainable format...")
    
    try:
        # Try loading with BitsAndBytesConfig
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: "28GB"},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_cache=False,
            local_files_only=True,
            quantization_config=quantization_config
        )
        
    except (AttributeError, TypeError) as e:
        logger.warning(f"BitsAndBytesConfig failed: {e}")
        logger.info("Trying alternative approach - loading without quantization config...")
        
        # Alternative: Load the model normally and let it use native quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            max_memory={0: "28GB"},
            torch_dtype="auto",  # Let model use its native dtype
            low_cpu_mem_usage=True,
            use_cache=False,
            local_files_only=True
        )
        
        # Check if model supports training
        logger.info("Checking model quantization compatibility for training...")
        if hasattr(model.config, 'quantization_config'):
            quant_type = getattr(model.config.quantization_config, 'quant_method', 'unknown')
            logger.info(f"Model uses quantization: {quant_type}")
            
            if 'mxfp4' in str(quant_type).lower():
                logger.error("MXFP4 quantization detected - this doesn't support training")
                logger.error("Consider using a different model or dequantizing manually")
                raise ValueError("MXFP4 quantization is not compatible with training")
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training if quantized
    try:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
        logger.info("Model prepared for k-bit training")
    except Exception as e:
        logger.warning(f"prepare_model_for_kbit_training failed: {e}")
        logger.info("Continuing with standard model preparation...")
    
    # Setup LoRA manually (since we're not using Unsloth)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=32,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        use_rslora=True,
    )
    
    model = get_peft_model(model, lora_config)
    model.train()
    
    # Setup tokenizer for training
    tokenizer.padding_side = "right"
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}### User:\n{{ message['content'] }}\n\n{% elif message['role'] == 'assistant' %}### Assistant:\n{{ message['content'] }}\n\n{% endif %}{% endfor %}{% if add_generation_prompt %}### Assistant:\n{% endif %}"
    
    clear_gpu_cache()
    logger.info("Native MXFP4 model loaded successfully")
    
    return model, tokenizer

def load_model_and_tokenizer(model_name: str, max_seq_length: int = 2048, use_mxfp4: bool = False):
    """Load GPT-OSS model with Unsloth optimizations and optional MXFP4 support."""
    # Check if we should use Unsloth or native MXFP4
    if use_mxfp4:
        logger.info("🔧 Loading model with native MXFP4 support (bypassing Unsloth)")
        return load_model_native_mxfp4(model_name, max_seq_length)
    
    
    FastLanguageModel, _, _ = setup_unsloth()
    
    logger.info(f"Loading model with Unsloth: {model_name}")
    
    # Clear GPU cache before loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Try different model loading approaches for compatibility
    try:
        # Conservative loading with CPU offloading for large models
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect dtype
            load_in_4bit=True,  # QLoRA 4-bit quantization
            trust_remote_code=True,
        )
        
        # Fix chat template for DeepSeek R1 if needed
        if "deepseek" in model_name.lower() or "qwen3" in model_name.lower():
            logger.info("🔧 Applying DeepSeek R1 chat template fix...")
            # Set a simple chat template that works with Unsloth
            tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}### User:\n{{ message['content'] }}\n\n{% elif message['role'] == 'assistant' %}### Assistant:\n{{ message['content'] }}\n\n{% endif %}{% endfor %}{% if add_generation_prompt %}### Assistant:\n{% endif %}"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"CUDA OOM during model loading: {e}")
            logger.info("Trying with smaller memory limit and reduced sequence length...")
            
            # Clear cache and try again with even smaller limits
            torch.cuda.empty_cache()
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length//2,  # Reduce sequence length
                dtype=None,
                load_in_4bit=True,
                trust_remote_code=True,
                device_map="auto", 
                max_memory={0: "24GB", "cpu": "32GB"},  # Larger CPU offloading
                offload_folder="./tmp_offload",
                low_cpu_mem_usage=True,
            )
        else:
            raise
    except AttributeError as e:
        if "get_loading_attributes" in str(e):
            logger.warning(f"Compatibility issue detected: {e}")
            logger.info("Trying alternative loading method...")
            # Try without explicit quantization config
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=None,
                trust_remote_code=True,
                # Let Unsloth handle quantization internally
            )
        else:
            raise
    
    # Setup LoRA configuration with RSLoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # LoRA rank - higher for better quality, lower for speed
        lora_alpha=128,  # LoRA scaling parameter
        lora_dropout=0.0,  # No dropout for better performance
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "embed_tokens", "lm_head",
        ],
        bias="none",  # No bias updates
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        random_state=42,
        use_rslora=True,  # Use Rank-Stabilized LoRA
    )
    
    # Setup chat template for conversation format
    tokenizer.padding_side = "right"  # Required for training
    
    # Use a simple chat template
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}### User:\n{{ message['content'] }}\n\n{% elif message['role'] == 'assistant' %}### Assistant:\n{{ message['content'] }}\n\n{% endif %}{% endfor %}{% if add_generation_prompt %}### Assistant:\n{% endif %}"
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer

def load_dataset(dataset_file: str, tokenizer, max_samples: Optional[int] = None, eval_split: float = 0.025):
    """Load and tokenize the combined training dataset with train/eval split."""
    from datasets import Dataset
    
    logger.info(f"Loading dataset from {dataset_file}")
    
    # Load JSONL data
    data = []
    with open(dataset_file, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line.strip()))
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_list(data)
    
    def formatting_prompts_func(examples):
        """Format conversations using chat template."""
        texts = []
        for messages in examples["messages"]:
            # Apply chat template
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    
    # Format dataset
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split into train/eval
    if eval_split > 0:
        split_dataset = dataset.train_test_split(test_size=eval_split, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        logger.info(f"Dataset split: {len(train_dataset)} train, {len(eval_dataset)} eval")
        return train_dataset, eval_dataset
    else:
        logger.info("Dataset formatted successfully (no eval split)")
        return dataset, None

def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint directory for auto-resumption."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    # Look for checkpoint-* directories
    checkpoints = []
    for item in checkpoint_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step_num = int(item.name.split("-")[1])
                checkpoints.append((step_num, str(item)))
            except (IndexError, ValueError):
                continue
    
    if not checkpoints:
        return None
    
    # Return the latest checkpoint
    latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
    logger.info(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

def get_custom_optimizer(model_parameters, optimizer_name: str, learning_rate: float, weight_decay: float = 0.01, **kwargs):
    """Get custom optimizer including Sophia variants."""
    if not SOPHIA_AVAILABLE and optimizer_name in ["sophia", "sophia_h"]:
        raise ValueError(f"Sophia optimizer requested but not available. Install with: pip install sophia-optimizer")
    
    # Sophia-specific hyperparameters
    sophia_lr = kwargs.get('sophia_lr', learning_rate * 0.5)  # Sophia typically needs lower LR
    sophia_betas = kwargs.get('sophia_betas', (0.965, 0.99))  # Sophia default betas
    sophia_rho = kwargs.get('sophia_rho', 0.04)  # Hessian update frequency
    sophia_eps = kwargs.get('sophia_eps', 1e-8)
    
    if optimizer_name == "sophia":
        logger.info(f"Using Sophia-G optimizer: lr={sophia_lr}, betas={sophia_betas}, rho={sophia_rho}")
        return SophiaG(
            model_parameters,
            lr=sophia_lr,
            betas=sophia_betas,
            rho=sophia_rho,
            weight_decay=weight_decay,
            eps=sophia_eps
        )
    elif optimizer_name == "sophia_h":
        logger.info(f"Using Sophia-H optimizer: lr={sophia_lr}, betas={sophia_betas}, rho={sophia_rho}")
        return SophiaH(
            model_parameters,
            lr=sophia_lr,
            betas=sophia_betas,
            rho=sophia_rho,
            weight_decay=weight_decay,
            eps=sophia_eps
        )
    else:
        return None

def setup_training_args(
    output_dir: str,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    batch_size: int = 4,
    eval_batch_size: int = None,
    gradient_accumulation_steps: int = 4,
    max_steps: int = -1,
    save_steps: int = 100,
    logging_steps: int = 1,
    warmup_steps: int = 100,
    optimizer: str = "adamw_8bit",
    lr_scheduler_type: str = "cosine",
    # Sophia-specific parameters
    sophia_lr: float = None,
    sophia_betas: tuple = (0.965, 0.99),
    sophia_rho: float = 0.04,
    sophia_eps: float = 1e-8,
) -> Dict[str, Any]:
    """Setup training arguments with checkpointing and custom optimizer support."""
    from transformers import TrainingArguments
    
    # Use eval_batch_size if provided, otherwise default to batch_size * 2 for efficiency
    if eval_batch_size is None:
        eval_batch_size = min(batch_size * 2, 8)  # Cap at 8 to avoid OOM during eval
    
    # Handle custom optimizers (Sophia)
    use_custom_optimizer = optimizer in ["sophia", "sophia_h"]
    
    # Store Sophia parameters for later use
    sophia_params = {
        'sophia_lr': sophia_lr if sophia_lr is not None else learning_rate * 0.5,
        'sophia_betas': sophia_betas,
        'sophia_rho': sophia_rho,
        'sophia_eps': sophia_eps
    }
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        optim=optimizer if not use_custom_optimizer else "adamw_8bit",  # Fallback for custom optimizers
        weight_decay=0.01,
        lr_scheduler_type=lr_scheduler_type,
        seed=42,
        
        # Checkpointing
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,  # Keep only 5 most recent checkpoints
        
        # Evaluation for best model tracking
        eval_strategy="steps",
        eval_steps=save_steps,
        
        # Best model saving
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Memory optimizations
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        
        # Progress bar and logging control
        disable_tqdm=False,  # Enable progress bars to show training progress
        log_level="error",  # Reduce HF logging verbosity
        
        # Reporting
        report_to=None,  # Disable wandb/tensorboard
        run_name=f"gpt-oss-finetune-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    
    # Store custom optimizer info in training args for later retrieval
    training_args._custom_optimizer = optimizer if use_custom_optimizer else None
    training_args._sophia_params = sophia_params if use_custom_optimizer else None
    
    return training_args

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS with Unsloth QLoRA")
    parser.add_argument("--model-name", default="unsloth/gpt-oss-20b-bnb-4bit", 
                       help="Model name/path")
    parser.add_argument("--dataset-file", default="gpt_oss_training_dataset.jsonl",
                       help="Training dataset JSONL file")
    parser.add_argument("--output-dir", default="./gpt-oss-finetuned",
                       help="Output directory for checkpoints")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size per device (default: 4)")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                       help="Evaluation batch size per device (default: batch_size * 2, capped at 8)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--save-steps", type=int, default=100,
                       help="Save checkpoint every N steps")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Limit training samples (for testing)")
    parser.add_argument("--resume", action="store_true",
                       help="Auto-resume from latest checkpoint")
    parser.add_argument("--use-mxfp4", action="store_true",
                       help="Use native MXFP4 quantization instead of Unsloth 4-bit")
    parser.add_argument("--eval-split", type=float, default=0.025,
                       help="Fraction of dataset to use for evaluation (default: 0.025)")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Number of warmup steps (default: 100)")
    parser.add_argument("--optimizer", type=str, default="adamw_8bit",
                       choices=["adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused", 
                               "adamw_anyprecision", "adafactor", "adamw_8bit", "lion_8bit", "lion_32bit", 
                               "sophia", "sophia_h"],
                       help="Optimizer to use (default: adamw_8bit)")
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine",
                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial", 
                               "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"],
                       help="Learning rate scheduler type (default: cosine)")
    
    # Sophia-specific parameters
    parser.add_argument("--sophia-lr", type=float, default=None,
                       help="Learning rate for Sophia optimizers (default: learning_rate * 0.5)")
    parser.add_argument("--sophia-beta1", type=float, default=0.965,
                       help="Beta1 for Sophia optimizers (default: 0.965)")
    parser.add_argument("--sophia-beta2", type=float, default=0.99,
                       help="Beta2 for Sophia optimizers (default: 0.99)")
    parser.add_argument("--sophia-rho", type=float, default=0.04,
                       help="Hessian update frequency for Sophia optimizers (default: 0.04)")
    parser.add_argument("--sophia-eps", type=float, default=1e-8,
                       help="Epsilon for Sophia optimizers (default: 1e-8)")
    
    args = parser.parse_args()
    
    # Setup (skip Unsloth if using MXFP4)
    if not args.use_mxfp4:
        FastLanguageModel, _, torch = setup_unsloth()
        if FastLanguageModel is None:
            return
    else:
        logger.info("🔧 Using native MXFP4 mode - skipping Unsloth setup")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for existing checkpoints if resume enabled
    resume_from_checkpoint = None
    if args.resume:
        resume_from_checkpoint = get_latest_checkpoint(args.output_dir)
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            # Clear GPU cache before loading checkpoint to reduce OOM risk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.info("Cleared GPU cache before checkpoint loading")
        else:
            logger.info("No existing checkpoints found, starting fresh")
    
    # Check if model needs dequantization first
    actual_model_path = args.model_name
    
    if args.use_mxfp4:
        # For GPT-OSS models, use Unsloth's pre-optimized version instead of manual dequantization
        logger.info("🔍 MXFP4 GPT-OSS model detected")
        
        # Check if it's a GPT-OSS model and if it's a local path
        if "gpt-oss" in args.model_name.lower():
            # Check if it's a local path (exists on disk)
            if os.path.exists(args.model_name):
                logger.info(f"🔧 Using local GPT-OSS model: {args.model_name}")
                actual_model_path = args.model_name
            else:
                logger.info("⚡ Switching to Unsloth's optimized GPT-OSS model for training")
                
                if "20b" in args.model_name:
                    actual_model_path = "unsloth/gpt-oss-20b-bnb-4bit"
                    logger.info(f"✅ Using Unsloth GPT-OSS-20B: {actual_model_path}")
                elif "120b" in args.model_name:
                    logger.warning("GPT-OSS-120B training requires significant resources")
                    actual_model_path = "unsloth/gpt-oss-120b-bnb-4bit"
                    logger.info(f"✅ Using Unsloth GPT-OSS-120B: {actual_model_path}")
                else:
                    logger.warning(f"Unknown GPT-OSS variant: {args.model_name}")
                    actual_model_path = args.model_name
        else:
            # For non-GPT-OSS MXFP4 models, keep the dequantization approach
            bf16_model_path = args.model_name + "-bf16"
            
            if not os.path.exists(bf16_model_path):
                logger.info("🔍 MXFP4 model detected - checking if dequantization needed...")
                
                # Check if the original model is MXFP4
                try:
                    config_path = os.path.join(args.model_name, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        if 'quantization_config' in config and 'mxfp4' in str(config['quantization_config']).lower():
                            logger.info("⚠️  MXFP4 quantization detected - dequantizing for training...")
                            
                            # Dequantize the model
                            if dequantize_mxfp4_model(args.model_name, bf16_model_path, max_memory_gb=28.0):
                                actual_model_path = bf16_model_path
                                logger.info(f"✅ Using dequantized model: {actual_model_path}")
                            else:
                                logger.error("❌ Failed to dequantize model")
                                return
                            
                except Exception as e:
                    logger.warning(f"Could not check model quantization: {e}")
            else:
                logger.info(f"✅ Using existing dequantized model: {bf16_model_path}")
                actual_model_path = bf16_model_path
    
    # Load model and tokenizer - use Unsloth for BF16 models (can load in 4-bit)
    model, tokenizer = load_model_and_tokenizer(actual_model_path, args.max_seq_length, use_mxfp4=False)
    
    # Load dataset with train/eval split
    train_dataset, eval_dataset = load_dataset(args.dataset_file, tokenizer, args.max_samples, args.eval_split)
    
    # Setup training arguments
    training_args = setup_training_args(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        warmup_steps=args.warmup_steps,
        optimizer=args.optimizer,
        lr_scheduler_type=args.lr_scheduler_type,
        # Sophia parameters
        sophia_lr=args.sophia_lr,
        sophia_betas=(args.sophia_beta1, args.sophia_beta2),
        sophia_rho=args.sophia_rho,
        sophia_eps=args.sophia_eps,
    )
    
    # Setup trainer with evaluation dataset
    from trl import SFTTrainer
    
    # Custom trainer class for Sophia optimizer support
    class SophiaSFTTrainer(SFTTrainer):
        def create_optimizer(self):
            """Override optimizer creation for Sophia support."""
            if hasattr(self.args, '_custom_optimizer') and self.args._custom_optimizer:
                optimizer_name = self.args._custom_optimizer
                sophia_params = self.args._sophia_params
                
                logger.info(f"Creating custom optimizer: {optimizer_name}")
                
                # Get model parameters
                decay_parameters = []
                no_decay_parameters = []
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        if 'bias' in name or 'LayerNorm' in name or 'layer_norm' in name:
                            no_decay_parameters.append(param)
                        else:
                            decay_parameters.append(param)
                
                param_groups = [
                    {"params": decay_parameters, "weight_decay": self.args.weight_decay},
                    {"params": no_decay_parameters, "weight_decay": 0.0}
                ]
                
                # Create Sophia optimizer
                self.optimizer = get_custom_optimizer(
                    param_groups, 
                    optimizer_name, 
                    self.args.learning_rate, 
                    self.args.weight_decay,
                    **sophia_params
                )
                
                if self.optimizer is None:
                    # Fallback to default optimizer creation
                    return super().create_optimizer()
                
                return self.optimizer
            else:
                # Use default optimizer creation
                return super().create_optimizer()
    
    # Choose trainer class based on whether we're using custom optimizers
    trainer_class = SophiaSFTTrainer if hasattr(training_args, '_custom_optimizer') and training_args._custom_optimizer else SFTTrainer
    
    # Different trainer setup for MXFP4 vs Unsloth
    if args.use_mxfp4:
        # MXFP4 mode - use minimal SFTTrainer parameters
        trainer = trainer_class(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
        # Set tokenizer manually
        trainer.tokenizer = tokenizer
    else:
        # Unsloth mode - includes all parameters
        trainer = trainer_class(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            dataset_num_proc=4,
            packing=False,
            args=training_args,
        )
    
    # Add custom callbacks
    best_model_callback = BestModelCallback(args.output_dir, args.use_mxfp4)
    
    trainer.add_callback(best_model_callback)
    
    # Memory stats before training
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = torch.cuda.memory_reserved() / 1024**3
        max_memory = gpu_stats.total_memory / 1024**3
        logger.info(f"GPU memory before training: {start_gpu_memory:.2f}/{max_memory:.2f} GB")
    
    # Additional memory management for resume
    if resume_from_checkpoint:
        logger.info("Applying additional memory optimizations for checkpoint resume...")
        if torch.cuda.is_available():
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Report memory after cleanup
            memory_used = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Memory after cleanup: {memory_used:.2f}/{memory_total:.2f} GB")
    
    # Start training
    logger.info("Starting training...")
    if resume_from_checkpoint:
        logger.info(f"DEBUG: Attempting to resume from: {resume_from_checkpoint}")
        logger.info(f"DEBUG: Checkpoint directory exists: {os.path.exists(resume_from_checkpoint)}")
        # List checkpoint contents for debugging
        if os.path.exists(resume_from_checkpoint):
            checkpoint_files = os.listdir(resume_from_checkpoint)
            logger.info(f"DEBUG: Checkpoint contains files: {checkpoint_files}")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Memory stats after training
    if torch.cuda.is_available():
        final_gpu_memory = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory after training: {final_gpu_memory:.2f}/{max_memory:.2f} GB")
    
    # Save final model
    final_output_dir = os.path.join(args.output_dir, "final-model")
    logger.info(f"Saving final model to {final_output_dir}")
    
    # Save with Unsloth format (includes LoRA adapters)
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    
    # Also save merged model for inference
    merged_output_dir = os.path.join(args.output_dir, "merged-model")
    logger.info(f"Saving merged model to {merged_output_dir}")
    
    # Merge LoRA weights back to base model for inference
    model = FastLanguageModel.for_inference(model)  # Enable inference mode
    model.save_pretrained_merged(merged_output_dir, tokenizer, save_method="merged_16bit")
    
    logger.info("Training completed successfully!")
    logger.info(f"LoRA model: {final_output_dir}")
    logger.info(f"Merged model: {merged_output_dir}")
    if best_model_callback.best_eval_loss != float('inf'):
        logger.info(f"Best model: {best_model_callback.best_model_dir} (loss: {best_model_callback.best_eval_loss:.4f})")
    else:
        logger.warning("No best model saved (no evaluation performed)")
    
    # Generate training summary
    summary = {
        "model_name": args.model_name,
        "dataset_file": args.dataset_file,
        "num_train_samples": len(train_dataset),
        "num_eval_samples": len(eval_dataset) if eval_dataset else 0,
        "best_eval_loss": best_model_callback.best_eval_loss if best_model_callback.best_eval_loss != float('inf') else None,
        "training_args": training_args.to_dict(),
        "output_dirs": {
            "checkpoints": args.output_dir,
            "lora_model": final_output_dir,
            "merged_model": merged_output_dir,
            "best_model": best_model_callback.best_model_dir
        },
        "completion_time": datetime.now().isoformat()
    }
    
    summary_file = os.path.join(args.output_dir, "training_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Training summary saved to {summary_file}")

if __name__ == "__main__":
    main()
