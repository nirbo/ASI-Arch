#!/bin/bash
# GPT-OSS Fine-tuning Launcher Script with MXFP4 Support

set -e

# Configuration options (modify these as needed)
USE_MXFP4=false  # DeepSeek R1 works better with standard QLoRA
TEST_RUN=false  # Set to true for quick test run

# Training hyperparameters (can be overridden with command line flags)
MANUAL_BATCH_SIZE="32"    # Significantly reduced for checkpoint resume OOM
MANUAL_GRAD_ACCUM="1"    # Increased to maintain effective batch size of 32
MANUAL_EVAL_BATCH="8"     # Reduced for checkpoint resume OOM

# New configurable training parameters
MANUAL_WARMUP_STEPS="50"    # Set to override default (e.g., "200")
MANUAL_OPTIMIZER="lion_8bit"       # Set to override default (e.g., "adamw_torch", "lion_32bit", "adafactor", "sophia", "sophia_h")  
MANUAL_LR_SCHEDULER="cosine"    # Set to override default (e.g., "linear", "cosine_with_restarts")

# Sophia optimizer parameters (only used when MANUAL_OPTIMIZER is "sophia" or "sophia_h")
SOPHIA_LR=""              # Learning rate for Sophia (default: learning_rate * 0.5)
SOPHIA_BETA1="0.965"      # Beta1 for Sophia (default: 0.965)
SOPHIA_BETA2="0.99"       # Beta2 for Sophia (default: 0.99)
SOPHIA_RHO="0.04"         # Hessian update frequency (default: 0.04)
SOPHIA_EPS="1e-8"         # Epsilon for numerical stability (default: 1e-8)

echo "🚀 DeepSeek R1 ASI-Arch Fine-tuning"
echo "==================================="
if [ "$USE_MXFP4" = true ]; then
    echo "Quantization: Native MXFP4 (no additional quantization)"
else
    echo "Quantization: Unsloth QLoRA 4-bit"
fi

# Check if shuffled dataset exists, create if needed
if [ ! -f "gpt_oss_training_dataset_shuffled.jsonl" ]; then
    if [ -f "gpt_oss_training_dataset.jsonl" ]; then
        echo "Creating shuffled dataset..."
        python shuffle_dataset.py gpt_oss_training_dataset.jsonl --output gpt_oss_training_dataset_shuffled.jsonl
    else
        echo "Training dataset not found: gpt_oss_training_dataset.jsonl"
        echo "Dataset contains ASI-Arch agent conversations + Orca Math reasoning samples"
        exit 1
    fi
fi

# Check GPU
nvidia-smi > /dev/null 2>&1 || {
    echo "NVIDIA GPU not detected or nvidia-smi not available"
    exit 1
}

# Get GPU memory
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "🔍 Detected GPU memory: ${GPU_MEMORY} MB"

# Set batch size based on manual override or GPU memory detection
if [ -n "$MANUAL_BATCH_SIZE" ]; then
    BATCH_SIZE=$MANUAL_BATCH_SIZE
    echo "Manual batch_size override: $BATCH_SIZE"
else
    # Auto-detect based on GPU memory and quantization type
    if [ "$USE_MXFP4" = true ]; then
        # MXFP4 is more memory efficient
        if [ "$GPU_MEMORY" -lt 16000 ]; then
            BATCH_SIZE=1
            echo "⚡ MXFP4 small GPU: auto batch_size=$BATCH_SIZE"
        elif [ "$GPU_MEMORY" -lt 24000 ]; then
            BATCH_SIZE=1
            echo "⚡ MXFP4 medium GPU: auto batch_size=$BATCH_SIZE"
        else
            BATCH_SIZE=2
            echo "⚡ MXFP4 large GPU: auto batch_size=$BATCH_SIZE"
        fi
    else
        # Unsloth QLoRA settings
        if [ "$GPU_MEMORY" -lt 16000 ]; then
            BATCH_SIZE=1
            echo "⚡ QLoRA small GPU: auto batch_size=$BATCH_SIZE"
        elif [ "$GPU_MEMORY" -lt 24000 ]; then
            BATCH_SIZE=2
            echo "⚡ QLoRA medium GPU: auto batch_size=$BATCH_SIZE"
        else
            BATCH_SIZE=4
            echo "⚡ QLoRA large GPU: auto batch_size=$BATCH_SIZE"
        fi
    fi
fi

# Set gradient accumulation based on manual override or auto-detection
if [ -n "$MANUAL_GRAD_ACCUM" ]; then
    GRAD_ACCUM=$MANUAL_GRAD_ACCUM
    echo "🎛️  Manual grad_accum override: $GRAD_ACCUM"
else
    # Auto-detect gradient accumulation to maintain effective batch size
    if [ "$USE_MXFP4" = true ]; then
        # MXFP4 gradient accumulation settings
        if [ "$GPU_MEMORY" -lt 16000 ]; then
            GRAD_ACCUM=32
        elif [ "$GPU_MEMORY" -lt 24000 ]; then
            GRAD_ACCUM=16
        else
            GRAD_ACCUM=8
        fi
    else
        # Unsloth QLoRA gradient accumulation settings
        if [ "$GPU_MEMORY" -lt 16000 ]; then
            GRAD_ACCUM=16
        elif [ "$GPU_MEMORY" -lt 24000 ]; then
            GRAD_ACCUM=8
        else
            GRAD_ACCUM=4
        fi
    fi
    echo "🎛️  Auto grad_accum: $GRAD_ACCUM"
fi

# Calculate effective batch size
EFFECTIVE_BATCH=$((BATCH_SIZE * GRAD_ACCUM))
echo "📊 Effective batch size: $EFFECTIVE_BATCH (batch_size=$BATCH_SIZE × grad_accum=$GRAD_ACCUM)"

# Set model name - use HuggingFace directly to avoid tokenizer issues
MODEL_NAME="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
echo "📦 Model: $MODEL_NAME"

# Training parameters
OUTPUT_DIR="./output/deepseek-r1-0528-qwen3-8B-finetuned"
EPOCHS=3
LEARNING_RATE="2e-4"
SAVE_STEPS=50

# Set training optimization parameters with defaults
WARMUP_STEPS=${MANUAL_WARMUP_STEPS:-100}
OPTIMIZER=${MANUAL_OPTIMIZER:-"adamw_8bit"}
LR_SCHEDULER=${MANUAL_LR_SCHEDULER:-"cosine"}

echo "💾 Output directory: $OUTPUT_DIR"
echo "🎯 Epochs: $EPOCHS"
echo "📈 Learning rate: $LEARNING_RATE"
echo "💾 Save every: $SAVE_STEPS steps"
echo "🔥 Warmup steps: $WARMUP_STEPS"
echo "⚡ Optimizer: $OPTIMIZER"
echo "📊 LR scheduler: $LR_SCHEDULER"

# Test run settings
if [ "$TEST_RUN" = true ]; then
    echo "🧪 Test run mode enabled - using 1 epoch, 100 samples"
fi

# Install dependencies if needed
# echo "📦 Checking dependencies..."
#python -c "import unsloth" 2>/dev/null || {
#    echo "Installing Unsloth..."
#    pip install "unsloth[cu128-torch270] @ git+https://github.com/unslothai/unsloth.git"
#}

# python -c "import trl" 2>/dev/null || {
#     echo "Installing TRL..."
#     pip install trl
# }

# Start training
echo ""
echo "Starting fine-tuning..."
echo "Press Ctrl+C to stop training (checkpoints will be saved)"
echo ""

# Build training command
TRAIN_CMD="python train_gpt_oss_unsloth.py \
    --model-name \"$MODEL_NAME\" \
    --dataset-file \"gpt_oss_training_dataset_shuffled.jsonl\" \
    --output-dir \"$OUTPUT_DIR\" \
    --batch-size $BATCH_SIZE \
    --gradient-accumulation-steps $GRAD_ACCUM \
    --learning-rate $LEARNING_RATE \
    --num-epochs $EPOCHS \
    --save-steps $SAVE_STEPS \
    --warmup-steps $WARMUP_STEPS \
    --optimizer $OPTIMIZER \
    --lr-scheduler-type $LR_SCHEDULER \
    --resume"

# Add Sophia-specific parameters if using Sophia optimizer
if [[ "$OPTIMIZER" == "sophia" || "$OPTIMIZER" == "sophia_h" ]]; then
    echo "🧠 Sophia optimizer detected - adding Sophia parameters"
    
    if [ -n "$SOPHIA_LR" ]; then
        TRAIN_CMD="$TRAIN_CMD --sophia-lr $SOPHIA_LR"
        echo "   Sophia LR: $SOPHIA_LR"
    fi
    
    TRAIN_CMD="$TRAIN_CMD --sophia-beta1 $SOPHIA_BETA1"
    TRAIN_CMD="$TRAIN_CMD --sophia-beta2 $SOPHIA_BETA2"
    TRAIN_CMD="$TRAIN_CMD --sophia-rho $SOPHIA_RHO"
    TRAIN_CMD="$TRAIN_CMD --sophia-eps $SOPHIA_EPS"
    
    echo "   Sophia Beta1: $SOPHIA_BETA1"
    echo "   Sophia Beta2: $SOPHIA_BETA2"
    echo "   Sophia Rho: $SOPHIA_RHO"
    echo "   Sophia Eps: $SOPHIA_EPS"
fi

# Add eval batch size if manually specified
if [ -n "$MANUAL_EVAL_BATCH" ]; then
    TRAIN_CMD="$TRAIN_CMD --eval-batch-size $MANUAL_EVAL_BATCH"
    echo "🎛️  Manual eval_batch_size: $MANUAL_EVAL_BATCH"
else
    # Let the script auto-calculate eval batch size (batch_size * 2, capped at 8)
    AUTO_EVAL_BATCH=$((BATCH_SIZE * 2))
    if [ $AUTO_EVAL_BATCH -gt 8 ]; then
        AUTO_EVAL_BATCH=8
    fi
    echo "🎛️  Auto eval_batch_size: $AUTO_EVAL_BATCH"
fi

# Add MXFP4 flag if enabled
if [ "$USE_MXFP4" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use-mxfp4"
fi

# Add test run flag if enabled
if [ "$TEST_RUN" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --max-samples 100"
    echo "🧪 Running test training with 100 samples..."
else
    echo "🚀 Running full training..."
fi

# Display final command
echo "📋 Training command:"
echo "   $TRAIN_CMD"
echo ""

# Execute training
eval $TRAIN_CMD

echo ""
echo "✅ Training completed!"
echo "📁 Models saved to: $OUTPUT_DIR"
echo ""
echo "🎯 Next steps:"
echo "  1. Test the fine-tuned model with ASI-Arch"
echo "  2. Compare performance with original GPT-OSS"
echo "  3. Use merged model for inference: $OUTPUT_DIR/merged-model"
