#!/usr/bin/env bash

# This script installs the correct dependencies for the ASI-Arch pipeline.

echo "Installing ASI-Arch dependencies..."

# Install uv if not present
if [ ! $(which uv) ]; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Determine numpy version if it's already installed
if python -c "import numpy" &> /dev/null; then
    INSTALL_NUMPY="numpy==$(python -c 'import numpy; print(numpy.__version__)')"
else
    INSTALL_NUMPY="numpy==2.1.3"
fi

echo "Installing Unsloth dependencies..."

# Install unsloth
uv pip install \
"unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
"unsloth[base] @ git+https://github.com/unslothai/unsloth"

echo "Installing core dependencies..."

# Install main packages
uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.8.0+cu128" \
    "torchaudio==2.8.0+cu128" \
    "torchvision==0.23.0+cu128" \
    "triton>=3.4.0" \
    "$INSTALL_NUMPY" \
    "transformers==4.52.4" \
    "tokenizers==0.21.1" \
    "safetensors==0.5.3" \
    "datasets==3.6.0" \
    "bitsandbytes==0.47.0" \
    "huggingface-hub==0.32.4" \
    "openai==1.84.0" \
    "openai-agents==0.0.17" \
    "openai-harmony" \
    "pydantic==2.11.5" \
    "pydantic-settings==2.9.1" \
    "fastapi" \
    "uvicorn==0.34.3" \
    "aiohttp==3.12.9" \
    "requests==2.32.3" \
    "httpx==0.28.1" \
    "PyYAML==6.0.2" \
    "python-dotenv==1.1.0" \
    "click==8.2.1" \
    "rich==14.0.0" \
    "tqdm==4.67.1" \
    "wandb==0.20.1" \
    "tensorboard==2.19.0" \
    "matplotlib==3.10.3" \
    "seaborn==0.13.2" \
    "pandas==2.3.0" \
    "scipy==1.15.3" \
    "pytorch-lightning==2.5.1.post0" \
    "torchmetrics==1.7.2" \
    "einops==0.8.1" \
    "flash-linear-attention==0.2.2" \
    "gym==0.26.2" \
    "ray==2.46.0" \
    "mcp==1.9.3" \
    "psutil==7.0.0" \
    "packaging==25.0" \
    "filelock==3.18.0" \
    "fsspec==2025.3.0" \
    "pillow==11.2.1" \
    "regex==2024.11.6" \
    "sympy==1.13.1" \
    "networkx==3.3" \
    "jsonschema==4.24.0" \
    "Jinja2==3.1.6" \
    "MarkupSafe==3.0.2" \
    "certifi==2025.4.26" \
    "urllib3==2.4.0" \
    "idna==3.10" \
    "charset-normalizer==3.4.2" \
    "anyio==4.9.0" \
    "sniffio==1.3.1" \
    "typing_extensions==4.14.0" \
    "six==1.17.0"

echo "Installing ML-specific dependencies..."

# Install additional ML dependencies
uv pip install \
    "causal-conv1d" \
    "mamba-ssm" \
    "tensorflow==2.19.0" \
    "tf_keras==2.19.0" \
    "keras==3.10.0" \
    "tensorflow-probability==0.25.0" \
    "absl-py==2.3.0" \
    "astunparse==1.6.3" \
    "flatbuffers==25.2.10" \
    "gast==0.6.0" \
    "google-pasta==0.2.0" \
    "grpcio==1.72.1" \
    "h5py==3.14.0" \
    "libclang==18.1.1" \
    "Markdown==3.8" \
    "ml_dtypes==0.5.1" \
    "opt_einsum==3.4.0" \
    "protobuf==5.29.5" \
    "tensorboard-data-server==0.7.2" \
    "tensorflow-io-gcs-filesystem==0.37.1" \
    "termcolor==3.1.0" \
    "wrapt==1.17.2" \
    "dm-tree==0.1.9" \
    "optree==0.16.0" \
    "namex==0.1.0"

echo "Installing CUDA dependencies..."

# NVIDIA CUDA 12.8 dependencies for RTX 5090 (Blackwell)
uv pip install \
    "nvidia-cublas-cu12>=12.8.0" \
    "nvidia-cuda-cupti-cu12>=12.8.0" \
    "nvidia-cuda-nvrtc-cu12>=12.8.0" \
    "nvidia-cuda-runtime-cu12>=12.8.0" \
    "nvidia-cudnn-cu12>=9.1.0" \
    "nvidia-cufft-cu12>=11.4.0" \
    "nvidia-curand-cu12>=10.3.5" \
    "nvidia-cusolver-cu12>=11.6.1" \
    "nvidia-cusparse-cu12>=12.5.0" \
    "nvidia-nccl-cu12>=2.21.5" \
    "nvidia-nvjitlink-cu12>=12.8.0" \
    "nvidia-nvtx-cu12>=12.8.0"

echo "Installing development and utility dependencies..."

# Development and utility dependencies
uv pip install \
    "GitPython==3.1.44" \
    "gitdb==4.0.12" \
    "smmap==5.0.2" \
    "griffe==1.7.3" \
    "lightning-utilities==0.14.3" \
    "sentry-sdk==2.29.1" \
    "setproctitle==1.3.6" \
    "hf-xet==1.1.3" \
    "markdown-it-py==3.0.0" \
    "mdurl==0.1.2" \
    "Pygments==2.19.1" \
    "ninja==1.11.1.4" \
    "mpmath==1.3.0" \
    "msgpack==1.1.0" \
    "cloudpickle==3.1.1" \
    "colorama==0.4.6" \
    "decorator==5.2.1" \
    "dill==0.3.8" \
    "distro==1.9.0" \
    "frozenlist==1.6.2" \
    "multidict==6.4.4" \
    "multiprocess==0.70.16" \
    "platformdirs==4.3.8" \
    "propcache==0.3.1" \
    "pyarrow==20.0.0" \
    "pydantic_core==2.33.2" \
    "pyparsing==3.2.3" \
    "python-dateutil==2.9.0.post0" \
    "python-multipart==0.0.20" \
    "pytz==2025.2" \
    "referencing==0.36.2" \
    "rpds-py==0.25.1" \
    "ruamel.yaml==0.18.13" \
    "ruamel.yaml.clib==0.2.12" \
    "xxhash==3.5.0" \
    "yarl==1.20.0"

echo "Installing optional visualization dependencies..."

# Visualization dependencies
uv pip install \
    "contourpy==1.3.2" \
    "cycler==0.12.1" \
    "fonttools==4.58.2" \
    "kiwisolver==1.4.8"

echo "Installing additional utility packages..."

# Additional utility packages
uv pip install \
    "aiohappyeyeballs==2.6.1" \
    "aiosignal==1.3.2" \
    "annotated-types==0.7.0" \
    "attrs==25.3.0" \
    "h11==0.16.0" \
    "httpcore==1.0.9" \
    "httpx-sse==0.4.0" \
    "jiter==0.10.0" \
    "jsonschema-specifications==2025.4.1" \
    "sse-starlette==2.3.6" \
    "starlette==0.47.0" \
    "types-requests==2.32.0.20250602" \
    "typing-inspection==0.4.1" \
    "tzdata==2025.2" \
    "Werkzeug==3.1.3"

echo ""
echo "Dependencies installation completed successfully!"
echo ""
echo "To activate the environment and verify installation:"
echo "  source venv/bin/activate  # or your venv path"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'"
echo "  python -c 'import transformers; print(f\"Transformers: {transformers.__version__}\")'"