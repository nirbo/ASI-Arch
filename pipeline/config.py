import os

class Config:
    """Configuration settings for the experiment."""
    
    # Project root directory
    PROJECT_ROOT: str = "/home/nir/ml-tools/ASI-Arch"

    # Target file - where evolved architectures are written
    SOURCE_FILE: str = os.path.join(PROJECT_ROOT, "pipeline/current_architecture.py")

    # Training script - script that trains and evaluates architectures
    BASH_SCRIPT: str = f"{os.path.join(PROJECT_ROOT, 'venv-asi-arch/bin/python')} {os.path.join(PROJECT_ROOT, 'train_architecture.py')}"

    # Experiment results
    RESULT_FILE: str = os.path.join(PROJECT_ROOT, "files/analysis/loss.csv")
    RESULT_FILE_TEST: str = os.path.join(PROJECT_ROOT, "files/analysis/benchmark.csv")

    # Debug file
    DEBUG_FILE: str = os.path.join(PROJECT_ROOT, "files/debug/training_error.txt")

    # Code pool directory
    CODE_POOL: str = "./pool"

    # Maximum number of debug attempts
    MAX_DEBUG_ATTEMPT: int = 3

    # Maximum number of retry attempts
    MAX_RETRY_ATTEMPTS: int = 10

    # RAG service URL
    RAG: str = "http://localhost:13142"

    # Database URL
    DATABASE: str = "http://localhost:8001"
