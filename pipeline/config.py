class Config:
    """Configuration settings for the experiment."""

    # Target file - where evolved architectures are written
    SOURCE_FILE: str = "./current_architecture.py"

    # Training script - script that trains and evaluates architectures
    BASH_SCRIPT: str = "./venv-asi-arch/bin/python ./train_architecture.py {name}"

    # Experiment results
    RESULT_FILE: str = "./files/analysis/loss.csv"
    RESULT_FILE_TEST: str = "./files/analysis/benchmark.csv"

    # Debug file
    DEBUG_FILE: str = "./files/debug/training_error.txt"

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
