class Config:
    """Configuration settings for the experiment."""
    # Target file - where evolved architectures are written
    SOURCE_FILE: str = "./current_architecture.py"
    
    # Training script - script that trains and evaluates architectures
    BASH_SCRIPT: str = "python ../train_architecture.py"
    
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
    RAG: str = "your rag url"
    
    # Database URL
    DATABASE: str = "http://localhost:8001"
    
    # OpenAI API Configuration
    OPENAI_API_KEY: str = "dummy"
    OPENAI_BASE_URL: str = "http://192.168.100.100:1234/v1"  # Default OpenAI, change for other providers
    OPENAI_MODEL: str = "openai/gpt-oss-20b"  # Model name: gpt-4o, claude-3-sonnet (OpenRouter), llama3.2 (Ollama), etc.
