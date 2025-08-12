class Config:
    """Configuration settings for the experiment."""
    # Target file - where evolved architectures are written
    SOURCE_FILE: str = "./current_architecture.py"
    
    # Training script - script that trains and evaluates architectures  
    BASH_SCRIPT: str = "/home/nir/ASI-Arch/venv-asi-arch/bin/python /home/nir/ASI-Arch/train_architecture.py {name}"
    
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
    
    # Local Model
    OPENAI_API_KEY: str = "dummy"
    OPENAI_BASE_URL: str = "http://localhost:11434/v1"  # Default OpenAI, change for other providers
    OPENAI_MODEL: str = "magistral"  # Model name: gpt-4o, claude-3-sonnet (OpenRouter), llama3.2 (Ollama), etc.
    
    # Openrouter
    # OPENAI_API_KEY: str = "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"
    # OPENAI_BASE_URL: str = "https://openrouter.ai/api/v1"  # Default OpenAI, change for other providers
    # OPENAI_MODEL: str = "deepseek/deepseek-chat-v3-0324:free"  # Model name: gpt-4o, claude-3-sonnet (OpenRouter), llama3.2 (Ollama), etc.
    
    # Model Adapter Configuration
    # Set to True to force Harmony adapter, False for Standard adapter, None for auto-detection
    FORCE_HARMONY_MODE: bool | None = None  # Auto-detect: gpt-oss models use Harmony, others use Standard
    
    # Harmony Model Configuration
    # Additional patterns to detect Harmony models (beyond default gpt-oss patterns)
    HARMONY_MODEL_PATTERNS: list[str] = []  # e.g., ["custom-harmony-model", "local-gpt-oss"]
    
    # Harmony Service Configuration
    # Whether to automatically start harmony services when harmony models are detected
    # Set to False since we use openai-harmony library directly, not a separate service
    AUTO_START_HARMONY_SERVICES: bool = False
    
    # Default harmony service configuration
    HARMONY_SERVICE_HOST: str = "localhost"
    HARMONY_SERVICE_PORT_START: int = 8080  # Starting port for harmony services
    HARMONY_SERVICE_STARTUP_TIMEOUT: float = 120.0  # Seconds to wait for service startup
    HARMONY_SERVICE_HEALTH_TIMEOUT: float = 10.0  # Seconds for health check timeout
    HARMONY_SERVICE_SHUTDOWN_TIMEOUT: float = 30.0  # Seconds to wait for graceful shutdown
    
    # Custom harmony service command (leave empty for default openai-harmony service)
    HARMONY_SERVICE_COMMAND: list[str] = []
    
    # Environment variables for harmony service processes
    HARMONY_SERVICE_ENVIRONMENT: dict[str, str] = {}
    
    # Working directory for harmony service processes (leave empty for current directory)
    HARMONY_SERVICE_WORKING_DIR: str = ""
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "doubao-embedding-large-text-240915"  # Embedding model for vector search
    
    # Agent Turn Limits Configuration
    # These control how many conversation turns each agent can have before timing out
    # Higher values allow more complex reasoning but take longer to complete
    
    # Evolution agents (most complex architectural tasks)
    MAX_TURNS_PLANNER: int = 50          # Architecture design and innovation
    MAX_TURNS_DEDUPLICATION: int = 50    # Analysis and differentiation from existing work
    MAX_TURNS_MOTIVATION_CHECKER: int = 50  # Motivation comparison and uniqueness validation
    MAX_TURNS_CODE_CHECKER: int = 50    # Code validation and correctness checking (already set)
    
    # Analysis agents
    MAX_TURNS_ANALYZER: int = 50         # Comprehensive result analysis and interpretation
    MAX_TURNS_SUMMARIZER: int = 50       # Context summarization for database elements (increased for complex synthesis)
    
    # Training and debugging agents
    MAX_TURNS_TRAINER: int = 50          # Training script execution and monitoring
    MAX_TURNS_DEBUGGER: int = 50         # Error analysis and code fixing

    RETRY_INTERVAL: int = 5