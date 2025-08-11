"""
Model Adapter Factory

Creates the appropriate model adapter based on model name and configuration.
Handles automatic detection of Harmony models and fallback strategies.
Integrates with HarmonyServiceManager for automatic service lifecycle management.
"""

import asyncio
import logging
from typing import Optional, Set, Tuple
from .base import ModelAdapter
from .standard import StandardModelAdapter  
from .harmony import HarmonyModelAdapter
from .harmony_service import get_service_manager, HarmonyServiceConfig

logger = logging.getLogger(__name__)


class ModelAdapterFactory:
    """
    Factory class for creating appropriate model adapters.
    
    This factory automatically detects the type of model based on naming patterns
    and configuration, then creates the appropriate adapter (Standard or Harmony).
    """
    
    # Known Harmony model patterns
    HARMONY_MODEL_PATTERNS: Set[str] = {
        "gpt-oss",
        "gpt-oss-20b",
        "gpt-oss-72b",
        "gpt-oss-405b"
    }
    
    @classmethod
    def create_adapter(
        cls,
        api_key: str,
        base_url: str,
        model: str,
        force_harmony: Optional[bool] = None
    ) -> ModelAdapter:
        """
        Create the appropriate model adapter based on model name and configuration.
        
        This is the synchronous version that doesn't handle service startup.
        For automatic service management, use create_adapter_async.
        
        Args:
            api_key: API key for the service
            base_url: Base URL for the API
            model: Model name to use
            force_harmony: Optional flag to force Harmony adapter (True) or Standard (False)
                          If None, auto-detect based on model name
            
        Returns:
            Appropriate ModelAdapter instance
            
        Raises:
            ValueError: If model configuration is invalid
            ImportError: If required dependencies are missing
        """
        try:
            # Check if harmony is explicitly forced
            if force_harmony is True:
                logger.info(f"Force creating Harmony adapter for model: {model}")
                return cls._create_harmony_adapter(api_key, base_url, model)
            
            if force_harmony is False:
                logger.info(f"Force creating Standard adapter for model: {model}")
                return cls._create_standard_adapter(api_key, base_url, model)
            
            # Auto-detect based on model name
            if cls._is_harmony_model(model):
                logger.info(f"Auto-detected Harmony model: {model}")
                return cls._create_harmony_adapter(api_key, base_url, model)
            else:
                logger.info(f"Using Standard adapter for model: {model}")
                return cls._create_standard_adapter(api_key, base_url, model)
                
        except Exception as e:
            logger.error(f"Error creating model adapter: {str(e)}")
            logger.info("Falling back to Standard adapter")
            return cls._create_standard_adapter(api_key, base_url, model)
    
    @classmethod
    async def create_adapter_async(
        cls,
        api_key: str,
        base_url: str,
        model: str,
        force_harmony: Optional[bool] = None,
        auto_start_service: bool = True
    ) -> Tuple[ModelAdapter, bool]:
        """
        Create the appropriate model adapter with automatic service management.
        
        This async version can automatically start harmony services when needed.
        
        Args:
            api_key: API key for the service
            base_url: Base URL for the API
            model: Model name to use
            force_harmony: Optional flag to force Harmony adapter (True) or Standard (False)
                          If None, auto-detect based on model name
            auto_start_service: Whether to automatically start harmony service if needed
            
        Returns:
            Tuple of (ModelAdapter instance, service_started: bool)
            
        Raises:
            ValueError: If model configuration is invalid
            ImportError: If required dependencies are missing
        """
        service_started = False
        
        try:
            # Determine if we need harmony
            need_harmony = force_harmony is True or (
                force_harmony is None and cls._is_harmony_model(model)
            )
            
            if need_harmony and auto_start_service:
                # Try to ensure harmony service is running
                service_manager = get_service_manager()
                success, result_url = await service_manager.ensure_service_running(model)
                
                if success:
                    logger.info(f"Harmony service ready for model {model} at {result_url}")
                    # Use the service URL instead of the provided base_url
                    effective_base_url = result_url
                    service_started = True
                else:
                    logger.warning(f"Failed to start harmony service for {model}: {result_url}")
                    logger.info("Proceeding with provided base_url")
                    effective_base_url = base_url
            else:
                effective_base_url = base_url
            
            # Create adapter using synchronous method
            if force_harmony is True:
                logger.info(f"Force creating Harmony adapter for model: {model}")
                adapter = cls._create_harmony_adapter(api_key, effective_base_url, model)
            elif force_harmony is False:
                logger.info(f"Force creating Standard adapter for model: {model}")
                adapter = cls._create_standard_adapter(api_key, effective_base_url, model)
            elif cls._is_harmony_model(model):
                logger.info(f"Auto-detected Harmony model: {model}")
                adapter = cls._create_harmony_adapter(api_key, effective_base_url, model)
            else:
                logger.info(f"Using Standard adapter for model: {model}")
                adapter = cls._create_standard_adapter(api_key, effective_base_url, model)
            
            return adapter, service_started
                
        except Exception as e:
            logger.error(f"Error creating model adapter: {str(e)}")
            logger.info("Falling back to Standard adapter")
            fallback_adapter = cls._create_standard_adapter(api_key, base_url, model)
            return fallback_adapter, service_started
    
    @classmethod
    def _is_harmony_model(cls, model: str) -> bool:
        """
        Determine if a model should use the Harmony adapter.
        
        Args:
            model: Model name to check
            
        Returns:
            True if this should use Harmony adapter
        """
        model_lower = model.lower()
        
        # Check exact matches
        if model_lower in cls.HARMONY_MODEL_PATTERNS:
            return True
        
        # Check partial matches
        for pattern in cls.HARMONY_MODEL_PATTERNS:
            if pattern in model_lower:
                return True
        
        # Check for gpt-oss prefix
        if model_lower.startswith('gpt-oss'):
            return True
        
        return False
    
    @classmethod
    def _create_standard_adapter(
        cls,
        api_key: str,
        base_url: str,
        model: str
    ) -> StandardModelAdapter:
        """
        Create a Standard model adapter.
        
        Args:
            api_key: API key for OpenAI
            base_url: Base URL for OpenAI API
            model: Model name
            
        Returns:
            StandardModelAdapter instance
        """
        return StandardModelAdapter(api_key, base_url, model)
    
    @classmethod
    def _create_harmony_adapter(
        cls,
        api_key: str,
        base_url: str,
        model: str
    ) -> HarmonyModelAdapter:
        """
        Create a Harmony model adapter.
        
        Args:
            api_key: API key (may be dummy for local models)
            base_url: Base URL for harmony service
            model: Model name
            
        Returns:
            HarmonyModelAdapter instance
            
        Raises:
            ImportError: If openai-harmony library is not available
        """
        try:
            return HarmonyModelAdapter(api_key, base_url, model)
        except ImportError as e:
            logger.error("openai-harmony library not found")
            logger.error("Install with: pip install openai-harmony")
            raise ImportError(
                f"Cannot create Harmony adapter for model '{model}': {str(e)}"
            ) from e
    
    @classmethod
    def get_supported_models(cls) -> dict:
        """
        Get information about supported model types.
        
        Returns:
            Dictionary with model type information
        """
        return {
            "standard_models": {
                "description": "Standard OpenAI-compatible models",
                "examples": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
            },
            "harmony_models": {
                "description": "Harmony-based models with 3-channel parsing", 
                "examples": list(cls.HARMONY_MODEL_PATTERNS)
            }
        }
    
    @classmethod
    def add_harmony_pattern(cls, pattern: str) -> None:
        """
        Add a new pattern for detecting Harmony models.
        
        Args:
            pattern: Model name pattern to add
        """
        cls.HARMONY_MODEL_PATTERNS.add(pattern.lower())
        logger.info(f"Added Harmony model pattern: {pattern}")
    
    @classmethod
    def remove_harmony_pattern(cls, pattern: str) -> None:
        """
        Remove a pattern for detecting Harmony models.
        
        Args:
            pattern: Model name pattern to remove
        """
        cls.HARMONY_MODEL_PATTERNS.discard(pattern.lower())
        logger.info(f"Removed Harmony model pattern: {pattern}")
    
    @classmethod
    def register_service_config(cls, config: HarmonyServiceConfig) -> None:
        """
        Register a service configuration for a specific model.
        
        This allows customizing how harmony services are started for different models.
        
        Args:
            config: Service configuration to register
        """
        service_manager = get_service_manager()
        service_manager.register_service_config(config)
        logger.info(f"Registered service config for model: {config.model}")
    
    @classmethod
    def create_default_service_config(
        cls,
        model: str,
        port: int = 8080,
        host: str = "localhost",
        command: Optional[list] = None
    ) -> HarmonyServiceConfig:
        """
        Create a default service configuration for a model.
        
        Args:
            model: Model name
            port: Port to run service on (default: 8080)
            host: Host to run service on (default: localhost)
            command: Custom command to start service (optional)
            
        Returns:
            Default HarmonyServiceConfig for the model
        """
        if command is None:
            # Create default command based on model
            command = [
                "python", "-m", "harmony_service",
                "--model", model,
                "--host", host,
                "--port", str(port),
                "--timeout", "300"
            ]
        
        return HarmonyServiceConfig(
            model=model,
            host=host,
            port=port,
            command=command
        )
    
    @classmethod
    async def get_service_status(cls, model: Optional[str] = None) -> dict:
        """
        Get status of harmony services.
        
        Args:
            model: Specific model to check (optional, checks all if None)
            
        Returns:
            Dictionary with service status information
        """
        service_manager = get_service_manager()
        if model:
            return await service_manager.get_service_status(model)
        else:
            return await service_manager.get_all_service_status()
    
    @classmethod
    async def shutdown_services(cls) -> bool:
        """
        Shutdown all harmony services managed by the factory.
        
        Returns:
            True if all services shut down successfully
        """
        service_manager = get_service_manager()
        return await service_manager.shutdown_all_services()