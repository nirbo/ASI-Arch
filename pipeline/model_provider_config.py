"""
Custom MultiProvider configuration for handling OpenRouter models with any prefix.
This module provides a flexible MultiProvider that can handle any model prefix
while maintaining full configurability for API key, base URL, and model names.
"""

from agents.models import MultiProvider, OpenAIProvider
from typing import Dict, Optional
import os


class FlexibleOpenRouterProvider(MultiProvider):
    """
    A flexible MultiProvider that handles any model prefix by mapping them
    to OpenAIProvider configured with OpenRouter settings.
    
    This provider removes the prefix limitation while maintaining full
    configurability for API key, base URL, and model names.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o",
        **kwargs
    ):
        """
        Initialize the flexible provider with OpenRouter configuration.
        
        Args:
            api_key: OpenRouter API key (defaults to environment variable)
            base_url: OpenRouter base URL (defaults to https://openrouter.ai/api/v1)
            default_model: Default model to use when none specified
            **kwargs: Additional arguments passed to OpenAIProvider
        """
        super().__init__()
        
        # Use provided values or fall back to environment/config
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self.default_model = default_model
        
        # Create a single OpenAIProvider for all models
        self._openai_provider = OpenAIProvider(
            api_key=self.api_key,
            base_url=self.base_url,
            **kwargs
        )
    
    def get_model(self, model_name: Optional[str] = None):
        """
        Get a model instance, handling any prefix by stripping it and using
        the underlying OpenAIProvider.
        
        Args:
            model_name: Model name (can include any prefix like "qwen/", "claude/", etc.)
            
        Returns:
            Model instance configured with OpenRouter
        """
        if model_name is None:
            model_name = self.default_model
        
        # Strip any prefix from the model name
        # Handle formats like "prefix/model-name" or just "model-name"
        if "/" in model_name:
            # Extract the actual model name after the prefix
            actual_model = model_name.split("/", 1)[1] if "/" in model_name else model_name
        else:
            actual_model = model_name
        
        # Use the OpenAIProvider with the actual model name
        return self._openai_provider.get_model(actual_model)


# Global instance for easy import and use
_flexible_provider: Optional[FlexibleOpenRouterProvider] = None


def get_flexible_provider(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    default_model: str = "gpt-4o"
) -> FlexibleOpenRouterProvider:
    """
    Get or create the global flexible provider instance.
    
    Args:
        api_key: OpenRouter API key
        base_url: OpenRouter base URL
        default_model: Default model name
        
    Returns:
        Configured FlexibleOpenRouterProvider instance
    """
    global _flexible_provider
    
    if _flexible_provider is None:
        _flexible_provider = FlexibleOpenRouterProvider(
            api_key=api_key,
            base_url=base_url,
            default_model=default_model
        )
    
    return _flexible_provider


def configure_flexible_provider_from_config():
    """
    Configure the flexible provider using values from pipeline.config.Config.
    
    This function imports the Config class and sets up the provider with
    the configured OpenRouter settings.
    """
    try:
        from pipeline.config import Config
        
        return get_flexible_provider(
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
            default_model=Config.OPENAI_MODEL
        )
    except ImportError:
        # Fallback to environment variables if Config is not available
        return get_flexible_provider()