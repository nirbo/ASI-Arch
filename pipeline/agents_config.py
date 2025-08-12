"""
Configuration module for the agents library to support OpenRouter models.
This module patches the MultiProvider class to handle any model prefix with OpenRouter settings.
"""

import os
from typing import Optional, Dict, Any
from agents.models.multi_provider import MultiProvider, MultiProviderMap
from agents.models.openai_provider import OpenAIProvider
from agents.models.interface import Model, ModelProvider
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI


class OpenRouterProvider(ModelProvider):
    """Custom provider for OpenRouter that preserves full model names including prefixes."""
    
    def __init__(self, **kwargs):
        self.api_key = kwargs.pop('api_key', None) or os.getenv("OPENAI_API_KEY")
        self.base_url = kwargs.pop('base_url', None) or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self._client = None
        
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._client
        
    def get_model(self, model_name: str | None) -> Model:
        """Get model, preserving the full model name for OpenRouter."""
        if model_name is None:
            raise ValueError("Model name is required")
            
        client = self._get_client()
        
        # For OpenRouter, we need to reconstruct the full model name with prefix
        # Since this provider is only used for specific prefixes, we know what prefix was stripped
        # We'll store the original prefix in the provider
        full_model_name = self._reconstruct_full_model_name(model_name)
        
        return OpenAIChatCompletionsModel(model=full_model_name, openai_client=client)
    
    def _reconstruct_full_model_name(self, stripped_name: str) -> str:
        """Reconstruct the full model name by adding back the prefix."""
        # This is a bit of a hack - we store the prefix that was used to create this provider
        if hasattr(self, '_prefix'):
            return f"{self._prefix}/{stripped_name}"
        # Fallback: try to guess the prefix from common patterns
        if "qwen" in stripped_name.lower():
            return f"qwen/{stripped_name}"
        elif "claude" in stripped_name.lower():
            return f"anthropic/{stripped_name}"
        elif "llama" in stripped_name.lower():
            return f"meta/{stripped_name}"
        else:
            # Default: assume it's already the full name
            return stripped_name


def create_openrouter_provider(prefix=None, **kwargs) -> OpenRouterProvider:
    """
    Create an OpenRouterProvider configured for OpenRouter.
    """
    # Pass API key and base URL explicitly if they're in kwargs
    provider_kwargs = {}
    if 'api_key' not in kwargs:
        provider_kwargs['api_key'] = os.getenv("OPENAI_API_KEY")
    if 'base_url' not in kwargs:
        provider_kwargs['base_url'] = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    provider_kwargs.update(kwargs)
    
    provider = OpenRouterProvider(**provider_kwargs)
    if prefix:
        provider._prefix = prefix
    return provider

def patch_agents_multi_provider():
    """
    Patch the MultiProvider to support OpenRouter models with any prefix.
    This adds support for qwen/ and other OpenRouter model prefixes.
    """
    # Import Config here to avoid circular imports
    try:
        from pipeline.config import Config
        api_key = Config.OPENAI_API_KEY
        base_url = Config.OPENAI_BASE_URL
    except ImportError:
        # Fallback to environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    
    # Create a custom provider map that includes qwen and other prefixes
    provider_map = MultiProviderMap()
    
    # Add common OpenRouter prefixes
    common_prefixes = [
        "qwen", "claude", "anthropic", "google", "gemini", "mistral", 
        "cohere", "meta", "llama", "deepseek", "perplexity"
    ]
    
    for prefix in common_prefixes:
        provider_map.add_provider(prefix, create_openrouter_provider(
            prefix=prefix, 
            api_key=api_key, 
            base_url=base_url
        ))
    
    # Store the original __init__ method
    original_init = MultiProvider.__init__
    
    def patched_init(self, **kwargs):
        # Use our custom provider map if none provided
        if 'provider_map' not in kwargs:
            kwargs['provider_map'] = provider_map
        
        # Ensure OpenRouter configuration
        if 'openai_api_key' not in kwargs:
            kwargs['openai_api_key'] = os.getenv("OPENAI_API_KEY")
        if 'openai_base_url' not in kwargs:
            kwargs['openai_base_url'] = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        if 'openai_use_responses' not in kwargs:
            kwargs['openai_use_responses'] = False  # Use chat completions instead of responses
        
        return original_init(self, **kwargs)
    
    # Apply the patch
    MultiProvider.__init__ = patched_init
    
    print("Successfully patched MultiProvider for OpenRouter compatibility")

# Apply the patch when this module is imported
patch_agents_multi_provider()