"""
Simplified configuration for the agents library to support any OpenRouter model.
This bypasses the prefix-based MultiProvider entirely.
"""

import os
from typing import Optional, Dict, Any
from agents.models.openai_provider import OpenAIProvider

class OpenRouterProvider(OpenAIProvider):
    """
    A simple OpenAI provider that uses OpenRouter configuration
    and accepts any model name without prefix filtering.
    """
    
    def __init__(self, **kwargs):
        # Ensure OpenRouter configuration
        api_key = kwargs.pop('api_key', None) or os.getenv("OPENAI_API_KEY")
        base_url = kwargs.pop('base_url', None) or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )

def patch_agents_with_openrouter():
    """
    Replace the MultiProvider with a simple OpenRouter provider
    that accepts any model name.
    """
    try:
        from agents.models import multi_provider
        
        # Store original MultiProvider
        original_multi_provider = multi_provider.MultiProvider
        
        # Create a simple replacement that always uses OpenRouter
        class SimpleOpenRouterProvider:
            def __init__(self, **kwargs):
                # Always use OpenRouter configuration
                self.provider = OpenRouterProvider(**kwargs)
            
            def get_model(self, model_name: str):
                # Return the OpenRouter provider for any model name
                return self.provider
            
            async def get_model_async(self, model_name: str):
                # Return the OpenRouter provider for any model name
                return self.provider
        
        # Replace MultiProvider with our simple version
        multi_provider.MultiProvider = SimpleOpenRouterProvider
        
        print("Successfully patched agents library to use OpenRouter for any model name")
        
    except ImportError as e:
        print(f"Warning: Could not patch agents library: {e}")

# Apply the patch when this module is imported
patch_agents_with_openrouter()