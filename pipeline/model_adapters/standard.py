"""
Standard Model Adapter

Handles traditional OpenAI API models by wrapping the OpenAI AsyncOpenAI client.
This adapter maintains 100% backward compatibility with existing code.
"""

import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .base import ModelAdapter
from ..agents_config import HarmonyAwareAsyncOpenAI

logger = logging.getLogger(__name__)


class StandardModelAdapter(ModelAdapter):
    """
    Adapter for standard OpenAI API models.
    
    This adapter wraps the AsyncOpenAI client and provides the same interface,
    ensuring 100% backward compatibility with existing code that uses OpenAI directly.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str):
        """
        Initialize the standard model adapter.
        
        Args:
            api_key: OpenAI API key
            base_url: Base URL for OpenAI API
            model: Model name to use
        """
        super().__init__(api_key, base_url, model)
        self.client = HarmonyAwareAsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        logger.info(f"Initialized StandardModelAdapter for model: {model}")
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """
        Create a chat completion using the standard OpenAI client.
        
        This method directly delegates to the OpenAI client, maintaining
        full compatibility with existing usage patterns.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to adapter's model)
            max_tokens: Maximum tokens in response  
            temperature: Sampling temperature
            tools: Available tools for the model
            tool_choice: Tool selection preference
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            ChatCompletion object or async iterator for streaming
        """
        try:
            # Use provided model or fall back to adapter's default model
            model_to_use = model or self.model
            
            # Prepare request parameters
            request_params = {
                "model": model_to_use,
                "messages": messages,
                "stream": stream,
                **kwargs
            }
            
            # Add optional parameters if provided
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens
            if temperature is not None:
                request_params["temperature"] = temperature
            if tools is not None:
                request_params["tools"] = tools
            if tool_choice is not None:
                request_params["tool_choice"] = tool_choice
            
            logger.debug(f"Creating chat completion with model: {model_to_use}")
            
            # Make the request using the OpenAI client
            response = await self.client.chat.completions.create(**request_params)
            
            logger.debug(f"Successfully created chat completion")
            return response
            
        except Exception as e:
            logger.error(f"Error creating chat completion: {str(e)}")
            raise
    
    def is_harmony_model(self) -> bool:
        """
        Check if this is a Harmony model adapter.
        
        Returns:
            False, as this is the standard adapter
        """
        return False
    
    async def health_check(self) -> bool:
        """
        Perform a health check on the OpenAI client.
        
        Returns:
            True if the client can successfully make requests
        """
        try:
            # Make a minimal request to test connectivity
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            logger.debug("StandardModelAdapter health check passed")
            return True
        except Exception as e:
            logger.warning(f"StandardModelAdapter health check failed: {str(e)}")
            return False