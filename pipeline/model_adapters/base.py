"""
Base ModelAdapter Abstract Class

Defines the common interface that all model adapters must implement.
This ensures compatibility with the existing OpenAI client interface
while allowing for different underlying implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncIterator
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage as CompletionMessage


class ModelAdapter(ABC):
    """
    Abstract base class for all model adapters.
    
    This class defines the interface that must be implemented by all adapters
    to ensure compatibility with the existing OpenAI client usage patterns.
    """
    
    def __init__(self, api_key: str, base_url: str, model: str):
        """
        Initialize the model adapter.
        
        Args:
            api_key: API key for the service
            base_url: Base URL for the API
            model: Model name to use
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
    
    @abstractmethod
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
        Create a chat completion request.
        
        This method must maintain compatibility with the OpenAI chat completions API.
        Different adapters can implement different underlying mechanisms while
        preserving the same interface.
        
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
        pass
    
    @abstractmethod 
    def is_harmony_model(self) -> bool:
        """
        Check if this adapter handles a Harmony model.
        
        Returns:
            True if this is a Harmony model adapter, False otherwise
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Perform a health check on the adapter.
        
        Returns:
            True if the adapter is healthy and can handle requests
        """
        pass
    
    def _create_completion_response(
        self,
        content: str,
        model: str,
        finish_reason: str = "stop",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        usage_data: Optional[Dict[str, int]] = None
    ) -> ChatCompletion:
        """
        Helper method to create a standardized ChatCompletion response.
        
        Args:
            content: Response content
            model: Model name used
            finish_reason: Reason for completion ending
            tool_calls: Any tool calls made
            usage_data: Token usage information
            
        Returns:
            Formatted ChatCompletion object
        """
        import time
        from openai.types.completion_usage import CompletionUsage
        
        message = ChatCompletionMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls
        )
        
        choice = Choice(
            index=0,
            message=message,
            finish_reason=finish_reason
        )
        
        usage = None
        if usage_data:
            usage = CompletionUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),  
                total_tokens=usage_data.get("total_tokens", 0)
            )
        
        return ChatCompletion(
            id=f"chatcmpl-{int(time.time())}", 
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage
        )