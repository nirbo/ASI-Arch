"""
Model Client Manager

Unified interface that replaces direct OpenAI client usage.
Provides seamless switching between Standard and Harmony adapters
while maintaining 100% backward compatibility.
Integrates with HarmonyServiceManager for automatic service lifecycle management.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, AsyncIterator, Tuple
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from .base import ModelAdapter
from .factory import ModelAdapterFactory

logger = logging.getLogger(__name__)


class ModelClientManager:
    """
    Unified client manager that handles both Standard and Harmony models.
    
    This class provides a single interface that can be used as a drop-in
    replacement for AsyncOpenAI client, automatically routing requests
    to the appropriate adapter based on model configuration.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: Optional[str] = None,
        force_harmony: Optional[bool] = None,
        auto_start_services: bool = True
    ):
        """
        Initialize the model client manager.
        
        Args:
            api_key: API key for the service
            base_url: Base URL for the API
            default_model: Default model to use
            force_harmony: Force Harmony (True) or Standard (False) adapter
            auto_start_services: Whether to automatically start harmony services when needed
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.force_harmony = force_harmony
        self.auto_start_services = auto_start_services
        
        # Cache for created adapters to avoid recreating them
        self._adapter_cache: Dict[str, ModelAdapter] = {}
        # Track which services we started so we can manage their lifecycle
        self._managed_services: Dict[str, bool] = {}
        # Lock for concurrent adapter creation
        self._adapter_locks: Dict[str, asyncio.Lock] = {}
        
        # Create chat interface for compatibility - ChatManager acts as "chat"
        self.chat = ChatManager(self)
        
        logger.info(f"Initialized ModelClientManager with default model: {default_model}")
    
    def get_adapter(self, model: Optional[str] = None) -> ModelAdapter:
        """
        Get or create the appropriate adapter for the given model.
        
        This is the synchronous version that doesn't handle service startup.
        For service management, use get_adapter_async.
        
        Args:
            model: Model name (uses default if not provided)
            
        Returns:
            Appropriate ModelAdapter instance
        """
        model_to_use = model or self.default_model
        if not model_to_use:
            raise ValueError("No model specified and no default model configured")
        
        # Check cache first
        if model_to_use in self._adapter_cache:
            return self._adapter_cache[model_to_use]
        
        # Create new adapter
        adapter = ModelAdapterFactory.create_adapter(
            api_key=self.api_key,
            base_url=self.base_url,
            model=model_to_use,
            force_harmony=self.force_harmony
        )
        
        # Cache the adapter
        self._adapter_cache[model_to_use] = adapter
        
        logger.debug(f"Created and cached adapter for model: {model_to_use}")
        return adapter
    
    async def get_adapter_async(self, model: Optional[str] = None) -> ModelAdapter:
        """
        Get or create the appropriate adapter for the given model with service management.
        
        This async version can automatically start harmony services when needed.
        
        Args:
            model: Model name (uses default if not provided)
            
        Returns:
            Appropriate ModelAdapter instance
        """
        model_to_use = model or self.default_model
        if not model_to_use:
            raise ValueError("No model specified and no default model configured")
        
        # Check cache first
        if model_to_use in self._adapter_cache:
            return self._adapter_cache[model_to_use]
        
        # Ensure we have a lock for this model
        if model_to_use not in self._adapter_locks:
            self._adapter_locks[model_to_use] = asyncio.Lock()
        
        # Use lock to prevent concurrent creation of same adapter
        async with self._adapter_locks[model_to_use]:
            # Double-check cache after acquiring lock
            if model_to_use in self._adapter_cache:
                return self._adapter_cache[model_to_use]
            
            # Create new adapter with service management
            adapter, service_started = await ModelAdapterFactory.create_adapter_async(
                api_key=self.api_key,
                base_url=self.base_url,
                model=model_to_use,
                force_harmony=self.force_harmony,
                auto_start_service=self.auto_start_services
            )
            
            # Cache the adapter and track service management
            self._adapter_cache[model_to_use] = adapter
            self._managed_services[model_to_use] = service_started
            
            logger.debug(f"Created and cached adapter for model: {model_to_use} (service_started={service_started})")
            return adapter
    
    async def health_check(self, model: Optional[str] = None) -> bool:
        """
        Perform a health check on the specified model adapter.
        
        Args:
            model: Model to check (uses default if not provided)
            
        Returns:
            True if the adapter is healthy
        """
        try:
            adapter = self.get_adapter(model)
            return await adapter.health_check()
        except Exception as e:
            logger.error(f"Health check failed for model {model}: {str(e)}")
            return False
    
    def is_harmony_model(self, model: Optional[str] = None) -> bool:
        """
        Check if the specified model uses Harmony adapter.
        
        Args:
            model: Model to check (uses default if not provided)
            
        Returns:
            True if the model uses Harmony adapter
        """
        try:
            adapter = self.get_adapter(model)
            return adapter.is_harmony_model()
        except Exception as e:
            logger.error(f"Error checking model type for {model}: {str(e)}")
            return False
    
    def clear_cache(self) -> None:
        """Clear the adapter cache, forcing recreating of adapters."""
        self._adapter_cache.clear()
        self._managed_services.clear()
        logger.info("Cleared adapter cache")
    
    def get_cached_models(self) -> List[str]:
        """
        Get list of models that have cached adapters.
        
        Returns:
            List of cached model names
        """
        return list(self._adapter_cache.keys())
    
    async def get_service_status(self, model: Optional[str] = None) -> dict:
        """
        Get status of harmony services managed by this client.
        
        Args:
            model: Specific model to check (optional, checks all if None)
            
        Returns:
            Dictionary with service status information
        """
        return await ModelAdapterFactory.get_service_status(model)
    
    async def shutdown_managed_services(self) -> bool:
        """
        Shutdown harmony services that were started by this client manager.
        
        Only shuts down services that this instance started, not all services.
        
        Returns:
            True if all managed services shut down successfully
        """
        if not self._managed_services:
            logger.info("No managed services to shut down")
            return True
        
        logger.info(f"Shutting down {len(self._managed_services)} managed services...")
        
        # We could implement selective shutdown, but for now shutdown all
        # since harmony services are typically shared across clients
        result = await ModelAdapterFactory.shutdown_services()
        
        # Clear our tracking
        self._managed_services.clear()
        
        return result
    
    def get_managed_models(self) -> Dict[str, bool]:
        """
        Get models managed by this client and whether we started their services.
        
        Returns:
            Dictionary mapping model names to service_started flags
        """
        return self._managed_services.copy()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with service cleanup."""
        await self.shutdown_managed_services()
    
    @property
    def completions(self):
        """Expose completions interface for backward compatibility."""
        # Return the underlying client's completions if available
        adapter = self.get_adapter()
        client = getattr(adapter, 'client', None)
        if client and hasattr(client, 'completions'):
            return client.completions
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'completions'")


class ChatManager:
    """
    Chat interface that provides the 'chat' namespace for the client.
    
    This class mimics AsyncOpenAI.chat and provides the completions property
    that contains the actual completions interface.
    """
    
    def __init__(self, client_manager: "ModelClientManager"):
        """
        Initialize the chat manager.
        
        Args:
            client_manager: Parent ModelClientManager instance
        """
        self.client_manager = client_manager
        # Create the completions interface
        self.completions = ChatCompletionsManager(client_manager)


class ChatCompletionsManager:
    """
    Chat completions interface that mimics AsyncOpenAI.chat.completions.
    
    This class provides the same interface as the OpenAI client's chat.completions,
    ensuring 100% backward compatibility with existing code.
    """
    
    def __init__(self, client_manager: "ModelClientManager"):
        """
        Initialize the chat completions manager.
        
        Args:
            client_manager: Parent ModelClientManager instance
        """
        self.client_manager = client_manager
    
    async def create(
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
        Create a chat completion using the appropriate adapter.
        
        This method provides the exact same interface as AsyncOpenAI.chat.completions.create,
        ensuring seamless compatibility with existing code.
        
        Args:
            messages: List of message dictionaries
            model: Model to use (defaults to manager's default model)
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
            # Get the appropriate adapter with service management
            adapter = await self.client_manager.get_adapter_async(model)
            
            # Delegate to the adapter
            return await adapter.create_chat_completion(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                stream=stream,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error in chat completion creation: {str(e)}")
            raise


class AsyncModelClientManager(ModelClientManager):
    """
    Async version of ModelClientManager for explicit async usage.
    
    This is an alias for ModelClientManager since all operations are already async.
    Provided for clarity when used in async contexts.
    """
    pass