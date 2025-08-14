"""
Model Adapters Package

This package provides a simplified architecture for handling both standard OpenAI models
and Harmony-based models (gpt-oss) within ASI-Arch.

The architecture consists of:
- ModelAdapter: Abstract base class defining the common interface
- StandardModelAdapter: Handles all models (harmony encoding handled automatically in agents_config.py)
- ModelAdapterFactory: Creates appropriate adapters based on model name
- ModelClientManager: Unified client interface for the pipeline
- HarmonyServiceManager: Manages harmony service lifecycle
- HarmonyServiceConfig: Configuration for harmony services

Note: Harmony models (gpt-oss) now use automatic unsloth encoding via the 
HarmonyAwareAsyncOpenAI wrapper in agents_config.py, eliminating the need for 
a separate HarmonyModelAdapter class.
"""

from .base import ModelAdapter
from .standard import StandardModelAdapter
from .factory import ModelAdapterFactory
from .client_manager import ModelClientManager
from .harmony_service import (
    HarmonyServiceManager, 
    HarmonyServiceConfig,
    get_service_manager,
    ensure_harmony_service,
    shutdown_harmony_services
)

__all__ = [
    "ModelAdapter",
    "StandardModelAdapter", 
    "ModelAdapterFactory",
    "ModelClientManager",
    "HarmonyServiceManager",
    "HarmonyServiceConfig", 
    "get_service_manager",
    "ensure_harmony_service",
    "shutdown_harmony_services"
]