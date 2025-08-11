"""
Model Adapters Package

This package provides a dual-mode architecture for handling both standard OpenAI models
and Harmony-based models (gpt-oss) within ASI-Arch.

The architecture consists of:
- ModelAdapter: Abstract base class defining the common interface
- StandardModelAdapter: Handles traditional OpenAI API models
- HarmonyModelAdapter: Handles gpt-oss models with 3-channel parsing
- ModelAdapterFactory: Creates appropriate adapters based on model name
- ModelClientManager: Unified client interface for the pipeline
- HarmonyServiceManager: Manages harmony service lifecycle
- HarmonyServiceConfig: Configuration for harmony services
"""

from .base import ModelAdapter
from .standard import StandardModelAdapter
from .harmony import HarmonyModelAdapter
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
    "HarmonyModelAdapter",
    "ModelAdapterFactory",
    "ModelClientManager",
    "HarmonyServiceManager",
    "HarmonyServiceConfig", 
    "get_service_manager",
    "ensure_harmony_service",
    "shutdown_harmony_services"
]