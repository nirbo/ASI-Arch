"""Generic provider connector for LLMs and RAG services.

This module loads configuration from `pipeline/config/provider.yml` and exposes helper
methods to retrieve the model parameters and the RAG parameters.  Environment
variables can override the values defined in the YAML file.
"""

from __future__ import annotations

import os
from pathlib import Path
import yaml
from typing import Dict, Any

class ProviderConnector:
    """Load provider configuration and expose it as dictionaries.

    The configuration file is expected to be located at
    ``pipeline/config/provider.yml`` relative to the repository root.
    Environment variables can override the values:

    - ``OPENAI_API_KEY`` overrides ``model.api_key``.
    - ``OPENAI_BASE_URL`` overrides ``model.base_url``.
    - ``OPENAI_MODEL`` overrides ``model.name``.
    """

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "provider.yml"
        self.config = self._load_config(config_path)
        # Environment overrides
        self.config["model"]["api_key"] = os.getenv("OPENAI_API_KEY", self.config["model"].get("api_key"))
        self.config["model"]["base_url"] = os.getenv("OPENAI_BASE_URL", self.config["model"].get("base_url"))
        self.config["model"]["name"] = os.getenv("OPENAI_MODEL", self.config["model"].get("name"))

    def _load_config(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Provider config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_model_params(self) -> Dict[str, Any]:
        return self.config.get("model", {})

    def get_rag_params(self) -> Dict[str, Any]:
        return self.config.get("rag", {})

# Example usage
# connector = ProviderConnector()
# model_params = connector.get_model_params()
# rag_params = connector.get_rag_params()
