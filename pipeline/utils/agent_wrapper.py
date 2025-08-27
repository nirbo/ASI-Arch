"""
Agent Output Validation Wrapper

This module provides a wrapper for agent output validation that includes
JSON sanitization to prevent Unicode character parsing errors.
"""

import logging
from typing import Any, Type
from agents.agent_output import AgentOutputSchema
from agents.exceptions import ModelBehaviorError
from agents.util._json import validate_json as original_validate_json
from .json_sanitizer import sanitize_json_string, validate_json_ascii_only

logger = logging.getLogger(__name__)


# Monkey patch the agents library's JSON validation to include sanitization
def sanitized_validate_json(json_str: str, type_adapter, partial: bool):
    """
    Replacement for agents.util._json.validate_json that includes sanitization.
    """
    original_str = json_str
    
    # First, check if the original string has non-ASCII characters
    is_ascii, ascii_error = validate_json_ascii_only(json_str)
    if not is_ascii:
        logger.warning(
            f"Agent output contains non-ASCII characters: {ascii_error}. "
            "Applying Unicode sanitization..."
        )
    
    # Sanitize Unicode characters to ASCII
    sanitized_str = sanitize_json_string(json_str)
    
    if sanitized_str != original_str:
        logger.info(
            f"JSON sanitization applied. Original length: {len(original_str)}, "
            f"sanitized length: {len(sanitized_str)}"
        )
        logger.debug(f"Original JSON: {original_str}")
        logger.debug(f"Sanitized JSON: {sanitized_str}")
    
    try:
        # Call the original validate_json with sanitized string
        return original_validate_json(sanitized_str, type_adapter, partial)
    except ModelBehaviorError as e:
        # Provide more context in error message
        error_msg = str(e)
        if sanitized_str != original_str:
            error_msg += (
                f"\n\nNote: Original JSON contained Unicode characters and was sanitized. "
                f"Unicode characters found: {ascii_error if not is_ascii else 'None detected'}"
            )
        
        raise ModelBehaviorError(error_msg) from e


class SanitizedAgentOutputSchema(AgentOutputSchema):
    """
    Extended AgentOutputSchema that sanitizes JSON input before validation.
    
    This wrapper addresses the issue where AI agents generate Unicode characters
    like en-dashes (‑) instead of regular hyphens (-), which breaks JSON parsing.
    """
    
    def validate_json(self, json_str: str) -> Any:
        """
        Validate JSON with automatic Unicode sanitization.
        
        Args:
            json_str: The JSON string from the agent
            
        Returns:
            The validated object
            
        Raises:
            ModelBehaviorError: If JSON is invalid even after sanitization
        """
        original_str = json_str
        
        # First, check if the original string has non-ASCII characters
        is_ascii, ascii_error = validate_json_ascii_only(json_str)
        if not is_ascii:
            logger.warning(
                f"Agent output contains non-ASCII characters: {ascii_error}. "
                "Applying Unicode sanitization..."
            )
        
        # Sanitize Unicode characters to ASCII
        sanitized_str = sanitize_json_string(json_str)
        
        if sanitized_str != original_str:
            logger.info(
                f"JSON sanitization applied. Original length: {len(original_str)}, "
                f"sanitized length: {len(sanitized_str)}"
            )
            logger.debug(f"Original JSON: {original_str}")
            logger.debug(f"Sanitized JSON: {sanitized_str}")
        
        try:
            # Call the parent validate_json with sanitized string
            return super().validate_json(sanitized_str)
        except ModelBehaviorError as e:
            # Provide more context in error message
            error_msg = str(e)
            if sanitized_str != original_str:
                error_msg += (
                    f"\n\nNote: Original JSON contained Unicode characters and was sanitized. "
                    f"Unicode characters found: {ascii_error if not is_ascii else 'None detected'}"
                )
            
            raise ModelBehaviorError(error_msg) from e


def apply_global_json_sanitization():
    """
    Apply global JSON sanitization by monkey-patching the agents library.
    
    This function replaces the agents library's JSON validation function
    with our sanitized version. This ensures all agent JSON parsing
    goes through sanitization.
    """
    try:
        import agents.util._json
        agents.util._json.validate_json = sanitized_validate_json
        logger.info("Global JSON sanitization applied to agents library")
    except ImportError as e:
        logger.error(f"Failed to apply global JSON sanitization: {e}")


def create_sanitized_agent_output_schema(output_type: Type[Any], strict_json_schema: bool = True) -> SanitizedAgentOutputSchema:
    """
    Factory function to create a sanitized agent output schema.
    
    Args:
        output_type: The type of the output
        strict_json_schema: Whether to use strict JSON schema mode
        
    Returns:
        A sanitized agent output schema instance
    """
    return SanitizedAgentOutputSchema(output_type, strict_json_schema)


def patch_agent_with_sanitization(agent):
    """
    Patch an existing Agent instance to use sanitized output validation.
    
    Args:
        agent: The Agent instance to patch
        
    Returns:
        The patched agent (modified in-place)
    """
    if hasattr(agent, 'output_type') and agent.output_type is not None:
        # Create a sanitized output schema
        if hasattr(agent, '_output_schema'):
            # If the agent already has an output schema, replace it
            original_schema = agent._output_schema
            if isinstance(original_schema, AgentOutputSchema):
                agent._output_schema = SanitizedAgentOutputSchema(
                    original_schema.output_type,
                    original_schema._strict_json_schema
                )
                logger.info(f"Patched agent '{agent.name}' with sanitized output validation")
        else:
            # Create new sanitized schema
            agent._output_schema = SanitizedAgentOutputSchema(agent.output_type)
            logger.info(f"Added sanitized output validation to agent '{agent.name}'")
    
    return agent